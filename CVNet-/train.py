import torch
import torch.nn as nn
from torchvision.models import resnet18
from thop import profile
import sys
import timm
import time
import open_clip
import ipdb
import numpy as np
from torch import optim
import torchvision.transforms as T
from torch.utils.data import DataLoader
from utils.data_loading import BasicDataset
from utils.path_hyperparameter import ph
import torch
from utils.losses import FCCDN_loss_without_seg
from utils.losses import cross_entropy_loss
from utils.losses import BCE_loss_without_seg
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import logging
import random
import wandb
from torchmetrics import MetricCollection
from torchmetrics.classification import  BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score,BinaryJaccardIndex
from utils.utils import train_val
from utils.dataset_process import compute_mean_std
from utils.dataset_process import image_shuffle, split_image
import onnx
import onnx.utils
import onnx.version_converter
import netron
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
from torch.optim.lr_scheduler import LambdaLR
import math
from models_bit.networks_me_multiscale_visualpromt import BASE_Transformer
from models_bit.promot_vit import PromotVisionTransformer

def lr_lambda(epoch):
    if epoch < 30:
        return 1.0
    else:
        return max(0.0, 1.0 - (epoch - 30) * 1e-3)

class DataLoaderX(DataLoader):
    """Using prefetch_generator to accelerate data loading
    Parameter:
        DataLoader(class): torch.utils.data.DataLoader.
    """

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def random_seed(SEED):
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    torch.backends.cudnn.deterministic = True  # keep convolution algorithm deterministic
    # torch.backends.cudnn.benchmark = False  # using fixed convolution algorithm to accelerate training
    # if model and input are fixed, set True to search better convolution algorithm
    torch.backends.cudnn.benchmark = True

def auto_experiment():
    random_seed(SEED=ph.random_seed)
    try:
        train_net(dataset_name=ph.dataset_name)
    except KeyboardInterrupt:
        logging.info('Interrupt')
        sys.exit(0)


def train_net(dataset_name):


    """
    This is the workflow of training model and evaluating model,
    note that the dataset should be organized as
    :obj:`dataset_name`/`train` or `val`/`t1` or `t2` or `label`

    Parameter:
        dataset_name(str): name of dataset

    Return:
        return nothing
    """
    # 1. Create dataset, checkpoint and best model path

    # compute mean and std of train dataset to normalize train/val dataset
    t1_mean, t1_std = compute_mean_std(images_dir=f'./{dataset_name}/train/t1/')
    t2_mean, t2_std = compute_mean_std(images_dir=f'./{dataset_name}/train/t2/')
    # t1_mean,t1_std=np.asarray([0.45026044,0.44666811,0.38134658]),np.asarray([0.17456748,0.16490024,0.15318057])
    # t2_mean, t2_std =np.asarray ([0.34552285,0.33819558,0.28881546]),np.asarray([0.12937804,0.12601846,0.1187869])

    # dataset path should be dataset_name/train or val/t1 or t2 or label
    dataset_args = dict(t1_mean=t1_mean, t1_std=t1_std, t2_mean=t2_mean, t2_std=t2_std)
    train_dataset = BasicDataset(t1_images_dir=f'./{dataset_name}/train/t1/',
                                 t2_images_dir=f'./{dataset_name}/train/t2/',
                                 labels_dir=f'./{dataset_name}/train/label/',
                                 train=True, **dataset_args)
    val_dataset = BasicDataset(t1_images_dir=f'./{dataset_name}/val/t1/',
                               t2_images_dir=f'./{dataset_name}/val/t2/',
                               labels_dir=f'./{dataset_name}/val/label/',
                               train=False, **dataset_args)

    # 2. Markdown dataset size
    n_train = len(train_dataset)
    n_val = len(val_dataset)

    # 3. Create data loaders

    loader_args = dict(num_workers=4,
                       prefetch_factor=5,
                       persistent_workers=True,
                       pin_memory=True,
                       )
    train_loader = DataLoaderX(train_dataset, shuffle=True, drop_last=False, batch_size=ph.batch_size, **loader_args)
    val_loader = DataLoaderX(val_dataset, shuffle=False, drop_last=False,
                             batch_size=ph.batch_size * ph.inference_ratio, **loader_args)

    # 4. Initialize logging

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # working device
    logging.basicConfig(level=logging.INFO)
    localtime = time.asctime(time.localtime(time.time()))
    hyperparameter_dict = ph.state_dict()
    hyperparameter_dict['time'] = localtime
    # using wandb to log hyperparameter, metrics and output
    # resume=allow means if the id is identical with the previous one, the run will resume
    # (anonymous=must) means the id will be anonymous

    logging.info(f'''Starting training:
        Epochs:          {ph.epochs}
        Batch size:      {ph.batch_size}
        Learning rate:   {ph.learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {ph.save_checkpoint}
        save best model: {ph.save_best_model}
        Device:          {device.type}
        Mixed Precision: {ph.amp}
    ''')

    # 5. Set up model, optimizer, warm_up_scheduler, learning rate scheduler, loss function and other things
    torch.backends.cudnn.enabled = True

    model, _, _ = open_clip.create_model_and_transforms(
        'ViT-B-16-SigLIP-256',
        device=device,
    )
    weights_path = '/home/zhengzhiyong/.cache/huggingface/hub/models--timm--ViT-B-16-SigLIP-256/snapshots/149fecaed17d2230c5b631c8b66f93cbfabcfcb9/open_clip_pytorch_model.bin'  # 请替换为你的实际权重文件路径
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    # # # tokenizer = open_clip.get_tokenizer('hf-hub:timm/ViT-B-16-SigLIP-256')
    visual_clip=model.visual.trunk
    pro_vit=PromotVisionTransformer().to(device=device).to(device=device)
    pro_vit.load_parent_params(visual_clip)
    net = BASE_Transformer(input_nc=3, output_nc=1, token_len=4, resnet_stages_num=4,
                           with_pos='learned', enc_depth=1, dec_depth=8, decoder_dim_head=8, visual_clip=pro_vit)

    net = net.to(device=device)

    optimizer = optim.AdamW([
    {'params': [net.clip_visual.deep_prompt_embeddings], 'lr': ph.learning_rate * 100},  # deep_prompt_embeddings, 100倍学习率
    {'params': [net.clip_visual.prompt_embeddings], 'lr': ph.learning_rate * 100},  # prompt_embeddings, 100倍学习率
    {'params': [param for name, param in net.named_parameters()
                if name != 'clip_visual.deep_prompt_embeddings' and name != 'clip_visual.prompt_embeddings'],
     'lr': ph.learning_rate, 'weight_decay': ph.weight_decay}  # 其他参数，使用默认学习率和权重衰减
    ])
    #11

    #
    optimizer = optim.AdamW(net.parameters(), lr=ph.learning_rate,
                            weight_decay=ph.weight_decay)  # optimizer
    warmup_lr = np.arange(1e-7, ph.learning_rate,
                          (ph.learning_rate - 1e-7) / ph.warm_up_step)  # warm up learning rate
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=ph.patience,
                                                     factor=ph.factor)  # learning rate scheduler
    grad_scaler = torch.cuda.amp.GradScaler()  # loss scaling for amp
    if ph.load:
        checkpoint = torch.load(ph.load, map_location=device)
        net.load_state_dict(checkpoint)
        # net.load_state_dict(checkpoint['net'])
        # logging.info(f'Model loaded from {ph.load}')
        # if 'optimizer' in checkpoint.keys():
        #     optimizer.load_state_dict(checkpoint['optimizer'])
        #     for g in optimizer.param_groups:
        #         g['lr'] = ph.learning_rate
        #     optimizer.param_groups[0]['capturable'] = True

    total_step = 0  # logging step
    lr = ph.learning_rate  # learning rate

    criterion = FCCDN_loss_without_seg  # loss function

    best_metrics = dict.fromkeys(['best_f1score', 'lowest loss'], 0)  # best evaluation metrics



    metric_collection = MetricCollection({
        'accuracy': BinaryAccuracy().to(device),
        'precision': BinaryPrecision().to(device),
        'recall': BinaryRecall().to(device),
        'f1score': BinaryF1Score().to(device),
        'IoU':BinaryJaccardIndex().to(device)
    })  # metrics calculator

    to_pilimg = T.ToPILImage()  # convert to PIL image to log in wandb

    # model saved path
    checkpoint_path = f'./{dataset_name}_checkpoint/'
    best_f1score_model_path = f'./{dataset_name}bit_best_f1score_model/'
    best_loss_model_path = f'./{dataset_name}_best_loss_model/'

    non_improved_epoch = 0  # adjust learning rate when non_improved_epoch equal to patience


    for epoch in range(ph.epochs):

        net, optimizer, grad_scaler, total_step, lr = \
            train_val(
                mode='train', dataset_name=dataset_name,
                dataloader=train_loader, device=device,  net=net,
                optimizer=optimizer, total_step=total_step, lr=lr, criterion=criterion,
                metric_collection=metric_collection, to_pilimg=to_pilimg, epoch=epoch,
                warmup_lr=warmup_lr, grad_scaler=grad_scaler,lr_scheduler=scheduler
            )

        if epoch >= ph.evaluate_epoch:
            with torch.no_grad():
                net, optimizer, total_step, lr, best_metrics, non_improved_epoch = \
                    train_val(
                        mode='val', dataset_name=dataset_name,
                        dataloader=val_loader, device=device, net=net,
                        optimizer=optimizer, total_step=total_step, lr=lr, criterion=criterion,
                        metric_collection=metric_collection, to_pilimg=to_pilimg, epoch=epoch,
                        best_metrics=best_metrics, checkpoint_path=checkpoint_path,
                        best_f1score_model_path=best_f1score_model_path, best_loss_model_path=best_loss_model_path,
                        non_improved_epoch=non_improved_epoch
                    )


if __name__ == '__main__':

    auto_experiment()