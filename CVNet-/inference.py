import sys
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils.data_loading import BasicDataset
import logging
from utils.path_hyperparameter import ph
import torch
from torchmetrics import MetricCollection
from torchmetrics.classification import  BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score,BinaryJaccardIndex
from utils.dataset_process import compute_mean_std
from tqdm import tqdm
import open_clip

import numpy as np
from models_bit.networks_me_multiscale_visualpromt import BASE_Transformer
import os
import cv2
from models_bit.promot_vit import PromotVisionTransformer

def train_net(dataset_name, load_checkpoint=True):
    # 1. Create dataset

    # compute mean and std of train dataset to normalize train/val/test dataset
    # t1_mean, t1_std = compute_mean_std(images_dir=f'./{dataset_name}/train/t1/')
    # t2_mean, t2_std = compute_mean_std(images_dir=f'./{dataset_name}/train/t2/')

    # levir_corp
    t1_mean,t1_std=np.asarray([0.45026044,0.44666811,0.38134658]),np.asarray([0.17456748,0.16490024,0.15318057])
    t2_mean, t2_std =np.asarray ([0.34552285,0.33819558,0.28881546]),np.asarray([0.12937804,0.12601846,0.1187869])


    # SYSU
    # t1_mean,t1_std=np.asarray([0.39659575,0.52846196,0.46540029]),np.asarray([0.20213537,0.15811189,0.15296703])
    # t2_mean, t2_std =np.asarray ([0.40202364,0.48766127,0.39895688]),np.asarray([0.18235275,0.15682769,0.1543715])

    #WHU
    # t1_mean,t1_std=np.asarray([0.48047181,0.44232931,0.38513029]),np.asarray([0.14913468,0.14363994,0.14289249])
    # t2_mean, t2_std =np.asarray ([0.48009379,0.48005141,0.45689122]),np.asarray([0.17925962,0.16984095,0.17603727])

    #CDD
    # t1_mean,t1_std=np.asarray([0.35390041,0.39103761,0.34307468]),np.asarray([0.15327417,0.15476148,0.14288178])
    # t2_mean, t2_std =np.asarray ([0.4732457 ,0.49861176,0.46873822]),np.asarray([0.16063922,0.16394015,0.15757924])


    dataset_args = dict(t1_mean=t1_mean, t1_std=t1_std, t2_mean=t2_mean, t2_std=t2_std)
    test_dataset = BasicDataset(t1_images_dir=f'./{dataset_name}/test/t1/',
                                t2_images_dir=f'./{dataset_name}/test/t2/',
                                labels_dir=f'./{dataset_name}/test/label/',
                                train=False, **dataset_args)
    # 2. Create data loaders
    loader_args = dict(num_workers=1,
                       prefetch_factor=2,
                       persistent_workers=True
                       )
    test_loader = DataLoader(test_dataset, shuffle=False, drop_last=False,
                             batch_size=ph.batch_size * ph.inference_ratio, **loader_args)

    # 3. Initialize logging
    logging.basicConfig(level=logging.INFO)

    # 4. Set up device, model, metric calculator
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logging.basicConfig(level=logging.INFO)
    logging.info(f'Using device {device}')


    model, _, _ = open_clip.create_model_and_transforms(
        'ViT-B-16-SigLIP-256',
        device=device,

    )

    weights_path = '/home/zhengzhiyong/.cache/huggingface/hub/models--timm--ViT-B-16-SigLIP-256/snapshots/149fecaed17d2230c5b631c8b66f93cbfabcfcb9/open_clip_pytorch_model.bin'  # 请替换为你的实际权重文件路径

    state_dict = torch.load(weights_path, map_location=device)
    #
    # # 将权重应用到模型
    # # 如果权重文件中的键与模型的键不完全匹配，可能需要进行一些键的重命名或删除
    model.load_state_dict(state_dict, strict=False)
    #
    # # # # tokenizer = open_clip.get_tokenizer('hf-hub:timm/ViT-B-16-SigLIP-256')
    visual_clip = model.visual.trunk
    pro_vit = PromotVisionTransformer().to(device=device).to(device=device)
    pro_vit.load_parent_params(visual_clip)
    # # # net = BASE_Transformer(input_nc=3, output_nc=1, token_len=4, resnet_stages_num=4,
    # # #                          with_pos='learned', enc_depth=1, dec_depth=8, decoder_dim_head=8)
    net = BASE_Transformer(input_nc=3, output_nc=1, token_len=4, resnet_stages_num=4,
                             with_pos='learned', enc_depth=1, dec_depth=8, decoder_dim_head=8,visual_clip=pro_vit)

    net=net.to(device=device)
    # before_update = net.clip_visual.prompt_embeddings.clone()
    # print(ph.load)
    assert ph.load, 'Loading model error, checkpoint ph.load'
    load_model = torch.load(ph.load, map_location=device)
    if load_checkpoint:
        net.load_state_dict(load_model['net'])
    else:
        net.load_state_dict(load_model,strict=False)
    logging.info(f'Model loaded from {ph.load}')
    torch.save(net.state_dict(), f'{dataset_name}_best_model.pth')


    metric_collection = MetricCollection({
        'accuracy': BinaryAccuracy().to(device),
        'precision': BinaryPrecision().to(device),
        'recall': BinaryRecall().to(device),
        'f1score': BinaryF1Score().to(device),
        'IoU':BinaryJaccardIndex().to(device)
    })  # metrics calculator

    net.eval()
    with torch.no_grad():
        for batch_img1, batch_img2, labels, names in tqdm(test_loader):
            batch_img1 = batch_img1.float().to(device)
            batch_img2 = batch_img2.float().to(device)
            labels = labels.float().to(device)
            # print(name)
            cd_preds = net(batch_img1, batch_img2)
            clip_preds=torch.sigmoid(cd_preds[1])
            cd_preds = torch.sigmoid(cd_preds[0])

            


            # Calculate and log other batch metrics
            cd_preds = cd_preds.float()
            # print(cd_preds.shape)
            labels = labels.int().unsqueeze(1)
            cd_preds=torch.round(cd_preds)
            metric_collection.update(cd_preds, labels)
            # print(f"Metrics on all data: {test_metrics}")
            batch_metrics = metric_collection.forward(cd_preds, labels)  # compute metric
            # print(f"Metrics on all data: {batch_metrics}")
            # print(name)



            # clear batch variables from memory
            del batch_img1, batch_img2, labels

        test_metrics = metric_collection.compute()
        print(f"Metrics on all data: {test_metrics}")
        metric_collection.reset()


    print('over')


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    try:
        train_net(dataset_name='levir_crop', load_checkpoint=False)
    except KeyboardInterrupt:
        logging.info('Error')
        sys.exit(0)