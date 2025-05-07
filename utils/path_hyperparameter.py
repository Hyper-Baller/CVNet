class Path_Hyperparameter:
    # random_seed = 803
    random_seed = 42
    # dataset hyper-parameter
    dataset_name = 'whu'
    # dataset_name = 'levir_crop'

    # training hyper-parameter
    epochs: int =100 # Number of epochs
    # epochs: int =150 # Number of epochs
    batch_size: int = 16 # Batch size
    inference_ratio = 1  # batch_size in val and test equal to batch_size*inference_ratio
    learning_rate: float = 1e-4 # Learning rate
    factor = 0.5                           # learning rate decreasing factor
    patience = 8  # schedular patience
    warm_up_step = 500  # warm up step
    weight_decay: float = 1e-3  # AdamW optimizer weight decay
    amp: bool = True  # if use mixed precision or not
    load: str = False  # Loa/home/zhengzhssiyong/offical-SGSLN-main/le"""""  """"  """vir_crop_best_f1sc"""  """ore_model/best/e_epoch425_Sat May 18 17:08:19 2024.pthsd model and/or optimizer from a .pth file for testing or continuing training
    # load: str = "/home/zhengzhiyong/offical-SGSLN-main/levir_crop_best_f1score_model/best_f1score_epoch139_Sat Apr 26 11:37:08 2025.pth"  # Load model and/or optimizer fom a .pth file for testing or continuing training
    max_norm: float = 10  # gradient clip max norm

    # evaluate hyper-parameter
    evaluate_epoch: int = 0 # start evaluate after training for evaluate epochs
    stage_epoch = [0, 0, 0, 0, 0]  # adjust learning rate after every stage epoch
    save_checkpoint: bool = False  # if save checkpoint of model or not
    save_interval: int = 10  # save checkpoint every interval epoch
    save_best_model: bool = True  # if save best model or not

    # log wandb hyper-parameter
    log_wandb_project: str = '4090_sysu'  # wandb project name


    # data transform hyper-parameter
    noise_p: float = 0.8  # probability of adding noise

    # model hyper-parameter
    dropout_p: float = 0.3  # probability of dropout
    patch_size: int = 256  # size of input image

    y = 2  # ECA-net parameter
    b = 1  # ECA-net parameter

    # inference parameter
    log_path = './log_feature/'

    def state_dict(self):
        return {k: getattr(self, k) for k, _ in Path_Hyperparameter.__dict__.items() \
                if not k.startswith('_')}

    # promot相关参数
    drop_out=0.1 #promot的参数
    num_promot_tokens=200#插入多少个 promot token
    promot_deep=True
    transformer_layers=12

    # vit相关参数
    img_size_vit=(256,256)
    patch_size_vit=16
    global_pool_vit='map'
    num_classes_vit=0
    embed_dim_vit=768
    depth_vit=12
    num_heads_vit=12
    mlp_ratio_vit=4.0
    qkv_bias_vit=True
    qk_norm_vit=False
    init_valus_vit=None
    no_embed_class_vit=False
    reg_tokens_vit=0
    pre_norm_vit=False
    fc_norm_vit=None
    dynamic_img_size_vit=False
    dynamic_img_pad_vit=False
    proj_drop_rate_vit=0.0
    proj_drop_rate_vit=0.0
    attn_drop_rate_vit=0.0
    weight_init_vit=''
    fix_init_vit=False
    norm_layer__vit=None
    act_layer_vit=None
    class_token_vit=False
    


ph = Path_Hyperparameter()
