from timm.models.vision_transformer import VisionTransformer,Block
import open_clip
import torch.nn as nn
import torch
from utils.path_hyperparameter import ph
from torch.nn import Conv2d, Dropout
import math
from functools import reduce
from operator import mul
from torch.nn.modules.utils import _pair

def copy_model_attributes(src, dest):
    for name, param in src.named_parameters():
        if hasattr(dest, name):
            getattr(dest, name).data.copy_(param.data)
        else:
            print(f"Attribute {name} not found in destination model.")



class PromotVisionTransformer(VisionTransformer):
    def __init__(self, *args, **kwargs):
        # 调用父类的初始化方法
        super().__init__(img_size=ph.img_size_vit,
                         num_classes=ph.num_classes_vit,
                         global_pool=ph.global_pool_vit,
                         class_token=ph.class_token_vit)
        
        # 添加子类的自定义层
        # print(ph.drop_out)
        self.prompt_dropout_me = Dropout(ph.drop_out)
        # print(self.prompt_dropout_me)
        self.num_tokens=ph.num_promot_tokens
        self.promot_patch_size = _pair(16)
        self.prompt_dim=768
        self.promot_deep=ph.promot_deep
        self.prompt_proj=nn.Identity()

        #第一层的
        val = math.sqrt(6. / float(3 * reduce(mul, self.promot_patch_size, 1) + self.prompt_dim))  # noqa
        self.prompt_embeddings = nn.Parameter(torch.zeros(
            1, self.num_tokens, self.prompt_dim))
        # xavier_uniform initialization
        nn.init.uniform_(self.prompt_embeddings.data, -val, val)

        #后面层的
        if self.promot_deep:
            total_d_layer = ph.transformer_layers-1
            self.deep_prompt_embeddings = nn.Parameter(torch.zeros(
            total_d_layer, self.num_tokens, self.prompt_dim))
            # xavier_uniform initialization
            nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)
    def load_parent_params(self, parent_model):
        # 直接加载父类模型的参数
        self.load_state_dict(parent_model.state_dict(),strict=False)
        self.frozen_paramters()
    def print_gradients_info(self):
        """直接打印每个参数的名称及其是否有梯度"""
        for name, param in self.named_parameters():
            # has_grad = param.grad is not None
            print(f"{name}: requires_grad={param.requires_grad}")
        
    def frozen_paramters(self):
        """冻结clip全部的参数"""
        # print("我的参数冻结了")
        for name, param in self.named_parameters():
            if "prompt_embeddings" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    def incorporate_prompt(self,x):
        b,c,h,w=x.shape
        x=self.patch_embed(x)  #打patch 加上 位置编码
        x=self._pos_embed(x)
        x=self.patch_drop(x) #这里其实没drop
        x=self.norm_pre(x) #这里其实也没有norm 
        # print(x.shape)
        # print(self.prompt_embeddings.shape)
        x = torch.cat((
            self.prompt_dropout_me(self.prompt_proj(self.prompt_embeddings).expand(b, -1, -1)),
            x[:, 0:, :]
        ), dim=1)
    # (batch_size, cls_token + n_prompt + n_patches, hidden_dim)
        return x
    def forward_deep_prompt(self,x):
        b,l,c=x.shape
        for i, blk in enumerate(self.blocks):
            # x = blk(x)
            if i == 0:
                # hidden_states = self.encoder.layer[i](x)
                x=blk(x)
            else:
                if i <= self.deep_prompt_embeddings.shape[0]:
                    deep_prompt_emb = self.prompt_dropout_me(self.prompt_proj(
                        self.deep_prompt_embeddings[i-1]).expand(b, -1, -1))

                    x = torch.cat((
                        deep_prompt_emb,
                        x[:, (self.num_tokens):, :]
                    ), dim=1)

                x=blk(x)
        return x
    def forward_me(self,x):
        # print("forward_me 到这里了")
        # pass
        x=self.incorporate_prompt(x) #第一层插入
        x=self.forward_deep_prompt(x)
        # print(x.shape)
        x_last_256 = x[:, -256:, :]
        # self.print_gradients_info()
        return x_last_256


    def forward(self, x):
        # 使用父类的 forward_features 方法
        x=self.forward_me(x)
        return x


if __name__=='__main__':
    pro_vit=PromotVisionTransformer()
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:timm/ViT-B-16-SigLIP-256') #可以有device这个参数

    visual_clip=model.visual.trunk
    pro_vit.__dict__ = visual_clip.__dict__.copy() #承接 完毕
    pro_vit.frozen_paramters() #冻结参数
    # print(pro_vit.dynamic_img_size)
    tensor=torch.rand(1,3,256,256)
    x=pro_vit(tensor)
    
    # pro_vit.forward_me(tensor)
    # pro_vit.load_parent_params(visual_clip)
    # compare_params(visual_clip,pro_vit)
    # for name, param in pro_vit.named_parameters():
    #     print(name, param.shape)
    # # self.has_class_token = class_token
    # print(pro_vit.has_class_token)
    # print(pro_vit.global_pool)
    # for name, param in pro_vit.named_parameters():
    #     print(name, param.requires_grad)
