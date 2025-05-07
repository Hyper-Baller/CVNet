import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module): #num_layers代表 多层感知机的层数  
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(
        self, input_dim, hidden_dim, output_dim, num_layers, affine_func=nn.Linear
    ):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            affine_func(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x: torch.Tensor):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
    


class Conv2_BN_Relu(nn.Module):
    """ Basic convolution."""

    def __init__(self, in_channel, out_channel, kernel, stride):
        super().__init__()
        self.conv_bn_relu = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=kernel,
                                                    padding=kernel // 2, bias=False, stride=stride),
                                          nn.BatchNorm2d(out_channel),
                                          nn.ReLU(inplace=True)
                                          )

    def forward(self, x):
        output = self.conv_bn_relu(x)

        return output
    

class Conv1_BN(nn.Module):
    """ Basic convolution."""

    def __init__(self, in_channel, out_channel, kernel, stride):
        super().__init__()
        self.conv_bn_relu = nn.Sequential(nn.Conv1d(in_channel, out_channel, kernel_size=kernel,
                                                    padding=kernel // 2, bias=False, stride=stride),
                                          nn.BatchNorm1d(out_channel),
                                          )

    def forward(self, x):
        output = self.conv_bn_relu(x)

        return output
    

def compare_params(parent_model, child_model):
# 对比两个参数是否相等 
    for name, parent_param in parent_model.named_parameters():
        # 获取子模型的参数
        child_param = dict(child_model.named_parameters()).get(name, None)
        
        if child_param is None:
            print(f"Parameter {name} not found in child model.")
        elif parent_param.shape != child_param.shape:
            print(f"Parameter {name} shape mismatch: parent {parent_param.shape}, child {child_param.shape}")
        else:
            # Check if the values are equal
            if torch.allclose(parent_param, child_param, atol=1e-6):
                print(f"Parameter {name} loaded successfully.")
            else:
                print(f"Parameter {name} values mismatch.")


