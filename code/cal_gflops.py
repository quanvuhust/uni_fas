from model import Net
import torch
from ptflops import get_model_complexity_info
import os
import json
from torch import nn
import timm

class Net(nn.Module):
  def __init__(self, device_id):
    super().__init__()
    self.model = timm.create_model('vit_large_patch14_reg4_dinov2.lvd142m', num_classes=2, pretrained=False, in_chans=3, dynamic_img_pad=True, dynamic_img_size=True)
    # self.model.set_grad_checkpointing()
    self.IMAGENET_DEFAULT_MEAN = torch.tensor([0.485, 0.456, 0.406]).to(device_id)
    self.IMAGENET_DEFAULT_STD = torch.tensor([0.229, 0.224, 0.225]).to(device_id)

  def forward(self, x):
    x = x.transpose(1, 2).transpose(1, 3).contiguous()
    x = x/255.0
    x = (x - self.IMAGENET_DEFAULT_MEAN[None,:, None, None])/self.IMAGENET_DEFAULT_STD[None,:, None, None]
    logit = self.model(x)
    
    return logit



device_id = "cuda:0"

with torch.cuda.device(0):
  net = Net(device_id).to(device_id)
  macs, params = get_model_complexity_info(net, (224, 224, 3), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
  print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
  print('{:<30}  {:<8}'.format('Number of parameters: ', params))
