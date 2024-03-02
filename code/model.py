import torch
from torch import nn
import timm

class Net(nn.Module):
    def __init__(self, back_bone, n_frames, device_id):
        super().__init__()
        print(back_bone)
        self.model = timm.create_model(back_bone, num_classes=2, pretrained=True, in_chans=3*n_frames, dynamic_img_pad=True, dynamic_img_size=True)
        
        
        self.model.set_grad_checkpointing()
        self.IMAGENET_DEFAULT_MEAN = torch.tensor([0.485, 0.456, 0.406]).to(device_id)
        self.IMAGENET_DEFAULT_STD = torch.tensor([0.229, 0.224, 0.225]).to(device_id)

    def forward(self, x):
        x = x.transpose(1, 2).transpose(1, 3).contiguous()
        x = x/255.0
        x = (x - self.IMAGENET_DEFAULT_MEAN[None,:, None, None])/self.IMAGENET_DEFAULT_STD[None,:, None, None]
        logit = self.model(x)
        
        return logit
