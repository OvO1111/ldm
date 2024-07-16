import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import random
from einops import rearrange
from functools import reduce, partial
from ldm.util import instantiate_from_config
from ldm.models.template import BasePytorchLightningTrainer
from ldm.modules.diffusionmodules.openaimodel import UNetModel
from torch.optim.lr_scheduler import LinearLR, LambdaLR
import medpy.metric.binary as bin

from monai.networks.nets.unet import UNet 
from monai.networks.nets.unetr import UNETR
from monai.networks.nets.basic_unetplusplus import BasicUNetPlusPlus
from monai.networks.nets.swin_unetr import SwinUNETR
from monai.networks.nets.dynunet import DynUNet


class Segmentator(BasePytorchLightningTrainer):
    def __init__(self,
                 num_classes,
                 image_key="image",
                 seg_key="mask",
                 backbone_name=None,
                 in_chns=1,
                 ckpt_path=None,
                 ignore_keys=[],
                 only_load_model=True,
                 monitor='val/loss',
                 dims=3,
                 image_size=(128, 128, 128)):
        super().__init__()
        self.dims = dims
        self.image_key = image_key
        self.seg_key = seg_key
        self.image_size = image_size
        self.num_classes = num_classes
        self.in_chns = in_chns
        self.backbone_name = backbone_name
        self._get_model(backbone_name)
        
        self.monitor = monitor
        if ckpt_path is not None: self.init_from_ckpt(ckpt_path, ignore_keys, only_load_model)
        
    def _get_model(self, model_name,):
        in_channels = self.in_chns
        
        if model_name == 'unet':
            self.model = UNet(self.dims, in_channels, self.num_classes, 
                              channels=(16, 32, 64, 128, 256),
                              strides=(2, 2, 2, 2))
        elif model_name == 'unetr':
            self.model = UNETR(spatial_dims=self.dims,
                               in_channels=in_channels,
                               out_channels=self.num_classes,
                               img_size=self.image_size)
        elif model_name == 'swinunetr':
            self.model = SwinUNETR(img_size=self.image_size,
                                   in_channels=in_channels,
                                   out_channels=self.num_classes,)
        elif model_name == 'unetpp':
            self.model = BasicUNetPlusPlus(self.dims, in_channels, self.num_classes,)
    
    def _multiclass_metrics(self, x, y, prefix=""):
        logs = {}
        for m in ["dc", "hd95"]:
            for i in range(1, self.num_classes):
                if (x == i).sum() == 0: result = 1 if (y == i).sum() == 0 else 0
                elif (y == i).sum() == 0: result = 1 if (x == i).sum() == 0 else 0
                else: result = getattr(bin, m, lambda *a: 0)(x == i, y == i)
                logs[f"{prefix}/{m}/{i}"] = result
            logs[f"{prefix}/{m}/mean"] = sum([logs[f"{prefix}/{m}/{j}"] for j in range(1, self.num_classes)]) / (self.num_classes - 1)
        return logs
    
    def _dice_loss(self, x, y, is_y_one_hot=False, num_classes=-1):
        smooth = 1e-5
        x = x.softmax(1)
        if not is_y_one_hot:
            y = rearrange(F.one_hot(y, num_classes), "b ... c -> b c ...")
        intersect = torch.sum(x * y)
        y_sum = torch.sum(y * y)
        z_sum = torch.sum(x * x)
        loss = 1 - (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        return loss
    
    def _ce_loss(self, x, y, **kw):
        return F.cross_entropy(x, y, **kw)
    
    def get_loss(self, preds, y=None, one_hot_y=False):
        ce_loss = self._ce_loss(preds, y)
        dice_loss = self._dice_loss(preds, y, one_hot_y, self.num_classes) 
            
        loss = ce_loss + dice_loss
        return loss
    
    def shared_step(self, batch):
        loss_dict = {}
        image, seg = map(lambda x: self.get_input(batch, x), [self.image_key, self.seg_key])
        seg = seg[:, 0].long()
        prefix = "train" if self.training else "val"
        model_outputs = self.model(image)
        if self.backbone_name == 'unetpp': model_outputs = model_outputs[-1]
                
        loss = self.get_loss(model_outputs, seg, one_hot_y=0)
        loss_dict[f"{prefix}/loss"] = loss
        # metrics
        loss_dict_nodisplay = self._multiclass_metrics(model_outputs.softmax(1).argmax(1).cpu().numpy(),
                                                       seg.cpu().numpy(),
                                                       prefix)
        self.log_dict(loss_dict_nodisplay, logger=True, prog_bar=True, on_step=True, on_epoch=True)
        return loss, loss_dict
    
    def log_images(self, batch, **kw):
        logs = {}
        image, seg = map(lambda x: self.get_input(batch, x), [self.image_key, self.seg_key])
        seg = seg.long()
        
        logs["inputs"] = image
        logs["gt"] = seg
        
        model_outputs = self.model(image)
        logs["seg"] = model_outputs.softmax(1).argmax(1)
        
        return logs
    
    def configure_optimizers(self,):
        opt = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=0.0001)
        sch = LinearLR(opt, start_factor=1, end_factor=0, total_iters=self.trainer.max_epochs, verbose=1)
        return [opt], [sch]