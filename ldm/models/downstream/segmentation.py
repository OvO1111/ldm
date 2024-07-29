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
        
    def get_input(self, batch, *keys):
        images = [batch[k] for k in keys]
        if self.training:
            return self.get_input_train(images)
        else:
            return self.get_input_val(images)
        
    def get_input_val(self, images):
        # return generator of patches
        image = images[0]
        (b, c, w, h, d) = image.shape
        outline = [self.image_size[i] * (image.shape[2 + i] // self.image_size[i] + 1) - image.shape[2 + i] for i in range(3)]
        padder = (lambda x: x) if sum(outline) == 0 else lambda x: torch.nn.functional.pad(x, (0, outline[2], 0, outline[1], 0, outline[0]), mode='constant', value=0)
        
        for i in range(0, w, self.image_size[0]):
            for j in range(0, h, self.image_size[1]):
                for k in range(0, d, self.image_size[2]):
                    patch = [padder(im)[:, :, i: i + self.image_size[0], j: j + self.image_size[1], k: k + self.image_size[2]] for im in images]
                    yield patch, (i, j, k)
        
    def get_input_train(self, images):
        # crop to patch
        image = images[0]
        output_size = self.image_size
        pw = max((output_size[0] - image.shape[1]) // 2 + 3, 0)
        ph = max((output_size[1] - image.shape[2]) // 2 + 3, 0)
        pd = max((output_size[2] - image.shape[3]) // 2 + 3, 0)
        image = torch.nn.functional.pad(image, (pd, pd, ph, ph, pw, pw), mode='constant', value=0)

        (b, c, w, h, d) = image.shape
        wl, wr = torch.where(torch.any(torch.any(image, 3), 3))[0][[0, -1]]
        hl, hr = torch.where(torch.any(torch.any(image, 2), -1))[0][[0, -1]]
        dl, dr = torch.where(torch.any(torch.any(image, 2), 2))[0][[0, -1]]
        if wl > w - output_size[0]: w1 = random.randint(0, w-output_size[0])
        else: w1 = random.randint(wl, min(w - output_size[0], wr))
        if hl > h - output_size[1]: h1 = random.randint(0, h-output_size[1])
        else: w1 = random.randint(hl, min(w - output_size[1], hr))
        if dl > d - output_size[2]: d1 = random.randint(0, d-output_size[2])
        else: d1 = random.randint(dl, min(w - output_size[2], dr))
        
        padder = (lambda x: x) if pw + ph + pd == 0 else lambda x: torch.nn.functional.pad(x, (pd, pd, ph, ph, pw, pw), mode='constant', value=0)
        cropper = [slice(w1, w1 + output_size[0]), slice(h1, h1 + output_size[1]), slice(d1, d1 + output_size[2])]
        inputs = [padder(im)[:, :, *cropper] for im in images]
        return inputs if len(inputs) > 1 else inputs[0]
        
    def _get_model(self, model_name,):
        in_channels = self.in_chns
        
        if model_name == 'unet':
            self.model = UNet(self.dims, in_channels, self.num_classes, 
                              channels=(16, 32, 64, 128, 256),
                              strides=(2, 2, 2, 2))
        elif model_name == 'unetr':
            self.model = UNETR(spatial_dims=self.dims,
                               feature_size=64,
                               in_channels=in_channels,
                               out_channels=self.num_classes,
                               img_size=self.image_size)
        elif model_name == 'swinunetr':
            self.model = SwinUNETR(img_size=self.image_size,
                                   feature_size=64,
                                   num_heads=(4, 8, 16, 32),
                                   in_channels=in_channels,
                                   out_channels=self.num_classes,)
        elif model_name == 'unetpp':
            self.model = BasicUNetPlusPlus(self.dims, in_channels, self.num_classes,)
    
    def _multiclass_metrics(self, x, y, prefix=""):
        logs = {}
        for m in ["dc", "hd95"]:
            for i in range(1, self.num_classes):
                if (x == i).sum() == 0 or (y == i).sum() == 0: result = 0
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
    
    def training_step(self, batch, batch_idx):
        loss_dict = {}
        image, seg = self.get_input(batch, self.image_key, self.seg_key)
        seg = seg[:, 0].long()
        prefix = "train" if self.training else "val"
        model_outputs = self.model(image)
        if self.backbone_name == 'unetpp': model_outputs = model_outputs[-1]
                
        loss = self.get_loss(model_outputs, seg, one_hot_y=0)
        loss_dict[f"{prefix}/loss"] = loss
        # metrics
        metric_dict = self._multiclass_metrics(model_outputs.softmax(1).argmax(1).cpu().numpy(),
                                                       seg.cpu().numpy(),
                                                       prefix)
        self.log_dict(metric_dict, logger=True, prog_bar=True, on_step=True, on_epoch=True)
        self.log_dict(loss_dict, prog_bar=True, on_step=True, on_epoch=True, logger=True)
        self.log("global_step", self.global_step, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss_dict = {}
        metric_dict = {}
        
        for itr, ((image, seg), vertex) in enumerate(self.get_input(batch, self.image_key, self.seg_key)):
            seg = seg[:, 0].long()
            prefix = "train" if self.training else "val"
            model_outputs = self.model(image)
            if self.backbone_name == 'unetpp': model_outputs = model_outputs[-1]
                    
            loss = self.get_loss(model_outputs, seg, one_hot_y=0)
            loss_dict[f"{prefix}/loss"] = loss
            # metrics
            iter_metric = self._multiclass_metrics(model_outputs.softmax(1).argmax(1).cpu().numpy(),
                                                    seg.cpu().numpy(),
                                                    prefix)
            for k, v in iter_metric.items():
                if k not in metric_dict: metric_dict[k] = v
                else: metric_dict[k] = (metric_dict[k] * itr + v) / (itr + 1)
                
        self.log_dict(metric_dict, logger=True, prog_bar=True, on_step=True, on_epoch=True)
        self.log_dict(loss_dict, prog_bar=False, on_step=False, on_epoch=True, logger=True)
        self.log("global_step", self.global_step, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        return loss
    
    def log_images(self, batch, **kw):
        logs = {}
        
        if self.training:
            image, seg = self.get_input(batch, self.image_key, self.seg_key)
            seg = seg.long()
            
            logs["inputs"] = image
            logs["gt"] = seg
            
            model_outputs = self.model(image)
            logs["seg"] = model_outputs.softmax(1).argmax(1, keepdim=True)
        else:
            images = torch.zeros_like(batch[self.image_key]) 
            segs = torch.zeros_like(batch[self.seg_key]) 
            model_outputs = torch.zeros_like(batch[self.seg_key]) 
            for (image, seg), vertex in self.get_input(batch, self.image_key, self.seg_key):   
                seg = seg.long()
                images[:, :, 
                    vertex[0]: vertex[0] + self.image_size[0], 
                    vertex[1]: vertex[1] + self.image_size[1], 
                    vertex[2]: vertex[2] + self.image_size[2]] =\
                    image[:, :, :min(vertex[0] + self.image_size[0], images.shape[-3]) - vertex[0], :min(vertex[1] + self.image_size[2], images.shape[-2]) - vertex[1], :min(images.shape[-1], vertex[2] + self.image_size[2]) - vertex[2]]
                segs[:, :, 
                    vertex[0]: vertex[0] + self.image_size[0], 
                    vertex[1]: vertex[1] + self.image_size[1], 
                    vertex[2]: vertex[2] + self.image_size[2]] =\
                    seg[:, :, :min(vertex[0] + self.image_size[0], images.shape[-3]) - vertex[0], :min(vertex[1] + self.image_size[2], images.shape[-2]) - vertex[1], :min(images.shape[-1], vertex[2] + self.image_size[2]) - vertex[2]]
                model_outputs[:, :, 
                    vertex[0]: vertex[0] + self.image_size[0], 
                    vertex[1]: vertex[1] + self.image_size[1], 
                    vertex[2]: vertex[2] + self.image_size[2]] =\
                    self.model(image).argmax(1, keepdim=True)[:, :, :min(vertex[0] + self.image_size[0], images.shape[-3]) - vertex[0], :min(vertex[1] + self.image_size[2], images.shape[-2]) - vertex[1], :min(images.shape[-1], vertex[2] + self.image_size[2]) - vertex[2]]
                

            logs["inputs"] = images
            logs['seg'] = model_outputs
            logs['gt'] = segs
        return logs
    
    def configure_optimizers(self,):
        opt = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=0.0001)
        sch = LinearLR(opt, start_factor=1, end_factor=0, total_iters=self.trainer.max_epochs, verbose=1)
        return [opt], [sch]