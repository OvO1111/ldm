import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import random
import numpy as np
from einops import rearrange, repeat
from ldm.models.template import BasePytorchLightningTrainer
from ldm.modules.diffusionmodules.model import Downsample, Upsample, make_attn, checkpoint
from torch.optim.lr_scheduler import LinearLR, LambdaLR
from scipy.ndimage import label
import medpy.metric.binary as bin

from monai.networks.nets.unet import UNet 
from monai.networks.nets.vnet import VNet
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
                 crop_input=False,
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
        self.crop_input = crop_input
        self.image_size = image_size
        self.num_classes = num_classes
        self.in_chns = in_chns
        self.backbone_name = backbone_name
        self._get_model(backbone_name)
        
        self.monitor = monitor
        if ckpt_path is not None: self.init_from_ckpt(ckpt_path, ignore_keys, only_load_model)
        
    def get_input(self, batch, *keys):
        images = [batch[k] for k in keys]
        if self.crop_input:
            if self.training:
                return self.get_input_train(images)
            else:
                return self.get_input_val(images)
        return images
        
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
                    
    def _patch_cropper(self, _image, _mode="random", _pad_value=0, _hazy=False):
        _image = _image[0]
        _output_size = self.image_size
        # maybe pad image if _output_size is larger than _image.shape
        ph = max((_output_size[0] - _image.shape[1]) // 2 + 3, 0)
        pw = max((_output_size[1] - _image.shape[2]) // 2 + 3, 0)
        pd = max((_output_size[2] - _image.shape[3]) // 2 + 3, 0)
        padder = lambda x: torch.nn.functional.pad(x, (pd, pd, ph, ph, pw, pw), mode='constant', value=0)
        _image = padder(_image)
        # padder = identity
            
        if _mode == "random":
            h_center = random.randint(ph, _image.shape[1] - ph)
            w_center = random.randint(pw, _image.shape[2] - pw)
            d_center = random.randint(pd, _image.shape[3] - pd)

        elif _mode == "foreground":
            if _image.sum() == 0: return self._patch_cropper(_image, _output_size, "random", _pad_value)
            
            _hl, _hr = torch.where(torch.any(torch.any(_image, 2), 2))[-1][[0, -1]]
            _wl, _wr = torch.where(torch.any(torch.any(_image, 1), -1))[-1][[0, -1]]
            _dl, _dr = torch.where(torch.any(torch.any(_image, 1), 1))[-1][[0, -1]]
            if _hazy: _label = _image
            else:
                _label, _n = label(_image[:, _hl: _hr + 1, _wl: _wr + 1, _dl: _dr + 1].data.cpu().numpy())
                _label = torch.tensor(_label == random.randint(1, _n), dtype=_image.dtype, device=_image.device)
            hl, hr = torch.where(torch.any(torch.any(_label, 2), 2))[-1][[0, -1]] + _hl
            wl, wr = torch.where(torch.any(torch.any(_label, 1), -1))[-1][[0, -1]] + _wl
            dl, dr = torch.where(torch.any(torch.any(_label, 1), 1))[-1][[0, -1]] + _dl
            hc, hd = (hl + hr) // 2, hr - hl + 1
            wc, wd = (wl + wr) // 2, wr - wl + 1
            dc, dd = (dl + dr) // 2, dr - dl + 1
            
            h_center = random.randint(hc - hd // 4, max(hc + hd // 4, hc - hd // 4 + 1))
            w_center = random.randint(wc - wd // 4, max(wc + wd // 4, wc - wd // 4 + 1))
            d_center = random.randint(dc - dd // 4, max(dc + dd // 4, dc - dd // 4 + 1))
        
        h_left = max(0, h_center - _output_size[0] // 2)
        h_right = min(_image.shape[1], h_center + _output_size[0] // 2)
        h_offset = (max(0, _output_size[0] // 2 - h_center), max(0, h_center + _output_size[0] // 2 - _image.shape[1]))
        w_left = max(0, w_center - _output_size[1] // 2)
        w_right = min(_image.shape[2], w_center + _output_size[1] // 2)
        w_offset = (max(0, _output_size[1] // 2 - w_center), max(0, w_center + _output_size[1] // 2 - _image.shape[2]))
        d_left = max(0, d_center - _output_size[2] // 2)
        d_right = min(_image.shape[3], d_center + _output_size[2] // 2) 
        d_offset = (max(0, _output_size[2] // 2 - d_center), max(0, d_center + _output_size[2] // 2 - _image.shape[3]))
        cropper = lambda x: x[:, :, h_left: h_right, w_left: w_right, d_left: d_right]
        post_padder = lambda x: torch.nn.functional.pad(x, 
                                                        (*d_offset, *w_offset, *h_offset),
                                                        mode='constant', value=_pad_value)
            
        return padder, cropper, post_padder 
        
    def get_input_train(self, images, fg_prob=0.7):
        mode = "foreground" if random.random() < fg_prob else "random"
        padder, cropper, post_padder = self._patch_cropper(images[1], _mode=mode, _pad_value=0)  # index 0 is image, 1 is seg
        inputs = [post_padder(cropper(padder(im))) for im in images]
        return inputs if len(inputs) > 1 else inputs[0]
        
    def _get_model(self, model_name,):
        in_channels = self.in_chns
        
        if model_name == 'unet':
            self.model = UNet(self.dims, in_channels, self.num_classes, 
                              channels=(16, 32, 64, 128, 256),
                              strides=(2, 2, 2, 2))
        elif model_name == 'vnet':
            self.model = VNet(self.dims, in_channels, self.num_classes)
        elif model_name == 'unetr':
            self.model = UNETR(spatial_dims=self.dims,
                               feature_size=64,
                               in_channels=in_channels,
                               out_channels=self.num_classes,
                               img_size=self.image_size)
        elif model_name == 'swinunetr':
            self.model = SwinUNETR(img_size=self.image_size,
                                   feature_size=48,
                                   num_heads=(4, 8, 16, 32),
                                   in_channels=in_channels,
                                   out_channels=self.num_classes,)
        elif model_name == 'unetpp':
            self.model = BasicUNetPlusPlus(self.dims, in_channels, self.num_classes,)
        elif model_name == 'sin':
            self.model = SinusoidalUNet(ch=16, ch_mult=(1, 2, 4, 8, 16), num_res_blocks=1, attn_resolutions=[], resolution=max(self.image_size),
                                        in_channels=in_channels, out_ch=self.num_classes, dims=self.dims, nonlinearity=SinNonlinearity())
    
    def _multiclass_metrics(self, x, y, prefix=""):
        logs = {}
        for m in ["dc"]:#, "hd95"]:
            for i in range(1, self.num_classes):
                if (x == i).sum() == 0 or (y == i).sum() == 0: result = 0
                else: result = getattr(bin, m, lambda *a: 0)(x == i, y == i)
                logs[f"{prefix}/{m}/{i}"] = result
            # logs[f"{prefix}/{m}/mean"] = sum([logs[f"{prefix}/{m}/{j}"] for j in range(1, self.num_classes)]) / (self.num_classes - 1)
        return logs
    
    def _dice_loss(self, x, y, is_y_one_hot=False, num_classes=-1):
        smooth = 1e-5
        x = x.softmax(1)
        if not is_y_one_hot:
            y = rearrange(F.one_hot(y, num_classes), "b ... c -> b c ...")
        intersect = torch.sum(x * y)
        x_sum = torch.sum(x * x)
        y_sum = torch.sum(y * y)
        loss = 1 - (2 * intersect) / (x_sum + y_sum + smooth)
        return loss
    
    def _ce_loss(self, x, y, **kw):
        return F.cross_entropy(x, y, **kw)
    
    def get_loss(self, preds, y=None, one_hot_y=False):
        ce_loss = self._ce_loss(preds, y)
        dice_loss = self._dice_loss(preds, y, one_hot_y, self.num_classes) 
            
        loss = ce_loss + dice_loss
        return loss * 10
    
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
        metric_dict = self._multiclass_metrics(model_outputs.argmax(1).cpu().numpy(),
                                               seg.cpu().numpy(),
                                               prefix)
        metric_dict['train/dc/mean'] = np.mean([metric_dict[f"train/dc/{i}"] for i in range(1, self.num_classes)])
        # metric_dict['train/hd95/mean'] = [metric_dict[f"train/hd95/{i}"] for i in range(1, self.num_classes)]
        self.log('train.dc.mean', metric_dict['train/dc/mean'], prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(metric_dict, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(loss_dict, prog_bar=True, on_step=True, on_epoch=True, logger=True, sync_dist=True)
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
        
        metric_dict['val/dc/mean'] = np.mean([metric_dict[f"val/dc/{i}"] for i in range(1, self.num_classes)])
        # metric_dict['train/hd95/mean'] = [metric_dict[f"train/hd95/{i}"] for i in range(1, self.num_classes)]
        self.log('val.dc.mean', metric_dict['val/dc/mean'], prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)      
        self.log_dict(metric_dict, logger=True, prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(loss_dict, prog_bar=False, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log("global_step", self.global_step, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        return loss
    
    def log_images(self, batch, **kw):
        logs = {}
        
        if self.training or not self.crop_input:
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
    
    
class SinusoidalUNet(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels, nonlinearity,
                 resolution, use_linear_attn=False, attn_type="vanilla", dims=3):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.out_ch = out_ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.conv_nd = getattr(nn, f"Conv{dims}d", nn.Identity)
        self.batchnorm_nd = torch.nn.BatchNorm2d if dims == 2 else torch.nn.BatchNorm3d if dims == 3 else None
        self.nonlinearity = nonlinearity
        
        # downsampling
        self.conv_in = self.conv_nd(in_channels,
                                    self.ch,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         dropout=dropout, dims=dims, nonlinearity=nonlinearity))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       dropout=dropout, dims=dims, nonlinearity=nonlinearity)
        # self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       dropout=dropout, dims=dims, nonlinearity=nonlinearity)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            skip_in = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                if i_block == self.num_res_blocks:
                    skip_in = ch*in_ch_mult[i_level]
                block.append(ResnetBlock(in_channels=block_in+skip_in,
                                         out_channels=block_out,
                                         dropout=dropout, dims=dims, nonlinearity=nonlinearity))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = self.batchnorm_nd(block_in)
        self.conv_out = self.conv_nd(block_in,
                                     out_ch,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)

    def forward(self, x, context=None):
        #assert x.shape[2] == x.shape[3] == self.resolution
        if context is not None:
            # assume aligned context, cat along channel axis
            x = torch.cat((x, context), dim=1)

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h)
        # h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1))
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = self.nonlinearity(h)
        h = self.conv_out(h)
        return h

    def get_last_layer(self):
        return self.conv_out.weight
    
    
class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, dims=3, nonlinearity):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.conv_nd = torch.nn.Conv2d if dims == 2 else torch.nn.Conv3d if dims == 3 else None
        self.batchnorm_nd = torch.nn.BatchNorm2d if dims == 2 else torch.nn.BatchNorm3d if dims == 3 else None

        self.norm1 = self.batchnorm_nd(in_channels)
        self.conv1 = self.conv_nd(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        self.norm2 = self.batchnorm_nd(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = self.conv_nd(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        self.nonlin1 = nonlinearity
        self.nonlin2 = nonlinearity
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = self.conv_nd(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = self.conv_nd(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)
    
    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), True)

    def _forward(self, x):
        h = x
        h = self.norm1(h)
        h = self.nonlin1(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = self.nonlin2(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h


class SinNonlinearity(nn.Module):
    def __init__(self, sinwave=10):
        self.waves = sinwave
        super().__init__()
        self.conv = torch.nn.Conv1d(self.waves, 1, 1)
        self.freq = nn.Parameter(torch.ones((self.waves,)), requires_grad=True)
    
    def forward(self, inputs):
        b, c, h, w, d = inputs.shape
        inputs = rearrange(inputs, 'b ... -> b 1 (...)')
        inputs = repeat(inputs, 'b d z -> b (d f) z', f=self.waves)
        inputs = torch.einsum('bfz,f->bfz', inputs, self.freq).sin()
        inputs = self.conv(inputs)
        outputs = rearrange(inputs[:, 0], 'b (c h w d) -> b c h w d', c=c, h=h, w=w, d=d)
        return outputs
            