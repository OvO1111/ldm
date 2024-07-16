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
        
        
class Classifier(nn.Module):
    def __init__(self, 
                 in_chns,
                 n_coarse,
                 n_fine,
                 *,
                 repeats=1,
                 use_separate_normalization=True,
                 use_prior_concatenation=True,
                 dims=3,):
        super().__init__()
        self.use_separate_normalization = use_separate_normalization
        self.use_prior_concatenation = use_prior_concatenation
        
        batchnorm_nd = getattr(nn, f"BatchNorm{dims}d", nn.Identity)
        conv_nd = getattr(nn, f"Conv{dims}d", nn.Identity)
        conv_block = nn.Sequential(*nn.ModuleList([conv_nd(in_chns, in_chns, kernel_size=3, padding=1) for _ in range(repeats)]))

        if use_separate_normalization:
            self.coarse_foreground = nn.Sequential(
                batchnorm_nd(in_chns),
                nn.ReLU(),
                conv_block
            )
            self.coarse_feat2logit = conv_nd(in_chns, 1, 1)
            self.coarse_feat2feat = conv_nd(in_chns + 1, n_fine - 1, 1) if use_prior_concatenation else conv_nd(in_chns, n_fine - 1, 1)
            self.coarse_background = nn.Sequential(
                batchnorm_nd(in_chns),
                nn.ReLU(),
                conv_block,
                conv_nd(in_chns, 1, 1),
            )
        
        elif use_prior_concatenation:
            self.conv = nn.Sequential(
                batchnorm_nd(in_chns),
                nn.ReLU(inplace=True),
                conv_block
            )
            self.coarse = conv_nd(in_chns, n_coarse, 1)
            self.fine = conv_nd(in_chns + n_coarse, n_fine, 1)

        else:
            self.conv = nn.Sequential(
                batchnorm_nd(in_chns),
                nn.ReLU(inplace=True),
                conv_block,
            )
            self.coarse = conv_nd(in_chns, n_coarse, 1)
            self.fine = conv_nd(in_chns, n_fine, 1)
        
    def forward(self, inputs):
        if self.use_separate_normalization:
            foreground = self.coarse_foreground(inputs)
            fg_logit = self.coarse_feat2logit(foreground)
            bg_logit = self.coarse_background(inputs)
            
            fg_concat = torch.cat([foreground, fg_logit], dim=1)
            fine_split = self.coarse_feat2feat(fg_concat if self.use_prior_concatenation else foreground)
            coarse = torch.cat([bg_logit, fg_logit], dim=1)
            fine = torch.cat([bg_logit, fine_split], dim=1)
            
        elif self.use_prior_concatenation:
            inputs = self.conv(inputs)
            coarse = self.coarse(inputs)
            fine = torch.cat([inputs, coarse], dim=1)
            fine = self.fine(fine)
        else:
            inputs = self.conv(inputs)
            coarse = self.coarse(inputs)
            fine = self.fine(inputs)
        return coarse, fine
        

class BasicUNet(nn.Module):
    def __init__(self, 
                 unet_config,
                 model_sn=False, 
                 model_pc=False,
                 n_coarse=2,
                 n_fine=4,):
        super().__init__()
        self.backbone = instantiate_from_config(unet_config)
        in_chns = self.backbone.out_ch
        self.classifier = Classifier(in_chns, n_coarse, n_fine, 
                                     use_separate_normalization=model_sn, use_prior_concatenation=model_pc)
    
    def forward(self, inputs):
        feature = self.backbone(inputs)
        coarse, fine = self.classifier(feature)
        return coarse, fine


class EfficientSubclassSegmentation(BasePytorchLightningTrainer):
    def __init__(self,
                 unet_config,
                 image_key="image",
                 coarse_key="coarse",
                 fine_key="fine",
                 use_data_augmentation=False,
                 use_mixup=False,
                 use_pseudo=False,
                 use_separate_normalization=False,
                 use_prior_concatenation=False,
                 use_ldm_mixup=True,
                 ckpt_path=None,
                 ignore_keys=[],
                 only_load_model=True,
                 pseudo_label_consensus_threshold=.4,
                 mixup_alphas=.5,
                 monitor='val/loss',
                 n_coarse=-1,
                 n_fine=-1):
        super().__init__()
        self.image_key = image_key
        self.coarse_key = coarse_key
        self.fine_key = fine_key
        self.use_ldm_mixup = use_ldm_mixup and use_mixup
        
        self.use_data_augmentation = use_data_augmentation
        self.use_mixup = use_mixup and use_data_augmentation
        self.use_pseudo = use_pseudo and use_data_augmentation
        self.use_separate_normalization = use_separate_normalization
        self.use_prior_concatenation = use_prior_concatenation
        
        self.monitor = monitor
        self.mixup_alphas = mixup_alphas
        self.pseudo_label_consensus_threshold = pseudo_label_consensus_threshold
        
        self.n_fine = n_fine
        self.n_coarse = n_coarse
        
        self.model = BasicUNet(unet_config,
                               self.use_separate_normalization,
                               self.use_prior_concatenation,
                               self.n_coarse, self.n_fine,)
        if ckpt_path is not None: self.init_from_ckpt(ckpt_path, ignore_keys, only_load_model)
        
    def on_pretrain_routine_start(self):
        if hasattr(self.trainer.datamodule, "batch_sampler"):
            self.primary_batch_size = self.trainer.datamodule.batch_sampler.primary_batch_size
        else: self.primary_batch_size = 0
        
    @staticmethod
    def _get_foreground_bbox(tensor, use_shape_on_background=False):
        cropped = []
        for it, t in enumerate(tensor):
            # c h w d
            if torch.any(t):
                crop_x = torch.where(torch.any(torch.any(t, dim=2), dim=2))[1][[0, -1]]
                crop_y = torch.where(torch.any(torch.any(t, dim=1), dim=2))[1][[0, -1]]
                crop_z = torch.where(torch.any(torch.any(t, dim=1), dim=1))[1][[0, -1]]
                cropped.append([slice(it, it + 1), slice(None, None),
                                slice(crop_x[0], crop_x[1] + 1),
                                slice(crop_y[0], crop_y[1] + 1),
                                slice(crop_z[0], crop_z[1] + 1)])
            else:
                if use_shape_on_background: cropped.append(None)
                else: cropped.append([slice(it, it+1)] + [slice(0, t.shape[i] + 1) for i in range(1, len(t.shape))])
        return cropped
    
    @staticmethod
    def _interpolate(tensor, *args, **kwargs):
        mode = kwargs.get("mode", "nearest")
        if mode == "nearest":
            kwargs["mode"] = "trilinear"
            return nn.functional.interpolate(tensor, align_corners=True, *args, **kwargs).round()
        return nn.functional.interpolate(tensor, *args, **kwargs)
        
    def mixup(self, 
              mixup_im1, mixup_im2,
              mixup_lf1, mixup_lc2,
              alpha=.5):
        # perform image mixup of im1 (primary) and im2 (secondary)
        secondary_batch_size = mixup_im2.shape[0]
        im_mix, mask_mix = [], []
        # crop foreground region
        im_fine_masked_cropped = self._get_foreground_bbox(mixup_lf1)
        im_coarse_masked_cropped = self._get_foreground_bbox(mixup_lc2)
        # resize fine foregrounds to coarse's size
        im_fine_reshaped, mask_fine_reshaped = [], []
        for i in range(secondary_batch_size):
            if im_fine_masked_cropped[0] is not None and im_coarse_masked_cropped[i] is not None:
                im_fine_reshaped.append(self._interpolate(mixup_im1[*im_fine_masked_cropped[0]], mixup_im2[*im_coarse_masked_cropped[i]].shape[2:], mode="trilinear"))
                mask_fine_reshaped.append(self._interpolate(mixup_lf1[*im_fine_masked_cropped[0]], mixup_lc2[*im_coarse_masked_cropped[i]].shape[2:], mode="nearest"))
            else:
                if im_coarse_masked_cropped[i] is None:
                    im_coarse_masked_cropped[i] = [slice(0, mixup_im2[i].shape[j] + 1) for j in range(len(mixup_im2[i].shape))]
                im_fine_reshaped.append(mixup_im2[*im_coarse_masked_cropped[i]])
                mask_fine_reshaped.append(mixup_lc2[*im_coarse_masked_cropped[i]])
        # mixup
        im_local_mix, mask_local_mix = [], []
        for i in range(secondary_batch_size):
            im_fine_reshaped[i][:, :, mixup_lc2[*im_coarse_masked_cropped[i]][0, 0] == 0] = mixup_im2[*im_coarse_masked_cropped[i]][:, :, mixup_lc2[*im_coarse_masked_cropped[i]][0, 0] == 0]
            mask_fine_reshaped[i][:, :, mixup_lc2[*im_coarse_masked_cropped[i]][0, 0] == 0] = 0
            im_local_mix.append(im_fine_reshaped[i] * (1 - alpha) + mixup_im2[*im_coarse_masked_cropped[i]] * alpha)
            mask_local_mix.append(mask_fine_reshaped[i])
            
        for i in range(secondary_batch_size): mixup_im2[i, *im_coarse_masked_cropped[i][1:]] = im_local_mix[i]
        for i in range(secondary_batch_size): mixup_lc2[i, *im_coarse_masked_cropped[i][1:]] = mask_local_mix[i]
        im_mix.append(mixup_im2)
        mask_mix.append(mixup_lc2)
        
        im_mix, lf_mix = torch.cat(im_mix, dim=0), torch.cat(mask_mix, dim=0)
        return im_mix, lf_mix
    
    def transform(self, im):
        rot_angle = random.randint(0, 3)
        flip_axis = random.randint(2, 3)
        
        im_trans = torch.flip(torch.rot90(im, k=rot_angle, dims=(2, 3)), dims=(flip_axis,))
        inv = lambda x: torch.rot90(torch.flip(x, dims=(flip_axis,)), k=-rot_angle, dims=(2, 3))
        return im_trans, inv
        
    def apply_model(self, inputs):
        coarse_preds, fine_preds = self.model(inputs)
        return coarse_preds, fine_preds
    
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
    
    def _ce_loss(self, x, y, is_y_one_hot=False, **kw):
        return F.cross_entropy(x, y, **kw)
    
    def get_loss(self, preds, coarse=None, fine=None, one_hot_y=False, mixup_only=False):
        if preds.shape[0] == 0: return 0
        if mixup_only: 
            one_hot_y = True
            if coarse is not None: coarse = rearrange(F.one_hot(coarse, self.n_coarse), "b 1 ... c -> b c ...")
            if fine is not None: fine = rearrange(F.one_hot(fine, self.n_fine), "b 1 ... c -> b c ...")
        if coarse is not None:
            ce_loss = self._ce_loss(preds, coarse, one_hot_y)
            dice_loss = self._dice_loss(preds, coarse, one_hot_y, self.n_coarse)  
        elif fine is not None:
            ce_loss = self._ce_loss(preds, fine, one_hot_y, reduction='none' if mixup_only else 'mean')
            dice_loss = self._dice_loss(preds, fine, one_hot_y, self.n_fine) 
            
        loss = ce_loss + dice_loss
        return loss
    
    def _multiclass_metrics(self, x, y, prefix=""):
        logs = {}
        for m in ["dc", "precision", "recall"]:
            for i in range(1, self.n_fine):
                logs[f"{prefix}/{m}/{i}"] = getattr(bin, m, lambda *a: 0)(x == i, y == i)
            logs[f"{prefix}/{m}/mean"] = sum([logs[f"{prefix}/{m}/{j}"] for j in range(1, self.n_fine)]) / (self.n_fine - 1)
        return logs
    
    def shared_step(self, batch):
        loss_dict = {}
        image, coarse, fine = map(lambda x: self.get_input(batch, x), 
                                  [self.image_key, self.coarse_key, self.fine_key])
        coarse, fine = coarse.long(), fine.long()
        prefix = "train" if self.training else "val"
        model_outputs_coarse, model_outputs_fine = self.apply_model(image)
        
        # coarse loss
        loss_dict[f"{prefix}/coarse"] = self.get_loss(model_outputs_coarse, coarse=coarse)

        # fine loss
        if self.training:
            # primary batch
            if self.primary_batch_size > 0:
                loss_dict[f"{prefix}/fine"] = self.get_loss(preds=model_outputs_fine[:self.primary_batch_size], fine=fine[:self.primary_batch_size])
            else:
                random_index = random.randint(0, coarse.shape[0] - 1)
                loss_dict[f"{prefix}/fine"] = self.get_loss(preds=model_outputs_fine[random_index: random_index+1], fine=fine[random_index: random_index+1])
            # secondary batch
            model_outputs = model_outputs_fine
            model_preds = model_outputs.softmax(1)
        
            if self.use_mixup:
                if self.use_ldm_mixup:
                    mixed_image, mixed_fine = map(lambda x: self.get_input(batch, x)[self.primary_batch_size:], ["mixed_image", "mixed_fine"])
                    mixed_fine = mixed_fine.long()
                    check_mix_validity = torch.argwhere(mixed_image.sum((1, 2, 3, 4)) == image[self.primary_batch_size:].sum((1, 2, 3, 4))).flatten()
                if self.primary_batch_size > 0 and check_mix_validity.numel() > 0:
                    random_primary_index = random.choice(list(_ for _ in range(self.primary_batch_size)))
                    mixed_image_, mixed_fine_ = self.mixup(image[random_primary_index:random_primary_index+1], image[self.primary_batch_size:][check_mix_validity],
                                                           fine[random_primary_index:random_primary_index+1][:, None].float(), coarse[self.primary_batch_size:][check_mix_validity][:, None].float(),
                                                           self.mixup_alphas)
                    mixed_image[check_mix_validity] = mixed_image_
                    mixed_fine[check_mix_validity] = mixed_fine_[:, 0].long()
                mixed_fine = rearrange(F.one_hot(mixed_fine, self.n_fine), "b ... c -> b c ...")
                _, model_outputs_mixup = self.apply_model(mixed_image)
                
            if self.use_pseudo:
                trans_image, inv = self.transform(image[self.primary_batch_size:])
                # consensus
                _, model_outputs_trans = self.apply_model(trans_image)
                model_outputs_trans_preds = inv(model_outputs_trans).softmax(1)
                
                trans_fine = torch.repeat_interleave(torch.zeros_like(coarse[self.primary_batch_size:])[:, None], self.n_fine, 1)
                for i_label in range(self.n_fine):
                    i_ = 0 if i_label == 0 else 1
                    trans_fine[:, i_label] = reduce(lambda x, y: x & y, [
                        (coarse[self.primary_batch_size:] == i_),
                        (model_preds[self.primary_batch_size:, i_label] > self.pseudo_label_consensus_threshold),
                        (model_outputs_trans_preds[:, i_label] > self.pseudo_label_consensus_threshold),
                    ])
                    
            if self.use_mixup and self.use_pseudo:
                loss_dict[f"{prefix}/agg"] = self.get_loss(model_outputs_mixup,
                                                 fine=self.mixup_alphas * trans_fine + (1 - self.mixup_alphas) * mixed_fine,
                                                 one_hot_y=True) / (1 + math.exp(-self.global_step // 1000))
            elif self.use_pseudo:
                loss_dict[f"{prefix}/pseudo"] = self.get_loss(model_outputs,
                                                    fine=trans_fine, one_hot_y=True) / (1 + math.exp(-self.global_step // 1000))
            elif self.use_mixup:
                loss_dict[f"{prefix}/mixup"] = self.get_loss(model_outputs,
                                                   fine=mixed_fine, one_hot_y=True, mixup_only=True) / (1 + math.exp(-self.global_step // 1000))
        else:
            loss_dict[f"{prefix}/fine"] = self.get_loss(preds=model_outputs_fine, fine=fine)
                
        loss = sum(loss_dict.values())
        loss_dict[f"{prefix}/loss"] = loss
        # metrics
        if not self.training:
            loss_dict_nodisplay = self._multiclass_metrics(model_outputs_fine.softmax(1).argmax(1).cpu().numpy(),
                                                           fine.cpu().numpy(), prefix)
            self.log_dict(loss_dict_nodisplay, logger=True, prog_bar=False, on_step=True, on_epoch=True)
        return loss, loss_dict
    
    def log_images(self, batch, **kw):
        logs = {}
        image, coarse, fine = map(lambda x: self.get_input(batch, x), 
                                  [self.image_key, self.coarse_key, self.fine_key])
        
        logs["inputs"] = image
        logs["gt_coarse"] = coarse
        logs["gt_fine"] = fine
        
        fine = fine[:self.primary_batch_size]
        model_outputs_coarse, model_outputs_fine = self.apply_model(image)
        logs["seg_coarse"] = model_outputs_coarse.softmax(1).argmax(1)
        logs["seg_fine"] = model_outputs_fine.softmax(1).argmax(1)
        
        return logs
    
    def configure_optimizers(self,):
        opt = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=0.0001)
        sch = LinearLR(opt, start_factor=1, end_factor=0, total_iters=self.trainer.max_epochs, verbose=1)
        return [opt], [sch]