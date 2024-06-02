import sys
import torch
import numpy as np
import torch.distributed
import torch.nn as nn
import torch.nn.functional as f

from tqdm import tqdm
from typing import Optional

from omegaconf import OmegaConf
from functools import partial
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ldm.modules.diffusionmodules.util import make_beta_schedule

import pytorch_lightning as pl
from ldm.modules.ema import LitEma
from einops import repeat, rearrange
from contextlib import contextmanager
from ldm.modules.losses.lpips import LPIPS
from ldm.modules.diffusionmodules.util import extract_into_tensor

sys.path.append("/mnt/workspace/dailinrui/code/multimodal/trajectory_generation/ccdm")
from ldm.util import get_obj_from_str, instantiate_from_config
from ldm.models.diffusion.ddim import make_ddim_timesteps


def identity(x, *args, **kwargs):
    return x


def default(x, dval=None):
    if not exists(x): return dval
    else: return x

def exists(x):
    return x is not None


class OneHotCategoricalBCHW(torch.distributions.OneHotCategorical):
    """Like OneHotCategorical, but the probabilities are along dim=1."""

    def __init__(
            self,
            probs: Optional[torch.Tensor] = None,
            logits: Optional[torch.Tensor] = None,
            validate_args=None):

        if probs is not None and probs.ndim < 2:
            raise ValueError("`probs.ndim` should be at least 2")

        if logits is not None and logits.ndim < 2:
            raise ValueError("`logits.ndim` should be at least 2")

        probs = self.channels_last(probs) if probs is not None else None
        logits = self.channels_last(logits) if logits is not None else None

        super().__init__(probs, logits, validate_args)

    def sample(self, sample_shape=torch.Size()):
        res = super().sample(sample_shape)
        return self.channels_second(res)

    @staticmethod
    def channels_last(arr: torch.Tensor) -> torch.Tensor:
        """Move the channel dimension from dim=1 to dim=-1"""
        dim_order = (0,) + tuple(range(2, arr.ndim)) + (1,)
        return arr.permute(dim_order)

    @staticmethod
    def channels_second(arr: torch.Tensor) -> torch.Tensor:
        """Move the channel dimension from dim=-1 to dim=1"""
        dim_order = (0, arr.ndim - 1) + tuple(range(1, arr.ndim - 1))
        return arr.permute(dim_order)

    def max_prob_sample(self):
        """Sample with maximum probability"""
        num_classes = self.probs.shape[-1]
        res = torch.nn.functional.one_hot(self.probs.argmax(dim=-1), num_classes)
        return self.channels_second(res)

    def prob_sample(self):
        """Sample with probabilities"""
        return self.channels_second(self.probs)


class CategoricalDiffusion(pl.LightningModule):
    def __init__(self, unet_config, loss_config, *,
                 conditional_encoder_config=None,
                 train_ddim_sigmas=False,
                 is_conditional=True,
                 data_key="mask",
                 cond_key="context",
                 timesteps=1000,
                 use_scheduler=False,
                 scheduler_config=None,
                 monitor=None,
                 ckpt_path=None,
                 ignore_keys=None,
                 load_only_unet=False,
                 conditioning_key="crossattn",
                 num_classes=12,
                 given_betas=None,
                 loss_type='l2',
                 beta_schedule="cosine",
                 linear_start=1e-2,
                 linear_end=2e-1,
                 cosine_s=8e-3,
                 use_ema=False,
                 class_weights=None,
                 p_x1_sample="majority",
                 parameterization="kl",
                 cond_stage_trainable=False,
                 cond_stage_forward=None,
                 use_automatic_optimization=True,
                 **kwargs) -> None:
        super().__init__()
        self.data_key = data_key
        self.cond_key = cond_key
        self.loss_type = loss_type
        self.timesteps = timesteps
        self.conditioning_key = conditioning_key
        self.is_conditional = is_conditional
        self.use_scheduler = use_scheduler
        self.loss_config = loss_config
        self.train_ddim_sigmas = train_ddim_sigmas
        if self.use_scheduler:
            self.scheduler_config = scheduler_config
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, only_model=load_only_unet)
        self.num_classes = num_classes
        self.automatic_optimization = use_automatic_optimization
        if not exists(class_weights):
            class_weights = torch.ones((self.num_classes,))
        self.register_buffer("class_weights", torch.tensor(class_weights))
        print(f"setting class weights as {self.class_weights}")
        self.use_ema = use_ema
        self.parameterization = parameterization
        self.cond_stage_forward = cond_stage_forward
        self.cond_stage_trainable = cond_stage_trainable
        
        self.loss_fn = dict()
        self.lpips = LPIPS().eval()
        self.p_x1_sample = p_x1_sample
        
        self.register_schedule(given_betas=given_betas, beta_schedule=beta_schedule, timesteps=timesteps,
                               linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)
        
        unet_config["in_channels"] = self.num_classes
        unet_config["out_channels"] = self.num_classes
        self.model = DiffusionWrapper(unet_config, conditioning_key)
        self.dims = getattr(self.model, "dims", 3)
        
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")
            
        if self.is_conditional:
            if conditional_encoder_config is None: self.cond_stage_model = nn.Identity()
            else: self.cond_stage_model = instantiate_from_config(**conditional_encoder_config)
        
    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")
    
    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        cd = torch.load(path, map_location="cpu")
        if "state_dict" in list(cd.keys()):
            cd = cd["state_dict"]
        keys = list(cd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del cd[k]
        missing, unexpected = self.load_state_dict(cd, strict=False) if not only_model else self.model.load_state_dict(
            cd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")
        
    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                       cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas', to_torch(alphas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))
     
    def get_input(self, batch, data_key, cond_key):
        x = batch.get(data_key)
        x = rearrange(f.one_hot((x * (self.num_classes - 1)).long(), self.num_classes), "b 1 h w d x -> b x h w d")
        c = self.cond_stage_model(batch.get(cond_key))
        c = {f"c_{self.conditioning_key}": [c.float()]}
        ret = [x.float(), c]
        return ret
    
    def lpips_loss(self, x0pred, x0):
        x, y = map(lambda i: repeat(i.argmax(1, keepdim=True), 'b 1 d h w -> b c d h w', c=3), [x0pred, x0])
        if self.dims == 3:
            lpips_x = self.lpips(rearrange(x, "b c d h w -> (b d) c h w"),
                                rearrange(y, "b c d h w -> (b d) c h w")).mean()
            lpips_y = self.lpips(rearrange(x, "b c d h w -> (b h) c d w"),
                                rearrange(y, "b c d h w -> (b h) c d w")).mean()
            lpips_z = self.lpips(rearrange(x, "b c d h w -> (b w) c d h"),
                                rearrange(y, "b c d h w -> (b w) c d h")).mean()
            lpips = (lpips_x + lpips_y + lpips_z) / 3
        elif self.dims == 2:
            lpips = self.lpips(x, y)
        return lpips
    
    def q_xt_given_xtm1(self, xtm1, t, noise=None):
        betas = extract_into_tensor(self.betas, t, xtm1.shape)
        if noise is None: probs = (1 - betas) * xtm1 + betas / self.num_classes
        else: probs = (1 - betas) * xtm1 + betas * noise
        return probs

    def q_xt_given_x0(self, x0, t, noise=None):
        alphas_cumprod = extract_into_tensor(self.alphas_cumprod, t, x0.shape)
        if noise is None: probs = alphas_cumprod * x0 + (1 - alphas_cumprod) / self.num_classes
        else: probs = alphas_cumprod * x0 + (1 - alphas_cumprod) * noise
        return probs

    def q_xtm1_given_x0_xt(self, xt, x0, t):
        # computes q_xtm1 given q_xt, q_x0, noise
        alphas_t = extract_into_tensor(self.alphas, t, x0.shape)
        alphas_cumprod_tm1 = extract_into_tensor(self.alphas_cumprod_prev, t, x0.shape)
        theta = ((alphas_t * xt + (1 - alphas_t) / self.num_classes) * (alphas_cumprod_tm1 * x0 + (1 - alphas_cumprod_tm1) / self.num_classes))
        return theta / theta.sum(dim=1, keepdim=True)

    def q_xtm1_given_x0pred_xt(self, xt, theta_x0, t):
        """
        This is equivalent to calling theta_post with all possible values of x0
        from 0 to C-1 and multiplying each answer times theta_x0[:, c].

        This should be used when x0 is unknown and what you have is a probability
        distribution over x0. If x0 is one-hot encoded (i.e., only 0's and 1's),
        use theta_post instead.
        """
        alphas_t = extract_into_tensor(self.alphas, t, xt.shape)
        alphas_cumprod_tm1 = extract_into_tensor(self.alphas_cumprod_prev, t, xt.shape)[..., None]

        # We need to evaluate theta_post for all values of x0
        x0 = torch.eye(self.num_classes, device=xt.device)[None, :, :, None, None]
        if self.dims == 3: x0 = x0[..., None]
        # theta_xt_xtm1.shape == [B, C, H, W]
        theta_xt_xtm1 = alphas_t * xt + (1 - alphas_t) / self.num_classes
        # theta_xtm1_x0.shape == [B, C1, C2, H, W]
        theta_xtm1_x0 = alphas_cumprod_tm1 * x0 + (1 - alphas_cumprod_tm1) / self.num_classes

        aux = theta_xt_xtm1[:, :, None] * theta_xtm1_x0
        # theta_xtm1_xtx0 == [B, C1, C2, H, W]
        theta_xtm1_xtx0 = aux / aux.sum(dim=1, keepdim=True)
        return torch.einsum("bcd...,bd...->bc...", theta_xtm1_xtx0, theta_x0)
    
    def shared_step(self, batch, **kwargs):
        x, c = self.get_input(batch, self.data_key, self.cond_key)
        loss = self(x, c, class_id=batch.get("class_id"))
        return loss
    
    def forward(self, x, c, *args, **kwargs):
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        if self.model.conditioning_key is not None:
            assert c is not None
            if self.cond_stage_trainable:
                c = self.get_learned_conditioning(c)
        return self.p_losses(x, c, t, *args, **kwargs)
    
    def get_learned_conditioning(self, c):
        if self.cond_stage_forward is None:
            if hasattr(self.cond_stage_model, 'encode') and callable(self.cond_stage_model.encode):
                c = self.cond_stage_model.encode(c)
            else:
                c = self.cond_stage_model(c)
        else:
            assert hasattr(self.cond_stage_model, self.cond_stage_forward)
            c = getattr(self.cond_stage_model, self.cond_stage_forward)(c)
        return c
    
    def p_losses(self, x0, c, t, class_id=None, *args, **kwargs):
        noise = OneHotCategoricalBCHW(logits=torch.zeros_like(x0)).sample()
        q_xt = self.q_xt_given_x0(x0, t, noise)
        model_outputs = self.model(q_xt, t, **c)
        if isinstance(model_outputs, dict): 
            label_cond = model_outputs.get("label_out")
            model_outputs = model_outputs["diffusion_out"]
            
        loss_log = dict()
        log_prefix = 'train' if self.training else 'val'
        if self.parameterization == "kl":
            xt = OneHotCategoricalBCHW(q_xt).sample()
            q_xtm1_given_xt_x0 = self.q_xtm1_given_x0_xt(xt, x0, t,)
            q_xtm1_given_xt_x0pred = self.q_xtm1_given_x0pred_xt(xt, model_outputs, t,)
            
            kl_loss = f.kl_div(torch.log(torch.clamp(q_xtm1_given_xt_x0pred, min=1e-12)),
                                q_xtm1_given_xt_x0,
                                reduction='none')
            kl_loss_per_class = kl_loss.sum(1) * self.class_weights[x0.argmax(1)]
            if (kl_loss_per_class.sum() < -1e-3).any():
                print(f"negative KL divergence {kl_loss_per_class.sum()} encountered in loss")
            batch_loss = kl_loss_per_class.sum() / x0.shape[0]
            loss_log[f"{log_prefix}/kl_div"] = batch_loss.item() * self.loss_config["kl_div"].get("coeff", 1)
        
        elif self.parameterization in ['x0', 'eps']:
            target = x0 if self.parameterization == 'x0' else noise
            if self.loss_type == 'l1':
                dir_loss = (model_outputs - target).sum()
            elif self.loss_type == 'l2':
                dir_loss = f.mse_loss(model_outputs, target, reduction='none').sum()
            batch_loss = dir_loss
            loss_log[f"{log_prefix}/dir_loss"] = batch_loss.item() * self.loss_config["dir_loss"].get("coeff", 1)
            
        if hasattr(self.model.diffusion_model, "use_label_predictor") and self.model.diffusion_model.use_label_predictor:
            ce_loss = f.cross_entropy(label_cond, class_id)
            loss_log[f"{log_prefix}/ce"] = ce_loss
            batch_loss += ce_loss
            
        loss_log["debug"] = model_outputs.argmax(1).max()

        if not self.automatic_optimization and self.training:
            opt = self.optimizers()
            if self.use_scheduler:
                lr = opt.param_groups[0]['lr']
                self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)
                
            opt.zero_grad()
            self.manual_backward(batch_loss)
            opt.step()
        
        return batch_loss, loss_log
    
    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch)

        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("global_step", self.global_step, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _, loss_dict_no_ema = self.shared_step(batch)
        with self.ema_scope():
            _, loss_dict_ema = self.shared_step(batch)
            loss_dict_ema = {key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}
        self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
    
    def on_train_epoch_end(self,):
        if self.use_scheduler:
            sch = self.lr_schedulers()
            if isinstance(sch, ReduceLROnPlateau): sch.step(self.trainer.callback_metrics["loss"])
            else: sch.step()
        
        if self.use_ema:
            self.model_ema(self.model)
    
    @torch.no_grad()
    def log_images(self, batch, split="train", sample=True, **kwargs):
        x0, c = self.get_input(batch, self.data_key, self.cond_key)
        noise = OneHotCategoricalBCHW(logits=torch.zeros_like(x0)).sample()
        
        logs = dict()
        b = x0.shape[0]
        logs["inputs"] = x0.argmax(1)
        if split == "train":
            t = torch.randint(0, self.num_timesteps, (b,), device=self.device).long()
            q_xt = self.q_xt_given_x0(x0, t, noise)
            model_outputs = self.model(q_xt, t, **c)
            if isinstance(model_outputs, dict): 
                model_outputs = model_outputs["diffusion_out"]
            
            logs["t"] = str(t.cpu().numpy().tolist())
            logs["noise"] = noise.argmax(1)
            logs["samples"] = OneHotCategoricalBCHW(model_outputs).max_prob_sample().argmax(1)
        elif split == "val":
            t = torch.tensor((self.timesteps,) * b, device=self.device)
            x0pred = self.p_sample(noise, c, shape=x0.shape, **kwargs)
            logs["samples"] = x0pred.argmax(1)
            
        lpips = self.lpips_loss(model_outputs, x0)
        logs["conditioning"] = str(batch["text"])
        self.log(f"{split}/lpips_metric", lpips, prog_bar=True, on_step=True)
            
        return logs
    
    @torch.no_grad()
    def p_sample(self, q_xT=None, c=None, use_ddim=False, ddim_steps=50, shape=None, verbose=False,
                 log_denoising=False, log_every_t=100):
        logs = dict()
        with self.ema_scope():
            c = c if exists(c) else dict()
            q_xT = q_xT if exists(q_xT) else OneHotCategoricalBCHW(logits=torch.zeros_like(shape)).sample()
            p_xtm1_given_xt, b = q_xT, q_xT.shape[0]
            if not use_ddim:
                t_values = reversed(range(1, self.timesteps)) if not verbose else tqdm(reversed(range(1, self.timesteps)), total=self.timesteps, desc="sampling progress")
                for t in t_values:
                    t_ = torch.full(size=(b,), fill_value=t, device=q_xT.device)
                    model_outputs = self.model(p_xtm1_given_xt, t_, **c)
                    if isinstance(model_outputs, dict): 
                        model_outputs = model_outputs["diffusion_out"]
                    p_xtm1_given_xt = torch.clamp(self.q_xtm1_given_x0_xt(q_xT, model_outputs, t_), min=1e-12)
                    # xt = OneHotCategoricalBCHW(probs=p_xtm1_given_xt).sample()
                    
                if self.p_x1_sample == "majority":
                    x0pred = OneHotCategoricalBCHW(probs=p_xtm1_given_xt).max_prob_sample()
                elif self.p_x1_sample == "confidence":
                    x0pred = OneHotCategoricalBCHW(probs=p_xtm1_given_xt).prob_sample()
                return x0pred
            else:
                ddim_timesteps = make_ddim_timesteps("uniform", num_ddim_timesteps=ddim_steps, num_ddpm_timesteps=self.timesteps, verbose=verbose)
                t_values = reversed(range(0, ddim_steps)) if not verbose else tqdm(reversed(range(0, ddim_steps)), total=self.timesteps, desc="ddim sampling progress")
                for t in t_values:
                    t_ = torch.full(size=(q_xT.shape[0],), fill_value=ddim_timesteps[t], device=q_xT.device)
                    model_outputs = self.model(q_xT, t_, **c)
                    if isinstance(model_outputs, dict): 
                        model_outputs = model_outputs["diffusion_out"]
                    p_xtm1_given_xt = torch.clamp(self.q_xtm1_given_x0pred_xt(q_xT, model_outputs, t_), min=1e-12)

                    if t > 1:
                        q_xT = OneHotCategoricalBCHW(probs=p_xtm1_given_xt).sample()
                    else:
                        if self.p_x1_sample == "majority":
                            q_xT = OneHotCategoricalBCHW(probs=p_xtm1_given_xt).max_prob_sample()
                        elif self.p_x1_sample == "confidence":
                            q_xT = OneHotCategoricalBCHW(probs=p_xtm1_given_xt).prob_sample()
                return q_xT
            
    @torch.no_grad()
    def p_sample_ddim(self, xt, c, ddim_t, t):
        ddim_betas = extract_into_tensor(self.betas, ddim_t, xt.shape)
        ddim_alphas = extract_into_tensor(self.alphas, ddim_t, xt.shape)
        ddim_alpha_cumprods = extract_into_tensor(self.alpha_cumprods, ddim_t, xt.shape)
        
        t = torch.full(size=(xt.shape[0],), fill_value=t, device=xt.device)
        ddim_t = torch.full(size=(xt.shape[0],), fill_value=ddim_t, device=xt.device)
        
        x0pred = self.model(xt, ddim_t, **c)
        if isinstance(x0pred, dict): 
            x0pred = x0pred["diffusion_out"]
        xtm1 = ...
        return xtm1
            
    def configure_optimizers(self):
        lr = self.learning_rate
        opt_parameters = list(self.model.parameters())
        if self.cond_stage_trainable:
            opt_parameters = opt_parameters + list(self.cond_stage_model.parameters())
        opt = torch.optim.AdamW(opt_parameters, lr=lr)
        if self.use_scheduler:
            scheduler = get_obj_from_str(self.scheduler_config["target"])(opt, **self.scheduler_config["params"])
            cfg = {
                "optimizer": opt,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "train/loss"
                }
            }
            return cfg
        return opt
    

class LossWrapper(nn.Module):
    def __init__(self, coeff, module):
        super().__init__()
        self.coeff = coeff
        self.module = module
        
    def __call__(self, *args, **kwargs):
        return self.coeff * self.module(*args, **kwargs)
    
    
class DiffusionWrapper(pl.LightningModule):
    def __init__(self, diff_model_config, conditioning_key):
        super().__init__()
        self.diffusion_model = instantiate_from_config(diff_model_config)
        self.conditioning_key = conditioning_key
        assert self.conditioning_key in [None, 'concat', 'crossattn', 'hybrid']

    def forward(self, x, t, c_concat: list = None, c_crossattn: list = None):
        if self.conditioning_key is None:
            out = self.diffusion_model(x, t)
        elif self.conditioning_key == 'concat':
            xc = torch.cat([x] + c_concat, dim=1)
            out = self.diffusion_model(xc, t)
        elif self.conditioning_key == 'crossattn':
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(x, t, context=cc)
        elif self.conditioning_key == 'hybrid':
            xc = torch.cat([x] + c_concat, dim=1)
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(xc, t, context=cc)
        elif self.conditioning_key == 'adm':
            cc = c_crossattn[0]
            out = self.diffusion_model(x, t, y=cc)
        else:
            raise NotImplementedError()

        return out


if __name__ == "__main__":
    spec = OmegaConf.to_container(OmegaConf.load("./run/train_ruijin_ccdm.yaml"))
            