
import pytorch_lightning as pl
import numpy as np
import einops
import os
import torch
import torch.nn as nn
import SimpleITK as sitk
from tqdm import tqdm
from einops import rearrange
from ldm.util import instantiate_from_config
from ldm.modules.diffusionmodules.util import make_beta_schedule
from torch.distributions.one_hot_categorical import OneHotCategorical


def set_precision(precision):
    if precision in [16, "16", "fp16", torch.float16]:
        p = torch.float16
        fn = lambda tensor: tensor.half()
    elif precision in ["bf16", torch.bfloat16]:
        p = torch.bfloat16
        fn = lambda tensor: tensor.bfloat16()
    elif precision in [32, "32", "fp32", torch.float32]:
        p = torch.float32
        fn = lambda tensor: tensor.float()
    elif precision in [64, "64", "fp64", "double", torch.double, torch.float64]:
        p = torch.float64
        fn = lambda tensor: tensor.double()
    else:
        raise ValueError(f"Unsupported precision: {precision}")
    
    def _impl(x):   
        x.dtype = p
        for module in x.modules():
            if isinstance(module, nn.modules.conv._ConvNd):
                module.weight.data = fn(module.weight.data)
                module.bias.data = fn(module.bias.data)
            elif isinstance(module, (
                nn.modules.normalization.GroupNorm,
                nn.modules.normalization.LayerNorm,
                nn.modules.batchnorm._BatchNorm
            )):
                module.weight.data = fn(module.weight.data)
                module.bias.data = fn(module.bias.data)
            elif isinstance(module, nn.Linear):
                module.weight.data = fn(module.weight.data) 
                if module.bias is not None:
                    module.bias.data = fn(module.bias.data)
    return p, _impl


class OneHotCategoricalBCHW(OneHotCategorical):
    """Like OneHotCategorical, but the probabilities are along dim=1."""

    def __init__(
            self,
            probs: torch.Tensor = None,
            logits: torch.Tensor = None,
            validate_args=None):

        if probs is not None and probs.ndim < 2:
            raise ValueError("`probs.ndim` should be at least 2")

        if logits is not None and logits.ndim < 2:
            raise ValueError("`logits.ndim` should be at least 2")

        probs = self.channels_last(probs) if probs is not None else None
        logits = self.channels_last(logits) if logits is not None else None

        super().__init__(probs, logits, validate_args)
        self.chwd_probs = None if probs is None else einops.rearrange(probs, "b h w d c -> b c h w d")

    def sample(self, sample_shape=torch.Size()):
        res = super().sample(sample_shape)
        return self.channels_second(res).float()

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
    def __init__(self, 
                 unet_config,
                 num_classes=4,
                 mask_key="mask",
                 concat_key=None,
                 crossattn_key='context',
                 conditioning_key='hybrid',
                 concat_encoder_config=None,
                 crossattn_encoder_config=None,
                 cond_stage_trainable=False,
                 timesteps=1000,
                 schedule='cosine',
                 step_T_sample='majority',
                 monitor="val/loss",
                 foreground_weight=1,
                 ckpt_path=None,
                 dims=3,
                 linear_start=.0001,
                 linear_end=.02,
                 cosine_s=.008, ignore_keys=[], **kw):
        super().__init__()
        betas = make_beta_schedule(schedule, timesteps, linear_start, linear_end, cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        self.dims = dims
        self.conditioning_key = conditioning_key
        self.register_buffer("betas", torch.tensor(betas))
        self.register_buffer("alphas", torch.tensor(alphas))
        self.register_buffer("cumalphas", torch.tensor(alphas_cumprod))
        self.register_buffer("cumalphas_prev", torch.tensor(alphas_cumprod_prev))
        
        self.num_classes = num_classes
        self.cond_stage_trainable = cond_stage_trainable
        
        unet_config['params']['dims'] = dims
        self.model = DiffusionWrapper(unet_config, conditioning_key)
        self.monitor = monitor
        self.foreground_weight = foreground_weight
        
        self.timesteps = timesteps
        self.mask_key = mask_key
        self.concat_key = concat_key
        self.crossattn_key = crossattn_key
        self.step_T_sample = step_T_sample
        
        if ckpt_path is not None:
            print(f"init model weights from {ckpt_path}")
            self.init_from_ckpt(ckpt_path, ignore_keys)
        
        if concat_encoder_config is not None: self.concat_encoder = instantiate_from_config(concat_encoder_config)
        if crossattn_encoder_config is not None: self.crossattn_encoder = instantiate_from_config(crossattn_encoder_config)
        
        self.weight = torch.tensor((1,) + (foreground_weight,) * (self.num_classes - 1))
        
    def get_loss(self, xt, x0, x0_pred, t, noise=None):
        prob_xtm1_given_xt_x0 = self.q_xtm1_given_x0_xt(xt, x0, t, noise)
        prob_xtm1_given_xt_x0pred = self.q_xtm1_given_x0prob_xt(xt, x0_pred, t, noise)
        
        loss = nn.functional.kl_div(
            torch.log(torch.clamp(prob_xtm1_given_xt_x0pred, min=1e-12)),
            prob_xtm1_given_xt_x0,
            reduction='none'
        )
        loss = loss.sum(dim=1)  * self.weight[x0.argmax(1)]
        loss = loss.sum()
        return loss
        
    def q_xt_given_xtm1(self, xtm1: torch.Tensor, t: torch.Tensor, noise=None) -> OneHotCategoricalBCHW:
        t = t - 1
        betas = self.betas[t]
        betas = betas[..., None, None, None]
        if self.dims == 3: betas = betas[..., None]
        if noise is None: probs = (1 - betas) * xtm1 + betas / self.num_classes
        else: probs = (1 - betas) * xtm1 + betas * noise
        return OneHotCategoricalBCHW(probs)

    def q_xt_given_x0(self, x0: torch.Tensor, t: torch.Tensor, noise=None) -> OneHotCategoricalBCHW:
        t = t - 1
        cumalphas = self.cumalphas[t]
        cumalphas = cumalphas[..., None, None, None]
        if self.dims == 3: cumalphas = cumalphas[..., None]
        if noise is None: probs = cumalphas * x0 + (1 - cumalphas) / self.num_classes
        else: probs = cumalphas * x0 + (1 - cumalphas) * noise
        return OneHotCategoricalBCHW(probs)

    def q_xtm1_given_x0_xt(self, xt: torch.Tensor, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor=None) -> torch.Tensor:
        # computes q_xtm1 given q_xt, q_x0, noise
        t = t - 1
        smooth = 1e-8
        alphas_t = self.alphas[t][..., None, None, None]
        cumalphas_tm1 = self.cumalphas[t - 1][..., None, None, None]
        if self.dims == 3:
            alphas_t = alphas_t[..., None]
            cumalphas_tm1 = cumalphas_tm1[..., None]
        alphas_t[t == 0] = 0.0
        cumalphas_tm1[t == 0] = 1.0
        theta = ((alphas_t * xt + (1 - alphas_t) / self.num_classes) * (cumalphas_tm1 * x0 + (1 - cumalphas_tm1) / self.num_classes))
        return (theta + smooth) / (theta.sum(dim=1, keepdim=True) + smooth)

    def q_xtm1_given_x0prob_xt(self, xt: torch.Tensor, theta_x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor=None) -> torch.Tensor:
        t = t - 1
        smooth = 1e-8
        alphas_t = self.alphas[t][..., None, None, None]
        cumalphas_tm1 = self.cumalphas[t - 1][..., None, None, None, None]
        if self.dims == 3:
            alphas_t = alphas_t[..., None]
            cumalphas_tm1 = cumalphas_tm1[..., None]
        alphas_t[t == 0] = 0.0
        cumalphas_tm1[t == 0] = 1.0

        x0 = torch.eye(self.num_classes, device=xt.device)[None, :, :, None, None]
        if self.dims == 3: x0 = x0[..., None]
        theta_xt_xtm1 = alphas_t * xt + (1 - alphas_t) / self.num_classes
        theta_xtm1_x0 = cumalphas_tm1 * x0 + (1 - cumalphas_tm1) / self.num_classes
        
        aux = theta_xt_xtm1[:, :, None] * theta_xtm1_x0
        theta_xtm1_xtx0 = (aux + smooth) / (aux.sum(dim=1, keepdim=True) + smooth)
        
        out = torch.einsum("bcdlhw,bdlhw->bclhw", theta_xtm1_xtx0.float(), theta_x0.float()) if self.dims == 3 else\
            torch.einsum("bcdhw,bdhw->bchw", theta_xtm1_xtx0, theta_x0)
            
        return out
        
    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.encoder.load_state_dict(
            sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")
            
    def set_precision(self, precision):
        p, fn = set_precision(precision)
        self.type(p)
        self.model.diffusion_model.apply(fn)
        return self
        
    def on_fit_start(self):
        self.weight = self.weight.to(self.device)
        if hasattr(self, "concat_encoder"): self.concat_encoder.to(self.device)
        if hasattr(self, "crossattn_encoder"): self.crossattn_encoder.to(self.device)
        
    def get_input(self, batch, key=None):
        if key is None: return None
        out = batch[key]
        if key == self.mask_key:
            out = rearrange(nn.functional.one_hot(out.long(), self.num_classes), "b 1 ... n -> b n ...")
        if key == self.concat_key:
            return getattr(self, "concat_encoder", torch.nn.Identity())(out)
        if key == self.crossattn_key:
            return getattr(self, "crossattn_encoder", torch.nn.Identity())(out)
        else: 
            return out
        
    def get_noise(self, x):
        return OneHotCategoricalBCHW(logits=torch.zeros(x.shape, device=self.device)).sample()
        
    def shared_step(self, batch):
        mask_x0 = self.get_input(batch, self.mask_key)
        if self.conditioning_key == 'concat': conditions = {"c_concat": [self.get_input(batch, self.concat_key)]}
        elif self.conditioning_key == 'crossattn': conditions = {'c_crossattn': [self.get_input(batch, self.crossattn_key)]}
        elif self.conditioning_key == 'hybrid':
            conditions = {"c_concat": [self.get_input(batch, self.concat_key)]} | {'c_crossattn': [self.get_input(batch, self.crossattn_key)]}
        else: conditions = {}
            
        b = mask_x0.shape[0]
        if self.training:
            loss_prefix = "train"
            t = torch.multinomial(torch.arange(self.timesteps, device=mask_x0.device) ** 1.5, b)
        else:
            loss_prefix = "val"
            t = torch.full((b,), fill_value=self.timesteps-1, device=mask_x0.device)
            
        mask_xt = self.q_xt_given_x0(mask_x0, t).sample()
        mask_x0pred = self.model(mask_xt, t, **conditions)
        mask_x0pred = mask_x0pred.softmax(1)
        debug = mask_x0pred.argmax(1).max()
        loss = self.get_loss(mask_xt, mask_x0, mask_x0pred, t)
        loss_dict = {f"{loss_prefix}/kl": loss, f"{loss_prefix}/debug": debug}
        return loss, loss_dict
    
    def training_step(self, batch, _):
        loss, loss_dict = self.shared_step(batch)
        self.log_dict(loss_dict, prog_bar=True, on_step=True, on_epoch=True, logger=True)
        self.log("global_step", self.global_step, prog_bar=False, on_step=True, on_epoch=False, logger=True)
        return loss
    
    def validation_step(self, batch, _):
        loss, loss_dict = self.shared_step(batch)
        self.log("val/loss", loss.item())
        self.log_dict(loss_dict, prog_bar=True, on_step=True, on_epoch=True, logger=True)
        self.log("global_step", self.global_step, prog_bar=False, on_step=True, on_epoch=False, logger=True)
    
    def denoising(self, xT, init_t: int=None, end_t: int=0, conditions=None, verbose=False):
        xt = xT.clone()
        if init_t is None: init_t = self.timesteps
        t_iterator = range(init_t, end_t, -1)
        if verbose: t_iterator = tqdm(t_iterator)

        shape = xt.shape
        for t in t_iterator:
            t_ = torch.full(size=(shape[0],), fill_value=t, device=xt.device)
            x0pred = self.model(xt, t_.float(), **conditions)
            x0pred = x0pred.softmax(1)
            xtm1probs = self.q_xtm1_given_x0prob_xt(xt, x0pred, t_.long()).clamp(min=1e-12)
            
            if t > 1:
                xt = OneHotCategoricalBCHW(probs=xtm1probs).sample()
            else:
                if self.step_T_sample is None or self.step_T_sample == "majority":
                    xt = OneHotCategoricalBCHW(probs=xtm1probs).max_prob_sample()
                elif self.step_T_sample == "confidence":
                    xt = OneHotCategoricalBCHW(probs=xtm1probs).prob_sample()
        return xt
            
    def configure_optimizers(self):
        parameters = list(self.model.parameters())
        if self.cond_stage_trainable == 'concat':
            parameters += list(self.concat_encoder.parameters())
        elif self.cond_stage_trainable == 'crossattn':
            parameters += list(self.crossattn_encoder.parameters())
        elif self.cond_stage_trainable == 'all':
            parameters += list(self.concat_encoder.parameters())
            parameters += list(self.crossattn_encoder.parameters())
        opt = torch.optim.AdamW(parameters, self.learning_rate)
        return opt
        
    def log_images(self, batch, split="train", init_t=None, end_t=0, verbose=False, **kw):
        logs = {}
        x0 = self.get_input(batch, self.mask_key)
        conditions = {"c_crossattn": [self.get_input(batch, self.crossattn_key)],
                      "c_concat": [self.get_input(batch, self.concat_key)]}
        b = x0.shape[0]
        if 'c_concat' in conditions and self.concat_key is not None:
            logs['conditioning'] = batch.get(self.concat_key)
        
        if split == "train":
            t = torch.multinomial(torch.arange(self.timesteps, device=x0.device) ** 1.5, b)
            xt = self.q_xt_given_x0(x0, t).sample()
                
            x0pred = self.model(xt, t, **conditions)
            
            logs["inputs"] = x0.argmax(1)
            logs[f"xt{t.cpu().numpy().tolist()}"] = xt.argmax(1)
            logs["samples"] = x0pred.argmax(1)
            
            return logs
        else:
            eps = self.get_noise(x0)
            x0pred = self.denoising(eps, init_t=init_t, end_t=end_t, conditions=conditions, verbose=verbose)
            
            logs["inputs"] = x0.argmax(1)
            logs["noise"] = eps.argmax(1)
            logs["samples"] = x0pred.argmax(1)
            
            return logs
        
        
class DiffusionWrapper(pl.LightningModule):
    def __init__(self, diff_model_config, conditioning_key):
        super().__init__()
        self.diffusion_model = instantiate_from_config(diff_model_config)
        self.conditioning_key = conditioning_key
        assert self.conditioning_key in [None, 'concat', 'crossattn', 'hybrid', 'adm']

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
            xc = torch.cat([x] + [c_concat], dim=1)
            cc = torch.cat([c_crossattn], 1)
            out = self.diffusion_model(xc, t, context=cc)
        elif self.conditioning_key == 'adm':
            cc = c_crossattn[0]
            out = self.diffusion_model(x, t, y=cc)
        else:
            raise NotImplementedError()

        return out         
    
    
if __name__ == '__main__':
    model = CategoricalDiffusion(dict(
        target="ldm.modules.diffusionmodules.openaimodel.RiskUNetModel",
        params=dict(
            image_size=[64, 64, 64],
            in_channels=1,
            out_channels=1,
            model_channels=64,
            attention_resolutions=[],
            num_res_blocks=2,
            channel_mult=[1, 2, 4],
            num_head_channels=32,
            use_checkpoint=True, # always use ckpt
            return_latents=True))
    )
    inputs = torch.zeros((1, 20, 1, 1, 1))
    inputs[:, 0] = 1
    probs = [model.q_xt_given_x0(inputs, t).probs for t in range(1, 1001, 100)]
    print(["\t".join(probs[i]) for i in range(len(probs))])
            