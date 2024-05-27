import os
import torch
import torch.nn as nn
import numpy as np
import SimpleITK as sitk

from enum import IntEnum
from einops import rearrange, repeat
from torch.nn.functional import interpolate
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddpm import LatentDiffusion, Img2MaskDiffusion


class MetricType(IntEnum):
    lpips = 1
    fid = 2
    psnr = 3
    fvd = 4
    
    
class BaseInferDiffusion(LatentDiffusion):
    def __init__(self, eval_scheme, **diffusion_kwargs):
        super().__init__(**diffusion_kwargs)
        
        self.eval_scheme = eval_scheme
        if MetricType.lpips in self.eval_scheme:
            from ldm.modules.losses.lpips import LPIPS
            self.perceptual = LPIPS().eval()
        if MetricType.fid in self.eval_scheme:
            from torchmetrics.image.fid import FrechetInceptionDistance
            self.fid = FrechetInceptionDistance(feature=2048, normalize=True)
            # self.fid.set_dtype(torch.float64)
        if MetricType.psnr in self.eval_scheme:
            from torchmetrics.image.psnr import PeakSignalNoiseRatio
            self.psnr = PeakSignalNoiseRatio()
        if MetricType.fvd in self.eval_scheme:
            from torchmetrics.image.fid import FrechetInceptionDistance
            self.fvd_module = instantiate_from_config(diffusion_kwargs.get("fvd_config"))
            self.fvd = FrechetInceptionDistance(feature=self.fvd_module, normalize=True)
            
        self.eval()
        
    def on_test_start(self, *args):
        if MetricType.fid in self.eval_scheme:
            self.fid = self.fid.to(self.device)
        if MetricType.fvd in self.eval_scheme:
            self.fvd = self.fvd.to(self.device)
        
    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        pass
    
    @torch.no_grad()
    def log_eval(self, x, y, log_group_metrics_in_2d=False):
        metrics = dict()
        b, c, *shp = x.shape
        
        if MetricType.lpips in self.eval_scheme:
            # lower = better (0, ?)
            perceptual = self.perceptual
            x, y = map(lambda i: repeat(i, 'b 1 d h w -> b c d h w', c=3), [x, y])
            if len(shp) == 3:
                lpips_x = perceptual(rearrange(x, "b c d h w -> (b d) c h w"),
                                    rearrange(y, "b c d h w -> (b d) c h w")).mean()
                lpips_y = perceptual(rearrange(x, "b c d h w -> (b h) c d w"),
                                    rearrange(y, "b c d h w -> (b h) c d w")).mean()
                lpips_z = perceptual(rearrange(x, "b c d h w -> (b w) c d h"),
                                    rearrange(y, "b c d h w -> (b w) c d h")).mean()
                lpips = (lpips_x + lpips_y + lpips_z) / 3
            elif len(shp) == 2:
                lpips = perceptual(x, y)
                
            metrics["LPIPS"] = lpips.item()
            
        if log_group_metrics_in_2d:
            if MetricType.fid in self.eval_scheme:
                # lower = better (0, inf)
                assert len(shp) == 3
                x = rearrange(x, "b c h ... -> (b h) 1 c ...")
                y = rearrange(y, "b c h ... -> (b h) 1 c ...")
                # resize x and y to (3, 299, 299)
                x, y = map(lambda x: torch.nn.functional.interpolate(x, (3, 299, 299)).squeeze(1), [x, y])
                self.fid.update(x.float(), real=False)
                self.fid.update(y.float(), real=True)
                fid = self.fid.compute()
            
                metrics["FID"] = fid.item()
                
            if MetricType.psnr in self.eval_scheme:
                # larger = better (0, inf)
                x = rearrange(x, "b c h ... -> (b h) c ...")
                y = rearrange(y, "b c h ... -> (b h) c ...")
                psnr = self.psnr(x, y)
            
                metrics["PSNR"] = psnr.item()
        return metrics

    @torch.no_grad()
    def log_eval_group(self, xs, ys, group_size=10, sample=False, sample_num=40):
        if isinstance(xs, str | os.PathLike):
            xs = [os.path.join(xs, p) for p in os.listdir(xs)]
        if isinstance(ys, str | os.PathLike):
            ys = [os.path.join(ys, p) for p in os.listdir(ys)]
        
        metrics = dict()
        ext = '.'.join(xs[0].split('.')[1:])
        if ext == "nii.gz": load_fn = lambda x: sitk.GetArrayFromImage(sitk.ReadImage(x))
        elif ext == "npz": load_fn = lambda x: np.load(x)[np.load(x).files[0]]
        
        x_buffer, y_buffer, i_buffer_fills = [], [], 0
        for x, y in zip(xs, ys):
            if len(x_buffer) < group_size:
                x, y = map(lambda i: torch.tensor(load_fn(i)), [x, y])
                x_buffer.append(x)
                y_buffer.append(y)
            else:
                i_buffer_fills += 1
                x, y = torch.cat(x_buffer, dim=0), torch.cat(y_buffer, dim=0)
                xp, yp = map(lambda i: rearrange(i, "b c h w d -> (b h) c w d"), [x, y])
                if sample:
                    random_indices = torch.randperm(xp.shape[0])[:min(xp.shape[0], sample_num)]
                    xp, yp = map(lambda i: i[random_indices], [xp, yp])
                    
                if MetricType.fid in self.eval_scheme:
                    xp, yp = map(lambda i: torch.nn.functional.interpolate(
                        rearrange(i, "b c h w -> b 1 c h w"), (3, 299, 299)).squeeze())
                    self.fid.update(xp, real=False)
                    self.fid.update(yp, real=True)
                    fid = self.fid.compute()
                    metrics["FID"] = metrics.get("FID", 0) + fid.item()
                if MetricType.psnr in self.eval_scheme:
                    psnr = self.psnr(xp, yp)
                    metrics["PSNR"] = metrics.get("PSNR", 0) + psnr.item()
                    
        metrics = {k: v / i_buffer_fills for k, v in metrics.items()}
        return metrics
    
    
class InferMixedDiffusion(BaseInferDiffusion):
    def __init__(self, **diffusion_kwargs):
        super().__init__(**diffusion_kwargs)
        
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
                else: cropped.append([slice(0, t.shape[i] + 1) for i in range(len(t.shape))])
        return cropped
    
    @staticmethod
    def _interpolate(tensor, *args, **kwargs):
        mode = kwargs.get("mode", "nearest")
        if mode == "nearest":
            kwargs["mode"] = "trilinear"
            return nn.functional.interpolate(tensor, align_corners=True, *args, **kwargs).round()
        return nn.functional.interpolate(tensor, *args, **kwargs)
    
    @torch.no_grad()
    def log_images(self, batch, log_metrics=False, log_group_metrics_in_2d=False, *args, **kwargs):
        logs = dict()
        z, c, x, xrec, xc = self.get_input(batch, self.first_stage_key,
                                            return_first_stage_outputs=True,
                                            force_c_encode=True,
                                            return_original_cond=True,)
        logs["inputs"] = x
        logs["reconstruction"] = xrec
        b, *shp = x.shape
        
        assert b > 1
        alpha = kwargs.get("mixup_ratio", .2)  # lower -> more fine gt ; higher -> more generated
        eta = kwargs.get("ddim_eta", 1)
        sample = kwargs.get("sample", True)
        ddim_steps = kwargs.get("ddim_steps", 200)
        use_ddim = kwargs.get("ddim_steps", 200) is not None
        # the first item is finely labeled, while the rest are not
        cf = None if not self.first_stage_model.is_conditional else super(LatentDiffusion, self).get_input(batch, self.first_stage_model.cond_key)
        
        if sample:
            samples, _ = self.sample_log(cond=c, batch_size=b, ddim=use_ddim, ddim_steps=ddim_steps, eta=eta)
            logs["samples"] = self.decode_first_stage(samples, cf)
            
            z_fine = z[0:1]
            z_coarse = samples[1:]
            # convert mask shape to match latent shape
            mask_fine = self._interpolate((cf[0:1] > 0).to(torch.float32), z_fine.shape[2:], mode="nearest")
            mask_coarse = self._interpolate((cf[1:] > 0).to(torch.float32), z_coarse.shape[2:], mode="nearest")
            # crop foreground region
            z_fine_masked_cropped = self._get_foreground_bbox(mask_fine)
            z_coarse_masked_cropped = self._get_foreground_bbox(mask_coarse, use_shape_on_background=True)
            # resize fine foregrounds to coarse's size
            z_fine_reshaped, mask_fine_reshaped, mix_log = [], [], []
            for i in range(b - 1):
                if z_fine_masked_cropped[0] is not None and z_coarse_masked_cropped[i] is not None:
                    mix_log.append("ok")
                    z_fine_reshaped.append(self._interpolate(z_fine[*z_fine_masked_cropped[0]], z_coarse[*z_coarse_masked_cropped[i]].shape[2:], mode="trilinear"))
                    mask_fine_reshaped.append(self._interpolate(mask_fine[*z_fine_masked_cropped[0]], mask_coarse[*z_coarse_masked_cropped[i]].shape[2:], mode="nearest"))
                else:
                    if z_coarse_masked_cropped[i] is None:
                        mix_log.append("nocrop-C")
                        z_coarse_masked_cropped[i] = [slice(0, z_coarse[i].shape[j] + 1) for j in range(len(z_coarse[i].shape))]
                    mix_log.append("nocrop-F")
                    z_fine_reshaped.append(z_coarse[*z_coarse_masked_cropped[i]])
                    mask_fine_reshaped.append(mask_coarse[*z_coarse_masked_cropped[i]])
            # mixup
            z_local_mix, mask_local_mix = [], []
            for i in range(b - 1):
                z_fine_reshaped[i][:, :, mask_fine_reshaped[i][0, 0] == 0] = z_coarse[*z_coarse_masked_cropped[i]][:, :, mask_fine_reshaped[i][0, 0] == 0]
                mask_fine_reshaped[i][:, :, mask_fine_reshaped[i][0, 0] == 0] = mask_coarse[*z_coarse_masked_cropped[i]][:, :, mask_fine_reshaped[i][0, 0] == 0]
                z_local_mix.append(z_fine_reshaped[i] * (1 - alpha) + z_coarse[*z_coarse_masked_cropped[i]] * alpha)
                mask_local_mix.append(mask_fine_reshaped[i] * (1 - alpha))
                
            for i in range(b - 1): z_coarse[i, *z_coarse_masked_cropped[i][1:]] = z_local_mix[i]
            for i in range(b - 1): mask_coarse[i, *z_coarse_masked_cropped[i][1:]] = mask_local_mix[i]
            z_mix = torch.cat([z_fine, z_coarse], dim=0)
            mask_mix = self._interpolate(torch.cat([mask_fine, mask_coarse], dim=0), cf.shape[2:], mode="trilinear")
            
            x_samples = self.decode_first_stage(z_mix, cf)
            logs["samples_mixed"] = torch.cat([x_samples, mask_mix], dim=1)
            logs["mask_mixed"] = mask_mix
            logs["mix_log"] = ','.join(mix_log)
            
        x = logs["inputs"]
        x_samples = logs["samples"]
        if self.eval_scheme is not None and len(self.eval_scheme) > 0 and log_metrics:
            metrics = self.log_eval(x_samples, x, log_group_metrics_in_2d)
            # print(metrics)
            self.log_dict(metrics, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            return metrics, logs
        
        return None, logs


class InferLatentDiffusion(BaseInferDiffusion):
    def __init__(self, **diffusion_kwargs):
        super().__init__(**diffusion_kwargs)
    
    @torch.no_grad()
    def log_images(self, batch, log_metrics=False, log_group_metrics_in_2d=False, *args, **kwargs):
        logs = super(BaseInferDiffusion, self).log_images(batch, *args, **kwargs)
        
        x = logs["inputs"]
        x_samples = logs["samples"]
        if self.eval_scheme is not None and len(self.eval_scheme) > 0 and log_metrics:
            metrics = self.log_eval(x_samples, x, log_group_metrics_in_2d)
            # print(metrics)
            self.log_dict(metrics, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            return metrics, logs
        
        return None, logs