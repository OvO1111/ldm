import os
import json
import h5py
import torch
import omegaconf
import torch.distributed
import torch.nn as nn
import numpy as np
import SimpleITK as sitk
import scipy.ndimage as ndm

from enum import IntEnum
from tqdm import tqdm
from medpy.metric import binary
from einops import rearrange, repeat
from ldm.data.utils import load_or_write_split
from ldm.util import instantiate_from_config
from ldm.models.autoencoder import AutoencoderKL, VQModelInterface
from ldm.models.diffusion.ddpm import LatentDiffusion, ContrastiveDiffusion
from ldm.models.diffusion.cdpm_legacy import CategoricalDiffusion, OneHotCategoricalBCHW
from ldm.models.diffusion.classifier import CharacteristicClassifier
from ldm.models.downstream.efficient_subclass import EfficientSubclassSegmentation

from ldm.models.diffusion.ddim import make_ddim_timesteps
from ldm.modules.diffusionmodules.util import extract_into_tensor


def exists(x):
    return x is not None


class print_once:
    counter = 0
    def __call__(self, msg):
        if counter != 0: return
        print(msg)
        counter += 1


class MetricType(IntEnum):
    lpips = 1
    fid = 2
    psnr = 3
    fvd = 4
    
    
class ComputeMetrics:
    def __init__(self, eval_scheme, fvd_config=None):
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
            self.fvd_module: nn.Module = instantiate_from_config(fvd_config)
            self.fvd = FrechetInceptionDistance(feature=self.fvd_module, normalize=True)
        self.metrics = {"carry": []}
    
    @torch.no_grad()
    def log_eval(self, x, y, log_group_metrics_in_2d=False):
        metrics = dict()
        assert x.shape == y.shape
        b, c, *shp = x.shape
        
        if MetricType.lpips in self.eval_scheme:
            # lower = better (0, ?)
            perceptual = self.perceptual
            if c != 1: x, y = map(lambda i: i[:, 0:1], [x, y])
            x, y = map(lambda i: repeat(i, 'b 1 ... -> b c ...', c=3), [x, y])
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
                x, y = map(lambda i: rearrange(i, "b c h ... -> (b h) 1 c ..."), [x, y])
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
                
        self.metrics["carry"].append(metrics)
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
    
    
class MakeDataset:
    def __init__(self, 
                 dataset_base,
                 include_keys,
                 suffixes,
                 bs=1, create_split=False, dims=3, desc=None, overwrite=False):
        self.bs = bs
        self.dims = dims
        self.desc = desc
        self.base = dataset_base
        self.suffixes = suffixes
        self.include_keys = include_keys
        self.create_split = create_split
        self.overwrite = False
        
        self.dataset = dict()
        
    def add(self, samples, sample_names=None, dtypes={}):
        if not exists(sample_names): sample_names = [f"case_{len(self.dataset) + i}" for i in range(self.bs)]
        if isinstance(sample_names, str): sample_names = [sample_names]
        for i in range(len(sample_names)): 
            while sample_names[i] in self.dataset: sample_names[i] = sample_names[i] + "0"
            self.dataset[sample_names[i]] = {}
        
        for key in self.include_keys:
            value = samples[key]
            for b in range(self.bs):
                sample_name_b = sample_names[b]
                if not isinstance(value, dict):
                    value = {"value": value}
                for k, v in value.items():
                    value_b = v[b]
                    k = "_".join([key, k])
                    if isinstance(value_b, torch.Tensor):
                        suffix = self.suffixes.get(key, self.suffixes.get(k, ""))
                        f = os.path.join(self.base, k, sample_name_b + suffix)
                        os.makedirs(os.path.dirname(f), exist_ok=True)
                        if os.path.exists(f) and not self.overwrite:
                            time = [(file, os.path.getmtime(os.path.join(os.path.dirname(f), file))) for file in os.listdir(os.path.dirname(f))]
                            maxtimefile = max(time, key=lambda x: x[1])[0]
                            if maxtimefile[-len(suffix) - 2] == 'v': 
                                maxtime = int(maxtimefile[-len(suffix) - 1])
                            else: maxtime = 0
                            f = f.replace(suffix, f'_v{maxtime + 1}' + suffix)
                            # print(f"found existent file, saving new one at {f}. if this is not desired, u can set MakeDataset().overwrite to be True")
                        im = value_b.cpu().data.numpy().astype(dtypes.get(key, np.float32))
                        if suffix == '.nii.gz':
                            assert im.ndim == self.dims + 1, f"desired ndim {self.dims} and actual ndim {im.shape} not match"
                            sitk.WriteImage(sitk.GetImageFromArray(rearrange(self.postprocess(im), "c ... -> ... c")), f)
                            continue
                        elif suffix == '.npy':
                            np.save(f, im)
                            continue
                        self.dataset[sample_name_b][k] = "not saved"
                    else:
                        self.dataset[sample_name_b][k] = value_b
        return 
                    
    def postprocess(self, sample):
        # c h w d
        return sample
        
    def finalize(self, dt=None, **kw):
        dataset = {} | kw
        collect_dt = self.dataset if dt is None else dt
        dataset["data"] = collect_dt
        dataset["desc"] = self.desc
        dataset["keys"] = omegaconf.OmegaConf.to_container(self.include_keys)
        dataset["length"] = len(collect_dt)
        dataset["format"] = {k: self.suffixes.get(k, "raw") for k, v in self.suffixes.items()}
        
        if self.create_split:
            keys = list(collect_dt.keys())
            load_or_write_split(self.base, force=True, 
                                train=keys[:round(len(keys)*.7)],
                                val=keys[round(len(keys)*.7):round(len(keys)*.8)],
                                test=keys[round(len(keys)*.8):],)
        with open(os.path.join(self.base, "dataset.json"), "w") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=4)
    

class InferAutoencoderKL(AutoencoderKL, ComputeMetrics):
    def __init__(self, eval_scheme=[1], **autoencoder_kwargs):
        AutoencoderKL.__init__(self, **autoencoder_kwargs)
        ComputeMetrics.__init__(self, eval_scheme)
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
    def log_images(self, batch, log_metrics=False, log_group_metrics_in_2d=False, *args, **kwargs):
        logs = super(InferAutoencoderKL, self).log_images(batch, *args, **kwargs)
        x = logs["inputs"]
        x_recon = logs["reconstructions"]
        
        if self.eval_scheme is not None and len(self.eval_scheme) > 0 and log_metrics:
            metrics = self.log_eval(x_recon, x, log_group_metrics_in_2d)
            # print(metrics)
            self.log_dict(metrics, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            return metrics, logs
        return {}, logs
    
    
class InferAutoencoderVQ(VQModelInterface, ComputeMetrics, MakeDataset):
    def __init__(self, 
                 eval_scheme=[1],
                 save_dataset=False,
                 save_dataset_path=None,
                 include_keys=["image"],
                 suffix_keys={"image":".nii.gz",},
                 **diffusion_kwargs):
        if save_dataset:
            self.save_dataset = save_dataset
            assert exists(save_dataset_path)
            MakeDataset.__init__(self, save_dataset_path, include_keys, suffix_keys)
        VQModelInterface.__init__(self, **diffusion_kwargs)
        ComputeMetrics.__init__(self, eval_scheme)
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
    def log_images(self, batch, log_metrics=False, log_group_metrics_in_2d=False, *args, **kwargs):
        logs = super(VQModelInterface, self).log_images(batch, *args, **kwargs)
        x = logs["inputs"]
        x_recon = logs["reconstructions"]
        
        if self.save_dataset:
            self.add({"image": x_recon}, batch.get("casename"), dtypes={"image": np.uint8})
        
        if self.eval_scheme is not None and len(self.eval_scheme) > 0 and log_metrics:
            metrics = self.log_eval(x_recon, x, log_group_metrics_in_2d)
            # print(metrics)
            self.log_dict(metrics, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            return metrics, logs
        return {}, logs
        
    
class InferLatentDiffusion(LatentDiffusion, ComputeMetrics, MakeDataset):
    def __init__(self, 
                 eval_scheme=[1],
                 save_dataset=False,
                 save_dataset_path=None,
                 include_keys=["data", "text"],
                 suffix_keys={"data":".nii.gz",},
                 **diffusion_kwargs):
        if save_dataset:
            self.save_dataset = save_dataset
            assert exists(save_dataset_path)
            MakeDataset.__init__(self, save_dataset_path, include_keys, suffix_keys)
        LatentDiffusion.__init__(self, **diffusion_kwargs)
        ComputeMetrics.__init__(self, eval_scheme)
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
    def log_images(self, batch, log_metrics=False, log_group_metrics_in_2d=False, *args, **kwargs):
        logs = super(InferLatentDiffusion, self).log_images(batch, *args, **kwargs)
        
        x = logs["inputs"]
        x_samples = logs["samples"]
        if self.save_dataset:
            self.add(logs | batch, batch.get("casename"), dtypes={"image": np.float32})
            
        if self.eval_scheme is not None and len(self.eval_scheme) > 0 and log_metrics:
            metrics = self.log_eval(x_samples, x, log_group_metrics_in_2d)
            # print(metrics)
            self.log_dict(metrics, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            return metrics, logs
        
        return None, logs
    

class InferCategoricalDiffusion(CategoricalDiffusion, ComputeMetrics, MakeDataset):
    def __init__(self, 
                 eval_scheme=[1],
                 save_dataset=False,
                 save_dataset_path=None,
                 include_keys=["data", "text"],
                 suffix_keys={"data":".nii.gz",},
                 **diffusion_kwargs):
        CategoricalDiffusion.__init__(self, **diffusion_kwargs)
        ComputeMetrics.__init__(self, eval_scheme)
        self.save_dataset = save_dataset
        if save_dataset:
            assert exists(save_dataset_path)
            MakeDataset.__init__(self, save_dataset_path, include_keys, suffix_keys)
        self.eval()
    
    def on_test_start(self, *args):
        if MetricType.fid in self.eval_scheme:
            self.fid = self.fid.to(self.device)
        if MetricType.fvd in self.eval_scheme:
            self.fvd = self.fvd.to(self.device)
    
    def on_test_end(self, *args):
        if self.save_dataset:
            datasets = torch.distributed.gather_object(self.dataset)
            metrics = torch.distributed.gather_object(self.metrics)
            mean_metrics = {"mean_metrics": {m: np.mean([c[m] for c in metrics]) for m in metrics[0].keys()},
                            "ind_metrics": metrics}
            self.finalize(datasets, mean_metrics)
        else:
            self.finalize()
        
    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        pass
    
    # @torch.no_grad()
    # def _log_images(self, batch, split="train", sample=True, use_ddim=False, ddim_steps=50, ddim_eta=1., verbose=False, **kwargs):
    #     x0, c = self.get_input(batch, self.data_key, self.cond_key)
    #     noise = self.get_noise(x0)
        
    #     logs = dict()
    #     b = x0.shape[0]
    #     logs["inputs"] = x0.argmax(1)
    #     logs["conditioning"] = str(batch["text"])
    #     if sample:
    #         t = torch.randint(950, self.num_timesteps, (b,), device=self.device).long()
    #         q_xt = self.q_xt_given_x0(x0, t, noise if self.parameterization != "kl" else None)
    #         model_outputs = self.model(q_xt, t, **c)
    #         if isinstance(model_outputs, dict): 
    #             model_outputs = model_outputs["diffusion_out"]
    #         x0pred = OneHotCategoricalBCHW(model_outputs).max_prob_sample()
            
    #         logs["t"] = str(t.cpu().numpy().tolist())
    #         logs["xt"] = OneHotCategoricalBCHW(q_xt).sample().argmax(1)
    #         logs["x0pred"] = x0pred.argmax(1)
            
    #         lpips = self.lpips_loss(x0pred.argmax(1, keepdim=True), x0.argmax(1, keepdim=True))
    #         self.log(f"{split}/lpips_metric", lpips, prog_bar=True, on_step=True)
            
    #         if not use_ddim: logs = logs | self.p_sample(q_xt, c, verbose=verbose, **kwargs)
    #         else: logs = logs | self.p_sample_ddim(q_xt, c, make_ddim_timesteps("uniform", 
    #                                                                             num_ddim_timesteps=ddim_steps, 
    #                                                                             num_ddpm_timesteps=self.timesteps,
    #                                                                             verbose=verbose), verbose=verbose, **kwargs)
    #         x0pred = logs["samples"]

    #         lpips = self.lpips_loss(x0pred.unsqueeze(1), x0.argmax(1, keepdim=True))
    #         self.log(f"{split}/lpips_metric", lpips, prog_bar=True, on_step=True)
            
    #     return logs
    
    @torch.no_grad()
    def p_sample(self, q_xT=None, c=None, verbose=False, timesteps=None,
                 plot_progressive_rows=False, 
                 plot_denoising_rows=False, plot_diffusion_every_t=200):
        logs = dict()
        with self.ema_scope():
            c = c if exists(c) else dict()
            p_xt, b = q_xT, q_xT.shape[0]
            timesteps = self.timesteps if not exists(timesteps) else timesteps
            t_values = reversed(range(1, timesteps)) if not verbose else tqdm(reversed(range(1, timesteps)), total=timesteps-1, desc="sampling progress")
            
            if plot_denoising_rows: denoising_rows = []
            if plot_progressive_rows: progressive_rows = []
            for t in t_values:
                t_ = torch.full(size=(b,), fill_value=t, device=q_xT.device)
                model_outputs = self.model(p_xt, t_, **c)
                if isinstance(model_outputs, dict): 
                    model_outputs = model_outputs["diffusion_out"]
                    
                if self.parameterization == "eps":
                    alphas_t = extract_into_tensor(self.alphas, t_, p_xt.shape)
                    p_x0_given_xt = (p_xt - (1 - alphas_t) * model_outputs) / alphas_t
                elif self.parameterization == "x0":
                    p_x0_given_xt = model_outputs
                
                if self.parameterization == "kl":
                    p_x0_given_xt = model_outputs
                    p_xt = torch.clamp(self.q_xtm1_given_x0pred_xt(p_xt, p_x0_given_xt, t_), min=1e-12)
                    p_xt = OneHotCategoricalBCHW(probs=p_xt).sample()
                else:
                    alphas_t = extract_into_tensor(self.alphas, t_, p_xt.shape)
                    cumalphas_t = extract_into_tensor(self.alphas_cumprod, t_, p_xt.shape)
                    # cumalphas_tm1 = cumalphas_t / alphas_t
                    # xt_coeff = (alphas_t - cumalphas_t) / (1 - cumalphas_t)
                    # x0_coeff = alphas_t * (cumalphas_tm1 - cumalphas_t) ** 2 / (1 - cumalphas_t) ** 2
                    # p_xt = torch.clamp(xt_coeff * p_xt ** 2 + x0_coeff * (p_x0_given_xt * p_xt - p_x0_given_xt ** 2), min=1e-12)
                    # p_xt = p_xt / p_xt.sum(1, keepdim=True)
                    
                    # eps = self.get_noise(torch.zeros_like(q_xT))
                    
                    eps = OneHotCategoricalBCHW(logits=(p_xt - p_x0_given_xt * cumalphas_t) / (1 - cumalphas_t)).sample()
                    p_xt = torch.clamp(self.q_xtm1_given_x0_xt(p_xt, p_x0_given_xt, t_, eps), min=1e-12)
                
                if plot_denoising_rows and t % plot_diffusion_every_t == 0: denoising_rows.append(p_x0_given_xt)
                if plot_progressive_rows and t % plot_diffusion_every_t == 0: progressive_rows.append(p_xt)

        if self.p_x1_sample == "majority":
            x0pred = OneHotCategoricalBCHW(probs=p_xt).max_prob_sample()
        elif self.p_x1_sample == "confidence":
            x0pred = OneHotCategoricalBCHW(probs=p_xt).prob_sample()
            
        logs["samples"] = x0pred.argmax(1)
        if plot_denoising_rows and len(denoising_rows) > 0:
            denoising_rows = OneHotCategoricalBCHW(probs=torch.cat(denoising_rows, dim=0)).max_prob_sample()
            logs["p(x0|xt) at different timestep"] = denoising_rows.argmax(1)
        if plot_progressive_rows and len(progressive_rows) > 0:
            progressive_rows = OneHotCategoricalBCHW(probs=torch.cat(progressive_rows, dim=0)).max_prob_sample()
            logs["p(x_{t-1}|xt) at different timestep"] = progressive_rows.argmax(1)
        
        return logs
    
    @torch.no_grad()
    def p_sample_ddim(self, q_xT, c, ddim_timesteps, verbose=False, **kwargs):
        logs = dict()
        
        def q_xtm1_given_x0_xt_ddim(xt, x0, ddim_t, ddim_tm1, noise=None):
            # computes q_xtm1 given q_xt, q_x0, noise
            alphas_t = extract_into_tensor(self.alphas, ddim_t, x0.shape)
            alphas_cumprod_tm1 = extract_into_tensor(self.alphas_cumprod_prev, ddim_tm1, x0.shape)
            if exists(noise): theta = ((alphas_t * xt + (1 - alphas_t) * noise) * (alphas_cumprod_tm1 * x0 + (1 - alphas_cumprod_tm1) * noise))
            else: theta = ((alphas_t * xt + (1 - alphas_t) / self.num_classes) * (alphas_cumprod_tm1 * x0 + (1 - alphas_cumprod_tm1) / self.num_classes))
            return theta / theta.sum(dim=1, keepdim=True)
        
        def q_xtm1_given_x0pred_xt_ddim(xt, x0pred, ddim_t, ddim_tm1, noise=None):
            alphas_t = extract_into_tensor(self.alphas, ddim_t, xt.shape)
            alphas_cumprod_tm1 = extract_into_tensor(self.alphas_cumprod_prev, ddim_tm1, xt.shape)
            
            x0 = torch.eye(self.num_classes, device=xt.device)[None, :, :, None, None]
            if self.dims == 3: x0 = x0[..., None]
            theta_xt_xtm1 = alphas_t * xt + (1 - alphas_t) / self.num_classes
            theta_xtm1_x0 = alphas_cumprod_tm1 * x0 + (1 - alphas_cumprod_tm1) / self.num_classes
            aux = theta_xt_xtm1[:, :, None] * theta_xtm1_x0
            theta_xtm1_xtx0 = aux / aux.sum(dim=1, keepdim=True)
            return torch.einsum("bcd...,bd...->bc...", theta_xtm1_xtx0, x0pred)
        
        p_xt, b = q_xT, q_xT.shape[0]
        t_values = list(reversed(range(1, len(ddim_timesteps)))) 
        iterator = t_values if not verbose else tqdm(t_values, total=len(ddim_timesteps), desc="ddim sampling progress")
        for index, t in enumerate(iterator):
            t_ = torch.full(size=(b,), fill_value=ddim_timesteps[t], device=q_xT.device)
            t_m1 = torch.full(size=(b,), fill_value=ddim_timesteps[t_values[index + 1] if index + 1 < len(t_values) else 0], device=q_xT.device)
            
            model_outputs = self.model(p_xt, t_, **c)
            if isinstance(model_outputs, dict): 
                model_outputs = model_outputs["diffusion_out"]
                
            if self.parameterization == "eps":
                alphas_t = extract_into_tensor(self.alphas, t_, p_xt.shape)
                p_x0_given_xt = (p_xt - (1 - alphas_t) * model_outputs) / alphas_t
            elif self.parameterization == "x0":
                p_x0_given_xt = model_outputs
                
            if self.parameterization == "kl":
                p_x0_given_xt = model_outputs
                p_xt = torch.clamp(q_xtm1_given_x0pred_xt_ddim(p_xt, p_x0_given_xt, t_, t_m1), min=1e-12)
                p_xt = OneHotCategoricalBCHW(probs=p_xt).sample()
            else:
                eps = self.get_noise(torch.zeros_like(q_xT))
                p_xt = torch.clamp(q_xtm1_given_x0_xt_ddim(p_xt, p_x0_given_xt, t_, t_m1, eps), min=1e-12)
            
        if self.p_x1_sample == "majority":
            x0pred = OneHotCategoricalBCHW(probs=p_xt).max_prob_sample()
        elif self.p_x1_sample == "confidence":
            x0pred = OneHotCategoricalBCHW(probs=p_xt).prob_sample()
            
        logs["samples"] = x0pred.argmax(1)
        return logs
    
    @torch.no_grad()
    def log_images(self, batch, log_metrics=False, log_group_metrics_in_2d=False, **kwargs):
        logs = super(InferCategoricalDiffusion, self).log_images(batch, **kwargs)
        x = logs["inputs"]
        x_recon = logs["samples"]
        if x.ndim < self.dims + 2: x = x[:, None]
        if x_recon.ndim < self.dims + 2: x_recon = x_recon[:, None]
        
        if self.save_dataset:
            self.add({"inputs": x, "samples": x_recon}, batch.get("casename"), dtypes={"inputs": np.uint8, "samples": np.uint8})
        
        if self.eval_scheme is not None and len(self.eval_scheme) > 0 and log_metrics:
            metrics = self.log_eval(x_recon, x, log_group_metrics_in_2d)
            # print(metrics)
            self.log_dict(metrics, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            return metrics, logs
        return {}, logs
    
    def postprocess(self, sample):
        # n h w d
        all_labels = np.unique(sample).flatten()
        for i in all_labels:
            labeled_sample, _ = ndm.label(sample[0] == i)
            labeled_sample_counts = np.bincount(labeled_sample.flatten())
            few_labels = np.argwhere(labeled_sample_counts < 50).flatten()
            
            for j in few_labels:
                sample[:, labeled_sample == j] = 255
            while (sample == 255).sum() > 0:
                itr_surrounding = ndm.binary_dilation(sample[0] == 255)
                itr_uniques = np.unique(sample[0, itr_surrounding]).flatten()
                for isur in itr_uniques:
                    if isur != 255: sample[:, itr_surrounding & ndm.binary_dilation(sample[0] == isur, mask=(sample[0] == 255))] = isur
        return sample.astype(np.uint8)
        

class InferMixedDiffusion(InferLatentDiffusion):
    def __init__(self, 
                 mix_scheme="latent", 
                 **diffusion_kwargs):
        self.mix_scheme = mix_scheme
        super().__init__(**diffusion_kwargs)
        
    def add(self, samples, sample_names=None, dtypes={}, b_mapping={}):
        if not exists(sample_names): sample_names = [f"case_{len(self.dataset) + i}" for i in range(self.bs)]
        if isinstance(sample_names, str): sample_names = [sample_names]
        for i in range(len(sample_names)): 
            while sample_names[i] in self.dataset: sample_names[i] = sample_names[i] + "0"
        B = samples[self.include_keys[0]].shape[0]
        
        for b in range(B):
            sample_name_b = 'syn' + sample_names[b]
            f = os.path.join(self.base, "data", sample_name_b + ".h5")
            h5 = h5py.File(f, "w")
            for key in self.include_keys:
                b_ = b_mapping[sample_names[b]].get(key, b_mapping[sample_names[b]]["others"])
                value_b = samples[key][b_]
                if isinstance(value_b, torch.Tensor):
                    im = value_b.cpu().data.numpy().astype(dtypes.get(key, np.float32))
                    h5.create_dataset(key, data=self.postprocess(im), compression="gzip")
            self.dataset[sample_name_b] = f
            h5.close()
        
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
    
    def on_test_end(self,):
        return
    
    @staticmethod
    def _interpolate(tensor, *args, **kwargs):
        mode = kwargs.get("mode", "nearest")
        if mode == "nearest":
            kwargs["mode"] = "trilinear"
            return nn.functional.interpolate(tensor, align_corners=True, *args, **kwargs).round()
        return nn.functional.interpolate(tensor, *args, **kwargs)
    
    @torch.no_grad()
    def log_images(self, batch, log_metrics=False, log_group_metrics_in_2d=False, alpha=.2, *args, **kwargs):
        # alpha is the mixup ratio, mix=alpha*sample+(1-alpha)*gt lower -> more fine gt ; higher -> more generated
        logs = dict()
        z, c, x, xrec, xc = self.get_input(batch, self.first_stage_key,
                                            return_first_stage_outputs=True,
                                            force_c_encode=True,
                                            return_original_cond=True,)
        logs["inputs"] = x
        logs["reconstruction"] = xrec
        B, *shp = x.shape
        
        assert B > 1
        eta = kwargs.get("ddim_eta", 1)
        sample = kwargs.get("sample", True)
        ddim_steps = kwargs.get("ddim_steps", 200)
        use_ddim = kwargs.get("ddim_steps", 200) is not None
        # the first item is finely labeled, while the rest are not
        cx = batch["mask"].clone()
        cf = super(LatentDiffusion, self).get_input(batch, self.first_stage_model.cond_key).to(self.device) if self.is_conditional_first_stage else None
        
        if sample:
            inputs = x
            samples, _ = self.sample_log(cond=c, batch_size=B, ddim=use_ddim, ddim_steps=ddim_steps, eta=eta)
            logs["samples"] = self.decode_first_stage(samples, cf)
            primary_batch_size = self.trainer.datamodule.batch_sampler.primary_batch_size
            print(f"mixing {batch['casename'][:primary_batch_size]} ({1-alpha:.2f}*gt) with {batch['casename'][primary_batch_size:]} ({alpha:.2f}*samples)")
            
            if self.mix_scheme == "latent":
                z_fines = z[:primary_batch_size]
                z_mix, mask_mix, mix_log = [], [], []
                for b in range(primary_batch_size):
                    z_fine = z_fines[b: b + 1]
                    z_coarse = z[primary_batch_size:]
                    # convert mask shape to match latent shape
                    mask_fine = cx[b:b+1].float()
                    mask_coarse = cx[primary_batch_size:].float()
                    mask_fine_down = self._interpolate(cx[b: b+1].to(torch.float32), z_fine.shape[2:], mode="nearest")
                    mask_coarse_down = self._interpolate((cx[primary_batch_size:] > 0).to(torch.float32), z_coarse.shape[2:], mode="nearest")
                    # crop foreground region
                    i_fine_masked_cropped = self._get_foreground_bbox(mask_fine)
                    i_coarse_masked_cropped = self._get_foreground_bbox(mask_coarse, use_shape_on_background=True)
                    z_fine_masked_cropped = self._get_foreground_bbox(mask_fine_down)
                    z_coarse_masked_cropped = self._get_foreground_bbox(mask_coarse_down, use_shape_on_background=True)
                    # resize fine foregrounds to coarse's size
                    z_fine_reshaped, mask_fine_reshaped, mask_fine_reshaped_down = [], [], []
                    for i in range(B - primary_batch_size):
                        if z_fine_masked_cropped[0] is not None and z_coarse_masked_cropped[i] is not None:
                            mix_log.append("ok")
                            z_fine_reshaped.append(self._interpolate(z_fine[*z_fine_masked_cropped[0]], z_coarse[*z_coarse_masked_cropped[i]].shape[2:], mode="trilinear"))
                            mask_fine_reshaped_down.append(self._interpolate(mask_fine[*z_fine_masked_cropped[0]], mask_coarse[*z_coarse_masked_cropped[i]].shape[2:], mode="nearest"))
                            mask_fine_reshaped.append(self._interpolate(mask_fine[*i_fine_masked_cropped[0]], mask_coarse[*i_coarse_masked_cropped[i]].shape[2:], mode="nearest"))
                        else:
                            if z_coarse_masked_cropped[i] is None:
                                mix_log.append("nocrop-C")
                                z_coarse_masked_cropped[i] = [slice(0, z_coarse[i].shape[j] + 1) for j in range(len(z_coarse[i].shape))]
                            mix_log.append("nocrop-F")
                            z_fine_reshaped.append(z_coarse[*z_coarse_masked_cropped[i]])
                            mask_fine_reshaped_down.append(mask_coarse[*z_coarse_masked_cropped[i]])
                            mask_fine_reshaped.append(mask_coarse[*i_coarse_masked_cropped[i]])
                    # mixup
                    z_local_mix, mask_local_mix = [], []
                    for i in range(B - primary_batch_size):
                        z_fine_reshaped[i][:, :, mask_fine_reshaped_down[i][0, 0] == 0] = z_coarse[*z_coarse_masked_cropped[i]][:, :, mask_fine_reshaped_down[i][0, 0] == 0]
                        # mask_fine_reshaped[i][:, :, mask_fine_reshaped[i][0, 0] == 0] = mask_coarse[*z_coarse_masked_cropped[i]][:, :, mask_fine_reshaped[i][0, 0] == 0]
                        mask_fine_reshaped[i][:, :, mask_coarse[*i_coarse_masked_cropped[i]][0, 0] == 0] = 0
                        z_local_mix.append(z_fine_reshaped[i] * (1 - alpha) + z_coarse[*z_coarse_masked_cropped[i]] * alpha)
                        mask_local_mix.append(mask_fine_reshaped[i])
                        
                    for i in range(B - primary_batch_size): z_coarse[i, *z_coarse_masked_cropped[i][1:]] = z_local_mix[i]
                    # for i in range(B - primary_batch_size): mask_coarse[i, *z_coarse_masked_cropped[i][1:]] = mask_local_mix[i]
                    for i in range(B - primary_batch_size): mask_coarse[i, *i_coarse_masked_cropped[i][1:]] = mask_local_mix[i]
                    z_mix.append(z_coarse)
                    mask_mix.append(self._interpolate(mask_coarse, cx.shape[2:], mode="trilinear").round())
                
                z_mix, mask_mix = map(lambda x: torch.cat(x, dim=0), [z_mix, mask_mix])
                x_samples = self.decode_first_stage(z_mix, cx[primary_batch_size:])
                logs["image"] = logs["inputs"][:primary_batch_size] # fine image
                logs["label"] = batch["fine"][:primary_batch_size].float() # fine label
                logs["premix"] = logs["inputs"][primary_batch_size:] # coarse image
                logs["premix_label"] = batch["coarse"][primary_batch_size:].float() # coarse label
                logs["mixed_samples"] = x_samples  # mixed image
                logs["mixed_fine"] = mask_mix      # mixed fine label
                # logs["mixed_coarse"] = ((mask_mix > 0) | (cx[primary_batch_size:] > 0)).float()
                logs["mix_log"] = ','.join(mix_log)
            
            elif self.mix_scheme == "direct":
                i_fine, i_coarse = logs["inputs"][:primary_batch_size].clone(), logs["samples"][primary_batch_size:].clone()
                i_mix, mask_mix, mix_log = [], [], []
                mask_coarse = cx[primary_batch_size:].float()
                for b in range(primary_batch_size):
                    mask_fine = cx[b:b+1].float()
                    # crop foreground region
                    i_fine_masked_cropped = self._get_foreground_bbox(mask_fine)
                    i_coarse_masked_cropped = self._get_foreground_bbox(mask_coarse, use_shape_on_background=True)
                    # resize fine foregrounds to coarse's size
                    i_fine_reshaped, mask_fine_reshaped = [], []
                    for i in range(B - primary_batch_size):
                        if i_fine_masked_cropped[0] is not None and i_coarse_masked_cropped[i] is not None:
                            mix_log.append("ok")
                            i_fine_reshaped.append(self._interpolate(i_fine[*i_fine_masked_cropped[0]], i_coarse[*i_coarse_masked_cropped[i]].shape[2:], mode="trilinear"))
                            mask_fine_reshaped.append(self._interpolate(mask_fine[*i_fine_masked_cropped[0]], mask_coarse[*i_coarse_masked_cropped[i]].shape[2:], mode="nearest"))
                        else:
                            if i_coarse_masked_cropped[i] is None:
                                mix_log.append("nocrop-C")
                                i_coarse_masked_cropped[i] = [slice(0, i_coarse[i].shape[j] + 1) for j in range(len(i_coarse[i].shape))]
                            mix_log.append("nocrop-F")
                            i_fine_reshaped.append(i_coarse[*i_coarse_masked_cropped[i]])
                            mask_fine_reshaped.append(mask_coarse[*i_coarse_masked_cropped[i]])
                    # mixup
                    i_local_mix, mask_local_mix = [], []
                    for i in range(B - primary_batch_size):
                        i_fine_reshaped[i][:, :, mask_coarse[*i_coarse_masked_cropped[i]][0, 0] == 0] = i_coarse[*i_coarse_masked_cropped[i]][:, :, mask_coarse[*i_coarse_masked_cropped[i]][0, 0] == 0]
                        mask_fine_reshaped[i][:, :, mask_coarse[*i_coarse_masked_cropped[i]][0, 0] == 0] = 0
                        i_local_mix.append(i_fine_reshaped[i] * (1 - alpha) + i_coarse[*i_coarse_masked_cropped[i]] * alpha)
                        mask_local_mix.append(mask_fine_reshaped[i])
                        
                    for i in range(B - primary_batch_size): i_coarse[i, *i_coarse_masked_cropped[i][1:]] = i_local_mix[i]
                    for i in range(B - primary_batch_size): mask_coarse[i, *i_coarse_masked_cropped[i][1:]] = mask_local_mix[i]
                    i_mix.append(i_coarse)
                    mask_mix.append(mask_coarse)
                
                i_mix, mask_mix = torch.cat(i_mix, dim=0), torch.cat(mask_mix, dim=0)
                z_mix = self.encode_first_stage(i_mix, cf[primary_batch_size:])
                # z_mix = self.p_sample_loop(cond=c[primary_batch_size:], shape=z_mix.shape, x_T=z_mix, timesteps=100, verbose=False)
                i_mix = self.decode_first_stage(z_mix, cf[primary_batch_size:])
                logs["samples"] = logs["samples"][primary_batch_size:]
                logs["mixed_samples"] = i_mix
                logs["mixed_fine"] = mask_mix
                logs["mixed_coarse"] = ((mask_mix > 0) | (cx[primary_batch_size:] > 0)).float()
                logs["mix_log"] = ','.join(mix_log)
                logs["alpha"] = torch.tensor([alpha,] * i_mix.shape[0])[:, None]
                
        x = logs["inputs"]
        x_samples = logs["samples"]
        
        if self.save_dataset:
            primary_casename = batch.get("casename")[:primary_batch_size]
            secondary_casename = batch.get("casename")[primary_batch_size:]
            casenames = ['+'.join([p, s]) for p in primary_casename for s in secondary_casename]
            mapping = {'+'.join([p, s]): {"samples": ip, "others": iis} for ip, p in enumerate(primary_casename) for iis, s in enumerate(secondary_casename)}
            self.add(logs, 
                     casenames, 
                     dtypes={"mixed_fine": np.uint8, "mixed_coarse": np.uint8},
                     b_mapping=mapping)
        
        if self.eval_scheme is not None and len(self.eval_scheme) > 0 and log_metrics:
            metrics = self.log_eval(x_samples, x, log_group_metrics_in_2d)
            # print(metrics)
            self.log_dict(metrics, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            return metrics, logs
        
        return None, logs
    
    
class InferMixedAutoencoderVQ(VQModelInterface, ComputeMetrics, MakeDataset):
    def __init__(self, 
                 eval_scheme=[1],
                 save_dataset=False,
                 save_dataset_path=None,
                 include_keys=["data", "text"],
                 suffix_keys={"data":".nii.gz",},
                 **diffusion_kwargs):
        VQModelInterface.__init__(self, **diffusion_kwargs)
        ComputeMetrics.__init__(self, eval_scheme)
        self.save_dataset = save_dataset
        if save_dataset:
            assert exists(save_dataset_path)
            MakeDataset.__init__(self, save_dataset_path, include_keys, suffix_keys)
        self.eval()
        
    def add(self, samples, sample_names=None, dtypes={}, b_mapping={}):
        if not exists(sample_names): sample_names = [f"case_{len(self.dataset) + i}" for i in range(self.bs)]
        if isinstance(sample_names, str): sample_names = [sample_names]
        for i in range(len(sample_names)): 
            while sample_names[i] in self.dataset: sample_names[i] = sample_names[i] + "0"
        B = samples[self.include_keys[0]].shape[0]
        
        for b in range(B):
            sample_name_b = 'syn' + sample_names[b]
            f = os.path.join(self.base, "data", sample_name_b + ".h5")
            h5 = h5py.File(f, "w")
            for key in self.include_keys:
                b_ = b_mapping[sample_names[b]].get(key, b_mapping[sample_names[b]]["others"])
                value_b = samples[key][b_]
                if isinstance(value_b, torch.Tensor):
                    im = value_b.cpu().data.numpy().astype(dtypes.get(key, np.float32))
                    h5.create_dataset(key, data=self.postprocess(im), compression="gzip")
            self.dataset[sample_name_b] = f
            h5.close()
        
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
    def test_step(self, batch, batch_idx):
        pass
    
    @torch.no_grad()
    def log_images(self, batch, log_metrics=False, log_group_metrics_in_2d=False, alpha=.2, *args, **kwargs):
        logs = dict()
        logs["inputs"] = self.get_input(batch, self.image_key)
        B = logs["inputs"].shape[0]
        primary_batch_size = self.trainer.datamodule.batch_sampler.primary_batch_size
        cx = batch["mask"].clone()
        cf = self.get_input(batch, self.cond_key) if self.is_conditional else None
        
        i_fine, i_coarse = logs["inputs"][:primary_batch_size].clone(), logs["inputs"][primary_batch_size:].clone()
        mask_fine_all = cx[:primary_batch_size].float()
        mask_coarse = cx[primary_batch_size:].float()
        i_fine_masked_cropped_all = self._get_foreground_bbox(mask_fine_all)
        i_coarse_masked_cropped = self._get_foreground_bbox(mask_coarse, use_shape_on_background=True)
        z_fine_masked_cropped_all = [self.encode(i_fine[*i_fine_masked_cropped_all[i]], cf[i][None][*i_fine_masked_cropped_all[i]]) for i in range(primary_batch_size)]
        z_coarse_masked_cropped_all = [self.encode(i_coarse[*i_coarse_masked_cropped[i]], cf[primary_batch_size+i][None][*i_coarse_masked_cropped[i]]) for i in range(B - primary_batch_size)]
        
        mask_mix, mix_log = [], []
        for b in range(primary_batch_size):
            # convert mask shape to match latent shape
            mask_fine = cx[b:b+1].float()
            mask_coarse = cx[primary_batch_size:].float()
            # resize fine foregrounds to coarse's size
            z_fine_reshaped = self._interpolate(z_fine_masked_cropped_all[b], (16, 16, 16), mode='trilinear')
            z_coarse_reshaped = [self._interpolate(z_coarse_masked_cropped_all[i], (16, 16, 16), mode='trilinear') for i in range(B - primary_batch_size)]
            mask_fine_reshaped = [self._interpolate(mask_fine[*i_fine_masked_cropped_all[b]], mask_coarse[*i_coarse_masked_cropped[i]].shape[2:], mode="nearest") for i in range(B - primary_batch_size)]
            mask_fine_reshaped_down = self._interpolate(mask_fine[*i_fine_masked_cropped_all[b]], (16, 16, 16), mode="nearest")
            mask_coarse_reshaped_down = [self._interpolate(mask_coarse[*i_coarse_masked_cropped[i]], (16, 16, 16), mode="nearest") for i in range(B - primary_batch_size)]

            z_local_mix, mask_local_mix = [], []
            for i in range(B - primary_batch_size):
                z_fine_reshaped[:, :, mask_fine_reshaped_down[0, 0] == 0] = z_coarse_reshaped[i][:, :, mask_fine_reshaped_down[0, 0] == 0]
                mask_fine_reshaped[i][:, :, mask_coarse[*i_coarse_masked_cropped[i]][0, 0] == 0] = 0
                z_local_mix.append(z_fine_reshaped[i] * (1 - alpha) + z_coarse_reshaped[i] * alpha)
                mask_local_mix.append(mask_fine_reshaped[i])
            
            i_local_mix, i_mix = [], i_coarse.clone()
            for i in range(B - primary_batch_size): 
                i_mix[*i_coarse_masked_cropped[i]] = self._interpolate(self.decode(z_local_mix[i], 
                                                                                      self._interpolate(((mask_coarse_reshaped_down[i] > 0) | (mask_fine_reshaped_down > 0)).float(), (64, 64, 64), mode="nearest")),
                                                                          mask_coarse[*i_coarse_masked_cropped[i]].shape[2:], mode="trilinear")
                
            for i in range(B - primary_batch_size): mask_coarse[i, *i_coarse_masked_cropped[i][1:]] = mask_local_mix[i]
            mask_mix.append(self._interpolate(mask_coarse, cx.shape[2:], mode="trilinear").round())
        
        mask_mix = torch.cat(mask_mix, dim=0)
        logs["image"] = logs["inputs"][:primary_batch_size] # fine image
        logs["label"] = batch["fine"][:primary_batch_size].float() # fine label
        logs["premix"] = logs["inputs"][primary_batch_size:] # coarse image
        logs["premix_label"] = batch["coarse"][primary_batch_size:].float() # coarse label
        logs["mixed_samples"] = i_mix  # mixed image
        logs["mixed_fine"] = mask_mix      # mixed fine label
        # logs["mixed_coarse"] = ((mask_mix > 0) | (cx[primary_batch_size:] > 0)).float()
        logs["mix_log"] = ','.join(mix_log)
        
        if self.save_dataset:
            primary_casename = batch.get("casename")[:primary_batch_size]
            secondary_casename = batch.get("casename")[primary_batch_size:]
            casenames = ['+'.join([p, s]) for p in primary_casename for s in secondary_casename]
            mapping = {'+'.join([p, s]): {"samples": ip, "others": iis} for ip, p in enumerate(primary_casename) for iis, s in enumerate(secondary_casename)}
            self.add(logs, 
                     casenames, 
                     dtypes={"mixed_fine": np.uint8, "mixed_coarse": np.uint8},
                     b_mapping=mapping)
        
        if self.eval_scheme is not None and len(self.eval_scheme) > 0 and log_metrics:
            metrics = self.log_eval(i_mix, i, log_group_metrics_in_2d)
            # print(metrics)
            self.log_dict(metrics, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            return metrics, logs
        
        return None, logs
    


class InferSubclassSegmentation(EfficientSubclassSegmentation, MakeDataset):
    def __init__(self, 
                 save_dataset=False,
                 save_dataset_path=None,
                 include_keys=["metrics"],
                 suffix_keys={},
                 **kw):
        if save_dataset:
            self.save_dataset = save_dataset
            assert exists(save_dataset_path)
            MakeDataset.__init__(self, save_dataset_path, include_keys, suffix_keys)
        EfficientSubclassSegmentation.__init__(self, **kw)
        self.eval()
        
    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        pass
    
    def on_test_end(self,):
        if self.save_dataset:
            datasets = torch.distributed.gather_object(self.dataset)
        mean_mt = {f"test_mean/{fn}/{c}": np.mean([d["metrics"][f"test/{fn}/{c}"] for d in datasets.values()]) for fn in ["dc", "precision", "recall"] for c in range(1, self.n_fine)}
        mean_mt = mean_mt | {f"test_mean/{fn}": np.mean([mean_mt[f"test_mean/{fn}/{c}"] for c in range(1, self.n_fine)]) for fn in ["dc", "precision", "recall"]}
        self.finalize(datasets, mean_metrics=mean_mt)
    
    @torch.no_grad()
    def log_images(self, batch, *args, **kwargs):
        logs = super().log_images(batch, *args, **kwargs)
        logs["metrics"] = self._multiclass_metrics(logs["seg_fine"].cpu().numpy(), logs["gt_fine"].cpu().numpy(), "test")
        
        if self.save_dataset:
            self.add(logs, batch.get("casename"), dtypes={"image": np.uint8})

        return logs["metrics"], logs


class InferCharacteristicClassifier(CharacteristicClassifier, MakeDataset):
    def __init__(self, 
                 save_dataset=False,
                 save_dataset_path=None,
                 include_keys=["metrics"],
                 suffix_keys={},
                 **kw):
        if save_dataset:
            self.save_dataset = save_dataset
            assert exists(save_dataset_path)
            MakeDataset.__init__(self, save_dataset_path, include_keys, suffix_keys)
        CharacteristicClassifier.__init__(self, **kw)
        self.eval()
        
    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        pass
    
    def on_test_end(self,):
        if self.save_dataset:
            datasets = torch.distributed.gather_object(self.dataset)
        mean_mt = {f"test_mean/{fn}/{c}": np.mean([d["metrics"][f"test/{fn}/{c}"] for d in datasets.values()]) for fn in ["dc", "precision", "recall"] for c in range(1, self.n_fine)}
        mean_mt = mean_mt | {f"test_mean/{fn}": np.mean([mean_mt[f"test_mean/{fn}/{c}"] for c in range(1, self.n_fine)]) for fn in ["dc", "precision", "recall"]}
        self.finalize(datasets, mean_metrics=mean_mt)
    
    @torch.no_grad()
    def log_images(self, batch, *args, **kwargs):
        logs = {}
        image = self.get_input(batch, self.data_key)
        feature = self.get_input(batch, self.feature_key)
        model_outputs = self.encoder(image)
        
        feat_log = {}
        for k, classifier in self.classifiers.items():
            preds = classifier[0](model_outputs)
            preds = classifier[1](preds.view(image.shape[0], -1))
            feat_log[k] = preds
            
        logs["inputs"] = image
        logs["feature_gt"] = self.dataset_connector.rev_parse(feature)
        logs["feature_pred"] = self.dataset_connector.rev_parse(feat_log)
        logs["feature_gt_onehot"] = {fk: feature[fk].argmax(1) for fk in self.classifiers.keys()}
        logs["feature_pred_onehot"] = {fk: feat_log.argmax(1) for fk in self.classifiers.keys()}
        
        if self.save_dataset:
            self.add(logs, batch.get("casename"), dtypes={})

        return logs["metrics"], logs
    

class InferContDiffusion(ContrastiveDiffusion, ComputeMetrics, MakeDataset):
    def __init__(self, 
                 eval_scheme=[1],
                 save_dataset=False,
                 save_dataset_path=None,
                 include_keys=["data", "text"],
                 suffix_keys={"data":".nii.gz",},
                 **diffusion_kwargs):
        ContrastiveDiffusion.__init__(self, **diffusion_kwargs)
        ComputeMetrics.__init__(self, eval_scheme)
        self.save_dataset = save_dataset
        if save_dataset:
            assert exists(save_dataset_path)
            MakeDataset.__init__(self, save_dataset_path, include_keys, suffix_keys)
        self.eval()
        
    def test_step(self, *a):
        pass
    
    def log_images(self, batch, split="train", log_metrics=False, log_group_metrics_in_2d=False, **kw):
        logs = dict()
        inputs = super(ContrastiveDiffusion, self).get_input(batch, self.first_stage_key)
        mask = super(ContrastiveDiffusion, self).get_input(batch, "mask") # b c h w d
        logs["inputs"] = inputs
        logs["conditioning"] = mask
        samples = torch.zeros_like(mask)
        
        for i in range(mask.shape[2]):
            cond = torch.cat([mask[:, :, i], samples[:, :, max(0, i-1)]], dim=1)
            log_ = super().log_images({self.first_stage_key: torch.randn_like(inputs[:, :, 0]),
                                       self.cond_stage_key: cond}, **kw)
            samples[:, :, i] = log_["samples"]
        logs["samples"] = samples
        
        if self.save_dataset:
            self.add({"samples": samples, "mask": mask}, batch.get("casename"), dtypes={"image": np.uint8})
        
        if self.eval_scheme is not None and len(self.eval_scheme) > 0 and log_metrics:
            metrics = self.log_eval(inputs, samples, log_group_metrics_in_2d)
            # print(metrics)
            self.log_dict(metrics, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            return metrics, logs
        return None, logs
