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

from tqdm import tqdm
from enum import IntEnum
from medpy.metric import binary
from collections import defaultdict
from einops import rearrange, repeat
from ldm.data.utils import load_or_write_split

from ldm.models import (
    AutoencoderKL,
    VQModelInterface,
    LatentDiffusion,
    CategoricalDiffusion,
    OneHotCategoricalBCHW,
    Segmentator
)
from ldm.modules.diffusionmodules.util import extract_into_tensor


def exists(x):
    return x is not None


def save_helper(fn):
    def _fn(*args, **kw):
        save_dict = fn(*args, **kw)
        
        for suffix, data_dict in save_dict.items():
            if suffix.endswith('npy'): 
                for key, (path, data) in data_dict.items():
                    np.save(path, data)
            if suffix.endswith('nii.gz'):
                for key, (path, data) in data_dict.items():
                    sitk.WriteImage(sitk.GetImageFromArray(data), path)
            path = list(data_dict.keys())[0]
            if suffix.endswith('npz'):
                npz_parent = os.path.join(os.path.dirname(os.path.dirname(path)), 'npz')
                np.savez(npz_parent, **{key: data for key, (path, data) in data_dict.items()})
            if suffix.endswith('h5'):
                h5_parent = os.path.join(os.path.dirname(os.path.dirname(path)), 'h5')
                h5 = h5py.File(h5_parent, 'w')
                for key, (path, data) in data_dict.items():
                    h5.create_dataset(key, data=data)
                h5.close()
    return _fn

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
    
    
    
class MakeDataset:
    def __init__(self, 
                 dataset_base,
                 suffix_keys,
                 create_split=False, dims=3, desc=None, overwrite=False):
        self.dims = dims
        self.desc = desc
        self.base = dataset_base
        self.suffix_keys = suffix_keys
        self.create_split = create_split
        self.overwrite = overwrite
        
        self.dataset = defaultdict(dict)
        
        for key, suffix in self.suffix_keys.items():
            if suffix not in ['raw', 'npz', 'h5']:
                os.makedirs(os.path.join(self.base, key), exist_ok=True)
            elif suffix in ['npz', 'h5']:
                os.makedirs(os.path.join(self.base, suffix), exist_ok=True)
        
    def __add_version(self, olds, new, is_file=False):
        if (new in olds or os.path.basename(new) in olds) and not self.overwrite:
            if is_file:
                # olds: os.listdir, new: os.path.abspath
                basename = os.path.basename(new)
                dirname = os.path.dirname(new)
                mtime = [(file, os.path.getmtime(os.path.join(dirname, file))) for file in olds if file.startswith(basename.split('.')[0])]
                max_time_file = max(mtime, key=lambda x: x[1])[0]
                if 'version' in max_time_file: 
                    maxtime = int(max_time_file.split('.')[0][max_time_file.find('version') + len("version"):])
                else: maxtime = 0
                new = os.path.join(dirname, basename.split('.')[0] + f"_version{maxtime + 1}" + ".".join(basename.split('.')[1:]))
            else:
                mpath = [(x, int(x.split('.')[0][x.find('version') + len('version'):]) if 'version' in x else 0) for x in olds if x.startswith(new.split('.')[0])]
                max_version = max(mpath, key=lambda x: x[1])[1]
                new = new.split('.')[0] + f'_version{max_version + 1}' + ".".join(new.split('.')[1:])
        return new
    
    @save_helper 
    def add(self, samples, sample_names=None, dtypes={}, nb=1):
        if not exists(sample_names): sample_names = [f"case_{len(self.dataset) + i}" for i in range(nb)]
        if isinstance(sample_names, str): sample_names = [sample_names]
        
        # handle batch dim
        if nb > 1:
            if len(sample_names) == 1:
                for b in range(nb):
                    self.add({k: v[b] for k, v in samples.items()}, sample_names, dtypes, nb=1)
            elif len(sample_names) < nb:
                print(f'expected {nb} sample names, got {sample_names}')
                for b in range(nb):
                    self.add({k: v[b] for k, v in samples.items()}, [sample_names[0]], dtypes, nb=1)
            else:
                for b in range(nb):
                    self.add({k: v[b] for k, v in samples.items()}, [sample_names[b]], dtypes, nb=1)

        for i in range(len(sample_names)): 
            if sample_names[i] in self.dataset: 
                sample_names[i] = self.__add_version(list(self.dataset.keys()), sample_names[i])
                
        # samples: {key1: data1, ...}
        ret_dict = defaultdict(dict)
        for key, suffix in self.suffix_keys.items():
            data = samples[key]
            if isinstance(data, torch.Tensor):
                if data.ndim == 5:
                    assert data.shape[0] == 1
                    data = data[0]
                if data.ndim == 4:
                    data = rearrange(data, 'c ... -> ... c')
                data = data.cpu().numpy().astype(dtypes.get(key, np.float32))
                path = os.path.join(self.base, key, sample_names[0] + suffix)
                path = self.__add_version(os.listdir(os.path.join(self.base, key)), path, is_file=True)
                ret_dict[suffix][key] = (path, self.postprocess(data))
                self.dataset[sample_names[0]][key] = path
                
            elif isinstance(data, (str, tuple, list, dict)) and suffix == 'raw':
                self.dataset[sample_names[0]][key] = str(data)
        return ret_dict
                    
    def postprocess(self, sample):
        # c h w d
        return sample
        
    def finalize(self, dt=None, **kw):
        dataset = {} | kw
        collect_dt = self.dataset if dt is not None else dt
        dataset["data"] = collect_dt
        dataset["desc"] = self.desc
        dataset["keys"] = omegaconf.OmegaConf.to_container(self.suffix_keys)
        dataset["length"] = len(collect_dt)
        dataset["format"] = {k: self.suffix_keys.get(k, "raw") for k, v in self.suffix_keys.items()}
        
        if self.create_split:
            keys = list(collect_dt.keys())
            load_or_write_split(self.base, force=True, 
                                train=keys[:round(len(keys)*.7)],
                                val=keys[round(len(keys)*.7):round(len(keys)*.8)],
                                test=keys[round(len(keys)*.8):],)
        with open(os.path.join(self.base, "dataset.json"), "w") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=4)
    

class InferAutoencoderKL(AutoencoderKL):
    def __init__(self, **autoencoder_kwargs):
        AutoencoderKL.__init__(self, **autoencoder_kwargs)
        self.eval()
        
    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        pass
    
    @torch.no_grad()   
    def log_images(self, batch, *args, **kwargs):
        logs = super(InferAutoencoderKL, self).log_images(batch, *args, **kwargs)
        return logs
    
    
class InferAutoencoderVQ(VQModelInterface, MakeDataset):
    def __init__(self, 
                 save_dataset=False,
                 save_dataset_path=None,
                 suffix_keys={"image":".nii.gz",},
                 **diffusion_kwargs):
        if save_dataset:
            self.save_dataset = save_dataset
            assert exists(save_dataset_path)
            MakeDataset.__init__(self, save_dataset_path, suffix_keys)
        VQModelInterface.__init__(self, **diffusion_kwargs)
        self.eval()
        
    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        pass
    
    @torch.no_grad()   
    def log_images(self, batch, *args, **kwargs):
        logs = super(VQModelInterface, self).log_images(batch, *args, **kwargs)
        x = logs["inputs"]
        x_recon = logs["reconstructions"]
        
        if self.save_dataset:
            self.add({"image": x_recon}, batch.get("casename"), dtypes={"image": np.uint8})
        return logs
        
    
class InferLatentDiffusion(LatentDiffusion, MakeDataset):
    def __init__(self, 
                 save_dataset=False,
                 save_dataset_path=None,
                 suffix_keys={"data":".nii.gz",},
                 **diffusion_kwargs):
        if save_dataset:
            self.save_dataset = save_dataset
            assert exists(save_dataset_path)
            MakeDataset.__init__(self, save_dataset_path, suffix_keys)
        LatentDiffusion.__init__(self, **diffusion_kwargs)
        self.eval()

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        pass
    
    @torch.no_grad()
    def log_images(self, batch, *args, **kwargs):
        logs = super(InferLatentDiffusion, self).log_images(batch, *args, **kwargs)
        if self.save_dataset:
            self.add(logs | batch, batch.get("casename"), dtypes={"image": np.float32})
        return logs
    

class InferCategoricalDiffusion(CategoricalDiffusion, MakeDataset):
    def __init__(self, 
                 save_dataset=False,
                 save_dataset_path=None,
                 suffix_keys={"data":".nii.gz",},
                 **diffusion_kwargs):
        CategoricalDiffusion.__init__(self, **diffusion_kwargs)
        self.save_dataset = save_dataset
        if save_dataset:
            assert exists(save_dataset_path)
            MakeDataset.__init__(self, save_dataset_path, suffix_keys)
        self.eval()
        
    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        pass
    
    @torch.no_grad()
    def p_sample(self, q_xT=None, c=None, verbose=False, timesteps=None,
                 plot_progressive_rows=False, 
                 plot_denoising_rows=False, plot_diffusion_every_t=200):
        logs = dict()
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
                
            p_x0_given_xt = model_outputs
            p_xt = torch.clamp(self.q_xtm1_given_x0_xt(p_xt, p_x0_given_xt, t_), min=1e-12)
            p_xt = OneHotCategoricalBCHW(probs=p_xt).sample()
            
            if plot_denoising_rows and t % plot_diffusion_every_t == 0: denoising_rows.append(p_x0_given_xt)
            if plot_progressive_rows and t % plot_diffusion_every_t == 0: progressive_rows.append(p_xt)

        if self.step_T_sample == "majority":
            x0pred = OneHotCategoricalBCHW(probs=p_xt).max_prob_sample()
        elif self.step_T_sample == "confidence":
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
    def log_images(self, batch, **kwargs):
        logs = super(InferCategoricalDiffusion, self).log_images(batch, **kwargs)
        x = logs["inputs"]
        x_recon = logs["samples"]
        if x.ndim < self.dims + 2: x = x[:, None]
        if x_recon.ndim < self.dims + 2: x_recon = x_recon[:, None]
        
        if self.save_dataset:
            self.add({"inputs": x, "samples": x_recon}, batch.get("casename"), dtypes={"inputs": np.uint8, "samples": np.uint8})
        return logs
    

class InferSegmentation(Segmentator, MakeDataset):
    def __init__(self, 
                 save_dataset=False,
                 save_dataset_path=None,
                 suffix_keys={"data":".nii.gz",},
                 **segmentation_kwargs):
        Segmentator.__init__(self, **segmentation_kwargs)
        self.save_dataset = save_dataset
        if save_dataset:
            assert exists(save_dataset_path)
            MakeDataset.__init__(self, save_dataset_path, suffix_keys)
        self.eval()
        
    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        pass
    
    @torch.no_grad()
    def log_images(self, batch, **kwargs):
        logs = super(InferSegmentation, self).log_images(batch, **kwargs)
        x = logs["inputs"]
        x_recon = logs["samples"]
        if x.ndim < self.dims + 2: x = x[:, None]
        if x_recon.ndim < self.dims + 2: x_recon = x_recon[:, None]
        
        if self.save_dataset:
            self.add({"inputs": x, "samples": x_recon}, batch.get("casename"), dtypes={"inputs": np.uint8, "samples": np.uint8})
        return logs