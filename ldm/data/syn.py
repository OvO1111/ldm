import os, sys
sys.path.append("/ailab/user/dailinrui/code/latentdiffusion")
import h5py, SimpleITK as sitk, numpy as np, json

import torch, random
import torchio as tio
from functools import reduce
from ldm.util import instantiate_from_config
from omegaconf import ListConfig
from torch.utils.data import Dataset
from ldm.data.utils import TorchioSequentialTransformer


class GeneratedDataset(Dataset):
    def __init__(self, base_folder, max_size=None, split="all", seed=1234, val_num=5):
        self.base = base_folder
        with open(os.path.join(base_folder, "dataset.json")) as f:
            self.desc = json.load(f)
        self.split = split
        self.keys = self.desc["keys"]
        self.data = self.desc["data"]
        self.format = self.desc["format"]
        self.all_keys = list(self.data.keys())
        
        if self.split != "all":
            random.seed(seed)
            random.shuffle(self.all_keys)
            self.train_keys = self.all_keys[:-val_num]
            self.val_keys = self.all_keys[-val_num:]
        
        self.transforms = TorchioSequentialTransformer({})
        self.split_keys = getattr(self, f"{split}_keys")[:max_size]
        
    def __len__(self):
        return len(self.split_keys)
    
    def _read_nifti(self, x):
        return self._possible_expand_dims(sitk.GetArrayFromImage(sitk.ReadImage(x)))
    
    def _read_h5(self, x):
        s = dict()
        h5 = h5py.File(x)
        for k in h5.keys():
            s[k] = self._possible_expand_dims(h5[k][:])
        return s
    
    @staticmethod
    def _possible_expand_dims(x):
        if hasattr(x, "ndim") and x.ndim == 3: x = x[None]
        return torch.tensor(x)
    
    def __getitem__(self, idx):
        casename = self.split_keys[idx]
        item = self.data[casename]
        sample = {}
        image_modalities = []
        for k in self.keys:
            if k not in self.format:
                sample[k] = item[k]
                continue
            if self.format[k] == "raw": sample[k] = item[k]
            elif self.format[k] == '.nii.gz': sample[k] = self._read_nifti(item[k])
            elif self.format[k] == '.h5': sample = sample | self._read_h5(item[k])
            if self.format[k] != 'raw': image_modalities.append(k)
        
        if len(image_modalities) > 0:
            subject = tio.Subject(**{k: tio.ScalarImage(tensor=sample[k]) for k in image_modalities})
            self.transforms(subject)
            sample.update({k: v.data for k, v in subject.items()})
            
        sample["casename"] = casename
        return sample
    
    
class GeneratedAndRealDataset(Dataset):
    def __init__(self, generated_ds, real_ds,
                 split="train"):
        self.ds = []
        if not isinstance(generated_ds, (list, ListConfig)):
            generated_ds = [generated_ds]
        if not isinstance(real_ds, (list, ListConfig)):
            real_ds = [real_ds]
        for ds in generated_ds:
            ds["split"] = split
            self.ds.append(instantiate_from_config(ds))
        self.gen_real_split = len(self.ds)
        for ds in real_ds:
            ds["split"] = split
            self.ds.append(instantiate_from_config(ds))
        
        self.split_keys = reduce(lambda x, y: x + y, [[(k, ids) for k in ds.split_keys] for ids, ds in enumerate(self.ds)], [])
        
    def __len__(self):
        return len(self.split_keys)
    
    def __getitem__(self, idx):
        item = self.split_keys[idx]
        item_index, ds_index = item
        
        sample = self.ds[ds_index][item_index]
        sample = {"type": "syn" if ds_index < self.gen_real_split else "real"} | sample
        return sample
        