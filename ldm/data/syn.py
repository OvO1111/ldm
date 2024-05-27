import os, sys
sys.path.append("/ailab/user/dailinrui/code/latentdiffusion")
import h5py, SimpleITK as sitk, numpy as np

import torch, random
from tqdm import tqdm
from einops import rearrange
from torch.utils.data import Dataset


class SynthesizeDataset(Dataset):
    # for second stage image synthesis from first stage mask gen
    # default to be test only
    def __init__(self, syn_folder, static_folder=None, first_stage_model=None, also_use_as_train=False, split="test"):
        super().__init__()
        self.data_keys = os.listdir(syn_folder)
        self.static_folder = static_folder
        
        if also_use_as_train:
            self.train_split = self.data_keys[:round(len(self.data_keys) * .8)]
            self.test_split = self.data_keys[:round(len(self.data_keys) * .8)]
        else:
            self.train_split = []
            self.test_split = self.data_keys
            
        self.load_fn = lambda f: sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(syn_folder, f)))
        self.split_keys = getattr(self, f"{split}_keys")
        
        if first_stage_model is not None:
            ...
        
    def __len__(self):
        return len(self.split_keys)
    
    def __getitem__(self, idx):
        data = self.load_fn(self.split_keys[idx])
        placeholder = np.zeros_like(data)
        return {"image": placeholder, "mask": data}