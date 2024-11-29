import os, sys
sys.path.append("/ailab/user/dailinrui/code/latentdiffusion")
import random, torchio as tio
import h5py, numpy as np

import torch
from tqdm import tqdm
import SimpleITK as sitk
from torch.utils.data import Dataset, _utils
from einops import rearrange
from ldm.data.utils import identity, TorchioForegroundCropper, invertible_window_norm


def get_dataset_base(dataset_name):
    base = '/ailab/user/dailinrui/data/datasets/DiffTumorTrainDataset'
    dataset_path_file = {
        "gallbladder_tumor": "gallbladdercancer",
        "gallbladder_stone": "gallbladderstone",
        "esophagus_tumor": "esophagealcancer",
        "bladder_tumor": "bladdercancer",
        "kidney_tumor": "kits23_tumor",
        "kidney_stone": "kidneystone",
        "kidney_cyst": "kits23_cyst",
        "liver_cyst": "livercyst",
        "liver_tumor": "Task03_Liver",
        "lung_tumor": "Task06_Lung",
        "colon_tumor": "Task10_Colon",
        "stomach_tumor": "stomachtumor",
        "pancreas_cyst": "PancreasCyst",
        "pancreas_tumor": "PancreasTumor",
    }[dataset_name.lower()]
    return os.path.join(base, dataset_path_file)


def schedule_window_pos(dataset_name):
    dataset_name = dataset_name.lower()
    return {"window_pos": 0, "window_width": 400}


def sitk_load(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(path))


class SynTumorDataset(Dataset):
    def __init__(self, dataset_name, split='all', crop_to=[128, 128, 128], max_size=None):
        self.split = split
        self.dataset_name = dataset_name
        self.dataset_base = get_dataset_base(self.dataset_name)
        
        assert split == 'all', "should use all synthetic cases for generative training"
        with open(os.path.join(self.dataset_base, "image.txt")) as f,\
            open(os.path.join(self.dataset_base, "label.txt")) as g:
            self.all_keys = [{"image": line1.strip(), "tumorseg": line2.strip()} for line1, line2 in zip(f.readlines(), g.readlines())]
        
        self.split_keys = self.all_keys[:max_size]
        self.preprocess = {
            "norm": invertible_window_norm(in_minmax=(-1500, 1500), outlier_percentile=0.2, **schedule_window_pos(self.dataset_name)),
            "crop": TorchioForegroundCropper(crop_level='patch', crop_anchor='tumorseg', foreground_prob=1., output_size=crop_to),
            "mask_norm": tio.RescaleIntensity(include=['totalseg'], out_min_max=(0, 1), in_min_max=(0, 104))
        }
        
    def __len__(self):
        return len(self.split_keys)
    
    def __getitem__(self, idx):
        sample = self.split_keys[idx]
        image, tumorseg = map(lambda x: sitk_load(sample[x]), ['image', 'tumorseg'])
        image = self.preprocess['norm'].encode(image)
        
        subject = tio.Subject(image=tio.ScalarImage(tensor=image[None]),
                              tumorseg=tio.ScalarImage(tensor=tumorseg[None]),)
        subject = self.preprocess['crop'](subject)
        subject = self.preprocess['mask_norm'](subject)
        
        return subject
        