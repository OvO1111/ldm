import os
import sys
sys.path.append("/ailab/user/dailinrui/code/latentdiffusion")
import torch
import SimpleITK as sitk
import numpy as np
import nibabel as nib
import json
import torchio as tio

from tqdm import tqdm
from einops import rearrange
from torch.utils.data import Dataset, _utils
from ldm.util import instantiate_from_config

from ldm.data.utils import *


def as_float(x, divisor=1, max_val=15):
    if x in ['', ' ']: return -1
    else:
        return min(float(x) // divisor, max_val)


class TCIATransform:
    def __init__(self, 
                 window_level=60,
                 window_width=360,
                 in_minmax=(-1500, 1500),
                 out_minmax=(-1, 1),
                 resize_or_crop="crop",
                 output_size=(64, 64, 64),):
        window_norm = invertible_window_norm(window_level, window_width, in_minmax=in_minmax, out_minmax=out_minmax)
        self.preprocess = [
            TorchioForegroundCropper(crop_level="patch", 
                                     crop_anchor="tumorseg",
                                     foreground_prob=1.,
                                     output_size=output_size,
                                     parent_kwargs={"include": ["image", 'tumorseg']}) 
            if resize_or_crop == "crop" else tio.Resize(output_size),
            tio.Lambda(window_norm.encode, include=["image"]),
        ]
        self.postprocess = [
            tio.Lambda(window_norm.decode, include=["image"]),
        ]


class TCIA(Dataset):
    def __init__(self, 
                 split="train", 
                 eventime_strat=12,
                 max_size:int = None,
                 use_datasets: list=None,
                 tcia_transform_config = None,
                 base="/ailab/user/dailinrui/data/datasets/TCIA_processed",
        ):
        super().__init__()
        self.base = base
        self.ds = [_ for _ in os.listdir(base) if (use_datasets is None or _ in use_datasets) and os.path.isdir(os.path.join(base, _))]
        self.data_keys = []
        self.eventime_strat = eventime_strat
        # for ds in self.ds:
        #     with open(os.path.join(base, ds, "dataset.json")) as f:
        #         json_file = json.load(f)
        #         for k, v in json_file.items():
        #             self.data_keys.extend(v['patient_best_ct'])
        with open(os.path.join(base, "good_list.txt"), 'r') as f:
            data = f.readlines()
            for d in self.ds:
                self.data_keys.extend([_.strip() for _ in data if d in _])

        self.transforms = TCIATransform(**tcia_transform_config)

        self.split = split
        train_keys = self.data_keys[:round(len(self.data_keys) * 0.8)]
        val_keys = self.data_keys[round(len(self.data_keys) * 0.8):]
        test_keys = self.data_keys[round(len(self.data_keys) * 0.8):]

        self.train_keys, self.val_keys, self.test_keys = load_or_write_split(self.base,
                                                                             True,
                                                                             train=train_keys, 
                                                                             val=val_keys, test=test_keys)
        self.split_keys = getattr(self, f"{split}_keys")[slice(0, max_size)]
        self.mapping = {
            "TCGA-KIRC": 1,
            "TCGA-KIRP": 1,
            "TCGA-KICH": 1,
            "CPTAC-CCRCC": 1,
            "TCGA-LIHC": 2,
            "Colorectal-Liver-Metastases": 2,
            "HCC-TACE-Seg": 2,
            "TCGA-LUAD": 3,
            "TCGA-LUSC": 3,
            "CPTAC-LSCC": 3,
            "CPTAC-LUAD": 3,
            "NSCLC Radiogenomics": 3,
            "NSCLC-Radiomics": 3,
        }

    def __len__(self):
        return len(self.split_keys)
    
    def parse_from_clinicals(self, clinicals, ds_index):
        survival = clinicals["survival"]
        return {"survival": torch.tensor([
            as_float(survival["survival_time"], divisor=365, max_val=self.eventime_strat-1),
            as_float(survival["survival_event"]),
            ds_index,
        ])[:, None].float()}

    def __getitem__(self, idx):
        image_nii = self.split_keys[idx]
        ds_index = self.mapping[image_nii.split("/")[-3]]
        tumorseg_nii = image_nii.replace(".nii.gz", "_tumorseg.nii.gz")
        with open(image_nii.replace(".nii.gz", ".json")) as f:
            clinicals = json.load(f)
            
        image = nib.load(image_nii).get_fdata()
        tumorseg = nib.load(tumorseg_nii).get_fdata()
        sample = {
            "image": tio.ScalarImage(tensor=image[None]),
            "tumorseg": tio.ScalarImage(tensor=tumorseg[None]),
        }
            
        subject = tio.Subject(**sample)
        for fn in self.transforms.preprocess:
            subject = fn(subject)
            
        subject = {k: v.data for k, v in subject.items()} | self.parse_from_clinicals(clinicals, ds_index)
        subject["image_and_seg"] = torch.cat([subject["image"], subject["tumorseg"]], 0)
        
        return subject


if __name__ == "__main__":
    def validate():
        from pathlib import Path
        g_list = {"data": [], "no_mask": [], "small_mask": []}
        base = Path("/ailab/user/dailinrui/data/datasets/TCIA_processed")
        all_list = [str(_) for _ in base.rglob("*.nii.gz") if "tumorseg" not in str(_)]
        for subject in tqdm(all_list):
            if not os.path.exists(subject.replace(".nii.gz", "_tumorseg.nii.gz")):
                print(f"{subject} has no mask")
                g_list["no_mask"].append(str(subject))
                continue
            mask = sitk.GetArrayFromImage(sitk.ReadImage(subject.replace(".nii.gz", "_tumorseg.nii.gz")))
            if mask.sum() > 50 and mask.max() == 1:
                g_list["data"].append(str(subject))
            else:
                print(f"{subject} has {mask.sum()} voxels of max val {mask.max()} < threshold 50")
                g_list["small_mask"].append(str(subject))
        with open(base / "good_list.json", 'w') as f:
            json.dump(g_list, f)
        
    def test():
        ds = TCIA(split="train", max_size=100)
        ds[0]
    
    validate()