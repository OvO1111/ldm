from torch.utils.data import Dataset

import os
import h5py
import json
import torch
import random
import SimpleITK as sitk
import torchio as tio, nibabel as nib, numpy as np
from functools import partial
from ldm.data.utils import TorchioForegroundCropper, TorchioBaseResizer, \
    identity, window_norm
    
    
class MSDDatasetForSegmentation(Dataset):
    def __init__(self, name, base="/ailab/user/dailinrui/data/datasets", mapping={}, split="train", max_size=None, resize_to=(96,)*3, force_rewrite_split=False, info={}):
        base_folder = os.path.join(base, name)
        self.base_folder = base_folder
        self.get_spacing = lambda x: sitk.ReadImage(x).GetSpacing()
        if split == "test": split = "val"
        self.transforms = dict(
            resize_base=TorchioBaseResizer(),
            crop=TorchioForegroundCropper(crop_level="patch", 
                                          crop_anchor="image",
                                          crop_kwargs=dict(output_size=resize_to, foreground_boost=1/3)) if resize_to is not None else tio.Lambda(identity),
            normalize_image=tio.Lambda(partial(window_norm, window_pos=-500, window_width=1800)),
            remap_mask=tio.RemapLabels(mapping, include=["mask"])
        )

        self.split = split
        self.name = name
        if not os.path.exists(f"{base_folder}/train.list") or force_rewrite_split:
            train_val_cases = os.listdir(f"{base_folder}/imagesTr")
            random.shuffle(train_val_cases)
            train_cases = train_val_cases[:round(0.8 * len(train_val_cases))]
            val_cases = train_val_cases[round(0.8 * len(train_val_cases)):]
            test_cases = os.listdir(f"{base_folder}/imagesTs")
            
            for spt in ["train", "val", "test"]:
                with open(f"{base_folder}/{spt}.list", "w") as fp:
                    for c in locals().get(f"{spt}_cases"):
                        fp.write(c + "\n")
        
        for spt in ["train", "val", "test"]:
            with open(f"{base_folder}/{spt}.list") as fp:
                self.__dict__[f"{spt}_keys"] = [_.strip() for _ in fp.readlines()]
        else:
            self.split_keys = getattr(self, f"{split}_keys")[:max_size]
            
        self.__dict__.update(info)
        
    @staticmethod                
    def load_fn(p, return_spacing=0, transpose_code=None):
        image = nib.load(p)
        array = image.dataobj[:]
        if transpose_code is None:
            x, y, z = nib.aff2axcodes(image.affine)
        else:
            x, y, z = transpose_code
        dims = [0, 1, 2]
        for ianc, anc in enumerate([x, y, z]):
            dims[0 if anc in ['S', 'I'] else 1 if anc in ['P', 'A'] else 2 if anc in ['L', 'R'] else 10] = ianc
        array = array.transpose(*dims)
        if "S" not in [x, y, z]: array = np.flip(array, axis=0)
        if "P" not in [x, y, z]: array = np.flip(array, axis=1)
        if "L" not in [x, y, z]: array = np.flip(array, axis=2)
        array = array.copy()
        if return_spacing:
            return array, tuple(np.array(image.header.get_zooms())[dims].tolist()), [x, y, z]
        return array
        
    def __len__(self):
        return len(self.split_keys)
            
    def __getitem__(self, idx):
        item = self.split_keys[idx] if isinstance(idx, int) else idx
        
        if self.split in ["train", "val"]:
            image, spacing, code = self.load_fn(os.path.join(self.base_folder, "imagesTr", item), 1)
            mask, totalseg = map(lambda x: self.load_fn(os.path.join(self.base_folder, x, item), transpose_code=code), ["labelsTr", "totalsegTr"])
            subject = tio.Subject(image=tio.ScalarImage(tensor=image[None], spacing=spacing), 
                                  mask=tio.LabelMap(tensor=mask[None], spacing=spacing),
                                  totalseg=tio.LabelMap(tensor=totalseg[None], spacing=spacing))
        if self.split == "test":
            image, spacing = self.load_fn(os.path.join(self.base_folder, "imagesTs", item), 1)
            subject = tio.Subject(image=tio.ScalarImage(tensor=image[None], spacing=spacing))
        # resize based on spacing
        ori_size = subject.image.data.shape
        subject = self.transforms["resize_base"](subject)
        # crop
        subject = self.transforms["crop"](subject)
        # normalize
        subject = self.transforms["normalize_image"](subject)
        subject = self.transforms["remap_mask"](subject)
        # random aug
        subject = self.transforms.get("augmentation", tio.Lambda(identity))(subject)
        subject = {k: v.data for k, v in subject.items()} | {'casename': self.split_keys[idx] if isinstance(idx, int) else idx, "ori_size": ori_size, "text": "", "attr": ""}

        return subject
    
    
class KiTS(Dataset):
    def __init__(self, base="/ailab/user/dailinrui/data/datasets/kits23", 
                 split="train", max_size=None, crop_to=(96,96,96)):
        super().__init__()
        self.load_fn = lambda x: h5py.File(x)
        self.transforms = dict(
            crop=TorchioForegroundCropper(crop_level="patch", 
                                          crop_anchor="image",
                                          crop_kwargs=dict(output_size=crop_to),) if crop_to is not None else identity,
            normalize_image=tio.RescaleIntensity(out_min_max=(0, 1), in_min_max=None, include=["image"]),
            remap_mask=tio.RemapLabels({1:0, 2:1}, include=["mask"])
        )
        self.split = split
        self.max_size = max_size
        
        with open(f"{base}/{split}.list") as fp:
            self.__dict__[f"{split}_keys"] = [os.path.join(f"{base}/data", _.strip()) for _ in fp.readlines()]
        
    def __len__(self):
        return len(self.split_keys)
    
    def __getitem__(self, idx):
        item = self.load_fn(self.split_keys[idx])
        image, mask = map(lambda x: item[x][:], ["image", "label"])
        
        subject = tio.Subject(image=tio.ScalarImage(tensor=image[None]), 
                              mask=tio.ScalarImage(tensor=mask[None]),)
        # crop
        subject = self.transforms["crop"](subject)
        # normalize
        subject = self.transforms["normalize_image"](subject)
        subject = self.transforms["remap_mask"](subject)
        # random aug
        subject = self.transforms.get("augmentation", tio.Lambda(identity))(subject)
        subject = {k: v.data for k, v in subject.items()} | {"ids": idx,
                                                             'casename': os.path.basename(self.split_keys[idx]).split('.')[0]}

        return subject