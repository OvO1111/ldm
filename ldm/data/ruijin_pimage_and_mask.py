
import os, json, math
import numpy as np
import nibabel as nib

from torch.utils.data import Dataset, _utils

import torch
import random
import torchio as tio
import SimpleITK as sitk
from torchvision.transforms import v2

import h5py
import shutil
from tqdm import tqdm
from einops import rearrange
from scipy.ndimage import zoom


def flush(f, filename=None):
    if filename is not None:
        os.remove(filename)
        return 
    for file in os.listdir(f):
        os.remove(os.path.join(f, file))
    

def conserve_only_certain_labels(label, designated_labels=[1, 2, 3, 5, 6, 10, 55, 56, 57, 104]):
    # 6: stomach, 57: colon
    if designated_labels is None:
        return label.astype(np.uint8)
    label_ = np.zeros_like(label)
    for il, l in enumerate(designated_labels):
        label_[label == l] = il
    return label_


def window_norm(image, window_pos=60, window_width=360):
    window_min = window_pos - window_width // 2
    image = (image - window_min) / window_width
    image[image < 0] = 0
    image[image > 1] = 1
    return image


def write_split(basefolder, *splits):
    with open(os.path.join(basefolder, "splits.json"), "w") as f:
        json.dump(dict(zip(["train", "val", "test"], splits)), f, indent=4)
        

def use_split(basefolder):
    with open(os.path.join(basefolder, "splits.json")) as f:
        splits = json.load(f)
    use = dict(train=splits.get("train"), val=splits.get("val"), test=splits.get("test"))
    print([len(u) for u in use.values() if u is not None])
    return use


class PretrainDataset(Dataset):
    # pretrain: CT cond on report -> totalseg organ mask
    def __init__(self, split="train", toy=False, cache_len=0):
        with open('/mnt/data/oss_beijing/dailinrui/data/ruijin/records/dataset_crc_v2.json', 'rt') as f:
            self.data = json.load(f)
        self.get_spacing = lambda x: sitk.ReadImage(x).GetSpacing()
        self.advance_step = 3
        
        self.volume_transform = tio.Compose((
            tio.Lambda(window_norm),
        ))
        self.mask_transform = tio.Compose((
            tio.RescaleIntensity(out_min_max=(0, 1), in_min_max=(0, 255)),
        ))
        self.joined_transform = v2.Compose([
            v2.RandomAffine(degrees=10, translate=(0.2, 0.2), scale=(0.9, 1.1))
        ])
        
        self.split = split
        if cache_len is None:
            cache_len = len(self.split_keys)
        self.cache_len = cache_len
        self.split_keys = use_split("/mnt/workspace/dailinrui/data/pretrained/controlnet/train_val_split")[split]
        if toy: 
            self.split_keys = self.split_keys[:10]
        self.data_cache = {}
    
    def __len__(self):
        return len(self.split_keys)
    
    def get_from_cache(self, k):
        dataset_path = "/mnt/data/oss_beijing/dailinrui/data/dataset/ruijin"
        if (_cached_sample := self.data_cache.get(k, None)) is None:
            if os.path.exists(_cached_path := os.path.join(dataset_path, f"{k}.h5")):
                try:
                    _cached_sample = h5py.File(_cached_path, "r")
                except Exception:
                    return None
                subject = tio.Subject(seg=tio.LabelMap(tensor=_cached_sample["seg"][:]),
                                      image=tio.ScalarImage(tensor=_cached_sample["image"][:]))
                spacing = _cached_sample["spacing"]
                ret_dict = dict(subject=subject, raw_spacing=spacing)
                if len(self.data_cache) < self.cache_len:
                    self.data_cache[k] = ret_dict
                return None
                # return ret_dict
                   
    def load_fn(self, k, valid_keys=["ct", "totalseg", "crcseg"]):
        resized = {}
        if (_cached_sample := self.get_from_cache(k)) is None:
            for key, n in self.data[k].items():
                if key in valid_keys:
                    im = sitk.ReadImage(n)
                    dtype = im.GetPixelID()
                    spacing = im.GetSpacing()
                    raw = sitk.GetArrayFromImage(im)
                    zoom_coef = np.array([1, 1, spacing[-1]])[::-1]
                    # change spacing to 1 ? ?
                    # _tmp = torch.tensor(zoom(raw, zoom_coef, order=0 if dtype == 1 else 3))
                    _tmp = raw
                if key == "ct":
                    resized[key] = self.volume_transform(_tmp[None])
                elif key == "totalseg":
                    resized[key] = conserve_only_certain_labels(_tmp)
                elif key == "crcseg":
                    resized["totalseg"][_tmp > 0] = 11
                    resized["totalseg"] = self.mask_transform(resized["totalseg"][None])

            subject = tio.Subject(seg=tio.LabelMap(tensor=resized["totalseg"]),
                                  image=tio.ScalarImage(tensor=resized["ct"]))
            ret_dict = dict(subject=subject, raw_spacing=spacing)
            if len(self.data_cache) < self.cache_len:
                self.data_cache[k] = ret_dict
            return ret_dict
        return _cached_sample

    def __getitem__(self, idx):
        key = self.split_keys[idx]
        sample = self.load_fn(key)
        subject, spacing = sample["subject"], sample["raw_spacing"]

        image, seg = subject.image.data, subject.seg.data
        start_layer, end_layer = torch.where(seg.sum((0, 2, 3)))[0][[0, -1]]
        
        if self.split == "train":
            random_slice = random.randint(start_layer.item(), end_layer.item() - self.advance_step) if random.random() < .1 else start_layer.item()
            random_slices = slice(random_slice, random_slice + self.advance_step)
            previous_layer = image[:, random_slice - 1: random_slice] if random_slice > start_layer else torch.zeros_like(image[:, 0:1])
            image_slice = image[:, random_slices] 
            seg_slice = seg[:, random_slices]
            
            sample = rearrange(torch.cat([image_slice, seg_slice, previous_layer], dim=1), "1 c h w -> c 1 h w")
            sample = self.joined_transform(sample)
            image = sample[:self.advance_step]
            control = sample[self.advance_step:]
        else:
            random_slice = random.randint(start_layer.item(), end_layer.item())
            random_slices = slice(random_slice, random_slice + 1)
            previous_layer = image[:, random_slice - 1: random_slice] if random_slice > start_layer else torch.zeros_like(image[:, 0:1])
            image_slice = image[:, random_slices] 
            seg_slice = seg[:, random_slices]
            
            # autoencoder takes in (prev_layer, slice)
            sample = torch.cat([image_slice, previous_layer, seg_slice], dim=1)
            sample = self.joined_transform(sample)
            image = sample[:, :1]
            control = sample[:, 1:]
            
        return dict(image=image, mask=control, caseid=key)
                    # wholemask=subject.seg.data,
                    # wholeimage=subject.image.data)
                    
    def collate(self, batch):
        sample = _utils.collate.default_collate(batch)
        return sample
        
    def preload(self):
        _temp_path = "/mnt/data/smart_health_02/dailinrui/data/temp"
        dataset_path = "/mnt/data/oss_beijing/dailinrui/data/dataset/ruijin"
        
        for k in tqdm(self.split_keys):
            if os.path.exists(os.path.join(dataset_path, f"{k}.h5")): continue
            sample = self.load_fn(k)
            saved_sample = h5py.File(_x := os.path.join(_temp_path, f"{k}.h5"), "w")
            saved_sample.create_dataset("image", data=sample["subject"].image.data, compression="gzip")
            saved_sample.create_dataset("seg", data=sample["subject"].seg.data, compression="gzip")
            saved_sample.create_dataset("spacing", data=np.array(sample["raw_spacing"]), compression="gzip")
            saved_sample.close()
            shutil.copyfile(_x, os.path.join(dataset_path, f"{k}.h5"))
            flush(_temp_path, filename=_x)
            
            
if __name__ == "__main__":
    # train_ds = PretrainDataset()
    # train_ds.preload()
    val_ds = PretrainDataset("train")
    val_ds.preload()