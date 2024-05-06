import os, sys
sys.path.append("/mnt/workspace/dailinrui/code/latentdiffusion")
import json, torchio as tio, torchvision as tv, shutil, nibabel as nib
import re, SimpleITK as sitk, scipy.ndimage as ndimage, numpy as np, multiprocessing as mp

import torch
from functools import reduce, partial
from torch.utils.data import Dataset

from ldm.data.utils import conserve_only_certain_labels, window_norm, load_or_write_split, TorchioForegroundCropper


def identity(x, *a, **b): return x


class AutoencoderDataset(Dataset):
    def __init__(self, split="train", 
                force_rewrite_split=False, 
                resize_to=(64, 128, 128)):
        super().__init__()
        with open('/mnt/data/oss_beijing/dailinrui/data/ruijin/records/dataset_crc_v2.json', 'rt') as f:
            self.data = json.load(f)
            self.data_keys = list(self.data.keys())

        self.base_folder = "/mnt/data/oss_beijing/dailinrui/data/ruijin"
        self.load_fn = lambda x: sitk.GetArrayFromImage(sitk.ReadImage(x))
        self.transforms = dict(
            resize=tio.Resize(resize_to) if resize_to is not None else tio.Lambda(identity),
            crop=TorchioForegroundCropper(crop_level="mask_foreground", crop_kwargs=dict(foreground_hu_lb=1e-3,
                                                                                         foreground_mask_label=None,
                                                                                         outline=(0, 0, 0))),
            normalize_image=tio.Lambda(window_norm, include=["image"]),
            normalize_mask=tio.RescaleIntensity(out_min_max=(0, 1), in_min_max=(0, 11), include=["mask"])
        )

        self.split = split
        np.random.shuffle(self.data_keys)
        self.train_keys = self.data_keys[:round(len(self.data_keys) * 0.7)]
        self.val_keys = self.data_keys[round(len(self.data_keys) * 0.7):round(len(self.data_keys) * 0.8)]
        self.test_keys = self.data_keys[round(len(self.data_keys) * 0.8):]

        self.train_keys, self.val_keys, self.test_keys = load_or_write_split("/mnt/data/smart_health_02/dailinrui/data/pretrained/ldm/contrastive_exp_split",
                                                                             force_rewrite_split,
                                                                             train=self.train_keys, 
                                                                             val=self.val_keys, test=self.test_keys)
        self.split_keys = getattr(self, f"{split}_keys")

    def __len__(self):
        return len(self.split_keys)

    def __getitem__(self, idx):
        item = self.data[self.split_keys[idx]]
        data, totalseg, crcseg, text = map(lambda x: item[x], ["ct", "totalseg", "crcseg", "summary"])
        image, mask, crcmask = map(self.load_fn, [data, totalseg, crcseg])
        
        mask = conserve_only_certain_labels(mask)
        mask[crcmask > 0] = 11
        
        subject = tio.Subject(image=tio.ScalarImage(tensor=image[None]), mask=tio.ScalarImage(tensor=mask[None]),)
        # crop
        subject = self.transforms["crop"](subject)
        # normalize
        subject = self.transforms["normalize_image"](subject)
        subject = self.transforms["normalize_mask"](subject)
        # resize
        subject = self.transforms["resize"](subject)
        # random aug
        subject = self.transforms.get("augmentation", tio.Lambda(identity))(subject)
        subject = {k: v.data for k, v in subject.items()} ## | {"text": text}  | operator is available from python 3.9
        subject.update(dict(text=text))

        return subject


if __name__ == "__main__":
    import time
    ds = AutoencoderDataset(resize_to=None)
    st = time.time()
    test_case = ds[0]
    print(test_case["text"], time.time() - st)
    sitk.WriteImage(sitk.GetImageFromArray(test_case["image"][0].numpy()), "./test_proc_im.nii.gz")
    sitk.WriteImage(sitk.GetImageFromArray((test_case["mask"][0] * 11).numpy().astype(np.uint8)), "./test_proc_mask.nii.gz")