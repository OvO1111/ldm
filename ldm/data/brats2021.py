import os, sys
sys.path.append("/mnt/workspace/dailinrui/code/latentdiffusion")
import json, torchio as tio
import h5py, SimpleITK as sitk, numpy as np

import torch
from functools import reduce
from torch.utils.data import Dataset

from ldm.data.utils import identity, TorchioForegroundCropper


class BraTS2021_3D(Dataset):
    def __init__(self, split="train", 
                crop_to=(96, 96, 96)):
        super().__init__()
        self.load_fn = lambda x: h5py.File(x)
        self.transforms = dict(
            crop=TorchioForegroundCropper(crop_level="patch", crop_kwargs=dict(output_size=crop_to)) if crop_to is not None else identity,
            normalize_image=tio.RescaleIntensity(out_min_max=(0, 1), in_min_max=None, include=["image"]),
            normalize_mask=tio.RescaleIntensity(out_min_max=(0, 1), in_min_max=(0, 3), include=["mask"])
        )

        self.split = split
        with open(f"/mnt/data/oss_beijing/dailinrui/data/dataset/BraTS2021/{split}.list") as fp:
            self.split_keys = [os.path.join("/mnt/data/oss_beijing/dailinrui/data/dataset/BraTS2021/data", _.strip()) for _ in fp.readlines()]

    def __len__(self):
        return len(self.split_keys)

    def __getitem__(self, idx):
        item = self.load_fn(self.split_keys[idx])
        image, mask = map(lambda x: item[x][:], ["image", "label"])
        
        subject = tio.Subject(image=tio.ScalarImage(tensor=image), mask=tio.ScalarImage(tensor=mask[None]),)
        # crop
        subject = self.transforms["crop"](subject)
        # normalize
        subject = self.transforms["normalize_image"](subject)
        subject = self.transforms["normalize_mask"](subject)
        # random aug
        subject = self.transforms.get("augmentation", tio.Lambda(identity))(subject)
        subject = {k: v.data for k, v in subject.items()}

        return subject


if __name__ == "__main__":
    import time
    ds = BraTS2021_3D(crop_to=None)
    st = time.time()
    test_case = ds[0]
    print(time.time() - st)
    sitk.WriteImage(sitk.GetImageFromArray(test_case["image"][0].numpy()), "./test_proc_im.nii.gz")
    sitk.WriteImage(sitk.GetImageFromArray((test_case["mask"][0] * 3).numpy().astype(np.uint8)), "./test_proc_mask.nii.gz")