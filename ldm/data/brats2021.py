import os, sys
sys.path.append("/ailab/user/dailinrui/code/latentdiffusion")
import json, torchio as tio
import h5py, numpy as np

import torch
from tqdm import tqdm
from torch.utils.data import Dataset, _utils

from ldm.data.utils import identity, TorchioForegroundCropper

from einops import rearrange
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.font_manager


def minmax(x, xmin=None, xmax=None):
    if xmax is not None:
        x = min(x, xmax)
    if xmin is not None:
        x = max(x, xmin)
    return x


class BraTS2021_3D(Dataset):
    n_coarse = 1
    n_fine = 3
    def __init__(self, split="train", 
                crop_to=(96, 96, 96),
                use_shm=False,
                max_size=None,
                n_fine=None,
                base="/ailab/group/pjlab-smarthealth03/transfers/dailinrui/data/dataset/BraTS2021"):
        super().__init__()
        self.load_fn = lambda x: h5py.File(x)
        self.transforms = dict(
            crop=TorchioForegroundCropper(crop_level="patch", 
                                          crop_anchor="image",
                                          crop_kwargs=dict(output_size=crop_to),) if crop_to is not None else identity,
            normalize_image=tio.RescaleIntensity(out_min_max=(0, 1), in_min_max=None, include=["image"]),
            normalize_mask=tio.RescaleIntensity(out_min_max=(0, 1), in_min_max=(0, 3), include=["mask"])
        )

        self.split = split
        if use_shm: base = "/dev/shm/BraTS2021"
        
        for spt in ["train", "val", "test"]:
            with open(f"{base}/{spt}.list") as fp:
                self.__dict__[f"{spt}_keys"] = [os.path.join(f"{base}/data", _.strip()) for _ in fp.readlines()]
        else:
            self.split_keys = getattr(self, f"{split}_keys")[:max_size]
            
        for broken_file in [os.path.join(f"{base}/data", _) for _ in ["BraTS2021_00000.h5"]]: self.split_keys.remove(broken_file) if broken_file in self.split_keys else 0
        
        self.fine_labeled_indices = list(range(n_fine if n_fine is not None else len(self.split_keys)))
        self.coarse_labeled_indices = [i for i in range(len(self.split_keys)) if i not in self.fine_labeled_indices]

    def __len__(self):
        return len(self.split_keys)

    def __getitem__(self, idx):
        item = self.load_fn(self.split_keys[idx])
        image, mask = map(lambda x: item[x][:], ["image", "label"])
        
        subject = tio.Subject(image=tio.ScalarImage(tensor=image), coarse=tio.ScalarImage(tensor=(mask >= 1).astype(np.uint8)[None]), fine=tio.ScalarImage(tensor=mask[None]),)
        # crop
        subject = self.transforms["crop"](subject)
        # normalize
        subject = self.transforms["normalize_image"](subject)
        subject = self.transforms["normalize_mask"](subject)
        # random aug
        subject = self.transforms.get("augmentation", tio.Lambda(identity))(subject)
        subject = {k: v.data for k, v in subject.items()} | {"ids": idx, 
                                                             "mask": subject.fine.data if idx in self.fine_labeled_indices else subject.coarse.data, 
                                                             'casename': os.path.basename(self.split_keys[idx]).split('.')[0]}

        return subject
        
    def verify_dataset(self):
        iterator = tqdm(range(len(self.split_keys)))
        for idx in iterator:
            try:
                item = self.__getitem__(idx)
                iterator.set_postfix(shape=item["image"].shape)
            except Exception as e:
                print(self.split_keys[idx], e)
                
    def collate(self, batch):
        return _utils.collate.default_collate(batch)


if __name__ == "__main__":
    import time
    ds = BraTS2021_3D(crop_to=(96, 96, 96), split="val")
    ds.verify_dataset()