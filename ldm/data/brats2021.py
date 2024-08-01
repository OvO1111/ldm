import os, sys
sys.path.append("/ailab/user/dailinrui/code/latentdiffusion")
import random, torchio as tio
import h5py, numpy as np

import torch
from tqdm import tqdm
from torch.utils.data import Dataset, _utils
from einops import rearrange
from ldm.data.utils import identity, TorchioForegroundCropper


def minmax(x, xmin=None, xmax=None):
    if xmax is not None:
        x = min(x, xmax)
    if xmin is not None:
        x = max(x, xmin)
    return x


class BraTS2021_3D(Dataset):
    def __init__(self, split="train", 
                crop_to=(96, 96, 96),
                use_shm=False,
                max_size=None,
                n_fine=None,
                base="/ailab/user/dailinrui/data/datasets/BraTS2021"):
        super().__init__()
        self.load_fn = lambda x: h5py.File(x)
        self.transforms = dict(
            crop=TorchioForegroundCropper(crop_level="patch", 
                                          crop_anchor="image",
                                          crop_kwargs=dict(output_size=crop_to, foreground_prob=1.),) if crop_to is not None else identity,
            normalize_image=tio.RescaleIntensity(out_min_max=(0, 1), in_min_max=None, include=["image"]),
            normalize_mask=tio.RescaleIntensity(out_min_max=(0, 1), in_min_max=(0, 3), include=["mask"])
        )

        self.n_fine = n_fine
        self.split = split
        self.max_size = max_size
        if use_shm: base = "/dev/shm/BraTS2021"
        
        for spt in ["train", "val", "test"]:
            with open(f"{base}/{spt}.list") as fp:
                self.__dict__[f"{spt}_keys"] = [os.path.join(f"{base}/data", _.strip()) for _ in fp.readlines()]
        else:
            self.split_keys = getattr(self, f"{split}_keys")[:max_size]
            
        # for broken_file in [os.path.join(f"{base}/data", _) for _ in ["BraTS2021_00000.h5"]]: self.split_keys.remove(broken_file) if broken_file in self.split_keys else 0
        
        use_fine_labeling = n_fine is not None and self.split == "train"
        self.fine_labeled_indices = list(range(n_fine if use_fine_labeling else len(self.split_keys)))
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
        subject = subject | {'cond': torch.cat([subject['coarse'], subject['mask']])}
        subject = subject | {"cond_onehot": torch.cat([subject['coarse'],
                                                       rearrange(torch.nn.functional.one_hot(subject['fine'].long(), 4), '1 ... n -> n ...')[1:].float() if idx in self.fine_labeled_indices else torch.zeros((3,) + subject['coarse'].shape[1:])], dim=0)}

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
    

class BraTS2021_DA(BraTS2021_3D):
    def __init__(self, gen_train_folder=None, primary_batch_size=None, **kw):
        super().__init__(**kw)
        self.primary_batch_size = primary_batch_size
        self.transforms["normalize_image"] = tio.RescaleIntensity(out_min_max=(0, 1), in_min_max=None, include=["image", "raw"])
        if gen_train_folder is not None:
            self.train_keys += [os.path.join(gen_train_folder, _) for _ in os.listdir(gen_train_folder)]
            self.split_keys = getattr(self, f"{self.split}_keys")[:self.max_size]
            self.coarse_labeled_indices = [i for i in range(len(self.split_keys)) if i not in self.fine_labeled_indices]
        random.shuffle(self.split_keys)
        
    def __getitem__(self, idx):
        item = self.load_fn(self.split_keys[idx])
        image, fine = map(lambda x: item[x][:], ["image", "label"])
        coarse = (fine > 0).astype(np.uint8)
        fine = fine[None] if fine.ndim == 3 else fine
        coarse = coarse[None] if coarse.ndim == 3 else coarse
        if "samples" not in item: 
            subject = tio.Subject(image=tio.ScalarImage(tensor=image), 
                                  coarse=tio.LabelMap(tensor=coarse), 
                                  fine=tio.LabelMap(tensor=fine),)
        else: 
            samples, mixed_samples, mixed_fine = map(lambda x: item[x][:], ["samples", "mixed_samples", "mixed_fine"])
            remedyl = TorchioForegroundCropper(crop_level='patch', crop_kwargs=dict(output_size=(128,128,128)), crop_anchor='fine')(tio.Subject(coarse=tio.LabelMap(tensor=coarse), fine=tio.LabelMap(tensor=fine)))
            coarse, fine = remedyl.coarse.data, remedyl.fine.data
            subject = tio.Subject(mixed_samples=tio.ScalarImage(tensor=mixed_samples),
                                  mixed_fines=tio.LabelMap(tensor=mixed_fine),
                                  image=tio.ScalarImage(tensor=samples), 
                                  coarse=tio.LabelMap(tensor=coarse), 
                                  fine=tio.LabelMap(tensor=fine),)
        
        # crop
        subject = self.transforms["crop"](subject)
        # normalize
        subject = self.transforms["normalize_image"](subject)
        # random aug
        subject = self.transforms.get("augmentation", tio.Lambda(identity))(subject)
        subject = {k: v.data for k, v in subject.items()} | {"ids": idx, 
                                                             "mask": subject.fine.data if idx in self.fine_labeled_indices else subject.coarse.data, 
                                                             'casename': os.path.basename(self.split_keys[idx]).split('.')[0]}
        return subject
    
    def collate(self, batch):
        images, coarses, fines, mixed_images, mixed_fines = [], [], [], [], []
        for b in batch:
            images.append(b["image"][None])
            coarses.append(b["coarse"])
            fines.append(b["fine"])
            mixed_images.append((b["mixed_samples"] if "mixed_samples" in b else b["image"])[None])
            mixed_fines.append((b["mixed_fine"] if "mixed_fine" in b else b["fine"]))
        collated_batch = {"image": torch.cat(images, dim=0),
                          "coarse": torch.cat(coarses, dim=0), "fine": torch.cat(fines, dim=0),
                          "mixed_image": torch.cat(mixed_images, dim=0), 'mixed_fine': torch.cat(mixed_fines, 0)}
        return collated_batch
    
    
class BraTS2021_3DFG(Dataset):
    def __init__(self, split="train", 
                crop_to=(96, 96, 96),
                use_shm=False,
                max_size=None,
                n_fine=None,
                base="/ailab/user/dailinrui/data/datasets/BraTS2021"):
        super().__init__()
        self.load_fn = lambda x: h5py.File(x)
        self.transforms = dict(
            crop=TorchioForegroundCropper(crop_level="mask_foreground", 
                                          crop_anchor="coarse",
                                          crop_kwargs=dict(foreground_hu_lb=1e-3,
                                                            foreground_mask_label=None,
                                                            outline=(0, 0, 0)),),
            resize=tio.Resize(crop_to) if crop_to is not None else identity,
            normalize_image=tio.RescaleIntensity(out_min_max=(0, 1), in_min_max=None, include=["image"]),
        )

        self.n_fine = n_fine
        self.split = split
        self.max_size = max_size
        if use_shm: base = "/dev/shm/BraTS2021"
        
        for spt in ["train", "val", "test"]:
            with open(f"{base}/{spt}.list") as fp:
                self.__dict__[f"{spt}_keys"] = [os.path.join(f"{base}/data", _.strip()) for _ in fp.readlines()]
        else:
            self.split_keys = getattr(self, f"{split}_keys")[:max_size]
            
        # for broken_file in [os.path.join(f"{base}/data", _) for _ in ["BraTS2021_00000.h5"]]: self.split_keys.remove(broken_file) if broken_file in self.split_keys else 0
        
        use_fine_labeling = n_fine is not None and self.split == "train"
        self.fine_labeled_indices = list(range(n_fine if use_fine_labeling else len(self.split_keys)))
        self.coarse_labeled_indices = [i for i in range(len(self.split_keys)) if i not in self.fine_labeled_indices]

    def __len__(self):
        return len(self.split_keys)

    def __getitem__(self, idx):
        item = self.load_fn(self.split_keys[idx])
        image, mask = map(lambda x: item[x][:], ["image", "label"])
        
        subject = tio.Subject(image=tio.ScalarImage(tensor=image), coarse=tio.ScalarImage(tensor=(mask >= 1).astype(np.uint8)[None]), fine=tio.ScalarImage(tensor=mask[None]),)
        # crop
        subject = self.transforms["crop"](subject)
        # resize
        subject = self.transforms["resize"](subject)
        # normalize
        subject = self.transforms["normalize_image"](subject)
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