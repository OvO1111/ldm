from torch.utils.data import Dataset

import os
import torch
import random
import torchio as tio, nibabel as nib, numpy as np
from ldm.data.utils import TorchioForegroundCropper, TorchioBaseResizer, \
    load_or_write_split, identity, window_norm
    

class AMOS(Dataset):
    def __init__(self, base="/ailab/user/dailinrui/data/datasets/amos22", 
                 split="train", max_size=None, crop_to=(96,96,96)):
        super().__init__()
        self.transforms = dict(
            resize_base=TorchioBaseResizer(),
            crop=TorchioForegroundCropper(crop_level="patch", 
                                          crop_anchor="image",
                                          crop_kwargs=dict(output_size=crop_to),) if crop_to is not None else identity,
            normalize_image=tio.RescaleIntensity(out_min_max=(0, 1), in_min_max=None, include=["image"]),
            remap_mask=tio.RemapLabels({}, include=["mask"])
        )
        self.split = split
        self.max_size = max_size
        if split == 'train':
            self.base = os.path.join(base, "imagesTr")
        elif split == 'val':
            self.base = os.path.join(base, "imagesVa")
        elif split == 'test':
            self.base = os.path.join(base, "imagesTs")
        # id > 500 are MRI, id < 500 are CT
        self.split_keys = [os.path.join(self.base, _) for _ in os.listdir(self.base) 
                            if int(_.split("_")[-1].split(".")[0]) < 500][:max_size]
        
    def __len__(self):
        return len(self.split_keys)
    
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
    
    def __getitem__(self, idx):
        name = self.split_keys[idx]
        image, spacing, code = self.load_fn(os.path.join(self.base, name), return_spacing=1)
        mask = self.load_fn(os.path.join(self.base.replace("images", "labels"), name), transpose_code=code)
        
        subject = tio.Subject(image=tio.ScalarImage(tensor=image[None], spacing=spacing), 
                              mask=tio.ScalarImage(tensor=mask[None], spacing=spacing))
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
        subject = {k: v.data for k, v in subject.items()} | {"ids": idx, "ori_size": ori_size,
                                                             'casename': os.path.basename(self.split_keys[idx]).split('.')[0]}

        return subject
    
    
class BTCV(Dataset):
    def __init__(self, base="/ailab/user/dailinrui/data/datasets/BTCV", 
                 split="train", max_size=None, crop_to=(96,96,96), name='all'):
        super().__init__()
        self.transforms = dict(
            resize_base=TorchioBaseResizer(),
            crop=TorchioForegroundCropper(crop_level="patch", 
                                          crop_anchor="image",
                                          crop_kwargs=dict(output_size=crop_to),) if crop_to is not None else identity,
            normalize_image=tio.RescaleIntensity(out_min_max=(0, 1), in_min_max=None, include=["image"]),
            remap_mask=tio.RemapLabels({}, include=["mask"])
        )
        self.split = split
        self.max_size = max_size
        self.training_base = os.path.join(base, name, "Training")
        self.testing_base = os.path.join(base, name, "Testing")
        
        self.data_keys = os.listdir(os.path.join(self.training_base, "img"))
        random.shuffle(self.data_keys)
        self.train_keys = self.data_keys[:round(0.8*len(self.data_keys))]
        self.val_keys = self.data_keys[round(0.8*len(self.data_keys)):]
        self.test_keys = os.listdir(os.path.join(self.testing_base, "img"))
        
        self.train_keys, self.val_keys, self.test_keys = load_or_write_split(os.path.join(base, name), 
                                                                             train=self.train_keys,
                                                                             val=self.val_keys,
                                                                             test=self.test_keys)
        self.split_keys = getattr(self, f"{split}_keys")
        
    def __len__(self):
        return len(self.split_keys)
    
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
    
    def __getitem__(self, idx):
        name = self.split_keys[idx]
        image, spacing, code = self.load_fn(os.path.join(self.training_base if self.split != 'test' else self.testing_base, name), return_spacing=1)
        if self.split != "test":
            mask = self.load_fn(os.path.join(self.training_base, name.replace("image", "label").replace("Image", "Mask")), transpose_code=code)
            subject = tio.Subject(image=tio.ScalarImage(tensor=image[None], spacing=spacing), 
                                mask=tio.ScalarImage(tensor=mask[None], spacing=spacing))
        else:
            subject = tio.Subject(image=tio.ScalarImage(tensor=image[None], spacing=spacing),)
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
        subject = {k: v.data for k, v in subject.items()} | {"ids": idx, "ori_size": ori_size,
                                                             'casename': os.path.basename(self.split_keys[idx]).split('.')[0]}

        return subject