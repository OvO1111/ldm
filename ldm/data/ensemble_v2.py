from torch.utils.data import  Dataset
from ldm.data.utils import identity, window_norm, TorchioForegroundCropper, TorchioSequentialTransformer
import torch
import h5py
import torchio as tio
import os, json
from collections import OrderedDict
from functools import reduce, partial


class GatheredEnsembleDataset(Dataset):
    def __init__(self, base='/ailab/user/dailinrui/data/datasets/ensemble', 
                 split="train", 
                 resize_to=(128,128,128), 
                 max_size=None,):
        self.transforms = {
            "crop": TorchioForegroundCropper(crop_level="mask_foreground", 
                                             crop_anchor="totalseg",
                                             crop_kwargs=dict(foreground_hu_lb=1e-3,
                                                              foreground_mask_label=None,
                                                              outline=(0, 0, 0))),
            "resize": tio.Resize(resize_to) if resize_to is not None else tio.Lambda(identity),
            "norm": tio.Lambda(partial(window_norm, window_pos=0, window_width=1000), include=['image']),
        }
        self.base = base
        self.split = split
        
        self.train_keys = os.listdir(os.path.join(self.base, 'train_v2'))
        self.val_keys = self.test_keys = os.listdir(os.path.join(self.base, 'val_v2'))
        self.split_keys = getattr(self, f"{split}_keys")[:max_size]
        
    def __len__(self): return len(self.split_keys)
    
    def __getitem__(self, idx):
        sample = h5py.File(os.path.join(self.base, 'train_v2' if self.split == 'train' else 'val_v2', self.split_keys[idx]))
        attrs = sample.attrs
        ds = {k: sample[k][:] for k in sample.keys()}
        ds['prompt_context'] = ds["prompt_context"][0]
        
        subject = tio.Subject(image=tio.ScalarImage(tensor=ds['image']),
                              totalseg=tio.LabelMap(tensor=ds['totalseg']),
                              mask=tio.LabelMap(tensor=ds['mask']))
        subject = self.transforms['crop'](subject)
        subject = self.transforms['resize'](subject)
        subject = self.transforms['norm'](subject)
        subject = self.transforms.get('augmentation', lambda x: x)(subject)
        
        sample = dict(**attrs) | ds
        sample.update({k: getattr(subject, k).data for k in subject.keys()})
        sample.update({"cond": torch.cat([sample['totalseg'], sample['mask']], dim=0)})
        return sample

    
class GatheredDatasetForClassification(GatheredEnsembleDataset):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.transforms['norm'] = tio.Lambda(partial(window_norm, window_pos=60, window_width=360), include=['image'])
        self.transforms['augmentation'] = TorchioSequentialTransformer(OrderedDict({
            "first": tio.OneOf({
                tio.RandomFlip(axes=(0,), flip_probability=0.2): 1,
                tio.RandomFlip(axes=(1,), flip_probability=0.2): 1,
                tio.RandomFlip(axes=(2,), flip_probability=0.2): 1,
                tio.RandomAffine(scales=.2, degrees=30, translation=30): 2,
                tio.Lambda(identity): 5,
            }),
            "second": tio.OneOf({
                tio.RandomAnisotropy(0, downsampling=(1.5, 5), image_interpolation='linear'): 2,
                tio.RandomAnisotropy((1,2), downsampling=(1.5, 5), image_interpolation='linear'): 2,
                tio.RandomNoise(): 1,
                tio.Lambda(identity): 5
            }),
            "third": tio.OneOf({
                tio.RandomGamma(): 5,
                tio.Lambda(identity): 5
            })
        }))
        
        
class GatheredDatasetForGeneration(GatheredEnsembleDataset):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.transforms['norm'] = tio.Lambda(partial(window_norm, window_pos=0, window_width=1500), include=['image'])

