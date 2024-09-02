from torch.utils.data import  Dataset
import sys
sys.path.append('/ailab/user/dailinrui/code/latentdiffusion/')
from ldm.data.utils import identity, window_norm, TorchioForegroundCropper, TorchioSequentialTransformer
import torch
import h5py, json
import torchio as tio
import os, numpy as np
from collections import OrderedDict
from functools import reduce, partial


class GatheredEnsembleDataset(Dataset):
    def __init__(self, base='/ailab/user/dailinrui/data/datasets/ensemble', 
                 split="train", 
                 resize_to=(128,128,128), 
                 max_size=None, include_ds=None, include_cases=None):
        self.transforms = {
            "crop": TorchioForegroundCropper(crop_level="mask_foreground", 
                                             crop_anchor="totalseg",
                                             crop_kwargs=dict(foreground_hu_lb=1e-3,
                                                              foreground_mask_label=None,
                                                              outline=(0, 0, 0))),
            "resize": tio.Resize(resize_to) if resize_to is not None else tio.Lambda(identity),
            "norm": tio.Lambda(partial(window_norm, window_pos=0, window_width=2000), include=['image']),
        }
        self.base = base
        self.split = split
        
        self.train_keys = os.listdir(os.path.join(self.base, 'train'))
        self.val_keys = self.test_keys = os.listdir(os.path.join(self.base, 'val'))
        self.split_keys = getattr(self, f"{split}_keys")[:max_size]
        
        with open(os.path.join(base, 'mapping.json')) as f:
            mappings = json.load(f)
        
        if include_cases is not None:
            self.split_keys = [_ for _ in self.split_keys if _ in include_cases]
        else:
            if include_ds is not None:
                self.split_keys = [_ for _ in self.split_keys if reduce(lambda x, y: x | y, [x in mappings[_] for x in include_ds])]
        
    def __len__(self): return len(self.split_keys)
    
    def __getitem__(self, idx):
        sample = h5py.File(os.path.join(self.base, 'train' if self.split == 'train' else 'val', self.split_keys[idx]))
        attrs = sample.attrs
        ds = {k: sample[k][:] for k in sample.keys()}
        ds['prompt_context'] = ds["prompt_context"][0]
        
        subject = tio.Subject(image=tio.ScalarImage(tensor=ds['image']),
                              totalseg=tio.LabelMap(tensor=ds['totalseg']),
                              mask=tio.LabelMap(tensor=(ds['mask'] == 2).astype(np.float32) if ds['mask'].max() > 1 else ds['mask']))
        subject = self.transforms['crop'](subject)
        subject = self.transforms['resize'](subject)
        subject = self.transforms['norm'](subject)
        subject = self.transforms.get('augmentation', lambda x: x)(subject)
        
        sample = dict(**attrs) | ds
        sample.update({k: getattr(subject, k).data for k in subject.keys()})
        sample.update({"cond": torch.cat([sample['totalseg'], sample['mask']], dim=0)})
        # if sample['mask'].max() > 1: sample['mask'] = (sample['mask'] == 2).float()  # kits
        return sample

    
class GatheredDatasetForClassification(GatheredEnsembleDataset):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.transforms['norm'] = tio.Lambda(partial(window_norm, window_pos=60, window_width=360), include=['image'])
        self.transforms['augmentation'] = TorchioSequentialTransformer(OrderedDict({
            "first": tio.OneOf({
                tio.RandomAnisotropy(0, downsampling=(1.5, 5), image_interpolation='linear', include=['image']): 2,
                tio.RandomAnisotropy((1,2), downsampling=(1.5, 5), image_interpolation='linear', include=['image']): 2,
                tio.RandomNoise(include=['image']): 1,
                tio.Lambda(identity): 5
            }),
            "second": tio.OneOf({
                tio.RandomGamma(include=['image']): 5,
                tio.Lambda(identity): 5
            })
        }))
        
        
class GatheredDatasetForGeneration(GatheredEnsembleDataset):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.transforms['norm'] = tio.Lambda(partial(window_norm, window_pos=0, window_width=2400), include=['image'])


class GatheredDatasetForMaskGeneration(GatheredEnsembleDataset):
    def __init__(self, num_classes=20, **kw):
        super().__init__(**kw)
        self.transforms['norm'] = tio.RescaleIntensity(in_min_max=(0, num_classes), out_min_max=(0, 1), include=['totalseg'])
        
    def __getitem__(self, idx):
        sample = h5py.File(os.path.join(self.base, 'train' if self.split == 'train' else 'val', self.split_keys[idx]))
        attrs = sample.attrs
        ds = {k: sample[k][:] for k in sample.keys()}
        ds['prompt_context'] = ds["prompt_context"][0]
        
        subject = tio.Subject(totalseg=tio.ScalarImage(tensor=ds['totalseg']),
                              mask=tio.LabelMap(tensor=ds['mask'].astype(np.uint8) if ds['mask'].max() == 1 else (ds['mask'] == 1).astype(np.uint8)))
        subject = self.transforms['crop'](subject)
        subject = self.transforms['resize'](subject)
        subject = self.transforms['norm'](subject)
        subject = self.transforms.get('augmentation', lambda x: x)(subject)
        
        sample = dict(**attrs) | ds
        sample.update({k: getattr(subject, k).data for k in subject.keys()})
        return sample
    
    
class MedSynDataset(GatheredEnsembleDataset):
    def __getitem__(self, idx):
        sample = h5py.File(os.path.join(self.base, 'train' if self.split == 'train' else 'val', self.split_keys[idx]))
        attrs = sample.attrs
        ds = {k: sample[k][:] for k in sample.keys()}
        ds['prompt_context'] = ds["prompt_context"][0]
        
        subject = tio.Subject(image=tio.ScalarImage(tensor=ds['image']),
                              totalseg=tio.LabelMap(tensor=ds['totalseg']),
                              mask=tio.LabelMap(tensor=ds['mask'].astype(np.uint8) if ds['mask'].max() == 1 else (ds['mask'] == 2).astype(np.uint8)))
        subject = self.transforms['crop'](subject)
        subject = self.transforms['resize'](subject)
        subject = self.transforms['norm'](subject)
        subject = self.transforms.get('augmentation', lambda x: x)(subject)
        sample =  {"data": torch.cat([subject.image.data, subject.totalseg.data / 10 - 1, subject.mask.data], dim=0),
                   "prompt_context": torch.tensor(ds['prompt_context'])} | dict(**attrs)
        return sample


if __name__ == "__main__":
    # import SimpleITK as sitk
    # import numpy as np
    # from tqdm import tqdm
    # p = "/ailab/user/dailinrui/data/datasets/ensemble/image"
    # os.makedirs(p, exist_ok=1)
    # ds = GatheredDatasetForGeneration(split='train', resize_to=(128,128,128), )
    # for i in tqdm(range(1000)):
    #     x = ds[i]
    #     seg = x['mask'].max()
    #     if seg != 1: print(seg)
    #     sitk.WriteImage(sitk.GetImageFromArray(seg), os.path.join(p, x['casename'] + ".nii.gz"))
    GatheredEnsembleDataset()