from torch.utils.data import  Dataset
import sys
sys.path.append('/ailab/user/dailinrui-hdd/code/latentdiffusion/')
from ldm.data.utils import identity, window_norm, TorchioForegroundCropper, TorchioSequentialTransformer, LabelParser, OrganTypeBase
import torch
import h5py, json
import torchio as tio
import os, numpy as np
from collections import OrderedDict
from functools import reduce, partial


OrganTypes = [
    OrganTypeBase("Background", 0),
    OrganTypeBase("Spleen", 1),
    OrganTypeBase("Kidney", 2),
    OrganTypeBase("Liver", 3),
    OrganTypeBase("Stomach", 4),
    OrganTypeBase("Pancreas", 5),
    OrganTypeBase("Lung", 6),
    OrganTypeBase("SmallBowel", 7),
    OrganTypeBase("Duodenum", 8),
    OrganTypeBase("Colon", 9),
    OrganTypeBase("UrinaryBladder", 10),
    OrganTypeBase("Heart", 11),
    OrganTypeBase("Vertebrae", 12),
    OrganTypeBase("Rib", 13),
    OrganTypeBase("Adrenal", 14),
    OrganTypeBase("PortalVeinAndSplenicVein", 15),
    OrganTypeBase("Esophagus", 16),
    OrganTypeBase("Aorta", 17),
    OrganTypeBase("InferiorVenaCava", 18),
    OrganTypeBase("Gallbladder", 19),
]


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
    
    
class SemanticSynthesizerDataset(Dataset):
    def __init__(self, 
                 base='/ailab/user/dailinrui-hdd/data/datasets/ensemble',
                 resize_to=(128,)*3,
                 split='train',
                 max_num=None):
        self.base = f"{base}/{split}_v2"
        self.ds = [os.path.join(self.base, f) for f in os.listdir(self.base)][:max_num]
        self.transforms = {
            "crop": TorchioForegroundCropper(crop_level="mask_foreground", 
                                             crop_anchor="totalseg",
                                             crop_kwargs=dict(foreground_hu_lb=1e-3,
                                                              foreground_mask_label=None,
                                                              outline=(0, 0, 0))),
            "resize": tio.Resize(resize_to) if resize_to is not None else tio.Lambda(identity),
            "norm": tio.Lambda(partial(window_norm, window_pos=0, window_width=2000), include=['image']),
        }
        self.parser = LabelParser(totalseg_version='v1')
    
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        sample = h5py.File(self.ds[idx])
        attrs = sample.attrs
        ds = {k: sample[k][:] for k in sample.keys()}
        ds['prompt_context'] = ds["prompt_context"][0]
        
        subject = tio.Subject(image=tio.ScalarImage(tensor=ds['image']),
                              totalseg=tio.LabelMap(tensor=ds['totalseg']),
                              mask=tio.LabelMap(tensor=ds['mask'].astype(np.uint8)))
        subject = self.transforms['crop'](subject)
        subject = self.transforms['resize'](subject)
        subject = self.transforms['norm'](subject)
        subject = self.transforms.get('augmentation', lambda x: x)(subject)
        totalseg_and_tumorseg = self.parser.totalseg2mask(subject.totalseg.data, OrganTypes)
        totalseg_and_tumorseg[subject.mask.data > 0] = 20
        sample =  {"mask": totalseg_and_tumorseg.long(),
                   "prompt_context": torch.tensor(ds['prompt_context'])} | dict(**attrs)
        return sample
    
    
def gather():
    from tqdm import tqdm
    from pathlib import Path
    from scipy.ndimage import label
    from ldm.modules.encoders.modules import FrozenBERTEmbedder
    
    embedder = FrozenBERTEmbedder().cuda()
    base = '/ailab/user/dailinrui-hdd/data/datasets/ensemble/train_v2'
    for file in tqdm(Path(base).glob('*.h5'), total=len(os.listdir(base))):
        h5 = h5py.File(file, 'r+')
        sample = {'dataset': {k: h5[k][:] for k in h5.keys()}, 'attrs': {k: h5.attrs[k] for k in h5.attrs.keys()}}
        if sample['dataset']['mask'].max() > 1: sample['dataset']['mask'] = (sample['dataset']['mask'] > 1).astype(sample['dataset']['mask'].dtype)
        size_tumors = sample['dataset']['mask'].sum()
        _, n_tumors = label(sample['dataset']['mask'])
        
        prompt = sample['attrs']['prompt'] + f"。该患者肿瘤较{'大' if size_tumors > 40000 else '小'}，共有{n_tumors}个原发肿瘤"
        feature = embedder(prompt).data.cpu().numpy()
        sample['dataset']['prompt_context'] = feature
        sample['attrs']['prompt'] = prompt
        for dataset in sample['dataset']:
            h5[dataset][...] = sample['dataset'][dataset]
        for attr in sample['attrs']:
            h5.attrs[attr] = sample['attrs'][attr]
        h5.close()


if __name__ == "__main__":
    # sbatch -D $(pwd) -J pp -o ./outs/pp.txt -p smart_health_02 -N 1 -n 1 --cpus-per-task=16 --gpus=4 --mem=128G --wrap "python /ailab/user/dailinrui-hdd/code/latentdiffusion/ldm/data/guidegen.py"
    # sbatch -D $(pwd) -J msk -o ./outs/masksyn.txt -p smart_health_02 -N 1 -n 1 --cpus-per-task=24 --gpus=8 --mem=400G --wrap "python /ailab/user/dailinrui-hdd/code/latentdiffusion/main.py -t --base /ailab/user/dailinrui-hdd/code/latentdiffusion/configs/categorical-diffusion/rebuttal.yaml --debug --name guidegen_tcss_21label"
    gather()