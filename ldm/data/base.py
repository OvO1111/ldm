from abc import abstractmethod
from torch.utils.data import Dataset, IterableDataset

import os
import h5py
import json
import torch
import random
import SimpleITK as sitk
import torchio as tio, nibabel as nib, numpy as np
from functools import partial
from ldm.data.utils import TorchioForegroundCropper, TorchioBaseResizer, \
    load_or_write_split, identity, window_norm


class GenDataset(Dataset):
    def __init__(self, include_case=None, resize_to=(128, 128, 128), max_size=None, **kw):
        totalseg_gen = "/ailab/user/dailinrui/data/ccdm_pl/ensemblev2_128_128_128_anatomical/dataset/samples"
        tumorseg_gen = "/ailab/user/dailinrui/data/datasets/ensemble/val/"
        mapping = "/ailab/user/dailinrui/data/datasets/ensemble/mapping.json"
        with open(mapping) as f:
            self.mapping = {v: k for k, v in json.load(f).items()}
            
        if include_case is None:
            include_case = [_ for _ in os.listdir(totalseg_gen)]
            random.shuffle(include_case)
        
        self.keys = [{"totalseg": os.path.join(totalseg_gen, case), 
                      "tumorseg": os.path.join(tumorseg_gen, self.mapping[case.replace('.nii.gz', '.h5')])} for case in include_case][:max_size]
        
        self.transforms = dict(
            resize=tio.Resize(resize_to) if resize_to is not None else tio.Lambda(identity),
            crop=TorchioForegroundCropper(crop_level="mask_foreground", 
                                          crop_anchor="totalseg",
                                          crop_kwargs=dict(outline=(10, 10, 10))) if resize_to is not None else tio.Lambda(identity),
        )
        
    def __len__(self):
        return len(self.keys)
    
    def load_nifti(self, x):
        return sitk.GetArrayFromImage(sitk.ReadImage(x))
    
    def __getitem__(self, idx):
        item = self.keys[idx]
        totalseg = self.load_nifti(item['totalseg'])
        sample = h5py.File(item['tumorseg'])
        attrs = sample.attrs
        ds = {k: sample[k][:] for k in sample.keys()}
        ds['prompt_context'] = ds["prompt_context"][0]
        image = np.zeros_like(totalseg)
        spacing = (1, 1, 1)
        
        subject = tio.Subject(image=tio.ScalarImage(tensor=ds['image'], spacing=spacing), 
                              totalseg=tio.LabelMap(tensor=ds['totalseg'], spacing=spacing,),
                              mask=tio.LabelMap(tensor=(ds['mask'] == 2).astype(np.float32) if ds['mask'].max() > 1 else ds['mask'], spacing=spacing))
        
        # resize based on spacing
        ori_size = subject.image.data.shape
        # crop
        subject = self.transforms["crop"](subject)
        # resize
        subject = self.transforms["resize"](subject)
        # random aug
        subject = self.transforms.get("augmentation", tio.Lambda(identity))(subject)
        
        sample = dict(**attrs) | ds
        sample.update({k: getattr(subject, k).data for k in subject.keys()})
        sample['totalseg'] = torch.tensor(totalseg[None])
        sample.update({"cond": torch.cat([sample['totalseg'], sample['mask']], dim=0)})
        return sample
    

class MSDDataset(Dataset):
    def __init__(self, name, base="/ailab/user/dailinrui/data/datasets", mapping={}, split="train", max_size=None, resize_to=(96,)*3, force_rewrite_split=False, info={}):
        base_folder = os.path.join(base, name)
        self.base_folder = base_folder
        self.get_spacing = lambda x: sitk.ReadImage(x).GetSpacing()
        if split == "test": split = "val"
        self.transforms = dict(
            resize_base=TorchioBaseResizer(),
            resize=tio.Resize(resize_to) if resize_to is not None else tio.Lambda(identity),
            crop=TorchioForegroundCropper(crop_level="patch", 
                                          crop_anchor="image",
                                          crop_kwargs=dict(output_size=resize_to)) if resize_to is not None else tio.Lambda(identity),
            normalize_image=tio.Lambda(partial(window_norm, window_pos=-500, window_width=1800)),
            normalize_mask=tio.RemapLabels(mapping, include=["mask"])
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
        subject = self.transforms["normalize_mask"](subject)
        # resize
        subject = self.transforms["resize"](subject)
        # random aug
        subject = self.transforms.get("augmentation", tio.Lambda(identity))(subject)
        subject = {k: v.data for k, v in subject.items()} | {'casename': self.split_keys[idx] if isinstance(idx, int) else idx, "ori_size": ori_size, "text": "", "attr": ""}

        return subject
    
    
class TCIADataset(Dataset):
    def __init__(self, 
                 ds_name,
                 mapping={},
                 split="train", 
                 base="/ailab/public/pjlab-smarthealth03/dailinrui/guidegen",
                 resize_to=(128, 128, 128)):
        self.split = split
        self.name = ds_name
        super().__init__()
        self.ds_base = os.path.join(base, self.name)
        with open(os.path.join(self.ds_base, "dataset.json")) as f:
            raw_ds = json.load(f)
        
        self.ds = {}
        for pid, pitem in raw_ds.items():
            for i in range(len(pitem['patient_best_ct'])):
                image_path = pitem["patient_best_ct"][i].replace("/hdd3/share/nifti", "/ailab/public/pjlab-smarthealth03/dailinrui/guidegen")
                if not image_path.endswith(".nii.gz"): continue
                if image_path not in ["/ailab/public/pjlab-smarthealth03/dailinrui/guidegen/TCGA-LUAD/TCGA-LUAD_00056/608.000000-FUSED LUNG WINDOW-41697.nii.gz",
                                      "/ailab/public/pjlab-smarthealth03/dailinrui/guidegen/TCGA-LUAD/TCGA-LUAD_00056/607.000000-FUSED PET CT-09507.nii.gz",
                                      "/ailab/public/pjlab-smarthealth03/dailinrui/guidegen/TCGA-LIHC/TCGA-LIHC_00008/10.000000-AP_Routine  3.0  SPO  cor-13326.nii.gz"]:
                    self.ds.update({f"{pid}_{i:04d}": {"image": image_path, "text": pitem['clinical_info']}})
            
        self.data_keys = list(self.ds.keys())
        random.shuffle(self.data_keys)
        self.train_keys = self.data_keys[:round(len(self.data_keys) * 0.8)]
        # self.val_keys = self.data_keys[round(len(self.data_keys) * 0.7):round(len(self.data_keys) * 0.8)]
        self.val_keys = self.test_keys = self.data_keys[round(len(self.data_keys) * 0.8):]
        self.train_keys, self.val_keys, self.test_keys = load_or_write_split(self.ds_base,
                                                                             force=False,
                                                                             train=self.train_keys, 
                                                                             val=self.val_keys, test=self.test_keys)
        
        self.transforms = dict(
            resize_base=TorchioBaseResizer(),
            resize=tio.Resize(resize_to) if resize_to is not None else tio.Lambda(identity),
            crop=TorchioForegroundCropper(crop_level="image_foreground", 
                                          crop_anchor="image",
                                          crop_kwargs=dict(foreground_hu_lb=1e-3,
                                                            foreground_mask_label=None,
                                                            outline=(0, 0, 0))),
            normalize_image=tio.RescaleIntensity(out_min_max=(0, 1), in_min_max=None, include=["image"]),
            normalize_mask=tio.RemapLabels(mapping, include=["mask"])
        )
        self.split_keys = getattr(self, f"{split}_keys")
        
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
        text = self.ds[item]["text"]
        if not os.path.exists(self.ds[item]["image"]) and ':' in self.ds[item]["image"]:
            self.ds[item]["image"] = self.ds[item]["image"].replace(":", "-")
        image, spacing, code = self.load_fn(self.ds[item]["image"], return_spacing=1)
        totalseg = self.load_fn(self.ds[item]["image"].replace(".nii.gz", "_totalseg.nii.gz"), transpose_code=code)
        mask = self.load_fn(self.ds[item]["image"].replace(".nii.gz", "_tumorseg.nii.gz"), transpose_code=code)
        subject = tio.Subject(image=tio.ScalarImage(tensor=image[None], spacing=spacing), 
                              totalseg=tio.LabelMap(tensor=totalseg[None], spacing=spacing,),
                              mask=tio.LabelMap(tensor=mask[None], spacing=spacing))
        
        # resize based on spacing
        ori_size = subject.image.data.shape
        subject = self.transforms["resize_base"](subject)
        # crop
        subject = self.transforms["crop"](subject)
        # normalize
        subject = self.transforms["normalize_image"](subject)
        subject = self.transforms["normalize_mask"](subject)
        # resize
        subject = self.transforms["resize"](subject)
        # random aug
        subject = self.transforms.get("augmentation", tio.Lambda(identity))(subject)
        subject = {k: v.data for k, v in subject.items()} | {'casename': self.split_keys[idx] if isinstance(idx, int) else idx, "ori_size": ori_size, "text": text["prompt"]}
        subject = subject | {"attr": text}

        return subject