from abc import abstractmethod
from torch.utils.data import Dataset, IterableDataset

import os
import random
import SimpleITK as sitk
import torchio as tio
from ldm.data.utils import TorchioForegroundCropper, identity, TorchioBaseResizer


class Txt2ImgIterableBaseDataset(IterableDataset):
    '''
    Define an interface to make the IterableDatasets for text2img data chainable
    '''
    def __init__(self, num_records=0, valid_ids=None, size=256):
        super().__init__()
        self.num_records = num_records
        self.valid_ids = valid_ids
        self.sample_ids = valid_ids
        self.size = size

        print(f'{self.__class__.__name__} dataset contains {self.__len__()} examples.')

    def __len__(self):
        return self.num_records

    @abstractmethod
    def __iter__(self):
        pass
    

class MSDDataset(Dataset):
    def __init__(self, base_folder, mapping={}, split="train", max_size=None, resize_to=(96,)*3, force_rewrite_split=False, info={}):
        self.load_fn = lambda x: sitk.GetArrayFromImage(sitk.ReadImage(x))
        self.get_spacing = lambda x: sitk.ReadImage(x).GetSpacing()
        self.transforms = dict(
            resize_base=TorchioBaseResizer(),
            resize=tio.Resize(resize_to) if resize_to is not None else tio.Lambda(identity),
            crop=TorchioForegroundCropper(crop_level="mask_foreground", 
                                          crop_anchor="totalseg",
                                          crop_kwargs=dict(foreground_hu_lb=1e-3,
                                                            foreground_mask_label=None,
                                                            outline=(0, 0, 0))),
            normalize_image=tio.RescaleIntensity(out_min_max=(0, 1), in_min_max=None, include=["image"]),
            normalize_mask=tio.RemapLabels(mapping, include=["mask"])
        )

        self.split = split
        self.base_folder = base_folder
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
            
    def __getitem__(self, idx):
        item = self.split_keys[idx] if isinstance(idx, int) else idx
        
        if self.split in ["train", "val"]:
            image, mask, totalseg = map(lambda x: self.load_fn(os.path.join(self.base_folder, x, item)), ["imagesTr", "labelsTr", "totalsegTr"])
            spacing = self.get_spacing(os.path.join(self.base_folder, "labelsTr", item))
            subject = tio.Subject(image=tio.ScalarImage(tensor=image[None], spacing=spacing), 
                                  mask=tio.LabelMap(tensor=mask[None], spacing=spacing),
                                  totalseg=tio.LabelMap(tensor=totalseg[None], spacing=spacing))
        if self.split == "test":
            image = self.load_fn(os.path.join(self.base_folder, "imagesTs", item))
            spacing = self.get_spacing(os.path.join(self.base_folder, "imagesTs", item))
            subject = tio.Subject(image=tio.ScalarImage(tensor=image[None], spacing=spacing))
        # resize based on spacing
        subject = self.transforms["resize_base"](subject)
        # crop
        subject = self.transforms["crop"](subject)
        ori_size = subject.image.data.shape
        # normalize
        subject = self.transforms["normalize_image"](subject)
        subject = self.transforms["normalize_mask"](subject)
        # resize
        subject = self.transforms["resize"](subject)
        # random aug
        subject = self.transforms.get("augmentation", tio.Lambda(identity))(subject)
        subject = {k: v.data for k, v in subject.items()} | {"ids": idx, 'casename': self.split_keys[idx] if isinstance(idx, int) else idx, "ori_size": ori_size}

        return subject