import os, sys
sys.path.append("/ailab/user/dailinrui/code/latentdiffusion")
import h5py, SimpleITK as sitk, numpy as np
import json, torchio as tio, torchvision.transforms.v2 as v2, shutil

import torch, random
from tqdm import tqdm
from einops import rearrange
from torch.utils.data import Dataset, _utils

from ldm.data.utils import conserve_only_certain_labels, identity, window_norm, load_or_write_split, TorchioForegroundCropper


class Ruijin_3D(Dataset):
    def __init__(self, split="train", 
                force_rewrite_split=True, 
                resize_to=(64, 128, 128),
                max_size=None,
                context_len=None,
                use_summary_level="short",
                text_encoder="CT_report_abstract_BLS_PULSE-20bv5_short"):
        super().__init__()
        with open('/ailab/user/dailinrui/data/records/dataset_crc_v2.json', 'rt') as f:
            self.data = json.load(f)
            self.data_keys = list(self.data.keys())
            self.data_keys.remove("RJ202302171638326937")

        self.use_summary_level = use_summary_level
        self.collate_context_len = context_len
        self.load_fn = lambda x: sitk.GetArrayFromImage(sitk.ReadImage(x))
        self.transforms = dict(
            resize=tio.Resize(resize_to) if resize_to is not None else tio.Lambda(identity),
            crop=TorchioForegroundCropper(crop_level="mask_foreground", 
                                          crop_anchor="mask",
                                          crop_kwargs=dict(foreground_hu_lb=1e-3,
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
        
        if "PULSE" in text_encoder:
            if "short" in text_encoder: self.use_summary_level = "short"
            if "medium" in text_encoder: self.use_summary_level = "medium"
            if "long" in text_encoder: self.use_summary_level = "long"
            
        self.context = {name: value for name, value in np.load(f"/ailab/user/dailinrui/data/dependency/{text_encoder}.npz").items()}

        self.train_keys, self.val_keys, self.test_keys = load_or_write_split("/ailab/user/dailinrui/data/ldm/",
                                                                             force_rewrite_split,
                                                                             train=self.train_keys, 
                                                                             val=self.val_keys, test=self.test_keys)
        self.split_keys = getattr(self, f"{split}_keys")[slice(0, max_size)]

    def __len__(self):
        return len(self.split_keys)
    
    @staticmethod
    def _get_class(text):
        class_id = -1
        if "升结肠" in text: class_id = 0
        elif "横结肠" in text: class_id = 1
        elif "降结肠" in text: class_id = 2
        elif "乙状结肠" in text: class_id = 3
        elif "直肠" in text: class_id = 4
        else: class_id = 5
        return torch.tensor(class_id)

    def __getitem__(self, idx):
        item = self.data[self.split_keys[idx]] if isinstance(idx, int) else idx
        context = torch.tensor(self.context[self.split_keys[idx] if isinstance(idx, int) else idx])
        data, totalseg, crcseg, text = map(lambda x: item[x], ["ct", "totalseg", "crcseg", "summary"])
        image, mask, crcmask = map(self.load_fn, [data, totalseg, crcseg])
        
        if self.use_summary_level == "short": text = item.get("summary", "").split("；")[0]
        elif self.use_summary_level == "medium": text = item.get("summary", "")
        elif self.use_summary_level == "long": text = item.get("text", "")
        
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
        subject = {k: v.data for k, v in subject.items()} | {"text": text, "context": context, "class_id": self._get_class(item.get("summary", "").split("；")[0])}

        return subject
    
    def collate(self, batch):
        context = [b["context"] for b in batch]
        for b in batch: del b["context"]
        collated = _utils.collate.default_collate(batch)
        longest_context = max([b.shape[0] for b in context]) if self.collate_context_len is None else self.collate_context_len
        collated_context = torch.cat([torch.nn.functional.pad(c, (0, 0, 0, longest_context - c.shape[1]), mode="constant", value=0) if c.shape[1] <= longest_context else c[:, :longest_context] for c in context], dim=0)
        collated["context"] = collated_context
        return collated
    
    
class RuijinJointDataset(Dataset):
    # pretrain: CT cond on report -> totalseg organ mask
    def __init__(self, split="train", toy=False, cache_len=0):
        with open('/mnt/data/oss_beijing/dailinrui/data/ruijin/records/dataset_crc_v2.json', 'rt') as f:
            self.data = json.load(f)
        self.get_spacing = lambda x: sitk.ReadImage(x).GetSpacing()
        self.advance_step = 3
        
        self.volume_transform = tio.Compose((
            tio.Lambda(window_norm),
        ))
        self.mask_transform = tio.Compose((
            tio.RescaleIntensity(out_min_max=(0, 1), in_min_max=(0, 255)),
        ))
        self.joined_transform = v2.Compose([
            v2.RandomAffine(degrees=10, translate=(0.2, 0.2), scale=(0.9, 1.1))
        ])
        
        self.split = split
        if cache_len is None:
            cache_len = len(self.split_keys)
        self.cache_len = cache_len
        self.split_keys = load_or_write_split("/mnt/workspace/dailinrui/data/pretrained/controlnet/train_val_split")[split]
        if toy: 
            self.split_keys = self.split_keys[:10]
        self.data_cache = {}
    
    def __len__(self):
        return len(self.split_keys)
    
    def get_from_cache(self, k):
        dataset_path = "/mnt/data/oss_beijing/dailinrui/data/dataset/ruijin"
        if (_cached_sample := self.data_cache.get(k, None)) is None:
            if os.path.exists(_cached_path := os.path.join(dataset_path, f"{k}.h5")):
                try:
                    _cached_sample = h5py.File(_cached_path, "r")
                except Exception:
                    return None
                subject = tio.Subject(seg=tio.LabelMap(tensor=_cached_sample["seg"][:]),
                                      image=tio.ScalarImage(tensor=_cached_sample["image"][:]))
                spacing = _cached_sample["spacing"]
                ret_dict = dict(subject=subject, raw_spacing=spacing)
                if len(self.data_cache) < self.cache_len:
                    self.data_cache[k] = ret_dict
                return None
                # return ret_dict
                   
    def load_fn(self, k, valid_keys=["ct", "totalseg", "crcseg"]):
        resized = {}
        if (_cached_sample := self.get_from_cache(k)) is None:
            for key, n in self.data[k].items():
                if key in valid_keys:
                    im = sitk.ReadImage(n)
                    dtype = im.GetPixelID()
                    spacing = im.GetSpacing()
                    raw = sitk.GetArrayFromImage(im)
                    zoom_coef = np.array([1, 1, spacing[-1]])[::-1]
                    # change spacing to 1 ? ?
                    # _tmp = torch.tensor(zoom(raw, zoom_coef, order=0 if dtype == 1 else 3))
                    _tmp = raw
                if key == "ct":
                    resized[key] = self.volume_transform(_tmp[None])
                elif key == "totalseg":
                    resized[key] = conserve_only_certain_labels(_tmp)
                elif key == "crcseg":
                    resized["totalseg"][_tmp > 0] = 11
                    resized["totalseg"] = self.mask_transform(resized["totalseg"][None])

            subject = tio.Subject(seg=tio.LabelMap(tensor=resized["totalseg"]),
                                  image=tio.ScalarImage(tensor=resized["ct"]))
            ret_dict = dict(subject=subject, raw_spacing=spacing)
            if len(self.data_cache) < self.cache_len:
                self.data_cache[k] = ret_dict
            return ret_dict
        return _cached_sample

    def __getitem__(self, idx):
        key = self.split_keys[idx]
        sample = self.load_fn(key)
        subject, spacing = sample["subject"], sample["raw_spacing"]

        image, seg = subject.image.data, subject.seg.data
        start_layer, end_layer = torch.where(seg.sum((0, 2, 3)))[0][[0, -1]]
        
        if self.split == "train":
            random_slice = random.randint(start_layer.item(), end_layer.item() - self.advance_step) if random.random() < .1 else start_layer.item()
            random_slices = slice(random_slice, random_slice + self.advance_step)
            previous_layer = image[:, random_slice - 1: random_slice] if random_slice > start_layer else torch.zeros_like(image[:, 0:1])
            image_slice = image[:, random_slices] 
            seg_slice = seg[:, random_slices]
            
            sample = rearrange(torch.cat([image_slice, seg_slice, previous_layer], dim=1), "1 c h w -> c 1 h w")
            sample = self.joined_transform(sample)
            image = sample[:self.advance_step]
            control = sample[self.advance_step:]
        else:
            random_slice = random.randint(start_layer.item(), end_layer.item())
            random_slices = slice(random_slice, random_slice + 1)
            previous_layer = image[:, random_slice - 1: random_slice] if random_slice > start_layer else torch.zeros_like(image[:, 0:1])
            image_slice = image[:, random_slices] 
            seg_slice = seg[:, random_slices]
            
            # autoencoder takes in (prev_layer, slice)
            sample = torch.cat([image_slice, previous_layer, seg_slice], dim=1)
            sample = self.joined_transform(sample)
            image = sample[:, :1]
            control = sample[:, 1:]
            
        return dict(image=image, mask=control, caseid=key)
                    # wholemask=subject.seg.data,
                    # wholeimage=subject.image.data)
        
    def preload(self):
        _temp_path = "/mnt/data/smart_health_02/dailinrui/data/temp"
        dataset_path = "/mnt/data/oss_beijing/dailinrui/data/dataset/ruijin"
        
        def flush(f, filename=None):
            if filename is not None:
                os.remove(filename)
                return 
            for file in os.listdir(f):
                os.remove(os.path.join(f, file))
        
        for k in tqdm(self.split_keys):
            if os.path.exists(os.path.join(dataset_path, f"{k}.h5")): continue
            sample = self.load_fn(k)
            saved_sample = h5py.File(_x := os.path.join(_temp_path, f"{k}.h5"), "w")
            saved_sample.create_dataset("image", data=sample["subject"].image.data, compression="gzip")
            saved_sample.create_dataset("seg", data=sample["subject"].seg.data, compression="gzip")
            saved_sample.create_dataset("spacing", data=np.array(sample["raw_spacing"]), compression="gzip")
            saved_sample.close()
            shutil.copyfile(_x, os.path.join(dataset_path, f"{k}.h5"))
            flush(_temp_path, filename=_x)


if __name__ == "__main__":
    # group ruijin dataset
    ds = Ruijin_3D(resize_to=None)
    ds.transforms["crop"] = TorchioForegroundCropper(crop_level="all")
    ds.transforms["normalize_mask"] = tio.Lambda(identity)
    iterator = tqdm(ds.data_keys)
    for key in iterator:
        data = ds[ds.data[key]]
        data = {k: v.cpu().numpy() for k, v in data.items() if torch.is_tensor(v)}
        np.savez(os.path.join("/ailab/user/dailinrui/data/datasets/ruijin", key + ".npz"), **data)
        iterator.set_postfix(save_path=os.path.join("/ailab/user/dailinrui/data/datasets/ruijin", key + ".npz"),
                             data_keys=list(data.keys()),
                             data_shape=list(map(lambda x: data[x].shape, data.keys())))