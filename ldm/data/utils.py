import os, os.path as path, pathlib as pb
import json, torchio as tio, shutil, nibabel as nib
import re, scipy.ndimage as ndimage, numpy as np

import torch, random

from typing import List
from datetime import datetime
from functools import reduce
from scipy.ndimage import label
from collections.abc import MutableMapping
from collections import defaultdict, namedtuple, OrderedDict


OrganTypeBase = namedtuple("OrganTypeBase", ["name", "label"])
    

class TotalsegOrganType:
    def __init__(self, path):
        self.max_split = 0
        self.organtypes = dict()
        self.nested_organtypes = dict()
        with open(path) as f:
            for line in f.readlines():
                index, name, alias = line.split('\t')
                index = int(index)
                name_split = name.split('_')
                self.max_split = max(self.max_split, len(name_split))
                key = reduce(lambda x, y: x + y[0].upper() + y[1:], name_split)
                self.organtypes[key[0].upper() + key[1:]] = index
                
                nested_type = self.nested_organtypes
                while len(name_split) > 0:
                    name_split_pop = name_split.pop(0)
                    name_split_pop = name_split_pop[0].upper() + name_split_pop[1:]
                    if name_split_pop not in nested_type: 
                        nested_type[name_split_pop] = {} 
                        if len(name_split) == 0: nested_type[name_split_pop] = index
                    nested_type = nested_type[name_split_pop]
                    
        self.organtypes["Background"] = 0
        self.nested_organtypes["Background"] = 0
        
        _flatten_dict = dict()
        def _flatten_dict_values(d: MutableMapping, parent_key='') -> MutableMapping:
            items = []
            for k, v in d.items():
                new_key = parent_key + k if parent_key else k
                if isinstance(v, MutableMapping):
                    items.extend(_flatten_dict_values(v, new_key))
                else:
                    items.append(v)
            if parent_key: _flatten_dict[parent_key] = items
            return items

        _flatten_dict_values(self.nested_organtypes)
        self.organtypes.update(_flatten_dict)
        
    def __getitem__(self, name):
        name = name if name in self.organtypes else [i for i in self.organtypes.keys() if i.lower() == name.lower()][0]
        return self.organtypes[name]

TotalsegOrganTypeV1 = TotalsegOrganType("./dependency/totalseg_v1_label_mapping.txt")
TotalsegOrganTypeV2 = TotalsegOrganType("./dependency/totalseg_v2_label_mapping.txt")


class LabelParser:
    def __init__(self, totalseg_version="v1"):
        self.totalseg_version = totalseg_version
        self.totalseg_decoder = TotalsegOrganTypeV1 if totalseg_version == "v1" else TotalsegOrganTypeV2
        # self.totalseg_mapping = self.totalseg_decoder.load(merge_level)
        
    def totalseg2mask(self, label, organtype: List[OrganTypeBase]=None):
        if organtype is None: return label
        label_ = np.zeros_like(label) if isinstance(label, np.ndarray) else torch.zeros_like(label) 
        for organ in organtype:
            label_index = organ.label
            totalseg_indices = self.totalseg_decoder[organ.name]
            if isinstance(totalseg_indices, int): totalseg_indices = [totalseg_indices]
            for totalseg_index in totalseg_indices:
                label_[label == totalseg_index] = label_index
        return label_
  

def identity(x, *a, **b): return x


def conserve_only_certain_labels(label,
                                 designated_labels=[1, 2, 3, 5, 6, 10, 55, 56, 57, 104],
                                 totalseg_version="v1"):
    if isinstance(label, np.ndarray):
        if designated_labels is None:
            return label.astype(np.uint8)
        label_ = np.zeros_like(label)
    elif isinstance(label, torch.Tensor):
        if designated_labels is None:
            return label.long()
        label_ = torch.zeros_like(label)
    for il, l in enumerate(designated_labels):
        label_[label == l] = il + 1
    return label_


def maybe_mkdir(p, destory_on_exist=False):
    if path.exists(p) and destory_on_exist:
        shutil.rmtree(p)
    os.makedirs(p, exist_ok=True)
    return pb.Path(p)

            
def get_date(date_string):
    date_pattern = r"\d{4}-\d{2}-\d{2}"
    _date_ymd = re.findall(date_pattern, date_string)[0]
    date = datetime.strptime(_date_ymd, "%Y-%m-%d") if len(_date_ymd) > 1 else None
    return date

def parse(i, target_res, raw=False):
    img = nib.load(i).dataobj[:].transpose(2, 1, 0)
    if raw:
        return img, np.zeros((3,))
    resize_coeff = np.array(target_res) / np.array(img.shape)
    resized = ndimage.zoom(img, resize_coeff, order=3)
    return resized, resize_coeff

    
def check_validity(file_ls):
    broken_ls = []
    for ifile, file in enumerate(file_ls):
        try:
            np.load(file)
        except Exception as e:
            print(f"{file} raised exception {e}, reprocessing")
            broken_ls.append(file.name.split("_")[1].split(".")[0])
        print(f"<{os.getpid()}> is processing {ifile}/{len(file_ls)}", end="\r")
    return broken_ls


def window_norm(image, window_pos=60, window_width=360, out=(-1, 1)):
    window_min = window_pos - window_width / 2
    image = (image - window_min) / window_width
    image = (out[1] - out[0]) * image + out[0]
    image = image.clamp(min=out[0], max=out[1])
    return image


class invertible_window_norm:
    def __init__(self, 
                 window_pos=60, 
                 window_width=360, 
                 in_minmax=(-1200, 1200), 
                 out_minmax=(-1, 1), 
                 outlier_percentile=0.1):
        self.window_pos = window_pos
        self.window_width = window_width
        self.window_max = window_pos + window_width / 2
        self.window_min = window_pos - window_width / 2
        self.outlier_percentile = outlier_percentile
        self.in_min, self.in_max = in_minmax
        self.out_min, self.out_max = out_minmax

    def encode(self, image):
        out_range = self.out_max - self.out_min
        outlier_offset = self.outlier_percentile / 2 * out_range

        normalized = np.zeros_like(image) if isinstance(image, np.ndarray) else torch.zeros_like(image)
        mask_lower = image < self.window_min
        mask_upper = image > self.window_max
        mask_window = (image >= self.window_min) & (image <= self.window_max)
        normalized[mask_lower] = (image[mask_lower] - self.window_min) / (self.window_min - self.in_min) * outlier_offset + (self.out_min + outlier_offset)
        normalized[mask_upper] = (image[mask_upper] - self.in_max) / (self.in_max - self.window_max) * outlier_offset + self.out_max
        normalized[mask_window] = (image[mask_window] - self.window_max) / self.window_width * (1 - self.outlier_percentile) * out_range + (self.out_max - outlier_offset)
        
        return normalized
    
    def decode(self, image):
        out_range = self.out_max - self.out_min
        outlier_offset = self.outlier_percentile / 2 * out_range
        
        recon = np.zeros_like(image) if isinstance(image, np.ndarray) else torch.zeros_like(image)
        mask_lower = image < self.out_min + outlier_offset
        mask_upper = image > self.out_max - outlier_offset
        mask_window = (image >= self.out_min + outlier_offset) & (image <= self.out_max - outlier_offset)
        recon[mask_lower] = -(image[mask_lower] - (self.out_min + outlier_offset)) / outlier_offset * (self.in_min - self.window_min) + self.window_min
        recon[mask_upper] = (image[mask_upper] - self.out_max) / outlier_offset * (self.in_max - self.window_max) + self.in_max
        recon[mask_window] = (image[mask_window] - (self.out_max - outlier_offset)) / ((1 - self.outlier_percentile) * out_range) * self.window_width + self.window_max
        
        return recon
        

def load_or_write_split(basefolder, force=False, **splits):
    splits_file = os.path.join(basefolder, "splits.json")
    if os.path.exists(splits_file) and not force:
        with open(splits_file, "r") as f:
            splits = json.load(f)
    else:
        with open(splits_file, "w") as f:
            json.dump(splits, f, indent=4)
    splits = list(splits.get(_) for _ in ["train", "val", "test"])
    return splits


class TorchioSequentialTransformer:
    def __init__(self, d: OrderedDict, force_include=False):
        self.transform_keys = d.keys()
        self.transforms = d.values()
        self.force_include = force_include
        
    def __call__(self, x: tio.Subject):
        for k, tr in zip(self.transform_keys, self.transforms):
            x = tr(x) if not self.force_include else tr(x, include=[k])
        return x


class TorchioBaseResizer(tio.transforms.Transform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @staticmethod
    def _interpolate(x, scale_coef, mode="trilinear"):
        x_rsz = torch.nn.functional.interpolate(x[None].float(), scale_factor=scale_coef, mode=mode)[0]
        if mode == "nearest":
            x_rsz = x_rsz.round()
        return x_rsz
        
    def apply_transform(self, data: tio.Subject):
        # data: c h w d
        subject_ = {k: v.data for k, v in data.items()}
        type_ = {k: v.type for k, v in data.items()}
        class_ = {k: tio.ScalarImage if isinstance(v, tio.ScalarImage) else tio.LabelMap for k, v in data.items()}
        
        local_spacing = np.array(data[list(subject_.keys())[0]]["spacing"])
        scale_coef = tuple(local_spacing / local_spacing.mean())[::-1]
        
        subject_ = {k: class_[k](tensor=self._interpolate(v, scale_coef, mode="nearest" if type_[k] == "label" else "trilinear"), type=type_[k]) for k, v in subject_.items()}
        return tio.Subject(subject_)


class TorchioForegroundCropper(tio.transforms.Transform):
    def __init__(self, 
                 crop_level="all", 
                 crop_anchor=None,
                 parent_kwargs={},
                 *args, **kwargs):
        self.crop_level = crop_level
        self.crop_kwargs = kwargs
        self.crop_anchor = crop_anchor
        super().__init__(**parent_kwargs)
        
    def _patch_cropper(self, _image, _output_size, _mode="random", _pad_value=0):
        # maybe pad image if _output_size is larger than _image.shape
        ph = max((_output_size[0] - _image.shape[1]) // 2 + 3, 0)
        pw = max((_output_size[1] - _image.shape[2]) // 2 + 3, 0)
        pd = max((_output_size[2] - _image.shape[3]) // 2 + 3, 0)
        padder = lambda x: torch.nn.functional.pad(x, (pd, pd, ph, ph, pw, pw), mode='constant', value=0)
        _image = padder(_image)
        # padder = identity
            
        if _mode == "random":
            h_center = random.randint(ph, _image.shape[0] - ph)
            w_center = random.randint(pw, _image.shape[1] - pw)
            d_center = random.randint(pd, _image.shape[2] - pd)

        elif _mode == "foreground":
            if _image.sum() == 0: return self._patch_cropper(_image, _output_size, "random", _pad_value)
            
            _hl, _hr = torch.where(torch.any(torch.any(_image, 2), 2))[-1][[0, -1]]
            _wl, _wr = torch.where(torch.any(torch.any(_image, 1), -1))[-1][[0, -1]]
            _dl, _dr = torch.where(torch.any(torch.any(_image, 1), 1))[-1][[0, -1]]
            _label, _n = label(_image[:, _hl: _hr + 1, _wl: _wr + 1, _dl: _dr + 1].numpy())
            _label = torch.tensor(_label == random.randint(1, _n), dtype=_image.dtype)
            hl, hr = torch.where(torch.any(torch.any(_label, 2), 2))[-1][[0, -1]] + _hl
            wl, wr = torch.where(torch.any(torch.any(_label, 1), -1))[-1][[0, -1]] + _wl
            dl, dr = torch.where(torch.any(torch.any(_label, 1), 1))[-1][[0, -1]] + _dl
            hc, hd = (hl + hr) // 2, hr - hl + 1
            wc, wd = (wl + wr) // 2, wr - wl + 1
            dc, dd = (dl + dr) // 2, dr - dl + 1
            
            h_center = random.randint(hc - hd // 4, max(hc + hd // 4, hc - hd // 4 + 1))
            w_center = random.randint(wc - wd // 4, max(wc + wd // 4, wc - wd // 4 + 1))
            d_center = random.randint(dc - dd // 4, max(dc + dd // 4, dc - dd // 4 + 1))
        
        h_left = max(0, h_center - _output_size[0] // 2)
        h_right = min(_image.shape[1], h_center + _output_size[0] // 2)
        h_offset = (max(0, _output_size[0] // 2 - h_center), max(0, h_center + _output_size[0] // 2 - _image.shape[1]))
        w_left = max(0, w_center - _output_size[1] // 2)
        w_right = min(_image.shape[2], w_center + _output_size[1] // 2)
        w_offset = (max(0, _output_size[1] // 2 - w_center), max(0, w_center + _output_size[1] // 2 - _image.shape[2]))
        d_left = max(0, d_center - _output_size[2] // 2)
        d_right = min(_image.shape[3], d_center + _output_size[2] // 2) 
        d_offset = (max(0, _output_size[2] // 2 - d_center), max(0, d_center + _output_size[2] // 2 - _image.shape[3]))
        cropper = lambda x: x[:, h_left: h_right, w_left: w_right, d_left: d_right]
        post_padder = lambda x: torch.nn.functional.pad(x, 
                                                        (*d_offset, *w_offset, *h_offset),
                                                        mode='constant', value=_pad_value)
            
        return padder, cropper, post_padder 
            
    def apply_transform(self, data: tio.Subject):
        # data: c h w d
        subject_ = {k: v.data for k, v in data.items()}
        type_ = {k: v.type for k, v in data.items()}
        class_ = {k: tio.ScalarImage if isinstance(v, tio.ScalarImage) else tio.LabelMap for k, v in data.items()}

        if self.crop_level == "raw":
            return data

        if self.crop_level == "patch":
            image_ = subject_[self.crop_anchor]
            output_size = self.crop_kwargs["output_size"]
            pad_value = self.crop_kwargs.get('pad_value', 0)
            foreground_prob = self.crop_kwargs.get("foreground_prob", 0)
            if callable(pad_value): pad_value = pad_value(subject_)
            
            mode = "foreground" if random.random() < foreground_prob else "random"
            padder, cropper, post_padder = self._patch_cropper(image_, output_size, mode, pad_value)
            subject_ = {k: class_[k](tensor=post_padder(cropper(padder(v))), type=type_[k]) for k, v in subject_.items()}
        
        elif "foreground" in self.crop_level:
            outline = self.crop_kwargs.get("outline", [0] * 6)
            if isinstance(outline, int): outline = [outline] * 6
            if len(outline) == 3: outline = reduce(lambda x, y: x + y, zip(outline, outline))
            if self.crop_level == "image_foreground":
                image_ = subject_[self.crop_anchor]
                s1, e1 = torch.where((image_ >= self.crop_kwargs.get('foreground_hu_lb', 0)).any(-1).any(-1).any(0))[0][[0, -1]]
                s2, e2 = torch.where((image_ >= self.crop_kwargs.get('foreground_hu_lb', 0)).any(1).any(-1).any(0))[0][[0, -1]]
                s3, e3 = torch.where((image_ >= self.crop_kwargs.get('foreground_hu_lb', 0)).any(1).any(1).any(0))[0][[0, -1]]
                cropper = [slice(max(0, s1 - outline[0]), min(e1 + 1 + outline[1], image_.shape[1])),
                        slice(max(0, s2 - outline[2]), min(e2 + 1 + outline[3], image_.shape[2])),
                        slice(max(0, s3 - outline[4]), min(e3 + 1 + outline[5], image_.shape[3]))]
                subject_ = {k: class_[k](tensor=v[:, cropper[0], cropper[1], cropper[2]], type=type_[k]) for k, v in subject_.items()}
            
            if self.crop_level == "mask_foreground":
                mask_ = conserve_only_certain_labels(subject_[self.crop_anchor], self.crop_kwargs.get("foreground_mask_label", None))
                s1, e1 = torch.where(mask_.any(-1).any(-1).any(0))[0][[0, -1]]
                s2, e2 = torch.where(mask_.any(1).any(-1).any(0))[0][[0, -1]]
                s3, e3 = torch.where(mask_.any(1).any(1).any(0))[0][[0, -1]]
                cropper = [slice(max(0, s1 - outline[0]), min(e1 + 1 + outline[1], mask_.shape[1])),
                        slice(max(0, s2 - outline[2]), min(e2 + 1 + outline[3], mask_.shape[2])),
                        slice(max(0, s3 - outline[4]), min(e3 + 1 + outline[5], mask_.shape[3]))]
                subject_ = {k: class_[k](tensor=v[:, cropper[0], cropper[1], cropper[2]], type=type_[k]) for k, v in subject_.items()}
            
        return tio.Subject(subject_)



