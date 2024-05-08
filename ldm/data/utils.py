import os, os.path as path, yaml, pathlib as pb
import json, torchio as tio, torchvision as tv, shutil, nibabel as nib
import re, SimpleITK as sitk, scipy.ndimage as ndimage, numpy as np, multiprocessing as mp

import torch

from tqdm import tqdm
from einops import rearrange
from datetime import datetime
from omegaconf import OmegaConf
from functools import reduce, partial
from collections import OrderedDict, defaultdict


def identity(x, *a, **b): return x


def conserve_only_certain_labels(label, designated_labels=[1, 2, 3, 5, 6, 10, 55, 56, 57, 104]):
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


def _mp_prepare(process_dict, save_dir, target_res, pid, raw=False):
    cumpred = 0
    dummy_dir = maybe_mkdir(f"/mnt/data/smart_health_02/dailinrui/data/temp/{pid}", destory_on_exist=True)
    for k, patient_imaging_history in process_dict.items():
        cumpred += 1
        _latent = defaultdict(list)
        # if path.exists(save_dir / f"case_{k}.npz"): continue
        valid_imaging_histories = [_ for _ in sorted(patient_imaging_history, key=lambda x: get_date(x["time"]))
                                   if len(_["abd_imagings"]) > 0]
        for img_index, img in enumerate(valid_imaging_histories):
            if len(img["abd_imagings"]) == 0: continue
            _latent["date"].append(get_date(img["time"]))
            parsed, coeff = parse(img["abd_imagings"][0], target_res, raw)
            _latent["resize_coeff"].append(coeff)
            _latent["img"].append(parsed)
        if len(_latent["date"]) == 0: continue
        
        dates = np.stack(_latent["date"], axis=0)
        if not raw:
            imgs = np.stack(_latent["img"], axis=0)
            coeffs = np.stack(_latent["resize_coeff"], axis=0)
            np.savez(dummy_dir / f"case_{k}.npz", date=dates, img=imgs, resize_coeff=coeffs)
            print(f"<{pid}> is processing {k}: {cumpred}/{len(process_dict)} cases {coeffs[0].tolist()}", end="\r")
        else:
            np.savez(dummy_dir / f"case_{k}.npz", *_latent["img"], date=dates)
            print(f"<{pid}> is processing {k}: {cumpred}/{len(process_dict)} cases {_latent['img'][0].shape}", end="\r") 
        shutil.copyfile(dummy_dir / f"case_{k}.npz", save_dir / f"case_{k}.npz")
        os.remove(dummy_dir / f"case_{k}.npz")
    shutil.rmtree(dummy_dir)
    
    
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


def window_norm(image, window_pos=60, window_width=360):
    window_min = window_pos - window_width // 2
    image = (image - window_min) / window_width
    image[image < 0] = 0
    image[image > 1] = 1
    return image


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


class TorchioForegroundCropper(tio.transforms.Transform):
    def __init__(self, crop_level="all", crop_kwargs=None,
                 *args, **kwargs):
        self.crop_level = crop_level
        self.crop_kwargs = crop_kwargs
        super().__init__(*args, **kwargs)

    def apply_transform(self, data: tio.Subject):
        # data: c h w d
        subject_ = {k: v.data for k, v in data.items()}
        type_ = {k: v.type for k, v in data.items()}

        if self.crop_level == "all":
            return data

        assert "image" in subject_
        if self.crop_level == "patch":
            image_ = subject_["image"]
            output_size = self.crop_kwargs["output_size"]
            
            pw = max((output_size[0] - image_.shape[1]) // 2 + 3, 0)
            ph = max((output_size[1] - image_.shape[2]) // 2 + 3, 0)
            pd = max((output_size[2] - image_.shape[3]) // 2 + 3, 0)
            image_ = torch.nn.functional.pad(image_, (pw, pw, ph, ph, pd, pd), mode='constant', value=0)

            (c, w, h, d) = image_.shape
            w1 = np.random.randint(0, w - output_size[0])
            h1 = np.random.randint(0, h - output_size[1])
            d1 = np.random.randint(0, d - output_size[2])
            
            padder = identity if pw + ph + pd == 0 else lambda x: torch.nn.functional.pad(x, (pw, pw, ph, ph, pd, pd), mode='constant', value=0)
            cropper = [slice(w1, w1 + output_size[0]), slice(h1, h1 + output_size[1]), slice(d1, d1 + output_size[2])]
            subject_ = {k: tio.Image(tensor=padder(v)[:, cropper[0], cropper[1], cropper[2]], type=type_[k]) for k, v in subject_.items()}
            
        outline = self.crop_kwargs.get("outline", [0] * 6)
        if isinstance(outline, int): outline = [outline] * 6
        if len(outline) == 3: outline = reduce(lambda x, y: x + y, zip(outline, outline))
        if self.crop_level == "image_foreground":
            image_ = subject_["image"]
            s1, e1 = torch.where((image_ >= self.crop_kwargs.get('foreground_hu_lb', 0)).any(-1).any(-1).any(0))[0][[0, -1]]
            s2, e2 = torch.where((image_ >= self.crop_kwargs.get('foreground_hu_lb', 0)).any(1).any(-1).any(0))[0][[0, -1]]
            s3, e3 = torch.where((image_ >= self.crop_kwargs.get('foreground_hu_lb', 0)).any(1).any(1).any(0))[0][[0, -1]]
            cropper = [slice(max(0, s1 - outline[0]), min(e1 + 1 + outline[1], image_.shape[1])),
                       slice(max(0, s2 - outline[2]), min(e2 + 1 + outline[3], image_.shape[2])),
                       slice(max(0, s3 - outline[4]), min(e3 + 1 + outline[5], image_.shape[3]))]
            subject_ = {k: tio.Image(tensor=v[:, cropper[0], cropper[1], cropper[2]], type=type_[k]) for k, v in subject_.items()}
        
        assert "mask" in subject_
        if self.crop_level == "mask_foreground":
            mask_ = conserve_only_certain_labels(subject_["mask"], self.crop_kwargs.get("foreground_mask_label", None))
            s1, e1 = torch.where(mask_.any(-1).any(-1).any(0))[0][[0, -1]]
            s2, e2 = torch.where(mask_.any(1).any(-1).any(0))[0][[0, -1]]
            s3, e3 = torch.where(mask_.any(1).any(1).any(0))[0][[0, -1]]
            cropper = [slice(max(0, s1 - outline[0]), min(e1 + 1 + outline[1], mask_.shape[1])),
                       slice(max(0, s2 - outline[2]), min(e2 + 1 + outline[3], mask_.shape[2])),
                       slice(max(0, s3 - outline[4]), min(e3 + 1 + outline[5], mask_.shape[3]))]
            subject_ = {k: tio.Image(tensor=v[:, cropper[0], cropper[1], cropper[2]], type=type_[k]) for k, v in subject_.items()}
            
        return tio.Subject(subject_)
            
            

