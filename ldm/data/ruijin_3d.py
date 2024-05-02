import os, os.path as path, yaml, pathlib as pb
import json, torchio as tio, torchvision as tv, shutil, nibabel as nib
import re, SimpleITK as sitk, scipy.ndimage as ndimage, numpy as np, multiprocessing as mp

import torch

from datetime import datetime
from tqdm import tqdm
from omegaconf import OmegaConf
from functools import reduce, partial
from collections import OrderedDict, defaultdict
from torch.utils.data import Dataset, _utils
from einops import rearrange


def conserve_only_certain_labels(label, designated_labels=[1, 2, 3, 5, 6, 10, 55, 56, 57, 104]):
    # 6: stomach, 57: colon
    if designated_labels is None:
        return label.astype(np.uint8)
    label_ = np.zeros_like(label)
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
    
    
class RandomCrop3d:
    def __init__(self, crop_size):
        self.crop_size = crop_size
    
    def __call__(self, image):
        pw = max((self.crop_size[0] - image.shape[0]) // 2 + 3, 0)
        ph = max((self.crop_size[1] - image.shape[1]) // 2 + 3, 0)
        pd = max((self.crop_size[2] - image.shape[2]) // 2 + 3, 0)
        image = np.pad(image, [(0, 0), (pw, pw), (ph, ph), (pd, pd)], mode="constant", constant_values=0)
        
        _, w, h, d = image.shape
        w_ = np.random.randint(0, w - self.crop_size[0])
        h_ = np.random.randint(0, h - self.crop_size[1])
        d_ = np.random.randint(0, d - self.crop_size[2])
        
        image = image[:, w_: w_ + self.crop_size[0], h_: h_ + self.crop_size[1], d_: d_ + self.crop_size[2]]
        return image
    
    
def window_norm(image, window_pos=60, window_width=360):
    window_min = window_pos - window_width // 2
    image = (image - window_min) / window_width
    image[image < 0] = 0
    image[image > 1] = 1
    return image


def write_split(basefolder, *splits):
    with open(os.path.join(basefolder, "splits.json"), "w") as f:
        json.dump(dict(zip(["train", "val", "test"], splits)), f, indent=4)
        

def use_split(basefolder):
    with open(os.path.join(basefolder, "splits.json")) as f:
        splits = json.load(f)
    return dict(train=splits.get("train"), val=splits.get("val"), test=splits.get("test"))
    
    
class AutoencoderDataset(Dataset):
    # pretrain: CT cond on report -> totalseg organ mask
    def __init__(self, split="train"):
        super().__init__()
        with open('/mnt/data/oss_beijing/dailinrui/data/ruijin/records/dataset_crc_v2.json', 'rt') as f:
            self.data = json.load(f)
            self.data_keys = list(self.data.keys())

        self.base_folder = "/mnt/data/oss_beijing/dailinrui/data/ruijin"
        self.load_fn = lambda x: sitk.GetArrayFromImage(sitk.ReadImage(x))
        self.volume_transform = tio.Compose((
            # tio.CropOrPad((64, 512, 512)),
            tio.Resize((64, 128, 128)),
        ))

        self.split = split
        np.random.shuffle(self.data_keys)
        self.train_keys = self.data_keys[:round(len(self.data_keys) * 0.99)]
        self.val_keys = self.data_keys[round(len(self.data_keys) * 0.99):]
        # self.val_keys = self.val_keys[:50]
        if not os.path.exists("/mnt/data/smart_health_02/dailinrui/data/pretrained/ldm/contrastive_exp_split/splits.json"):
            write_split("/mnt/data/smart_health_02/dailinrui/data/pretrained/ldm/contrastive_exp_split", self.train_keys, self.val_keys)
        else:
            _ = use_split("/mnt/data/smart_health_02/dailinrui/data/pretrained/ldm/contrastive_exp_split")
            self.train_keys, self.val_keys = _["train"], _["val"]

    def __len__(self):
        return len(self.train_keys) if self.split == "train" else len(self.val_keys)

    def __getitem__(self, idx):
        if self.split == "train":
            item = self.data[self.train_keys[idx]]  # {pacs, date, data, cond={<date>:<string>}}
        else:
            item = self.data[self.val_keys[idx]]  # {pacs, date, data, cond={<date>:<string>}}
        data = pb.Path(item["ct"])
        # totalseg = pb.Path(self.base_folder, "totalseg_8k",
        #                data.parts[-3], data.parts[-2].split("_")[0], data.parts[-1].split(".")[0],
        #                "all.nii.gz")
        totalseg = pb.Path(item["totalseg"])
        crcseg = pb.Path(item["crcseg"])
        image = self.load_fn(data)
        mask = self.load_fn(totalseg)
        crcmask = self.load_fn(crcseg)
        
        mask[crcmask > 0] = 255
        text = item["summary"]
        
        mask = conserve_only_certain_labels(mask) / 255.
        start_layer, end_layer = np.where(mask.sum((1, 2)))[0][[0, -1]]
        mask = mask[start_layer: end_layer + 1]
        image = image[start_layer: end_layer + 1]
        if self.volume_transform is not None:
            image = self.volume_transform(image[None])[0]
            mask = self.volume_transform(mask[None])[0]
        
        image = torch.tensor(window_norm(image))
        mask = torch.tensor(mask)
        
        # reshape to 3 channels to match pretrained weights
        # image = torch.cat([image[..., None], mask[..., None]], dim=-1)
        # image = rearrange(image, "d h w c -> c d h w")
        return dict(data=image, text=text, mask=mask[..., None])
    
    def collate(self, batch):
        return _utils.collate.default_collate(batch)


if __name__ == "__main__":
    c = AutoencoderDataset()
    c[0]