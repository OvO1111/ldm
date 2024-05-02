import os, os.path as path, yaml, pathlib as pb
import json, torchio as tio, torchvision as tv, shutil, nibabel as nib
import re, SimpleITK as sitk, scipy.ndimage as ndimage, numpy as np, multiprocessing as mp

import torch

from datetime import datetime
from tqdm import tqdm
from omegaconf import OmegaConf
from functools import reduce, partial
from collections import OrderedDict, defaultdict
from torch.utils.data import Dataset


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
    

# class RJDataBase:
#     ROOT_DIR = pb.Path("/mnt/data/oss_beijing/dailinrui/data/ruijin")
    
#     def __init__(self,
#                  data_root = None,
#                  target_image_res=[32, 64, 64],
#                  split="train",
#                  split_portion=[0.7],
#                  force=False,
#                  raw=False) -> None:
#         self.config = dict(
#             ndim=3, target_image_res=target_image_res, split_portion=split_portion, split=split
#         )
#         self.data_root = data_root or self.ROOT_DIR
#         if not isinstance(self.config, dict):
#             self.config = OmegaConf.to_container(self.config)
#         self.keep_orig_class_label = self.config.get("keep_orig_class_label", None)
#         self.seek_dir = pb.Path(self.data_root,
#                                 f"resized_{'x'.join(list(map(str, self.config['target_image_res'])))}") \
#                                     if not raw else pb.Path(self.data_root, f"raw")
#         self.raw = raw
#         self.force = force
#         self.image_transforms = tio.Compose(
#             [tio.RescaleIntensity(out_min_max=(0, 1)),] if not raw else [RandomCrop3d(target_image_res), 
#                                                                          tio.RescaleIntensity(out_min_max=(0, 1))]
#         )
#         self._prepare(force, raw)
#         with open(self.seek_dir / "dataset.json") as f:
#             self.data = json.load(f)[split]
        
#     def _prepare(self, f=False, r=False):
#         def load_json(f):
#             with open(f) as fp: return OrderedDict(**json.load(fp))
            
#         def listdir(d):
#             return list(map(lambda x: d / x, d.glob("case_*")))
            
#         npool = 32
#         all_imaging_history = load_json(self.data_root / "imaging_descrip.json")
#         if not path.exists(self.seek_dir) or f:
#             save_dir = maybe_mkdir(self.seek_dir)
#             all_case_keys = list(all_imaging_history.keys())
#             with mp.get_context("spawn").Pool(npool) as pool:
#                 pool.starmap(_mp_prepare, 
#                              ([{k: v for k, v in all_imaging_history.items() if k in all_case_keys[_::npool]},
#                                save_dir, self.config["target_image_res"], _, r] for _ in range(npool)))
                
#             with open(save_dir / "dataset.json", "w") as f:
#                 all_cases = list(save_dir.glob("case_*"))
#                 np.random.shuffle(all_cases)
#                 train_test_split = np.split(all_cases, 
#                                             np.round(len(all_cases) * np.asarray(self.config["split_portion"])).astype(int))
#                 train_ds = list(map(str, train_test_split[0]))
#                 val_ds = list(map(str, train_test_split[1]))
#                 test_ds = list(map(str, train_test_split[-1]))
#                 json.dump(dict(train=train_ds, val=val_ds, test=test_ds), f, indent=4)
            
#         with mp.get_context("spawn").Pool(npool) as pool:
#             dirs = listdir(self.seek_dir)
#             retval = pool.starmap(check_validity, 
#                                  ([dirs[_::npool]] for _ in range(npool)))
#         retval = reduce(lambda x, y: x + y, retval)
#         if len(retval) > 0:
#             print(f"got broken file list {retval}, reprocessing")
#             _mp_prepare({k: all_imaging_history[k] for k in retval}, self.seek_dir, self.config["target_image_res"], 0, r)

#     def __getitem__(self, i):
#         if not self.raw:
#             img_timeseq = np.load(self.data[i])["img"]
#             if img_timeseq.shape[0] > 1 and len(img_timeseq.shape) == 4:
#                 rand = np.random.randint(img_timeseq.shape[0]-1)
#                 image = img_timeseq[rand]
#                 cond = img_timeseq[rand + 1]
#             else:
#                 image = img_timeseq[0] if len(img_timeseq.shape) == 4 else img_timeseq
#                 cond = None
#             image = self.image_transforms(image[None])[0]
#             if cond is not None:
#                 cond = self.image_transforms(cond[None])[0]
#         else:
#             raw = np.load(self.data[i])
#             img_keys = [_ for _ in raw.files if _.startswith("arr")]
#             img_timeseq = []
#             for key in img_keys:
#                 img_timeseq.append(self.image_transforms(raw[key][None])[0])
#             img_timeseq = np.stack(img_timeseq, axis=0)
#             if img_timeseq.shape[0] > 1 and len(img_timeseq.shape) == 4:
#                 rand = np.random.randint(img_timeseq.shape[0]-1)
#                 image = img_timeseq[rand]
#                 cond = img_timeseq[rand + 1]
#             else:
#                 image = img_timeseq[0] if len(img_timeseq.shape) == 4 else img_timeseq
#                 cond = None
                
#         example = dict(image=image)  #, next_image_cond=cond)
#         return example
    
#     def __len__(self):
#         return len(self.data)
    
    
def window_norm(image, window_pos=60, window_width=360):
    window_min = window_pos - window_width // 2
    image = (image - window_min) / window_width
    image[image < 0] = 0
    image[image > 1] = 1
    return image
    
    
class AutoencoderDataset(Dataset):
    # pretrain: CT cond on report -> totalseg organ mask
    def __init__(self, split="train"):
        with open('/mnt/data/oss_beijing/dailinrui/data/ruijin/records/dataset_v2.json', 'rt') as f:
            self.data = json.load(f)
            self.data_keys = list(self.data.keys())

        self.base_folder = "/mnt/data/oss_beijing/dailinrui/data/ruijin"
        self.load_fn = lambda x: sitk.GetArrayFromImage(sitk.ReadImage(x))
        self.volume_transform = tio.Compose((
            tio.CropOrPad((64, 512, 512)),
            # tio.Resize((64, 256, 256)),
            tio.Lambda(window_norm),
            # tio.RescaleIntensity(out_min_max=(-1, 1)),  # normalize to 0-1
            tio.RandomFlip("LR", 0.3),
            tio.RandomFlip("AP", 0.3),
            tio.RandomFlip("IS", 0.3),
        ))
        self.split = split
        np.random.shuffle(self.data_keys)
        self.train_keys = self.data_keys[:round(len(self.data_keys) * 0.9)]
        self.val_keys = self.data_keys[round(len(self.data_keys) * 0.9):]
        self.val_keys = self.val_keys[:50]
        self.slice_transform = None

    def __len__(self):
        return len(self.train_keys) if self.split == "train" else len(self.val_keys)

    def __getitem__(self, idx):
        if self.split == "train":
            item = self.data[self.train_keys[idx]]  # {pacs, date, data, cond={<date>:<string>}}
        else:
            item = self.data[self.val_keys[idx]]  # {pacs, date, data, cond={<date>:<string>}}
        data = pb.Path(item["data"])
        # data = pb.Path(self.base_folder, "totalseg_8k",
        #                data.parts[-3], data.parts[-2].split("_")[0], data.parts[-1].split(".")[0],
        #                "all.nii.gz")
        image = self.load_fn(data)
        
        if self.volume_transform is not None:
            image = self.volume_transform(image[None])[0]
        n = image.shape[0]
        if self.slice_transform is not None:
            image = torch.stack([self.slice_transform(image[i][None])[0] for i in range(n)], 0)
        
        # reshape to 3 channels to match pretrained weights
        image = image[np.random.randint(n),...,None]#.repeat(3, axis=-1)
        
        return dict(image=image)


if __name__ == "__main__":
    c = AutoencoderDataset()
    c[0]