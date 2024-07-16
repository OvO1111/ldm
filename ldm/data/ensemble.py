import json, os, sys
sys.path.append("/ailab/user/dailinrui/code/latentdiffusion")
from re import findall
import h5py
import torch
import numpy as np
import torchio as tio

from torch.utils.data import  Dataset
from ldm.data.utils import identity, window_norm, LabelParser, OrganTypeBase, TorchioForegroundCropper, TorchioBaseResizer, TorchioSequentialTransformer
from ldm.data.ruijin import Ruijin_3D
from ldm.data.base import MSDDataset, TCIADataset
from functools import reduce, partial


OrganType = [
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
    OrganTypeBase("Gallbladder", 19)
]

CancerType = [
    OrganTypeBase("Background", 0),
    OrganTypeBase("Cancer", 1),
]


class LocalLabel:
    def __init__(self):
        for structure in OrganType + CancerType:
            self.__dict__[structure.name] = structure


class MSDDatasetForEnsemble(Dataset):
    def __init__(self, include_ds=None, **kwargs):
        super().__init__()
        if include_ds is None: include_ds = ["msd_liver", "msd_pancreas", "msd_lung", "msd_colon"]
        self.ds = [MSDDataset(ds, mapping={1:0, 2:1} if ds not in ['msd_colon', 'msd_lung'] else {1:1}, **kwargs) for ds in include_ds]
        self.split_keys = reduce(lambda x, y: x + y, [[(k, ids) for k in ds.split_keys] for ids, ds in enumerate(self.ds)], [])
        self.totalseg_parser = LabelParser(totalseg_version="v2")
        
        for ds in self.ds:
            ds.transforms["normalize_image"] = tio.Lambda(window_norm, include=["image"])
            
    def broadcast(self, fn):
        for ds in self.ds:
            fn(ds)
        
    def __getitem__(self, idx):
        item = self.split_keys[idx] if isinstance(idx, int) else idx
        sample = self.ds[item[1]][item[0]]
        sample["totalseg"] = self.totalseg_parser.totalseg2mask(sample["totalseg"], OrganType)
        return sample
    
    def get_clinical(self, sample, parse_fn):
        clinicals = dict(
            sex=parse_fn("male" if (sample["totalseg"] == 22).sum() > 0 else "female", "sex"),
            age=parse_fn(feat_name='age'),
            loc=parse_fn(sample["casename"], feat_name="loc"),
            tx=parse_fn(feat_name='tx'),
            nx=parse_fn(feat_name='nx'),
            mx=parse_fn(feat_name='mx'),
            race=parse_fn('unknown', feat_name='race'),
            alc=parse_fn('', feat_name='alc'),
            cig=parse_fn('', feat_name='cig'),
        )
        return clinicals
        
    
class RuijinForEnsemble(Dataset):
    def __init__(self, **kwargs):
        super().__init__()
        self.ds = [Ruijin_3D(**kwargs)]
        self.data = self.ds[0].data
        self.load_fn = self.ds[0].load_fn
        self.transforms = self.ds[0].transforms
        self.use_summary_level = self.ds[0].use_summary_level
        self.collate_context_len = self.ds[0].collate_context_len
        self.ds[0].transforms["resize_base"] = TorchioBaseResizer()
        self.ds[0].transforms["crop"] = TorchioForegroundCropper(crop_level="mask_foreground", 
                                                                crop_anchor="totalseg",
                                                                crop_kwargs=dict(foreground_hu_lb=1e-3,
                                                                                    foreground_mask_label=None,
                                                                                    outline=(0, 0, 0)))
        self.totalseg_parser = LabelParser(totalseg_version="v1")
        with open("/ailab/user/dailinrui/data/records/basic_info_dict.json") as f, \
            open("/ailab/user/dailinrui/data/records/survival_final.json") as g:
            self.ruijin_info = json.load(f)
            self.ruijin_survival = json.load(g)
        self.split_keys = self.ds[0].split_keys
            
    def broadcast(self, fn):
        for ds in self.ds:
            fn(ds)
        
    def __getitem__(self, idx):
        item = self.data[self.split_keys[idx]] if isinstance(idx, int) else self.data[idx]
        data, totalseg, crcseg, text = map(lambda x: item[x], ["ct", "totalseg", "crcseg", "summary"])
        image, spacing, code = self.load_fn(data, 1)
        mask, crcmask = map(lambda x: self.load_fn(x, transpose_code=code), [ totalseg, crcseg])
        
        if self.use_summary_level == "short": text = item.get("summary", "").split("；")[0]
        elif self.use_summary_level == "medium": text = item.get("summary", "")
        elif self.use_summary_level == "long": text = item.get("text", "")
        
        subject = tio.Subject(image=tio.ScalarImage(tensor=image[None], spacing=spacing), 
                              mask=tio.LabelMap(tensor=crcmask[None], spacing=spacing),
                              totalseg=tio.LabelMap(tensor=mask[None], spacing=spacing))
        # resize based on spacing
        subject = self.transforms["resize_base"](subject)
        # crop
        subject = self.transforms["crop"](subject)
        ori_size = subject.image.data.shape
        # normalize
        subject = self.transforms["normalize_image"](subject)
        # resize
        subject = self.transforms["resize"](subject)
        # random aug
        subject = self.transforms.get("augmentation", tio.Lambda(identity))(subject)
        subject = {k: v.data for k, v in subject.items()} | {"text": text, "casename": self.split_keys[idx] if isinstance(idx, int) else idx, "ori_size": ori_size}
        
        subject["totalseg"] = self.totalseg_parser.totalseg2mask(subject["totalseg"], OrganType)
        return subject | {"attr": ""}
    
    def get_clinical(self, sample, parse_fn):
        person = self.ruijin_info.get(sample["casename"], {"birth_date": "2018", "gender": "无"}) |\
            self.ruijin_survival.get(sample["casename"], {})
        tx, nx = findall(r"T\d", str(person["TN"])), findall(r"N\d", str(person["TN"]))
        if len(tx) == 0: tx = "tx"
        else: tx = tx[0].lower()
        if len(nx) == 0: nx = "nx"
        else: nx = "n0" if nx[0].lower() == 'n0' else 'n1'
        
        clinicals = dict(
            sex=parse_fn(person["gender"], feat_name="sex"),
            age=parse_fn(str(2018 - int(person["birth_date"].split("-")[0])), feat_name='age'),
            loc=parse_fn(sample["text"].split("；")[0].split("：")[-1], feat_name="loc"),
            tx=parse_fn(tx, feat_name='tx'),
            nx=parse_fn(nx, feat_name='nx'),
            mx=parse_fn('m0' if person["metastasis"] is not None else 'm1', feat_name='mx'),
            race=parse_fn('yellow', feat_name='race'),
            alc=parse_fn('', feat_name='alc'),
            cig=parse_fn('', feat_name='cig'),
        )
        return clinicals
    
    
class TCIAForEnsemble(Dataset):
    def __init__(self, include_ds=None, **kw):
        super().__init__()
        if include_ds is None: include_ds = os.listdir(self.base)
        self.ds = [TCIADataset(ds, mapping={1:0, 2:1} if "LI" in ds or "PD" in ds else {1:1}, **kw) for ds in include_ds]
        self.split_keys = reduce(lambda x, y: x + y, [[(k, ids) for k in ds.split_keys] for ids, ds in enumerate(self.ds)])
        self.totalseg_parser = LabelParser(totalseg_version="v2")
        
    def __len__(self):
        return len(self.split_keys)
    
    def broadcast(self, fn):
        for ds in self.ds:
            fn(ds)
    
    def __getitem__(self, idx):
        item = self.split_keys[idx] if isinstance(idx, int) else idx
        subject = self.ds[item[1]][item[0]]
        
        subject["totalseg"] = self.totalseg_parser.totalseg2mask(subject["totalseg"], OrganType)
        return subject
    
    def get_clinical(self, sample, parse_fn):
        clinicals = dict(
            sex=parse_fn(sample['attr']['demographic']['gender'], feat_name="sex"),
            age=parse_fn(sample['attr']['demographic']['age'], feat_name='age'),
            loc=parse_fn(sample['text'].split('\n')[0], feat_name="loc"),
            tx=parse_fn(sample['attr']['diagnostic']['ajcc_t'], feat_name='tx'),
            nx=parse_fn(sample['attr']['diagnostic']['ajcc_n'], feat_name='nx'),
            mx=parse_fn(sample['attr']['diagnostic']['ajcc_m'], feat_name='mx'),
            race=parse_fn(sample['attr']['demographic']['race'], feat_name='race'),
            cig=parse_fn(sample['attr']['exposure']['cig/day'], feat_name='cig'),
            alc=parse_fn(sample['attr']['exposure']['alchohol'], feat_name='alc'),
        )
        return clinicals


class EnsembleDataset(Dataset):
    def __init__(self, 
                 use_dataset_name=["ruijin", "msd_liver", "msd_pancreas", "msd_lung"],
                 split="train",
                 resize_to=(96, 192, 192),
                 max_size=None,
                 force_rewrite_split=False,
                 prompt_field=["年龄", "性别", "肿瘤位置"],
                 use_preprocessed_context=False,
                 preprocessed_context_dir=None):
        use_preprocessed_context = use_preprocessed_context & (preprocessed_context_dir is not None)
        dataset_generic_kwargs = dict(split=split,
                                      force_rewrite_split=force_rewrite_split,
                                      max_size=max_size,
                                      resize_to=resize_to, )
        if "ruijin" in use_dataset_name:
            self.ruijin_dataset = RuijinForEnsemble(**dataset_generic_kwargs)
            with open("/ailab/user/dailinrui/data/records/basic_info_dict.json") as f:
                self.ruijin_info = json.load(f)
            with open("/ailab/user/dailinrui/data/records/survival_final.json") as g:
                self.ruijin_survival = json.load(g)
        if "msd_liver" in use_dataset_name:
            self.liver_dataset = MSDDatasetForEnsemble(base_folder="/ailab/user/dailinrui/data/datasets/msd_liver", 
                                                       mapping={1: 0, 2: 1},        # liver label is given as per TotalSegmentator
                                                       **dataset_generic_kwargs)
        if "msd_pancreas" in use_dataset_name:
            self.pancreas_dataset = MSDDatasetForEnsemble(base_folder="/ailab/user/dailinrui/data/datasets/msd_pancreas", 
                                                          mapping={1: 0, 2: 1,},    # pancreas label is given as per TotalSegmentator
                                                          **dataset_generic_kwargs)
        self.split = split
        self.labels = LocalLabel()
        self.split_keys = reduce(lambda x, y: x + y, [[(_, k) for k in self.__dict__[_].split_keys]
                                                      for _ in self.__dict__ if _.endswith("_dataset")], [])
        self.prompt_field = prompt_field
        self.preprocessed_context = np.load(preprocessed_context_dir) if use_preprocessed_context else None
        
    def __len__(self):
        return len(self.split_keys)
    
    def get_ruijin_info(self, key):
        person = self.ruijin_info.get(key, {"birth_date": "2018", "gender": "无"})
        age = 2018 - int(person["birth_date"].split("-")[0])
        gender = person["gender"]
        return age, gender if gender in ["男", "女"] else "无"
    
    def get_msd_info(self, totalseg):
        check_sex = (totalseg == 22).sum() > 0  # check for Prostate label (v2)
        check_ct_cover = (totalseg == 20).sum() > 0
        if check_sex: return 0, "男"
        if check_ct_cover: return 0, "女"
        else: return 0, "无"
        
    def parse(self, feat, feat_name="age"):
        if isinstance(feat, str):
            if "男" in feat:
                return torch.tensor([0, 1, 0]).float()
            elif "女" in feat: 
                return torch.tensor([0, 0, 1]).float()
            elif "无" in feat:
                return torch.tensor([1, 0, 0]).float()
            elif "年龄" in feat: 
                feat = int(findall(r"[0-9]+", feat))
            else:
                # parse cancer location into integers of one-hot vectors of 1-7, 0 is reserved
                if "肝" in feat: return torch.tensor([0, 1,] + [0,] * 7).float()
                if "胰" in feat: return torch.tensor([0, 0, 1] + [0,] * 6).float()
                if "肺" in feat: return torch.tensor([0, 0, 0, 1] + [0,] * 5).float()
                vector = [0,] * 9
                if "升" in feat: vector[-5] = 1
                if "横" in feat: vector[-4] = 1
                if "降" in feat: vector[-3] = 1
                if "乙" in feat: vector[-2] = 1
                if "直" in feat: vector[-1] = 1
                if sum(vector) == 0: vector[0] = 1
                return torch.tensor(vector).float() / sum(vector)
        if isinstance(feat, int):
            # age
            age_group = [0,] * 11
            age_group[min(max(0, feat // 10), 100)] = 1
            return torch.tensor(age_group).float()
        
    def rev_parse(self, feat_dict):
        feat = ""
        for k, v in feat_dict.items():
            for vb in v:
                vb = vb.softmax(0)
                feat += str(k) + "="
                if k == 'age': feat += str(dict(zip(
                    ["-"] + [f"age{i*10}-{(i+1)*10}" for i in range(1, 10)] + ["age>100"], 
                    [round(_.item(), 3) for _ in vb]
                )))
                if k == 'sex': feat += str(dict(zip(
                    ["-", "male", "female"], 
                    [round(_.item(), 3) for _ in vb]
                )))
                elif k == "tumor_loc": feat += str(dict(zip(
                    ["-", "liver", "pancreas", "lung", "ascendant", "transversal", "descendant", "sigmoid", "rectum"],
                    [round(_.item(), 3) for _ in vb]
                )))
                feat += " ; "
            feat += " \n "
        return feat
        
    def __getitem__(self, idx):
        dataset_name, dataset_idx = self.split_keys[idx]
        item = self.__dict__[dataset_name][dataset_idx]
        
        prompt = {k: "" for k in self.prompt_field}
        if dataset_name == "ruijin_dataset":
            if "肿瘤位置" in prompt: prompt["肿瘤位置"] = item["text"].split("；")[0].split("：")[-1]
            age, sex = self.get_ruijin_info(item["casename"])
            if "年龄" in prompt: prompt["年龄"] = age
            if "性别" in prompt: prompt["性别"] = sex
            tumor_type = "Colorectal"
        elif dataset_name == "liver_dataset":
            if "肿瘤位置" in prompt: prompt["肿瘤位置"] = "肝脏"
            age, sex = self.get_msd_info(item["totalseg"])
            if "年龄" in prompt: prompt["年龄"] = age
            if "性别" in prompt: prompt["性别"] = sex
            tumor_type = "Liver"
        elif dataset_name == "pancreas_dataset":
            if "肿瘤位置" in prompt: prompt["肿瘤位置"] = "胰脏"
            age, sex = self.get_msd_info(item["totalseg"])
            if "年龄" in prompt: prompt["年龄"] = age
            if "性别" in prompt: prompt["性别"] = sex
            tumor_type = "Pancreatic"
        
        item["mask"] *= getattr(self.labels, f"{tumor_type}Cancer").label
        item["mask"][item["mask"] == 0] = item["totalseg"][item["mask"] == 0]
        sample = item | {"text": "，".join([key + "：" + str(value) for key, value in prompt.items()])}
        
        sample = sample | {"age": self.parse(age), "sex": self.parse(sex), "tumor_loc": self.parse(prompt["肿瘤位置"])}
        if self.preprocessed_context is not None: sample = sample | {"context": self.preprocessed_context[sample["casename"]]}
        return sample


class EnsembleDatasetV2(Dataset):
    def __init__(self, *, split="train",
                 use_ruijin=1, use_msd=1, use_tcia=0,
                 use_aug=0, use_norm=1, include_ds=[], **kw):
        msd_include = [_ for _ in include_ds if 'msd' in _]
        tcia_include = [_ for _ in include_ds if 'msd' not in _]
        if use_ruijin: self.ruijin_ds = RuijinForEnsemble(split=split, **kw)
        if use_msd: self.msd_ds = MSDDatasetForEnsemble(split=split, include_ds=msd_include, **kw)
        if use_tcia: self.tcia_ds = TCIAForEnsemble(split=split, include_ds=tcia_include, **kw)
        self.ds = [getattr(self, ds) for ds in self.__dict__.keys() if ds.endswith("ds")]
        
        self.split_keys = reduce(lambda x, y: x+y, [[(_, ids) for _ in ds.split_keys] for ids, ds in enumerate(self.ds)])
        for ds in self.ds:
            if use_norm: ds.broadcast(self.window_norm)
            if use_aug: ds.broadcast(self.add_aug)
            
    @staticmethod
    def window_norm(ds):
        body_window = partial(window_norm, window_pos=0, window_width=1000, out=(-1, 1))
        ds.transforms["normalize_image"] = tio.Lambda(body_window, include=['image'])
            
    @staticmethod
    def add_aug(ds):
        ds.transforms["augmentation"] = TorchioSequentialTransformer({
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
        })
    
    @staticmethod
    def parse(feat="", feat_name='age'):
        feat_index = 0
        feat = str(feat)
        if feat_name == "age":
            age = findall(r"[0-9]+", feat)
            if len(age) > 0:
                feat_index = min(max(0, int(age[0])), 100) // 10
            return torch.nn.functional.one_hot(torch.tensor(feat_index), 11)
        elif feat_name == "sex":
            if reduce(lambda x, y: x | y, [f.lower() in feat for f in ["男", "male"]]):
                feat_index = 1
            elif reduce(lambda x, y: x | y, [f.lower() in feat for f in ["女", "female"]]):
                feat_index = 2
            return torch.nn.functional.one_hot(torch.tensor(feat_index), 3)
        elif feat_name == "race":
            if feat == "white": feat_index = 1
            elif feat == "black": feat_index = 2
            elif feat == "yellow": feat_index = 3
            return torch.nn.functional.one_hot(torch.tensor(feat_index), 4)
        elif feat_name == "alc":
            if reduce(lambda x, y: x | y, [f.lower() in feat for f in ["nan"]]): feat_index = 1
            elif feat != "nan": feat_index = 2
            return torch.nn.functional.one_hot(torch.tensor(feat_index), 3)
        elif feat_name == "cig":
            if reduce(lambda x, y: x | y, [f.lower() in feat for f in ["nan"]]): feat_index = 1
            elif feat.isdecimal(): feat_index = 2
            return torch.nn.functional.one_hot(torch.tensor(feat_index), 3)
        elif feat_name == "loc":
            if reduce(lambda x, y: x | y, [f.lower() in feat for f in ["lung", "肺"]]): feat_index = 1
            elif reduce(lambda x, y: x | y, [f.lower() in feat for f in ["liver", "肝"]]): feat_index = 2
            elif reduce(lambda x, y: x | y, [f.lower() in feat for f in ["pancreas", "胰"]]): feat_index = 3
            elif reduce(lambda x, y: x | y, [f.lower() in feat for f in ["kidney", "肾"]]): feat_index = 4
            elif reduce(lambda x, y: x | y, [f.lower() in feat for f in ["colon", "升", "横", "降", "乙", "直"]]): feat_index = 5
            # elif reduce(lambda x, y: x | y, [f.lower() in feat for f in ["rectum", "直"]]): feat_index = 6
            return torch.nn.functional.one_hot(torch.tensor(feat_index), 6)
        elif feat_name == "tx":
            if reduce(lambda x, y: x | y, [f.lower() in feat for f in ["t0", "0"]]): feat_index = 1
            elif reduce(lambda x, y: x | y, [f.lower() in feat for f in ["t1", "1"]]): feat_index = 2
            elif reduce(lambda x, y: x | y, [f.lower() in feat for f in ["t2", "2"]]): feat_index = 3
            elif reduce(lambda x, y: x | y, [f.lower() in feat for f in ["t3", "3"]]): feat_index = 4
            elif reduce(lambda x, y: x | y, [f.lower() in feat for f in ["t4", "4"]]): feat_index = 5
            return torch.nn.functional.one_hot(torch.tensor(feat_index), 6)
        elif feat_name == "nx":
            if reduce(lambda x, y: x | y, [f.lower() in feat for f in ["n0", "0"]]): feat_index = 1
            elif reduce(lambda x, y: x | y, [f.lower() in feat for f in ["n1", "1"]]): feat_index = 2
            return torch.nn.functional.one_hot(torch.tensor(feat_index), 3)
        elif feat_name == "mx":
            if reduce(lambda x, y: x | y, [f.lower() in feat for f in ["m0", "0"]]): feat_index = 1
            elif reduce(lambda x, y: x | y, [f.lower() in feat for f in ["m1", "1"]]): feat_index = 2
            return torch.nn.functional.one_hot(torch.tensor(feat_index), 3)
        else:
            raise NotImplementedError()
        
    def promptify(self, clinicals, lang='zh'):
        def get_index(v): 
            out = v.argmax(0).item()
            return out
        age = get_index(clinicals['age']) * 10
        sex = get_index(clinicals['sex'])
        race = get_index(clinicals['race'])
        tx = get_index(clinicals['tx'])
        nx = get_index(clinicals['nx'])
        mx = get_index(clinicals['mx'])
        loc = get_index(clinicals['loc'])
        alc, cig = 0, 0
        # alc = get_index(clinicals['alc'])
        # cig = get_index(clinicals['cig'])
        if lang == 'zh':
            age = f'{age}岁' if age > 0 else '未知年龄'
            sex = '男性' if sex == 1 else '女性' if sex == 2 else '人'
            race = '黄种' if race == 3 else '黑种' if race == 2 else '白种' if race == 1 else '未知种族的'
            tx = f"T{tx-1}" if tx > 0 else 'TX'
            nx = f"N{nx-1}" if nx > 0 else 'NX'
            mx = f"M{mx-1}" if mx > 0 else 'MX'
            loc = '肺癌' if loc == 1 else '肝癌' if loc == 2 else '胰腺癌' if loc == 3 else '肾癌' if loc == 4 else '肠癌' if loc == 5 else '某种癌症'
            alc = '有' if alc == 2 else '无' if alc == 1 else '不详'
            cig = '有' if cig == 2 else '无' if cig == 1 else '不详'
            prompt = f"这名患者是一名{age}的{race}{sex}。在检查中发现患有T分期{tx}期,N分期{nx}期,M分期{mx}期的{loc}。该患者饮酒史{alc},烟史{cig}"
        
        return {"prompt": prompt}
    
    def __len__(self): 
        return len(self.split_keys)
    
    def __getitem__(self, idx):
        item = self.split_keys[idx]
        subject = self.ds[item[1]][item[0]]
        clinical = self.ds[item[1]].get_clinical(subject, self.parse)
        prompt = self.promptify(clinical)
        sample = subject | clinical | prompt
        return sample

    
class GatheredEnsembleDataset(Dataset):
    def __init__(self, base='/ailab/user/dailinrui/data/datasets/ensemble', 
                 split="train", 
                 resize_to=(128,128,128), 
                 max_size=None,
                 disable_aug=False):
        self.transforms = TorchioSequentialTransformer({
            "crop": TorchioForegroundCropper(crop_level="mask_foreground", 
                                             crop_anchor="totalseg",
                                             crop_kwargs=dict(foreground_hu_lb=1e-3,
                                                              foreground_mask_label=None,
                                                              outline=(0, 0, 0))),
            "resize": tio.Resize(resize_to) if resize_to is not None else tio.Lambda(identity),
            "norm": tio.Lambda(partial(window_norm, window_pos=0, window_width=1500), include=['image']),
            'augmentation': TorchioSequentialTransformer({
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
            }) if not disable_aug else tio.Lambda(identity)
        })
        self.base = base
        self.split = split
        
        self.train_keys = os.listdir(os.path.join(self.base, 'train'))
        self.val_keys = self.test_keys = os.listdir(os.path.join(self.base, 'val'))
        self.split_keys = getattr(self, f"{split}_keys")[:max_size]
        
    def __len__(self): return len(self.split_keys)
    
    def __getitem__(self, idx):
        sample = h5py.File(os.path.join(self.base, 'train' if self.split == 'train' else 'val', self.split_keys[idx]))
        attrs = sample.attrs
        ds = {k: sample[k][:] for k in sample.keys()}
        ds['prompt_context'] = ds["prompt_context"][0]
        
        subject = tio.Subject(image=tio.ScalarImage(tensor=ds['image']),
                              totalseg=tio.LabelMap(tensor=ds['totalseg']),
                              mask=tio.LabelMap(tensor=ds['mask']))
        subject = self.transforms(subject)
        
        sample = dict(**attrs) | ds
        sample.update({k: getattr(subject, k).data for k in subject.keys()})
        sample.update({"cond": torch.cat([sample['totalseg'], sample['mask']], dim=0)})
        return sample
        

def group_ensemble_dataset(save_dir="/ailab/user/dailinrui/data/datasets/ensemble",
                           split='train'):
    from tqdm import tqdm
    import pathlib as pb
    import gc, multiprocessing as mp
    from pytorch_lightning.utilities.memory import garbage_collection_cuda
    from ldm.modules.encoders.modules import FrozenBERTEmbedder
    embedder = FrozenBERTEmbedder(ckpt_path="/ailab/user/dailinrui/data/dependency/bert-ernie-health").cuda()
    include_ds = ['CPTAC-LSCC', 'TCGA-LUAD', 'TCGA-COAD', 'CMB-LCA', 'CMB-CRC', 'TCGA-LUSC', 'TCGA-READ', 'CPTAC-LSCC',
                  'TCGA-KICH', 'TCGA-KIRP', 'CPTAC-LUAD', 'TCGA-KIRC', 'CPTAC-CCRCC', 'CPTAC-PDA', 'TCGA-LIHC'] + ["msd_liver", "msd_pancreas", "msd_lung", "msd_colon"]
    
    def nullify_transforms(ds):
        for transform in ds.transforms.keys():
            if transform not in ["resize_base", "normalize_mask"]:
                ds.transforms[transform] = tio.Lambda(identity)
                
    def collect_garbage():
        garbage_collection_cuda()
        torch.cuda.empty_cache()
        garbage_collection_cuda()
        gc.collect()
        
    def process(cases, split='train', *datasets):
        for i in tqdm(cases):
            i = int(i)
            name = f"EnsembleV2{split}_{i:05d}.h5"
            if (pb.Path(save_dir) / split / name).exists(): continue
            sample = datasets[0][i] if i < len(datasets[0]) else datasets[1][i - len(datasets[0])]
            if sample['mask'].sum() == 0: 
                print(f"no valid predicted mask in sample {sample['casename']}")
                continue
            h5 = h5py.File(pb.Path(save_dir) / split / name, mode='w')
            for k, v in sample.items():
                if isinstance(v, str): 
                    h5.attrs[k] = v
                    continue
                if isinstance(v, dict):
                    h5.attrs[k] = json.dumps(v, ensure_ascii=False)
                    continue
                if isinstance(v, torch.Tensor): v = v.numpy()
                h5.create_dataset(name=k, data=v, compression='gzip')
            context = embedder(sample["prompt"]).cpu().numpy()
            h5.create_dataset("prompt_context", data=context)
            h5.close()
            gc.collect()
        
    os.makedirs(pb.Path(save_dir) / "train", exist_ok=1)
    os.makedirs(pb.Path(save_dir) / "val", exist_ok=1)
    # exist_len_train = len(os.listdir(pb.Path(save_dir) / "train"))
    # exist_len_val = len(os.listdir(pb.Path(save_dir) / "val"))
    if split == 'train':
        collect_garbage()
        dataset = EnsembleDatasetV2(split="train", use_tcia=1, use_ruijin=1, use_msd=1, include_ds=include_ds)
        for ds in dataset.ds:
            ds.broadcast(nullify_transforms)
            
        # last = max(int(x.split('_')[-1].split('.')[0]) for x in os.listdir(pb.Path(save_dir) / "train"))
        cases = [int(_.split("_")[-1].split('.')[0]) for _ in os.listdir(pb.Path(save_dir) / "train_v2")]
        process(cases, 'train', dataset)
        # pool = []
        # nproc = 2
        # for p in range(nproc):
        #     pool.append(mp.Process(target=process, args=(cases[p::nproc], 'train', dataset)))
        #     pool[-1].start()
            
        # for p in pool:
        #     p.join()
            
    if split == 'val':
        collect_garbage()
        dataset1 = EnsembleDatasetV2(split="val", use_tcia=1, use_ruijin=1, use_msd=1, include_ds=include_ds)
        dataset2 = EnsembleDatasetV2(split="test", use_tcia=0, use_ruijin=1, use_msd=0, include_ds=include_ds)
        total_len = len(dataset1.split_keys) + len(dataset2)
        for ds in dataset1.ds:
            ds.broadcast(nullify_transforms)
        for ds in dataset2.ds:
            ds.broadcast(nullify_transforms)
        
        # last = max(int(x.split('_')[-1].split('.')[0]) for x in os.listdir(pb.Path(save_dir) / "val"))
        cases = [int(_.split("_")[-1].split('.')[0]) for _ in os.listdir(pb.Path(save_dir) / "val_v2")]
        process(cases, 'val', dataset1, dataset2)
        # pool = []
        # nproc = 2
        # for p in range(nproc):
        #     pool.append(mp.Process(target=process, args=(cases[p::nproc], 'val', dataset1, dataset2)))
        #     pool[-1].start()
            
        # for p in pool:
        #     p.join()
    

if __name__ == "__main__": 
    group_ensemble_dataset(split="val")