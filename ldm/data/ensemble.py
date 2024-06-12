import json, os, sys
sys.path.append("/ailab/user/dailinrui/code/latentdiffusion")
import numpy as np
import torchio as tio

from torch.utils.data import default_collate, Dataset
from ldm.data.utils import identity, LabelParser, OrganTypeBase, TorchioForegroundCropper
from ldm.data.ruijin import Ruijin_3D
from ldm.data.base import MSDDataset
from functools import reduce


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
]

CancerType = [
    OrganTypeBase("ColorectalCancer", 11),
    OrganTypeBase("LiverCancer", 12),
    OrganTypeBase("PancreaticCancer", 13),
    OrganTypeBase("LungCancer", 14),
]


class LocalLabel:
    def __init__(self):
        for structure in OrganType + CancerType:
            self.__dict__[structure.name] = structure


class MSDDatasetForEnsemble(MSDDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.totalseg_parser = LabelParser(totalseg_version="v2")
        
    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        sample["totalseg"] = self.totalseg_parser.totalseg2mask(sample["totalseg"], OrganType)
        return sample
        
    
class RuijinForEnsemble(Ruijin_3D):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.transforms["crop"] = TorchioForegroundCropper(crop_level="mask_foreground", 
                                                            crop_anchor="totalseg",
                                                            crop_kwargs=dict(foreground_hu_lb=1e-3,
                                                                                foreground_mask_label=None,
                                                                                outline=(0, 0, 0)))
        self.totalseg_parser = LabelParser(totalseg_version="v1")
        
    def __getitem__(self, idx):
        item = self.data[self.split_keys[idx]] if isinstance(idx, int) else self.data[idx]
        data, totalseg, crcseg, text = map(lambda x: item[x], ["ct", "totalseg", "crcseg", "summary"])
        image, mask, crcmask = map(self.load_fn, [data, totalseg, crcseg])
        
        if self.use_summary_level == "short": text = item.get("summary", "").split("；")[0]
        elif self.use_summary_level == "medium": text = item.get("summary", "")
        elif self.use_summary_level == "long": text = item.get("text", "")
        
        subject = tio.Subject(image=tio.ScalarImage(tensor=image[None]), 
                              mask=tio.ScalarImage(tensor=crcmask[None]),
                              totalseg=tio.ScalarImage(tensor=mask[None]))
        # crop
        subject = self.transforms["crop"](subject)
        # normalize
        subject = self.transforms["normalize_image"](subject)
        # resize
        subject = self.transforms["resize"](subject)
        # random aug
        subject = self.transforms.get("augmentation", tio.Lambda(identity))(subject)
        subject = {k: v.data for k, v in subject.items()} | {"text": text, "casename": self.split_keys[idx] if isinstance(idx, int) else idx}
        
        subject["totalseg"] = self.totalseg_parser.totalseg2mask(subject["totalseg"], OrganType)
        return subject


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
                                                      for _ in self.__dict__ if _.endswith("_dataset")])
        self.prompt_field = prompt_field
        self.totalseg_parser = LabelParser(totalseg_version="v2")
        self.preprocessed_context = np.load(preprocessed_context_dir) if use_preprocessed_context else None
        
    def __len__(self):
        return len(self.split_keys)
    
    def get_ruijin_info(self, key):
        person = self.ruijin_info.get(key, {"birth_date": "2018", "gender": "无"})
        age = 2018 - int(person["birth_date"].split("-")[0])
        gender = person["gender"]
        return age, gender
    
    def get_msd_info(self, totalseg):
        check_sex = (totalseg == 22).sum() > 0  # check for Prostate label (v2)
        if check_sex: return 0, "男"
        else: return 0, "女"
        
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
        sample = item | {"prompt": "，".join([key + "：" + str(value) for key, value in prompt.items()])}
        if self.preprocessed_context is not None: sample = sample | {"context": self.preprocessed_context[sample["casename"]]}
        return sample
    

def save_text_features(save_dir="/ailab/user/dailinrui/data/dependency/",
                       bert_name="bert-ernie-health"):
    from tqdm import tqdm
    from ldm.modules.encoders.modules import FrozenBERTEmbedder
    
    embedder = FrozenBERTEmbedder(os.path.join(save_dir, bert_name))
    for spt in ["train", "val", "test"]:
        save_dict = dict()
        dataset = EnsembleDataset(split=spt)
        for sample in tqdm(dataset, total=len(dataset)):
            save_dict[sample["casename"]] = embedder(sample["prompt"])

        np.savez(os.path.join(save_dir, bert_name + f"_prompt_age_sex_loc_{spt}.npz"), **save_dict)
    

if __name__ == "__main__": save_text_features()