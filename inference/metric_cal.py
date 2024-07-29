import os
import json
import h5py
import torch
import torch.distributed
import torch.nn as nn
import numpy as np
import SimpleITK as sitk
import multiprocessing as mp

from tqdm import tqdm
from einops import rearrange, repeat
from scipy.ndimage import zoom

from medpy import metric
import sys
sys.path.append("/ailab/user/dailinrui/code/latentdiffusion")
from ldm.data.ensemble import LabelParser, OrganType


def exists(x):
    return x is not None


class MetricType:
    @staticmethod
    def dice(x, y):
        assert x.shape == y.shape
        if x.sum() == 0 and y.sum() == 0: return 1
        elif x.sum() == 0 or y.sum() == 0: return 0
        else: return metric.binary.dc(x, y)
    
    @staticmethod
    def precision(x, y):
        assert x.shape == y.shape
        if x.sum() == 0 and y.sum() == 0: return 1
        elif x.sum() == 0 or y.sum() == 0: return 0
        else: return metric.binary.precision(x, y)
    
    @staticmethod
    def recall(x, y):
        assert x.shape == y.shape
        if x.sum() == 0 and y.sum() == 0: return 1
        elif x.sum() == 0 or y.sum() == 0: return 0
        else: return metric.binary.recall(x, y)
    
    @staticmethod
    def hd95(x, y):
        assert x.shape == y.shape
        # resample, otherwise takes too long to compute hd95
        # zoom_coef = np.array((128, 128, 128)) / np.array(x.shape)
        # x, y = map(lambda i: zoom(i, zoom_coef, order=0), [x, y])
        if x.sum() == 0 and y.sum() == 0: return 1
        elif x.sum() == 0 or y.sum() == 0: return 0
        else: return metric.binary.hd95(x, y)
    
    @staticmethod
    def asd(x, y):
        assert x.shape == y.shape
        if x.sum() == 0 and y.sum() == 0: return 1
        elif x.sum() == 0 or y.sum() == 0: return 0
        else: return metric.binary.asd(x, y)
    
    def lpips(self, x, y, reset=False):
        if 'perceptual' not in self.__dict__:
            from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
            self.perceptual = LearnedPerceptualImagePatchSimilarity(net_type='vgg').cuda()
        shp = x.shape
        assert x.shape == y.shape
        if len(shp) == 3:
            x, y = map(lambda i: rearrange(i, "h ... -> h 1 ..."), [x, y])
            # resize x and y to (3, x, x)
            x, y = map(lambda i: repeat(i, "h c ... -> h (r c) ...", r=3), [x, y])
        x, y = map(lpips_clip, [x, y])
        if reset:
            self.perceptual.reset()
        self.perceptual.update(x.float(), y.float())
        lpips = self.perceptual.compute()
        return lpips.item()
    
    def fid(self, x, y, reset=False):
        if 'inception' not in self.__dict__:
            from torchmetrics.image.fid import FrechetInceptionDistance
            self.inception = FrechetInceptionDistance(feature=2048, normalize=True).cuda()
        shp = x.shape
        assert x.shape == y.shape
        if len(shp) == 3:
            x, y = map(lambda i: rearrange(i, "h ... -> h 1 ..."), [x, y])
            x, y = map(lambda i: repeat(i, "h c ... -> h (r c) ...", r=3), [x, y])
        # resize x and y to (3, 299, 299)
        x, y = map(lambda x: torch.nn.functional.interpolate(x, (299, 299)), [x, y])
        if reset:
            self.inception.reset()
        self.inception.update(x.float(), real=False)
        self.inception.update(y.float(), real=True)
        fid = self.inception.compute()
        return fid.item()
    
    def fvd(self, x, y, reset=False):
        if 'video' not in self.__dict__:
            from fvd import FrechetVideoDistance
            self.video = FrechetVideoDistance().cuda()
        b, c, *shp = x.shape
        assert x.shape == y.shape
        if len(shp) == 1: x, y = x[None, None], y[None, None]
        if len(shp) == 2: x, y = x[None], y[None]
        x, y = map(lambda i: repeat(i, "b c h w d -> b (c r) h w d", r=3), [x, y])
        if reset:
            self.video.reset()
        self.video.update(x.float(), y.float())
        fvd = self.video.compute()
        return fvd
        
    def clipscore(self, x, t, compute=None):
        if 'clip' not in self.__dict__:
            from torchmetrics.multimodal.clip_score import CLIPScore
            self.clip = CLIPScore().cuda()
        b, c, *shp = x.shape
        if len(shp) == 3:
            x = rearrange(x, "b c h ... -> (b h) 1 c ...")
            x = torch.nn.functional.interpolate(x, (3, 224, 224)).squeeze(1)
            clip_score = self.clip.update(x, [t for _ in range(x.shape[0])])
        return clip_score.item()
    
    
def load_nifti(p):
    return sitk.GetArrayFromImage(sitk.ReadImage(p))

    
class ComputeSegmentationMetrics:
    def __init__(self, source_folder, target_folder,
                 metrics=["dice", "precision", "recall", "hd95"],
                 num_classes=2,
                 source_transform=None,
                 target_transform=None,
                 classes_to_focus=[1],
                 case_to_focus=None):
        self.source = source_folder
        self.target = target_folder
        self.source_transform = source_transform if source_transform is not None else lambda x: x
        self.target_transform = target_transform if target_transform is not None else lambda x: x
        self.case = case_to_focus
        self.focus_classes = classes_to_focus
        self.num_classes = num_classes
        self.metrics = metrics
        self.results = {m: {_: [] for _ in range(1, self.num_classes)} for m in self.metrics}
        
    def load_data(self):
        if self.case is None:
            gen = dict(zip(sorted(os.listdir(self.source)), sorted(os.listdir(self.target))))
        if isinstance(self.case, (str, os.PathLike)) and os.path.exists(self.case):
            if self.case.endswith("json"):
                with open(self.case, 'r') as f:
                    self.case = json.load(f)
            else:
                with open(self.case, 'r') as f:
                    self.case = [_.strip() for _ in f.readlines()]
        if isinstance(self.case, list):
            gen = dict(zip(sorted([_ for _ in os.listdir(self.source) if _ in self.case]), 
                           sorted([_ for _ in os.listdir(self.target) if _ in self.case])))
        elif isinstance(self.case, dict):
            gen = self.case
            
        for i, (s, t) in enumerate(gen.items()):
            s, t = map(lambda x: load_nifti(x), [os.path.join(self.source, s), os.path.join(self.target, t)])
            yield self.source_transform(s), self.target_transform(t), (i + 1) / len(gen)
            
    def compute(self):
        gen = self.load_data()
        pbar = tqdm(gen, desc="eval progress")
        for s, t, pfix in pbar:
            for m in self.metrics:
                metric = getattr(MetricType, m)
                # with mp.get_context('spawn').Pool(self.num_classes - 1) as pool:
                #     ret = pool.map_async(metric, [[s == i, t == i] for i in range(1, self.num_classes)])
                # for i in range(1, self.num_classes):
                #     self.results[m][i] = ret[i]
                for i in range(1, self.num_classes):
                    self.results[m][i].append(metric(s == i, t == i))
            pbar.set_postfix({"data%": pfix} | {f"{m[:4]}@{i}": round(np.mean(self.results[m][i]), 2) for i in self.focus_classes for m in self.metrics})
        self.results['mean_per_class'] = {m: {i: round(np.mean(self.results[m][i]), 2) for i in range(1, self.num_classes)} for m in self.metrics}
        
        # print(json.dumps(self.results, indent=4))
        with open(os.path.join(self.source, "metrics.json"), "w") as f:
            json.dump(self.results, f, ensure_ascii=0, indent=4)
            
            
class ComputeGenerationMetrics:
    def __init__(self, source_folder, target_folder,
                 metrics=["lpips", "fid", "fvd"],
                 max_size=1024,
                 source_case=None,
                 target_case=None,
                 source_transform=None,
                 target_transform=None):
        self.source = source_folder
        self.target = target_folder
        self.source_case = source_case
        self.target_case = target_case
        self.metrics = metrics
        self.max_size = max_size
        self.metrictype = MetricType()
        self.results = {m: [] for m in self.metrics}
        
        self.source_transform = (lambda x: x) if source_transform is None else source_transform
        self.target_transform = (lambda x: x) if target_transform is None else target_transform
        
    def load_data(self):
        source_cases = sorted(os.listdir(self.source))
        target_cases = sorted(os.listdir(self.target))
        if isinstance(self.source_case, list):
            source_cases = sorted([_ for _ in os.listdir(self.source) if _ in self.source_case])
        if isinstance(self.target_case, list):
            target_cases = sorted([_ for _ in os.listdir(self.target) if _ in self.target_case])
        gen = dict(zip(source_cases, target_cases))
            
        for i, (s, t) in enumerate(gen.items()):
            s, t = map(lambda x: torch.tensor(load_nifti(x)).cuda(), [os.path.join(self.source, s), os.path.join(self.target, t)])
            s, t = map(lambda x: x[None] if x.ndim == 4 else x, [s, t])
            s = self.source_transform(s)
            t = self.target_transform(t)
            yield s, t, (i + 1) / len(gen)
            
    def compute(self):
        i = 0
        gen = self.load_data()
        pbar = tqdm(gen, desc="eval progress", total=self.max_size)
        for s, t, pfix in pbar:
            metric = {f"{m[:4]}": getattr(self.metrictype, m)(s, t) for m in self.metrics}
            for m in self.metrics:
                self.results[m].append(metric[f"{m[:4]}"])
            pbar.set_postfix({"data%": pfix, } | metric)
            if i > self.max_size: break
            else: i += 1
        
        # print(json.dumps(self.results, indent=4))
        self.results['mean'] = {m: round(np.mean(self.results[m]), 2) for m in self.metrics}
        with open(os.path.join(self.source, "metrics.json"), "w") as f:
            json.dump(self.results, f, ensure_ascii=0, indent=4)


def organ_transform(x: torch.Tensor):
    return x / 10 - 1

        
def ddpm_transform(x: torch.Tensor):
    return ((x - x.min()) / (x.max() - x.min()) * 20).clamp(0, 20).round().float() / 10 - 1


def lpips_clip(x: torch.Tensor):
    return x.clamp(-1, 1)


def noise_transform(x: torch.Tensor):
    y = torch.randn((128, 128, 128), device=x.device)
    return ((y - y.min()) / (y.max() - y.min())).round()


def tumor_transform(x: torch.Tensor):
    outline = [10, 10, 10]
    try:
        wl, wr = torch.where(torch.any(torch.any(x, 1), 1))[0][[0, -1]]
        hl, hr = torch.where(torch.any(torch.any(x, 0), -1))[0][[0, -1]]
        dl, dr = torch.where(torch.any(torch.any(x, 0), 0))[0][[0, -1]]
        wl, wr = max(0, wl - outline[0] - 1), min(x.shape[0], wr + outline[0])
        hl, hr = max(0, hl - outline[1] - 1), min(x.shape[0], hr + outline[1])
        dl, dr = max(0, dl - outline[2] - 1), min(x.shape[0], dr + outline[2])
        pad_x = torch.nn.functional.interpolate(x[None, None, wl: wr+1, hl: hr+1, dl: dr+1], size=(128, 128, 128), mode='nearest').float()
        return pad_x[0, 0]
    except Exception as e:
        print(e)
        return torch.zeros((128, 128, 128), device=x.device)
    

def totalsegv2_transform(x: torch.Tensor):
    remap = LabelParser('v2').totalseg2mask(x, OrganType)
    return remap
            
            
def main():
    # with open("/ailab/user/dailinrui/data/datasets/kits23/val.list") as f:
    #     cases = [line.strip().replace('.h5', '.nii.gz') for line in f.readlines()]
    # metric_calculator = ComputeSegmentationMetrics("/ailab/user/dailinrui/data/datasets/kits23/labels",
    #                                                "/ailab/user/dailinrui/data/datasets/kits23/preds",
    #                                                num_classes=3, case_to_focus=cases)
    # with open("/ailab/user/dailinrui/data/datasets/msd_liver/val.list") as f:
    #     cases = [line.strip() for line in f.readlines()]
    # metric_calculator = ComputeSegmentationMetrics("/ailab/user/dailinrui/data/datasets/msd_liver/labelsTr",
    #                                                "/ailab/user/dailinrui/data/datasets/msd_liver/predsVal",
    #                                                num_classes=3, case_to_focus=cases, classes_to_focus=[2], metrics=['hd95', 'dice'])
    
    # metric_calculator = ComputeGenerationMetrics("/ailab/user/dailinrui/data/ccdm_pl/ensemblev2_128_128_128_anatomical/dataset/samples",
    #                                              "/ailab/user/dailinrui/data/datasets/ensemble/totalseg",
    #                                              metrics=['lpips', 'fid'], source_transform=organ_transform, target_transform=organ_transform)
    # metric_calculator = ComputeGenerationMetrics("/ailab/user/dailinrui/data/ldm/ensemble_anatomical_ddpm_128_128_128/dataset/samples",
    #                                              "/ailab/user/dailinrui/data/datasets/ensemble/totalseg",
    #                                              metrics=['lpips', 'fid'], source_transform=ddpm_transform, target_transform=organ_transform)
    # metric_calculator = ComputeGenerationMetrics("/ailab/user/dailinrui/data/datasets/compr_syntumor",
    #                                              "/ailab/user/dailinrui/data/datasets/ensemble/tumorseg",
    #                                              metrics=['lpips', 'fid'], source_transform=noise_transform, target_transform=tumor_transform,)
                                                 #target_case=[_ for _ in os.listdir("/ailab/user/dailinrui/data/datasets/ensemble/tumorseg") if 'LI' in _ or 'liver' in _ ])
    # metric_calculator = ComputeGenerationMetrics("/ailab/user/dailinrui/data/ldm/guidegen_ldm_128_128_128/dataset/inputs_value",
    #                                              "/ailab/user/dailinrui/data/ldm/guidegen_ldm_128_128_128/dataset/samples_value",
    #                                              metrics=['lpips', 'fid', 'fvd'], max_size=200)
    
    # metric_calculator = ComputeSegmentationMetrics("/ailab/user/dailinrui/data/nnUNetv2/nnUNet_raw/Dataset201_BTCV_ABD/predsTs",
    #                                                "/ailab/user/dailinrui/data/nnUNetv2/nnUNet_raw/Dataset201_BTCV_ABD/labelsTs",
    #                                                num_classes=20, classes_to_focus=[], metrics=['dice', 'precision', 'recall'],)
    # metric_calculator = ComputeSegmentationMetrics("/ailab/user/dailinrui/data/nnUNetv2/nnUNet_raw/Dataset203_AMOS/predsTs",
    #                                                "/ailab/user/dailinrui/data/nnUNetv2/nnUNet_raw/Dataset203_AMOS/labelsTs",
    #                                                num_classes=20, classes_to_focus=[], metrics=['dice', 'precision', 'recall'],)
    # metric_calculator = ComputeSegmentationMetrics("/ailab/user/dailinrui/data/nnUNetv2/nnUNet_raw/Dataset204_MSD_COLON/predsTsV1",
    #                                                "/ailab/user/dailinrui/data/nnUNetv2/nnUNet_raw/Dataset204_MSD_COLON/labelsTs",
    #                                                num_classes=2, classes_to_focus=[1], metrics=['dice', 'precision', 'recall', 'hd95'],)
    metric_calculator = ComputeSegmentationMetrics("/ailab/user/dailinrui/data/ldm/guidegen_ldm_128_128_128/dataset/consistency/totalseg",
                                                   "/ailab/user/dailinrui/data/ldm/guidegen_ldm_128_128_128/dataset/consistency/cond_totalseg",
                                                   num_classes=20, classes_to_focus=[], metrics=['dice'], source_transform=totalsegv2_transform)
    metric_calculator.compute()
    

if __name__ == "__main__":
    main()