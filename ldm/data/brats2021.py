import os, sys
sys.path.append("/ailab/user/dailinrui/code/latentdiffusion")
import json, torchio as tio
import h5py, SimpleITK as sitk, numpy as np

import torch
from tqdm import tqdm
from functools import reduce
from torch.utils.data import Dataset

from ldm.data.utils import identity, TorchioForegroundCropper

from einops import rearrange
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.font_manager


def minmax(x, xmin=None, xmax=None):
    if xmax is not None:
        x = min(x, xmax)
    if xmin is not None:
        x = max(x, xmin)
    return x


class BraTS2021_3D(Dataset):
    n_coarse = 1
    n_fine = 3
    def __init__(self, split="train", 
                crop_to=(96, 96, 96)):
        super().__init__()
        self.load_fn = lambda x: h5py.File(x)
        self.transforms = dict(
            crop=TorchioForegroundCropper(crop_level="patch", 
                                          crop_anchor="image",
                                          crop_kwargs=dict(output_size=crop_to),) if crop_to is not None else identity,
            normalize_image=tio.RescaleIntensity(out_min_max=(0, 1), in_min_max=None, include=["image"]),
            normalize_mask=tio.RescaleIntensity(out_min_max=(0, 1), in_min_max=(0, 3), include=["mask"])
        )

        self.split = split
        with open(f"/ailab/group/pjlab-smarthealth03/transfers/dailinrui/data/dataset/BraTS2021/{split}.list") as fp:
            self.split_keys = [os.path.join("/ailab/group/pjlab-smarthealth03/transfers/dailinrui/data/dataset/BraTS2021/data", _.strip()) for _ in fp.readlines()]
            
        for broken_file in [os.path.join("/ailab/group/pjlab-smarthealth03/transfers/dailinrui/data/dataset/BraTS2021/data", _) for _ in ["BraTS2021_00000.h5"]]: self.split_keys.remove(broken_file) if broken_file in self.split_keys else 0
            
        self.datatypes = {"image": ["image"], "mask": ["coarse", "fine"], "text": []}
        self.logger_kwargs = {"coarse": {"n": self.n_coarse}, "fine": {"n": self.n_fine}}

    def __len__(self):
        return len(self.split_keys)

    def __getitem__(self, idx):
        item = self.load_fn(self.split_keys[idx])
        image, mask = map(lambda x: item[x][:], ["image", "label"])
        
        subject = tio.Subject(image=tio.ScalarImage(tensor=image), coarse=tio.ScalarImage(tensor=(mask > 1).astype(np.float32)[None]), fine=tio.ScalarImage(tensor=mask[None]),)
        # crop
        subject = self.transforms["crop"](subject)
        # normalize
        subject = self.transforms["normalize_image"](subject)
        subject = self.transforms["normalize_mask"](subject)
        # random aug
        subject = self.transforms.get("augmentation", tio.Lambda(identity))(subject)
        subject = {k: v.data for k, v in subject.items()}

        return subject
    
    def logger(self, log_path: str | os.PathLike, data: dict, global_step: int, current_epoch: int, batch_idx: int):
        def _logger(image, n_slices=10, is_mask=False, **kwargs):
            if len(image.shape) == 4:
                b, h = image.shape[:2]
                if h > n_slices: image = image[:, ::h // n_slices]
                image = rearrange(image, "b h w d -> (b h) 1 w d")
            image = make_grid(image, nrow=min(n_slices, h), normalize=not is_mask)

            if is_mask:
                n = kwargs.get("n")
                cmap = cm.get_cmap("viridis")
                rgb = torch.tensor([(0, 0, 0)] + [cmap(i)[:-1] for i in np.arange(0.1, n) / n], device=image.device)
                colored_mask = rearrange(rgb[image][0], "i j n -> 1 n i j")
                return colored_mask
            else:
                return image
        
        ind_vis = {}
        for k, v in data.items():
            if k in self.datatypes["image"] or k in self.datatypes["mask"]: 
                ind_vis[str(k)] = _logger(v, k in self.datatypes["mask"], **self.logger_kwargs.get(k)).squeeze().data.cpu().numpy()
            elif k in self.datatypes["text"]: ind_vis[str(k)] = v
            
        h = max([getattr(x, "shape", [0, 0, 0])[1] for x in ind_vis.values()])
        w = sum([getattr(x, "shape", [0, 0, 0])[2] for x in ind_vis.values()])
        fig = plt.figure(figsize=(minmax(w // 1024, 15, 30), minmax(h // 1024, 5, 10)))
        for i, (k, v) in enumerate(ind_vis.items()):
            ax = fig.add_subplot(1, len(data), i + 1)
            ax.set_title(k)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.xaxis.set_tick_params(labelbottom=False)
            ax.yaxis.set_tick_params(labelleft=False)
            if isinstance(v, np.ndarray):
                ax.imshow(rearrange(v, "c h w -> h w c"))
            if isinstance(v, str):
                ax.imshow(np.zeros((10, 10)))
                ax.text(0, 0, "\n".join([v[i * 20: (i + 1) * 20] for i in range(np.ceil(len(v) / 20).astype(int))]),
                        color="white",
                        fontproperties=matplotlib.font_manager.FontProperties(size=5,
                                                                              fname='/ailab/user/dailinrui/data/dependency/Arial-Unicode-Bold.ttf'))
        
        plt.savefig(os.path.join(log_path, "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                global_step,
                current_epoch,
                batch_idx)), dpi=300, bbox_inches="tight", pad_inches=0)
        plt.close()
        
    def verify_dataset(self):
        iterator = tqdm(range(len(self.split_keys)))
        for idx in iterator:
            try:
                item = self.__getitem__(idx)
                iterator.set_postfix(shape=item["image"].shape)
            except Exception as e:
                print(self.split_keys[idx], e)


if __name__ == "__main__":
    import time
    ds = BraTS2021_3D(crop_to=(96, 96, 96), split="val")
    ds.verify_dataset()