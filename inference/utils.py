import pathlib as pb

import re
import torch
import itertools
import numpy as np
from PIL import Image

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager
from matplotlib.cm import get_cmap
from torchvision.utils import make_grid
from einops import rearrange
from scipy.ndimage import sobel
from collections import namedtuple
from torch.utils.data.sampler import Sampler


OrganClass = namedtuple("OrganClass", ["label_name", "totalseg_id", "color"])
abd_organ_classes = [
    OrganClass("unlabeled", 0, (0, 0, 0)),
    OrganClass("spleen", 1, (0, 80, 100)),
    OrganClass("kidney_left", 2, (119, 11, 32)),
    OrganClass("kidney_right", 3, (119, 11, 32)),
    OrganClass("liver", 5, (250, 170, 30)),
    OrganClass("stomach", 6, (220, 220, 0)),
    OrganClass("pancreas", 10, (107, 142, 35)),
    OrganClass("small_bowel", 55, (255, 0, 0)),
    OrganClass("duodenum", 56, (70, 130, 180)),
    OrganClass("colon", 57, (0, 0, 255)),
    OrganClass("urinary_bladder", 104, (0, 255, 255)),
    OrganClass("colorectal_cancer", 255, (0, 255, 0))
]


def find_vacancy(path):
    path = pb.Path(path)
    d, f, s = path.parent, path.name, ".".join([""] + path.name.split(".")[1:])
    exist_files = list(_.name for _ in d.glob(f"*{s}"))
    file_num = list(int(([-1] + re.findall(r"\d+", _))[-1]) for _ in exist_files)
    fa = [i for i in range(1000) if i not in file_num]
    vacancy = d / (f.split(s)[0] + str(fa[0]) + s)
    print("found vacancy at ", f.split(s)[0] + str(fa[0]) + s)
    return vacancy
    

def custom_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.) / 2.
    x = x.permute(1, 2, 0).numpy()
    x = (255 * x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x


def custom_to_np(x):
    # saves the batch in adm style as in https://github.com/openai/guided-diffusion/blob/main/scripts/image_sample.py
    sample = x.detach().cpu()
    sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    return sample


def logs2pil(logs, keys=["sample"]):
    imgs = dict()
    for k in logs:
        try:
            if len(logs[k].shape) == 4:
                img = custom_to_pil(logs[k][0, ...])
            elif len(logs[k].shape) == 3:
                img = custom_to_pil(logs[k])
            else:
                print(f"Unknown format for key {k}. ")
                img = None
        except:
            img = None
        imgs[k] = img
    return imgs


def minmax(val, minimum=None, maximum=None):
    if maximum is not None:
        val = min(val, maximum)
    if minimum is not None:
        val = max(val, minimum)
    return val


def combine_mask_and_im(x, overlay_coef=0.2, colors=None, n=11, mask_normalied=True):
    # b (im (no-display) msk) h w (d)
    if len(x.shape) == 5: ndim = 3
    if len(x.shape) == 4: ndim = 2
    def find_mask_boundaries_nd(im, mask, color):
        boundaries = torch.zeros_like(mask)
        for i in range(1, 12):
            m = (mask == i).numpy()
            sobel_x = sobel(m, axis=1, mode='constant')
            sobel_y = sobel(m, axis=2, mode='constant')
            if ndim == 3:
                sobel_z = sobel(m, axis=3, mode='constant')

            boundaries = torch.from_numpy((np.abs(sobel_x) + np.abs(sobel_y) + (0 if ndim == 2 else np.abs(sobel_z))) * i) * (boundaries == 0) + boundaries * (boundaries != 0)
        im = color[boundaries.long()] * (boundaries[..., None] > 0) + im * (boundaries[..., None] == 0)
        return im
    
    image = 255 * x[:, 0, ..., None].clamp(0, 1).repeat(*([1] * (ndim+1)), 3)
    mask = x[:, -1].round() * n if mask_normalied else mask[:, 1]
    cmap = get_cmap("viridis")
    colors = torch.tensor([(0, 0, 0)] + [cmap(i)[:-1] for i in np.arange(0.3, n) / n] if colors is None else colors, device=image.device)
    colored_mask = (colors[mask.long()] * (mask[..., None] > 0) + image * (mask[..., None] == 0))
    colored_im = colored_mask * overlay_coef + image * (1-overlay_coef)
    colored_im = rearrange(find_mask_boundaries_nd(colored_im, mask, colors), "b ... c -> b c ...")
    return colored_im


def visualize(image: torch.Tensor, n: int=11, num_images=8):
    is_mask = image.dtype == torch.long
    if len(image.shape) == 5:
        image = image[:, 0] 
    if len(image.shape) == 4:
        b, h = image.shape[:2]
        if h > num_images: image = image[:, ::h // num_images]
        image = rearrange(image, "b h w d -> (b h) 1 w d")
    image = make_grid(image, nrow=min(num_images, h), normalize=not is_mask, )

    if is_mask:
        cmap = get_cmap("viridis")
        rgb = torch.tensor([(0, 0, 0)] + [cmap(i)[:-1] for i in np.arange(0.3, n) / n], device=image.device)
        colored_mask = rearrange(rgb[image][0], "i j n -> 1 n i j")
        return colored_mask
    else:
        return image


def image_logger(dict_of_images, path, n_labels=11, n_grid_images=8, log_separate=False, **kwargs):
    ind_vis = {}
    for k, v in dict_of_images.items():
        if isinstance(v, torch.Tensor): ind_vis[str(k)] = visualize(kwargs.get(k, lambda x: x)(v), n_labels, n_grid_images).squeeze().data.cpu().numpy()
        elif isinstance(v, str): ind_vis[str(k)] = v
    h = max([getattr(x, "shape", [0, 0, 0])[1] for x in ind_vis.values()])
    w = sum([getattr(x, "shape", [0, 0, 0])[2] for x in ind_vis.values()])
    if not log_separate:
        fig = plt.figure(figsize=(minmax(w // 1024, 15, 30), minmax(h // 1024, 15, 20)), dpi=300)
        for i, (k, v) in enumerate(ind_vis.items()):
            ax = fig.add_subplot(len(dict_of_images), 1, i + 1)
            ax.set_title(k)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.xaxis.set_tick_params(labelbottom=False)
            ax.yaxis.set_tick_params(labelleft=False)
            if isinstance(v, np.ndarray):
                ax.imshow(rearrange(v, "c h w -> h w c"))
            if isinstance(v, str):
                ax.imshow(np.zeros((5, 10)))
                ax.text(0, 0, "\n".join([v[i * 20: (i + 1) * 20] for i in range(np.ceil(len(v) / 20).astype(int))]),
                        color="white",
                        fontproperties=matplotlib.font_manager.FontProperties(size=5,
                                                                                fname='/ailab/user/dailinrui/data/dependency/Arial-Unicode-Bold.ttf'))
        
        plt.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0)
        image_from_plt = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plt = {"main": image_from_plt.reshape(fig.canvas.get_width_height()[::-1] + (3,))}
        plt.close(fig)
    else:
        image_from_plt = dict()
        assert callable(path)
        for i, (k, v) in enumerate(ind_vis.items()):
            fig = plt.figure(dpi=300)
            ax = fig.gca()
            ax.set_title(k)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.xaxis.set_tick_params(labelbottom=False)
            ax.yaxis.set_tick_params(labelleft=False)
            if isinstance(v, np.ndarray):
                ax.imshow(rearrange(v, "c h w -> h w c"))
            if isinstance(v, str):
                ax.imshow(np.zeros((5, 10)))
                ax.text(0, 0, "\n".join([v[i * 20: (i + 1) * 20] for i in range(np.ceil(len(v) / 20).astype(int))]),
                        color="white",
                        fontproperties=matplotlib.font_manager.FontProperties(size=5,
                                                                                fname='/ailab/user/dailinrui/data/dependency/Arial-Unicode-Bold.ttf'))
            plt.savefig(path(k), dpi=300, bbox_inches="tight", pad_inches=0)
            image_from_plt_step = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            image_from_plt[k] = image_from_plt_step.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)
    return image_from_plt


class TwoStreamBatchSampler(Sampler):
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size, **kwargs):
        self.batch_size = batch_size
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0,\
            f"condition {len(self.primary_indices)} >= {self.primary_batch_size} > 0 is not satisfied"
        if len(self.secondary_indices) < self.secondary_batch_size:
            self.secondary_indices = self.secondary_indices + self.primary_indices
            print("using coarse labels extracted from fine labels as supervision")
        # assert len(self.secondary_indices) >= self.secondary_batch_size >= 0,\
        #     f"condition {len(self.secondary_indices)} >= {self.secondary_batch_size} >= 0 is not satisfied"

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        if self.secondary_batch_size != 0:
            secondary_iter = iterate_eternally(self.secondary_indices)
            return (
                primary_batch + secondary_batch
                for (primary_batch, secondary_batch)
                in zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
            )
        else:
            return (primary_batch for primary_batch in grouper(primary_iter, self.primary_batch_size))
        
    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)
    