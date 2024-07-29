import cv2
import torch
import numpy as np
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager
from matplotlib.cm import get_cmap
from torchvision.utils import make_grid
from einops import rearrange
from scipy.ndimage import sobel, distance_transform_edt
from SimpleITK import GetArrayFromImage, ReadImage


def combine_mask_and_im_v2(x, 
                           overlay_coef=.8, 
                           colors=None, n_mask=11, mask_normalized=False, 
                           num_images=8, 
                           backend="cv2"):
    # x: b 2 h w (d)
    x = x.cpu().data.numpy()
    cmap = get_cmap("viridis")
    colors = [cmap(i)[:-1] for i in np.arange(0.3, n_mask) / n_mask] if colors is None else colors
    image = np.expand_dims(x[:, 0], -1).repeat(3, -1)
    image = (image - image.min()) / (image.max() - image.min())
    mask = (x[:, 1] * (n_mask if mask_normalized else 1)).astype(np.uint8)
    contours = np.zeros(mask.shape + (3,))
    if backend == "cv2":
        h = mask.shape[1]
        if x.ndim == 5: mask = rearrange(mask, "b h w d -> (b h) w d")
        for ib, b in enumerate(mask):
            for i in np.unique(b).flatten():
                if i != 0:
                    binary = (b == i).astype(np.uint8)
                    contour, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(contours[ib // h, ib % h] if x.ndim == 5 else contours[ib], contour, -1, colors[i-1], 1)
                    cv2.destroyAllWindows()
    elif backend == "edt":
        for ib, b in enumerate(mask):
            for i in np.unique(b).flatten():
                if i != 0:
                    outline = 5
                    binary = (b == i).astype(np.uint8)
                    x1, x2 = np.where(np.any(binary, (1, 2)))[0][[0, -1]]
                    y1, y2 = np.where(np.any(binary, (0, 2)))[0][[0, -1]]
                    z1, z2 = np.where(np.any(binary, (0, 1)))[0][[0, -1]]
                    box = [slice(max(0, x1 - outline), min(binary.shape[0] - 1, x2 + outline)),
                            slice(max(0, y1 - outline), min(binary.shape[1] - 1, y2 + outline)),
                            slice(max(0, z1 - outline), min(binary.shape[2] - 1, z2 + outline))]
                    box_binary = binary[*box]
                    contour = distance_transform_edt(box_binary == 0, )
                    contour = (contour > 0) & (contour < 1.8)
                    contours[ib, *box][contour] = np.expand_dims(np.array(colors[i-1]), 0).repeat(contour.sum(), 0)
    contours[contours[..., 0] == 0, :] = image[contours[..., 0] == 0]
    colored_image = image * (1 - overlay_coef) + contours * overlay_coef
    
    b, h = colored_image.shape[:2]
    if h > num_images: colored_image = colored_image[:, ::h // num_images]
    colored_image = rearrange(colored_image, "b h w d c -> (b h) c w d")
    colored_image = make_grid(torch.tensor(colored_image), nrow=1, normalize=False, pad_value=255, padding=3)
    return colored_image
        

def visualize(image: torch.Tensor, n_mask: int=20, num_images=8, is_mask=False):
    is_mask = is_mask or image.dtype == torch.long
    if len(image.shape) == 5:
        image = image[:, 0] 
    if len(image.shape) == 4:
        b, h = image.shape[:2]
        if h > num_images: image = image[:, ::h // num_images]
        image = rearrange(image, "b h w d -> (b h) 1 w d")
    else: return image
    image = make_grid(image, nrow=1, normalize=not is_mask, pad_value=255, padding=3)

    if is_mask:
        cmap = get_cmap("viridis")
        rgb = torch.tensor([(0, 0, 0)] + [cmap(i)[:-1] for i in (n_mask - np.arange(0., n_mask)) / n_mask] + [(255, 255, 255)], device=image.device)
        image = image.long().clip(0, n_mask + 1)
        colored_mask = rearrange(rgb[image.long()][0], "i j n -> n i j")
        return colored_mask
    else:
        image = (image - image.min()) / (image.max() - image.min())
        return image[0]


def image_logger(ind_vis, path, **kwargs):
    fig = plt.figure(figsize=(8, 15), dpi=300)
    for i, (k, v) in enumerate(ind_vis.items()):
        ax = fig.add_subplot(1, len(ind_vis), i + 1)
        ax.set_title(k)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.xaxis.set_tick_params(labelbottom=False)
        ax.yaxis.set_tick_params(labelleft=False)
        if isinstance(v, np.ndarray):
            ax.imshow(rearrange(v, "c h w -> h w c"))
        if isinstance(v, str):
            linewidth = 100
            ax.set_facecolor("black")
            ax.imshow(np.zeros((5, 20)))
            ax.text(.2, 2.5, "\n".join([v[i * linewidth: (i + 1) * linewidth] for i in range(np.ceil(len(v) / linewidth).astype(int))]),
                    color="white",
                    fontproperties=matplotlib.font_manager.FontProperties(size=7,
                                                                            fname='/ailab/user/dailinrui/data/dependency/Arial-Unicode-Bold.ttf'))
    
    plt.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0)
    image_from_plt = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plt = {"main": image_from_plt.reshape(fig.canvas.get_width_height()[::-1] + (3,))}
    plt.close(fig)
    return image_from_plt


def visualize_image_generative(cond_path, dict_of_image_paths):
    mask = GetArrayFromImage(ReadImage(cond_path))
    totalseg = mask[..., 0]
    tumorseg = mask[..., 1]
    dict_of_images = [visualize(torch.tensor(totalseg)[None, None], num_images=3, is_mask=1, n_mask=20), visualize(torch.tensor(tumorseg)[None, None], num_images=3, is_mask=1, n_mask=2)]
    for k, v in dict_of_image_paths.items():
        image = GetArrayFromImage(ReadImage(v))
        dict_of_images.append(combine_mask_and_im_v2(
            torch.cat([torch.tensor(image)[None, None], torch.tensor(totalseg)[None, None]], dim=1),
            overlay_coef=.5, n_mask=20, mask_normalized=False, num_images=3))
        
    images = torch.cat(dict_of_images, dim=2).cpu().numpy()
    image_logger({"image": images}, "./helpers/result.pdf")
    
    

if __name__ == '__main__':
    visualize_image_generative("/ailab/user/dailinrui/data/ldm/guidegen_ldm_128_128_128/dataset/cond_value/CMB-LCA_00012_0001.nii.gz",
                               {"guidegen": "/ailab/user/dailinrui/data/ldm/guidegen_ldm_128_128_128/dataset/samples_value/CMB-LCA_00012_0001.nii.gz",
                                "medsyn": "/ailab/user/dailinrui/data/ldm/guidegen_ldm_128_128_128/dataset/samples_value/CMB-LCA_00012_0001.nii.gz",
                                "medicalddpm": "/ailab/user/dailinrui/data/ldm/guidegen_ldm_128_128_128/dataset/samples_value/CMB-LCA_00012_0001.nii.gz",
                                "zhuang's": "/ailab/user/dailinrui/data/ldm/guidegen_ldm_128_128_128/dataset/samples_value/CMB-LCA_00012_0001.nii.gz",
                                "monaildm": "/ailab/user/dailinrui/data/ldm/guidegen_ldm_128_128_128/dataset/samples_value/CMB-LCA_00012_0001.nii.gz",
                                "generateCT": "/ailab/user/dailinrui/data/ldm/guidegen_ldm_128_128_128/dataset/samples_value/CMB-LCA_00012_0001.nii.gz",
                                "medicaldiffusion": "/ailab/user/dailinrui/data/ldm/guidegen_ldm_128_128_128/dataset/samples_value/CMB-LCA_00012_0001.nii.gz",})