import cv2, os
import torch
import numpy as np
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager
from torchvision.utils import make_grid
from einops import rearrange, repeat
from scipy.ndimage import sobel, distance_transform_edt
from SimpleITK import GetArrayFromImage, ReadImage


def normalize(x):
    return (x - x.min()) / (x.max() - x.min())


def combine_mask_and_im_v2(x, 
                           overlay_coef=1, 
                           colors=None, n_mask=11, mask_normalized=False, 
                           num_images=2, layer=None, 
                           backend="cv2"):
    # x: b 2 h w (d)
    x = x.cpu().data.numpy()
    cmap = matplotlib.colormaps.get_cmap("tab20")
    colors = [cmap(i)[:-1] for i in np.arange(0.3, n_mask) / n_mask] if colors is None else colors
    image = np.expand_dims(x[:, 0], -1).repeat(3, -1)
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
    if layer is not None: colored_image = colored_image[:, layer-4:layer+5:5]
    else: 
        if h > num_images: colored_image = colored_image[:, ::h // num_images]
    colored_image = rearrange(colored_image, "b h w d c -> (b h) c w d")
    colored_image = make_grid(torch.tensor(colored_image), nrow=1, normalize=False, pad_value=255, padding=0)
    return colored_image


def window_norm(image, window_pos=60, window_width=360, out=(0, 1)):
    window_min = window_pos - window_width / 2
    image = (image - window_min) / window_width
    image = (out[1] - out[0]) * image + out[0]
    image = image.clamp(min=out[0], max=out[1])
    return image
        

def visualize(image: torch.Tensor, n_mask: int=20, num_images=8, is_mask=False, color=None, wp=0, wn=10, layer=None):
    is_mask = is_mask or image.dtype == torch.long
    if len(image.shape) == 5:
        image = image[:, 0] 
    if len(image.shape) == 4:
        if layer is None:
            b, h = image.shape[:2]
            if h > num_images: image = image[:, ::h // num_images]
        else:
            image = image[:, layer-3:layer+4:6]
        image = rearrange(image, "b h w d -> (b h) 1 w d")
    else: return image
    image = make_grid(image, nrow=1, normalize=False, pad_value=255, padding=0)

    if is_mask:
        cmap = matplotlib.colormaps.get_cmap("viridis")
        rgb = torch.tensor([(0, 0, 0)] + ([cmap(i)[:-1] for i in (n_mask - np.arange(0., n_mask)) / n_mask] if color is None else color) + [(1, 1, 1)], device=image.device)
        image = image.long().clip(0, n_mask + 1)
        colored_mask = rearrange(rgb[image.long()][0], "i j n -> n i j")
        return colored_mask
    else:
        image = window_norm(image, wp, wn)
        return image


def image_logger(ind_vis, path, **kwargs):
    fig = plt.figure(figsize=(8, 15), dpi=300)
    for i, (k, v) in enumerate(ind_vis.items()):
        ax = fig.add_subplot(1, len(ind_vis), i + 1)
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
    image_from_plt = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    image_from_plt = {"main": image_from_plt.reshape(fig.canvas.get_width_height()[::-1] + (4,))[..., :4]}
    plt.close(fig)
    return image_from_plt


def _visualize_image_generative(cond_path, dict_of_image_paths, wc=0, wr=2):
    if isinstance(cond_path, str):
        mask = GetArrayFromImage(ReadImage(cond_path))
        totalseg = mask[..., 0]
        tumorseg = mask[..., 1]
    else:
        totalseg = GetArrayFromImage(ReadImage(cond_path[0]))
        tumorseg = GetArrayFromImage(ReadImage(cond_path[1]))
    if tumorseg.max() > 1: tumorseg = (tumorseg == 2).astype(np.uint8)
    layer = np.argmax(np.sum(np.sum(tumorseg, axis=1), axis=1))
    print(f"visualizing {layer}")
    list_of_images = [visualize(torch.tensor(totalseg)[None, None], num_images=3, is_mask=1, n_mask=20,
                                color=[matplotlib.colormaps.get_cmap("tab20")(i / 20)[:-1] for i in range(20)]), 
                      visualize(torch.tensor(tumorseg)[None, None], num_images=3, is_mask=1, n_mask=2, color=[(1, 1, 1), (1, 1, 1)])]
    
    for k, v in dict_of_image_paths.items():
        if isinstance(v, dict):
            wc = v.get('wc', wc)
            wr = v.get('wr', wr)
            layer = v.get('layer', layer)
            v = v['image']
        if not os.path.exists(v):
            print(f"{v} not found")
            continue
        image = normalize(GetArrayFromImage(ReadImage(v)))
        if image.ndim == 4:
            image = image[..., 0] # medsyn
        list_of_images.append(visualize(torch.rot90(torch.tensor(image), dims=(1,2))[None, None], is_mask=False, wp=wc, wn=wr, num_images=3))
        list_of_images.append(torch.zeros((list_of_images[-1].shape[0], list_of_images[-1].shape[1], 10)))
        
    images = torch.cat(list_of_images, dim=2)
    return images
    
    
def visualize_image_generative(list_of_cond_path, list_of_dict_of_image_paths, list_of_wc=None, list_of_wr=None):
    if list_of_wc is None: list_of_wc = [0.5] * len(list_of_cond_path)
    if list_of_wr is None: list_of_wr = [1] * len(list_of_cond_path)
    list_of_images = []
    for cond_path, dict_of_image_paths, wc, wr in zip(list_of_cond_path, list_of_dict_of_image_paths, list_of_wc, list_of_wr):
        list_of_images.append(_visualize_image_generative(cond_path, dict_of_image_paths, wc, wr))
        list_of_images.append(torch.cat([torch.zeros((list_of_images[-1].shape[0], 10, list_of_images[-1].shape[2])), 
                                        #  torch.ones((list_of_images[-1].shape[0], 10, 10)),
                                        #  torch.zeros((list_of_images[-1].shape[0], 10, 128)),
                                        #  torch.ones((list_of_images[-1].shape[0], 10, 10)),
                                        #  torch.zeros((list_of_images[-1].shape[0], 10, 128)),
                                        #  torch.ones((list_of_images[-1].shape[0], 10, 10)),
                                        #  torch.zeros((list_of_images[-1].shape[0], 10, 128)),
                                        #  torch.ones((list_of_images[-1].shape[0], 10, 10)),
                                        #  torch.zeros((list_of_images[-1].shape[0], 10, 128)),
                                        #  torch.ones((list_of_images[-1].shape[0], 10, 10)),
                                        #  torch.zeros((list_of_images[-1].shape[0], 10, 128)),
                                        #  torch.ones((list_of_images[-1].shape[0], 10, 10)),
                                        #  torch.zeros((list_of_images[-1].shape[0], 10, 128)),
                                        #  torch.ones((list_of_images[-1].shape[0], 10, 10)),
                                         ], dim=-1))
    
    image_logger({"image": torch.cat(list_of_images, dim=1).cpu().numpy()}, "./helpers/real.png")
    
    
def _visualize_image_segmentation(dict_of_seg_paths, image_path, wc=0.5, wr=1, no_flip=False):
    image = normalize(GetArrayFromImage(ReadImage(image_path)))
    if not no_flip: image = np.flip(image, axis=1).copy()
    gt = GetArrayFromImage(ReadImage(dict_of_seg_paths['gt']))
    layer = np.argmax(np.sum(np.sum((gt > 0), axis=1), axis=1))
    print(f"visualizing {layer}")
    list_of_images = []
    
    for k, v in dict_of_seg_paths.items():
        if not os.path.exists(v):
            print(f"{v} not found")
            continue
        seg = GetArrayFromImage(ReadImage(v))
        if not no_flip: seg = np.flip(seg, axis=1).copy()
        if image.ndim == 4:
            image = image[..., 0] # medsyn
        list_of_images.append(combine_mask_and_im_v2(
            torch.cat([window_norm(torch.tensor(image), wc, wr)[None, None], torch.tensor(seg)[None, None]], dim=1),
                overlay_coef=1, n_mask=(2 if gt.max() < 10 else 20), mask_normalized=False, layer=layer, colors=[(1, 0, 0), [(1, 0, 0)]]if gt.max() < 10 else None))
        list_of_images.append(torch.zeros((list_of_images[-1].shape[0], list_of_images[-1].shape[1], 10)))
        
    images = torch.cat(list_of_images, dim=2)
    return images
    
    
def visualize_image_segmentation(list_of_dict_of_seg_paths, list_of_image_paths, list_of_wc=None, list_of_wr=None):
    if list_of_wc is None: list_of_wc = [0.5] * len(list_of_dict_of_seg_paths)
    if list_of_wr is None: list_of_wr = [1] * len(list_of_dict_of_seg_paths)
    list_of_images = []
    i = 0
    for dict_of_seg_paths, image_path, wc, wr in zip(list_of_dict_of_seg_paths, list_of_image_paths, list_of_wc, list_of_wr):
        i += 1
        list_of_images.append(_visualize_image_segmentation(dict_of_seg_paths, image_path, wc, wr, no_flip=False if i != len(list_of_wr) else True))
        list_of_images.append(torch.cat([torch.zeros((list_of_images[-1].shape[0], 10, list_of_images[-1].shape[2]))], dim=-1))
    
    image_logger({"image": torch.cat(list_of_images, dim=1).cpu().numpy()}, "./helpers/segmentation.png")
    

if __name__ == '__main__':
    visualize_image_generative(
        [
            "/ailab/user/dailinrui/data/vis/guidegen_ldm_128_128_128/cond_value/TCGA-LUAD_00004_0004.nii.gz",
            # "/ailab/user/dailinrui/data/vis/guidegen_ldm_128_128_128/cond_value/CPTAC-LSCC_00002_0002.nii.gz",
            # "/ailab/user/dailinrui/data/vis/guidegen_ldm_128_128_128/cond_value/RJ202302171638326818.nii.gz",
            "/ailab/user/dailinrui/data/vis/guidegen_ldm_128_128_128/cond_value/CPTAC-CCRCC_00017_0000.nii.gz",
            "/ailab/user/dailinrui/data/vis/guidegen_ldm_128_128_128/cond_value/TCGA-KIRC_00119_0000.nii.gz",
        ],
        # [
        #     "/ailab/user/dailinrui/data/ldm/guidegen_ldm_128_128_128/dataset_noddim/cond_value/CPTAC-LUAD_00003_0000.nii.gz",
        #     "/ailab/user/dailinrui/data/ldm/guidegen_ldm_128_128_128/dataset_noddim/cond_value/RJ202302171638327237.nii.gz",
        #     "/ailab/user/dailinrui/data/ldm/guidegen_ldm_128_128_128/dataset_noddim/cond_value/CPTAC-CCRCC_00001_0004.nii.gz",
        # ],
        [
            {
                "guidegen": "/ailab/user/dailinrui/data/vis/guidegen_ldm_128_128_128/samples_value/TCGA-LUAD_00004_0004.nii.gz",
                "real": "/ailab/user/dailinrui/code/latentdiffusion/data_ln/vis/guidegen_ldm_128_128_128/inputs_value/TCGA-LUAD_00004_0004.nii.gz"
                # "pinaya's": {"image": "/ailab/user/dailinrui/data/ldm/medsyn_ddpm_128_128_128/dataset_noddim/samples_value/CPTAC-CCRCC_00001_0004.nii.gz", 'layer': 89},
                # "generateCT": {"image": "/ailab/user/dailinrui/data/vis/monai_kl_ldm_128_128_128/samples_value/RJ202302171638325045.nii.gz", 'layer': 89},
                # "monaildm": {"image": "/ailab/user/dailinrui/code/latentdiffusion/data_ln/vis/vis/samples_value/TCGA-LUAD_00004_0004.nii.gz", "wc": 0.6, "wr": 0.8},
                # "zhuang's": "/ailab/user/dailinrui/data/vis/contourddpm/image/TCGA-LUAD_00004_0004.nii.gz",
                # "medicalddpm": "/ailab/user/dailinrui/data/vis/medddpm/image/TCGA-LUAD_00004_0004.nii.gz",
                # "medsyn": "/ailab/user/dailinrui/data/vis/monai_kl_ldm_128_128_128/samples_value/TCGA-LUAD_00004_0004.nii.gz",
            },
            {
                "guidegen": "/ailab/user/dailinrui/data/vis/guidegen_ldm_128_128_128/samples_value/CPTAC-CCRCC_00017_0000.nii.gz",
                "real": "/ailab/user/dailinrui/code/latentdiffusion/data_ln/vis/guidegen_ldm_128_128_128/inputs_value/CPTAC-CCRCC_00017_0000.nii.gz"
                # "pinaya's": "/ailab/user/dailinrui/data/ldm/medsyn_ddpm_128_128_128/dataset_noddim/samples_value/RJ202302171638327237.nii.gz",
                # "generateCT": {"image": "/ailab/user/dailinrui/data/vis/monai_kl_ldm_128_128_128/samples_value/CPTAC-CCRCC_00068_0003.nii.gz", "layer": 43, "wc": 0.5, "wr": 0.5},
                # "monaildm": "/ailab/user/dailinrui/data/vis/monai_kl_ldm_128_128_128/samples_value/CPTAC-CCRCC_00017_0000.nii.gz",
                # "zhuang's": "/ailab/user/dailinrui/data/vis/contourddpm/image/CPTAC-CCRCC_00017_0000.nii.gz",
                # "medicalddpm": "/ailab/user/dailinrui/data/vis/medddpm/image/CPTAC-CCRCC_00017_0000.nii.gz",
                # "medsyn": "/ailab/user/dailinrui/data/vis/medddpm/image/RJ202302171638327237_new.nii.gz",
            },
            {
                "guidegen": "/ailab/user/dailinrui/data/vis/guidegen_ldm_128_128_128/samples_value/TCGA-KIRC_00119_0000.nii.gz",
                "real": "/ailab/user/dailinrui/code/latentdiffusion/data_ln/vis/guidegen_ldm_128_128_128/inputs_value/TCGA-KIRC_00119_0000.nii.gz"
                # "pinaya's": "/ailab/user/dailinrui/code/latentdiffusion/data_ln/ldm/monai_kl_ldm_128_128_128/dataset/samples_value/liver_36.nii.gz",
                # "generateCT": "/ailab/user/dailinrui/code/latentdiffusion/data_ln/ldm/monai_kl_ldm_128_128_128/dataset/samples_value/CPTAC-LUAD_00013_0001.nii.gz",
                # "monaildm": "/ailab/user/dailinrui/data/ldm/guidegen_ldm_128_128_128/dataset/samples_value/TCGA-KIRC_00119_0000.nii.gz",
                # "zhuang's": "/ailab/user/dailinrui/data/vis/contourddpm/image/TCGA-KIRC_00119_0000.nii.gz",
                # "medicalddpm": "/ailab/user/dailinrui/data/vis/medddpm/image/TCGA-KIRC_00119_0000.nii.gz",
                # "medsyn": "/ailab/user/dailinrui/data/vis/medddpm/image/CPTAC-CCRCC_00001_0004_new.nii.gz",
            },
        ],
        list_of_wc=[0.2, 0.5, 0.6,],
        list_of_wr=[0.6, 0.3, 0.3,],
    )
    
    # import json
    # base = "/ailab/user/dailinrui/code/latentdiffusion/data_ln/nnUNetv2/nnUNet_raw/Dataset301_BTCV_ABD/predsTs"
    # with open(os.path.join(base, '10S', "metrics.json")) as f:
    #     print(json.load(f).keys())
    #     json_max = np.array(json.load(f)["mean_per_person"]['dice'])
    #     json_d = {}
            
    # for setting in ["ContourDDPM", "MedDDPM", "MONAI"]:
    #     with open(os.path.join(base, setting, "metrics.json")) as f:
    #         json_d[setting] = json.load(f)["mean_per_person"]["dice"]
    
    # json_d = np.array([np.max([json_d[item][_] for item in ["ContourDDPM", "MedDDPM", "MONAI"]]) for _ in range(len(list(json_d["MONAI"])))])
    # json_abs = (json_max - json_d).argmax()
    # print(json_abs)
    # exit()
            
    # visualize_image_segmentation(
    #     [
    #         {
    #             "gt": "/ailab/user/dailinrui/code/latentdiffusion/data_ln/nnUNetv2/nnUNet_raw/Dataset301_BTCV_ABD/labelsTr/Dataset301_BTCV_ABD_00003.nii.gz",
    #             "guidegen": "/ailab/user/dailinrui/code/latentdiffusion/data_ln/nnUNetv2/nnUNet_raw/Dataset301_BTCV_ABD/predsTs/10S/Dataset301_BTCV_ABD_00003.nii.gz",
    #             "contourddpm": "/ailab/user/dailinrui/code/latentdiffusion/data_ln/nnUNetv2/nnUNet_raw/Dataset301_BTCV_ABD/predsTs/ContourDDPM/Dataset301_BTCV_ABD_00003.nii.gz",
    #             "medddpm": "/ailab/user/dailinrui/code/latentdiffusion/data_ln/nnUNetv2/nnUNet_raw/Dataset301_BTCV_ABD/predsTs/MedDDPM/Dataset301_BTCV_ABD_00003.nii.gz",
    #             "monai": "/ailab/user/dailinrui/code/latentdiffusion/data_ln/nnUNetv2/nnUNet_raw/Dataset301_BTCV_ABD/predsTs/MONAI/Dataset301_BTCV_ABD_00003.nii.gz",
    #             "medsyn": "/ailab/user/dailinrui/code/latentdiffusion/data_ln/nnUNetv2/nnUNet_raw/Dataset301_BTCV_ABD/predsTs/1S/Dataset301_BTCV_ABD_00003.nii.gz"
    #         },
    #         {
    #             "gt": "/ailab/user/dailinrui/code/latentdiffusion/data_ln/nnUNetv2/nnUNet_raw/Dataset303_AMOS/labelsTr/Dataset303_AMOS_00055.nii.gz",
    #             "guidegen": "/ailab/user/dailinrui/code/latentdiffusion/data_ln/nnUNetv2/nnUNet_raw/Dataset303_AMOS/predsTs/10S/Dataset303_AMOS_00055.nii.gz",
    #             "contourddpm": "/ailab/user/dailinrui/code/latentdiffusion/data_ln/nnUNetv2/nnUNet_raw/Dataset303_AMOS/predsTs/ContourDDPM/Dataset303_AMOS_00055.nii.gz",
    #             "medddpm": "/ailab/user/dailinrui/code/latentdiffusion/data_ln/nnUNetv2/nnUNet_raw/Dataset303_AMOS/predsTs/MedDDPM/Dataset303_AMOS_00055.nii.gz",
    #             "monai": "/ailab/user/dailinrui/code/latentdiffusion/data_ln/nnUNetv2/nnUNet_raw/Dataset303_AMOS/predsTs/MONAI/Dataset303_AMOS_00055.nii.gz",
    #             "medsyn": "/ailab/user/dailinrui/code/latentdiffusion/data_ln/nnUNetv2/nnUNet_raw/Dataset303_AMOS/predsTs/1S/Dataset303_AMOS_00055.nii.gz"
    #         },
    #         {
    #             "gt": "/ailab/user/dailinrui/code/latentdiffusion/data_ln/nnUNetv2/nnUNet_raw/Dataset305_MSD_LUNG/labelsTr/Dataset305_MSD_LUNG_00004.nii.gz",
    #             "guidegen": "/ailab/user/dailinrui/code/latentdiffusion/data_ln/nnUNetv2/nnUNet_raw/Dataset305_MSD_LUNG/predsTs/10S/Dataset305_MSD_LUNG_00004.nii.gz",
    #             "contourddpm": "/ailab/user/dailinrui/code/latentdiffusion/data_ln/nnUNetv2/nnUNet_raw/Dataset305_MSD_LUNG/predsTs/ContourDDPM/Dataset305_MSD_LUNG_00004.nii.gz",
    #             "medddpm": "/ailab/user/dailinrui/code/latentdiffusion/data_ln/nnUNetv2/nnUNet_raw/Dataset305_MSD_LUNG/predsTs/MedDDPM/Dataset305_MSD_LUNG_00004.nii.gz",
    #             "monai": "/ailab/user/dailinrui/code/latentdiffusion/data_ln/nnUNetv2/nnUNet_raw/Dataset305_MSD_LUNG/predsTs/MONAI/Dataset305_MSD_LUNG_00004.nii.gz",
    #             "medsyn": "/ailab/user/dailinrui/code/latentdiffusion/data_ln/nnUNetv2/nnUNet_raw/Dataset305_MSD_LUNG/predsTs/MONAI/Dataset305_MSD_LUNG_00004.nii.gz"
    #         },
    #         {
    #             "gt": "/ailab/user/dailinrui/code/latentdiffusion/data_ln/nnUNetv2/nnUNet_raw/Dataset304_MSD_COLON/labelsTr/Dataset304_MSD_COLON_00095.nii.gz",
    #             "guidegen": "/ailab/user/dailinrui/code/latentdiffusion/data_ln/nnUNetv2/nnUNet_raw/Dataset304_MSD_COLON/predsTs/10S/Dataset304_MSD_COLON_00095.nii.gz",
    #             "contourddpm": "/ailab/user/dailinrui/code/latentdiffusion/data_ln/nnUNetv2/nnUNet_raw/Dataset304_MSD_COLON/predsTs/ContourDDPM/Dataset304_MSD_COLON_00095.nii.gz",
    #             "medddpm": "/ailab/user/dailinrui/code/latentdiffusion/data_ln/nnUNetv2/nnUNet_raw/Dataset304_MSD_COLON/predsTs/MedDDPM/Dataset304_MSD_COLON_00095.nii.gz",
    #             "monai": "/ailab/user/dailinrui/code/latentdiffusion/data_ln/nnUNetv2/nnUNet_raw/Dataset304_MSD_COLON/predsTs/MONAI/Dataset304_MSD_COLON_00095.nii.gz",
    #             "medsyn": "/ailab/user/dailinrui/code/latentdiffusion/data_ln/nnUNetv2/nnUNet_raw/Dataset304_MSD_COLON/predsTs/MONAI/Dataset304_MSD_COLON_00095.nii.gz"
    #         },
    #         {
    #             "gt": "/ailab/user/dailinrui/code/latentdiffusion/data_ln/nnUNetv2/nnUNet_raw/Dataset308_KITS/labelsTr/Dataset308_KITS_00011.nii.gz",
    #             "guidegen": "/ailab/user/dailinrui/code/latentdiffusion/data_ln/nnUNetv2/nnUNet_raw/Dataset308_KITS/predsTs/10S/Dataset308_KITS_00011.nii.gz",
    #             "contourddpm": "/ailab/user/dailinrui/code/latentdiffusion/data_ln/nnUNetv2/nnUNet_raw/Dataset308_KITS/predsTs/ContourDDPM/Dataset308_KITS_00011.nii.gz",
    #             "medddpm": "/ailab/user/dailinrui/code/latentdiffusion/data_ln/nnUNetv2/nnUNet_raw/Dataset308_KITS/predsTs/MedDDPM/Dataset308_KITS_00011.nii.gz",
    #             "monai": "/ailab/user/dailinrui/code/latentdiffusion/data_ln/nnUNetv2/nnUNet_raw/Dataset308_KITS/predsTs/MONAI/Dataset308_KITS_00011.nii.gz",
    #             "medsyn": "/ailab/user/dailinrui/code/latentdiffusion/data_ln/nnUNetv2/nnUNet_raw/Dataset308_KITS/predsTs/1S/Dataset308_KITS_00011.nii.gz"
    #         },
    #     ],
    #     [
    #         "/ailab/user/dailinrui/code/latentdiffusion/data_ln/nnUNetv2/nnUNet_raw/Dataset301_BTCV_ABD/imagesTr/Dataset301_BTCV_ABD_00003_0000.nii.gz",
    #         "/ailab/user/dailinrui/code/latentdiffusion/data_ln/nnUNetv2/nnUNet_raw/Dataset303_AMOS/imagesTr/Dataset303_AMOS_00055_0000.nii.gz",
    #         "/ailab/user/dailinrui/code/latentdiffusion/data_ln/nnUNetv2/nnUNet_raw/Dataset305_MSD_LUNG/imagesTr/Dataset305_MSD_LUNG_00004_0000.nii.gz",
    #         "/ailab/user/dailinrui/code/latentdiffusion/data_ln/nnUNetv2/nnUNet_raw/Dataset304_MSD_COLON/imagesTr/Dataset304_MSD_COLON_00095_0000.nii.gz",
    #         "/ailab/user/dailinrui/code/latentdiffusion/data_ln/nnUNetv2/nnUNet_raw/Dataset308_KITS/imagesTr/Dataset308_KITS_00011_0000.nii.gz",
    #     ],
    #     list_of_wc=[0.3, 0.6, 0.15, 0.5, 0.3],
    #     list_of_wr=[0.3, 0.3, 0.3, 0.3, 0.3]
    # )
    