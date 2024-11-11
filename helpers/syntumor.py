### Tumor Generateion
import sys
sys.path.append("/ailab/user/dailinrui/code/latentdiffusion/")
import json
import random
import cv2, os
import elasticdeform
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm
from pathlib import Path
from scipy.ndimage import gaussian_filter
from ldm.data.utils import LabelParser, OrganTypeBase


def generate_prob_function(mask_shape):
    sigma = np.random.uniform(3,15)
    # uniform noise generate
    a = np.random.uniform(0, 1, size=(mask_shape[0],mask_shape[1],mask_shape[2]))

    # Gaussian filter
    # this taks some time
    a_2 = gaussian_filter(a, sigma=sigma)

    scale = np.random.uniform(0.19, 0.21)
    base = np.random.uniform(0.04, 0.06)
    a =  scale * (a_2 - np.min(a_2)) / (np.max(a_2) - np.min(a_2)) + base

    return a

# first generate 5*200*200*200

def get_texture(mask_shape):
    # get the prob function
    a = generate_prob_function(mask_shape) 

    # sample once
    random_sample = np.random.uniform(0, 1, size=(mask_shape[0],mask_shape[1],mask_shape[2]))

    # if a(x) > random_sample(x), set b(x) = 1
    b = (a > random_sample).astype(float)  # int type can't do Gaussian filter

    # Gaussian filter
    if np.random.uniform() < 0.7:
        sigma_b = np.random.uniform(3, 5)
    else:
        sigma_b = np.random.uniform(5, 8)

    # this takes some time
    b2 = gaussian_filter(b, sigma_b)

    # Scaling and clipping
    u_0 = np.random.uniform(0.5, 0.55)
    threshold_mask = b2 > 0.12    # this is for calculte the mean_0.2(b2)
    beta = u_0 / (np.sum(b2 * threshold_mask) / threshold_mask.sum())
    Bj = np.clip(beta*b2, 0, 1) # 目前是0-1区间
    
    return Bj


# here we want to get predefined texutre:
def get_predefined_texture(mask_shape, sigma_a, sigma_b):
    # uniform noise generate
    a = np.random.uniform(0, 1, size=(mask_shape[0],mask_shape[1],mask_shape[2]))
    a_2 = gaussian_filter(a, sigma=sigma_a)
    scale = np.random.uniform(0.19, 0.21)
    base = np.random.uniform(0.04, 0.06)
    a =  scale * (a_2 - np.min(a_2)) / (np.max(a_2) - np.min(a_2)) + base

    # sample once
    random_sample = np.random.uniform(0, 1, size=(mask_shape[0],mask_shape[1],mask_shape[2]))
    b = (a > random_sample).astype(float)  # int type can't do Gaussian filter
    b = gaussian_filter(b, sigma_b)

    # Scaling and clipping
    u_0 = np.random.uniform(0.5, 0.55)
    threshold_mask = b > 0.12    # this is for calculte the mean_0.2(b2)
    beta = u_0 / (np.sum(b * threshold_mask) / threshold_mask.sum())
    Bj = np.clip(beta*b, 0, 1) # 目前是0-1区间

    return Bj

# Step 1: Random select (numbers) location for tumor.
def random_select(mask_scan):
    # we first find z index and then sample point with z slice
    z_start, z_end = np.where(np.any(mask_scan, axis=(0, 1)))[0][[0, -1]]

    # we need to strict number z's position (0.3 - 0.7 in the middle of liver)
    # , which should not be necessary since we are no longer doing mask generation in liver only
    z = round(random.uniform(0.01, 0.99) * (z_end - z_start)) + z_start

    liver_mask = mask_scan[..., z]

    # erode the mask (we don't want the edge points)
    kernel = np.ones((5,5), dtype=np.uint8)
    liver_mask = cv2.erode(liver_mask, kernel, iterations=1)

    coordinates = np.argwhere(liver_mask == 1)
    if len(coordinates) == 0: 
        print(f"no valid points found on z={z}")
        return random_select(mask_scan)  # handle edge scenarios
    random_index = np.random.randint(0, len(coordinates))
    xyz = coordinates[random_index].tolist() # get x,y
    xyz.append(z)
    potential_points = xyz

    return potential_points

# Step 2 : generate the ellipsoid
def get_ellipsoid(x, y, z):
    """"
    x, y, z is the radius of this ellipsoid in x, y, z direction respectly.
    """
    sh = (4*x, 4*y, 4*z)
    out = np.zeros(sh, int)
    aux = np.zeros(sh)
    radii = np.array([x, y, z])
    com = np.array([2*x, 2*y, 2*z])  # center point

    # calculate the ellipsoid 
    bboxl = np.floor(com-radii).clip(0,None).astype(int)
    bboxh = (np.ceil(com+radii)+1).clip(None, sh).astype(int)
    roi = out[tuple(map(slice,bboxl,bboxh))]
    roiaux = aux[tuple(map(slice,bboxl,bboxh))]
    logrid = *map(np.square,np.ogrid[tuple(
            map(slice,(bboxl-com)/radii,(bboxh-com-1)/radii,1j*(bboxh-bboxl)))]),
    dst = (1-sum(logrid)).clip(0,None)
    mask = dst>roiaux
    roi[mask] = 1
    np.copyto(roiaux,dst,where=mask)

    return out

def get_fixed_geo(mask_scan, tumor_type):

    enlarge_x, enlarge_y, enlarge_z = 160, 160, 160
    geo_mask = np.zeros((mask_scan.shape[0] + enlarge_x, mask_scan.shape[1] + enlarge_y, mask_scan.shape[2] + enlarge_z), dtype=np.int8)
    # texture_map = np.zeros((mask_scan.shape[0] + enlarge_x, mask_scan.shape[1] + enlarge_y, mask_scan.shape[2] + enlarge_z), dtype=np.float16)
    tiny_radius, small_radius, medium_radius, large_radius = 4, 8, 16, 32

    if tumor_type == 'tiny':
        num_tumor = random.randint(3,10)
        for _ in range(num_tumor):
            # Tiny tumor
            x = random.randint(int(0.75*tiny_radius), int(1.25*tiny_radius))
            y = random.randint(int(0.75*tiny_radius), int(1.25*tiny_radius))
            z = random.randint(int(0.75*tiny_radius), int(1.25*tiny_radius))
            sigma = random.uniform(0.5,1)
            
            geo = get_ellipsoid(x, y, z)
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0,1))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(1,2))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0,2))
            point = random_select(mask_scan)
            new_point = [point[0] + enlarge_x//2, point[1] + enlarge_y//2, point[2] + enlarge_z//2]
            x_low, x_high = new_point[0] - geo.shape[0]//2, new_point[0] + geo.shape[0]//2 
            y_low, y_high = new_point[1] - geo.shape[1]//2, new_point[1] + geo.shape[1]//2 
            z_low, z_high = new_point[2] - geo.shape[2]//2, new_point[2] + geo.shape[2]//2 
            
            # paste small tumor geo into test sample
            geo_mask[x_low:x_high, y_low:y_high, z_low:z_high] += geo

    if tumor_type == 'small':
        num_tumor = random.randint(3,10)
        for _ in range(num_tumor):
            # Small tumor
            x = random.randint(int(0.75*small_radius), int(1.25*small_radius))
            y = random.randint(int(0.75*small_radius), int(1.25*small_radius))
            z = random.randint(int(0.75*small_radius), int(1.25*small_radius))
            sigma = random.randint(1, 2)
            
            geo = get_ellipsoid(x, y, z)
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0,1))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(1,2))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0,2))
            # texture = get_texture((4*x, 4*y, 4*z))
            point = random_select(mask_scan)
            new_point = [point[0] + enlarge_x//2, point[1] + enlarge_y//2, point[2] + enlarge_z//2]
            x_low, x_high = new_point[0] - geo.shape[0]//2, new_point[0] + geo.shape[0]//2 
            y_low, y_high = new_point[1] - geo.shape[1]//2, new_point[1] + geo.shape[1]//2 
            z_low, z_high = new_point[2] - geo.shape[2]//2, new_point[2] + geo.shape[2]//2 
            
            # paste small tumor geo into test sample
            geo_mask[x_low:x_high, y_low:y_high, z_low:z_high] += geo
            # texture_map[x_low:x_high, y_low:y_high, z_low:z_high] = texture

    if tumor_type == 'medium':
        num_tumor = random.randint(2, 5)
        for _ in range(num_tumor):
            # medium tumor
            x = random.randint(int(0.75*medium_radius), int(1.25*medium_radius))
            y = random.randint(int(0.75*medium_radius), int(1.25*medium_radius))
            z = random.randint(int(0.75*medium_radius), int(1.25*medium_radius))
            sigma = random.randint(3, 6)
            
            geo = get_ellipsoid(x, y, z)
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0,1))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(1,2))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0,2))
            # texture = get_texture((4*x, 4*y, 4*z))
            point = random_select(mask_scan)
            new_point = [point[0] + enlarge_x//2, point[1] + enlarge_y//2, point[2] + enlarge_z//2]
            x_low, x_high = new_point[0] - geo.shape[0]//2, new_point[0] + geo.shape[0]//2 
            y_low, y_high = new_point[1] - geo.shape[1]//2, new_point[1] + geo.shape[1]//2 
            z_low, z_high = new_point[2] - geo.shape[2]//2, new_point[2] + geo.shape[2]//2 
            
            # paste medium tumor geo into test sample
            geo_mask[x_low:x_high, y_low:y_high, z_low:z_high] += geo
            # texture_map[x_low:x_high, y_low:y_high, z_low:z_high] = texture

    if tumor_type == 'large':
        num_tumor = random.randint(1,3)
        for _ in range(num_tumor):
            # Large tumor
            x = random.randint(int(0.75*large_radius), int(1.25*large_radius))
            y = random.randint(int(0.75*large_radius), int(1.25*large_radius))
            z = random.randint(int(0.75*large_radius), int(1.25*large_radius))
            sigma = random.randint(5, 10)
            
            geo = get_ellipsoid(x, y, z)
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0,1))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(1,2))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0,2))
            # texture = get_texture((4*x, 4*y, 4*z))
            point = random_select(mask_scan)
            new_point = [point[0] + enlarge_x//2, point[1] + enlarge_y//2, point[2] + enlarge_z//2]
            x_low, x_high = new_point[0] - geo.shape[0]//2, new_point[0] + geo.shape[0]//2 
            y_low, y_high = new_point[1] - geo.shape[1]//2, new_point[1] + geo.shape[1]//2 
            z_low, z_high = new_point[2] - geo.shape[2]//2, new_point[2] + geo.shape[2]//2 
            
            # paste small tumor geo into test sample
            geo_mask[x_low:x_high, y_low:y_high, z_low:z_high] += geo
            # texture_map[x_low:x_high, y_low:y_high, z_low:z_high] = texture

    if tumor_type == "mix":
        # tiny
        num_tumor = random.randint(3,10)
        for _ in range(num_tumor):
            # Tiny tumor
            x = random.randint(int(0.75*tiny_radius), int(1.25*tiny_radius))
            y = random.randint(int(0.75*tiny_radius), int(1.25*tiny_radius))
            z = random.randint(int(0.75*tiny_radius), int(1.25*tiny_radius))
            sigma = random.uniform(0.5,1)
            
            geo = get_ellipsoid(x, y, z)
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0,1))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(1,2))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0,2))
            point = random_select(mask_scan)
            new_point = [point[0] + enlarge_x//2, point[1] + enlarge_y//2, point[2] + enlarge_z//2]
            x_low, x_high = new_point[0] - geo.shape[0]//2, new_point[0] + geo.shape[0]//2 
            y_low, y_high = new_point[1] - geo.shape[1]//2, new_point[1] + geo.shape[1]//2 
            z_low, z_high = new_point[2] - geo.shape[2]//2, new_point[2] + geo.shape[2]//2 
            
            # paste small tumor geo into test sample
            geo_mask[x_low:x_high, y_low:y_high, z_low:z_high] += geo

        # small
        num_tumor = random.randint(5,10)
        for _ in range(num_tumor):
            # Small tumor
            x = random.randint(int(0.75*small_radius), int(1.25*small_radius))
            y = random.randint(int(0.75*small_radius), int(1.25*small_radius))
            z = random.randint(int(0.75*small_radius), int(1.25*small_radius))
            sigma = random.randint(1, 2)
        
            geo = get_ellipsoid(x, y, z)
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0,1))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(1,2))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0,2))
            # texture = get_texture((4*x, 4*y, 4*z))
            point = random_select(mask_scan)
            new_point = [point[0] + enlarge_x//2, point[1] + enlarge_y//2, point[2] + enlarge_z//2]
            x_low, x_high = new_point[0] - geo.shape[0]//2, new_point[0] + geo.shape[0]//2 
            y_low, y_high = new_point[1] - geo.shape[1]//2, new_point[1] + geo.shape[1]//2 
            z_low, z_high = new_point[2] - geo.shape[2]//2, new_point[2] + geo.shape[2]//2 
            
            # paste small tumor geo into test sample
            geo_mask[x_low:x_high, y_low:y_high, z_low:z_high] += geo
            # texture_map[x_low:x_high, y_low:y_high, z_low:z_high] = texture
            
        # medium
        num_tumor = random.randint(2, 5)
        for _ in range(num_tumor):
            # medium tumor
            x = random.randint(int(0.75*medium_radius), int(1.25*medium_radius))
            y = random.randint(int(0.75*medium_radius), int(1.25*medium_radius))
            z = random.randint(int(0.75*medium_radius), int(1.25*medium_radius))
            sigma = random.randint(3, 6)
            
            geo = get_ellipsoid(x, y, z)
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0,1))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(1,2))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0,2))
            # texture = get_texture((4*x, 4*y, 4*z))
            point = random_select(mask_scan)
            new_point = [point[0] + enlarge_x//2, point[1] + enlarge_y//2, point[2] + enlarge_z//2]
            x_low, x_high = new_point[0] - geo.shape[0]//2, new_point[0] + geo.shape[0]//2 
            y_low, y_high = new_point[1] - geo.shape[1]//2, new_point[1] + geo.shape[1]//2 
            z_low, z_high = new_point[2] - geo.shape[2]//2, new_point[2] + geo.shape[2]//2 
            
            # paste medium tumor geo into test sample
            geo_mask[x_low:x_high, y_low:y_high, z_low:z_high] += geo
            # texture_map[x_low:x_high, y_low:y_high, z_low:z_high] = texture

        # large
        num_tumor = random.randint(1,3)
        for _ in range(num_tumor):
            # Large tumor
            x = random.randint(int(0.75*large_radius), int(1.25*large_radius))
            y = random.randint(int(0.75*large_radius), int(1.25*large_radius))
            z = random.randint(int(0.75*large_radius), int(1.25*large_radius))
            sigma = random.randint(5, 10)
            geo = get_ellipsoid(x, y, z)
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0,1))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(1,2))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0,2))
            # texture = get_texture((4*x, 4*y, 4*z))
            point = random_select(mask_scan)
            new_point = [point[0] + enlarge_x//2, point[1] + enlarge_y//2, point[2] + enlarge_z//2]
            x_low, x_high = new_point[0] - geo.shape[0]//2, new_point[0] + geo.shape[0]//2 
            y_low, y_high = new_point[1] - geo.shape[1]//2, new_point[1] + geo.shape[1]//2 
            z_low, z_high = new_point[2] - geo.shape[2]//2, new_point[2] + geo.shape[2]//2 
            
            # paste small tumor geo into test sample
            geo_mask[x_low:x_high, y_low:y_high, z_low:z_high] += geo
            # texture_map[x_low:x_high, y_low:y_high, z_low:z_high] = texture

    geo_mask = geo_mask[enlarge_x//2:-enlarge_x//2, enlarge_y//2:-enlarge_y//2, enlarge_z//2:-enlarge_z//2]
    # texture_map = texture_map[enlarge_x//2:-enlarge_x//2, enlarge_y//2:-enlarge_y//2, enlarge_z//2:-enlarge_z//2]
    geo_mask = (geo_mask * mask_scan) >=1
    
    return geo_mask


def get_tumor(mask_scan, tumor_type):
    geo_mask = get_fixed_geo(mask_scan, tumor_type)

    # sigma      = np.random.uniform(1, 2)
    # difference = np.random.uniform(65, 145)

    # # blur the boundary
    # geo_blur = gaussian_filter(geo_mask*1.0, sigma)
    # abnormally = (volume_scan - texture * geo_blur * difference) * mask_scan
    # # abnormally = (volume_scan - texture * geo_mask * difference) * mask_scan
    
    # abnormally_full = volume_scan * (1 - mask_scan) + abnormally
    abnormally_mask = mask_scan + geo_mask

    return abnormally_mask

def SynthesisTumor(mask_scan, tumor_type):
    # for speed_generate_tumor, we only send the liver part into the generate program
    x_start, x_end = np.where(np.any(mask_scan, axis=(1, 2)))[0][[0, -1]]
    y_start, y_end = np.where(np.any(mask_scan, axis=(0, 2)))[0][[0, -1]]
    z_start, z_end = np.where(np.any(mask_scan, axis=(0, 1)))[0][[0, -1]]

    # shrink the boundary
    x_start, x_end = max(0, x_start+1), min(mask_scan.shape[0], x_end-1)
    y_start, y_end = max(0, y_start+1), min(mask_scan.shape[1], y_end-1)
    z_start, z_end = max(0, z_start+1), min(mask_scan.shape[2], z_end-1)

    liver_mask   = mask_scan[x_start:x_end, y_start:y_end, z_start:z_end]
    liver_mask = get_tumor(liver_mask, tumor_type)
    mask_scan[x_start:x_end, y_start:y_end, z_start:z_end] = liver_mask
    return mask_scan


def syn(out="/ailab/user/dailinrui/data/datasets/compr_syntumor"):
    tumor_types = ["tiny", "small", "medium", "large", "mix"]
    for i in range(500):
        mask_scan = np.ones((50,)*3)
        tumor_scan = SynthesisTumor(mask_scan, tumor_types[random.randint(0, len(tumor_types)-1)])
        sitk.WriteImage(sitk.GetImageFromArray((tumor_scan == 2).astype(np.uint8)), os.path.join(out, f"case_{i:04d}.nii.gz"))
        
        
def lesion_syn(inputs="/ailab/user/dailinrui/code/latentdiffusion/helpers/template_organ/image",
               outputs="/ailab/user/dailinrui/data/datasets/compr_lesion/"):
    import torchio as tio
    from itertools import cycle
    from scipy.ndimage import label
    from ldm.data.utils import TorchioForegroundCropper
    def load(path):
        im = sitk.ReadImage(path)
        arr = sitk.GetArrayFromImage(im)
        return im, arr
    
    def add_version(olds, new, is_file=False, suffix='.nii.gz'):
        if is_file:
            # olds: os.listdir, new: os.path.abspath
            basename = os.path.basename(new)
            dirname = os.path.dirname(new)
            mtime = [(file, os.path.getmtime(os.path.join(dirname, file))) for file in olds if file.startswith(basename.split('.')[0])]
            if len(mtime) == 0: maxtime = 0
            else:
                max_time_file = max(mtime, key=lambda x: x[1])[0]
                if 'version' in max_time_file: 
                    maxtime = int(max_time_file.split(suffix)[0][max_time_file.find('version') + len("version"):])
                else: maxtime = 0
            new = os.path.join(dirname, basename.split(suffix)[0] + f"_version{maxtime + 1}" + suffix)
        else:
            mpath = [(x, int(x.split(suffix)[0][x.find('version') + len('version'):]) if 'version' in x else 0) for x in olds if x.startswith(new.split('.')[0])]
            max_version = max(mpath, key=lambda x: x[1])[1]
            new = new.split(suffix)[0] + f'_version{max_version + 1}' + suffix
        return new
    
    inputs = Path(inputs)
    outputs = Path(outputs)
    os.makedirs(outputs, exist_ok=1)
    parser = LabelParser("v1")
    cropper = TorchioForegroundCropper(crop_level='patch',
                                       crop_anchor='tumorseg',
                                       output_size=(280, 280, 280),
                                       foreground_prob=1.,
                                       pad_value=lambda x: x['image'].min())
    tumor_types = ["tiny", "small", "medium", "large", "mix"]
    ds_names = ["colon"]  # ["liver", "stomach", "kidney", "gallbladder", "esophagus", "pancreas", "urinarybladder"]
    for ds in inputs.glob("*.txt"):
        base = Path(outputs) / ds.name.split('.')[0]
        if base.name not in ds_names: continue
        os.makedirs(base / "image", exist_ok=1)
        os.makedirs(base / "totalseg", exist_ok=1)
        os.makedirs(base / "tumorseg", exist_ok=1)
        with open(ds) as f, open(ds.parent.parent / "totalseg" / ds.name) as g:
            scans = [_.strip() for _ in f.readlines()]
            scan_totalsegs = [_.strip() for _ in g.readlines()]
            
        obj = {}
        itr = 1024
        for scan, scan_totalseg in tqdm(cycle(zip(scans, scan_totalsegs)), total=1000, desc=f"generate {ds.name.split('.')[0]} mask"):
            path_to_image = Path(scan)
            path_to_totalseg = Path(scan_totalseg)
            if not path_to_totalseg.exists(): 
                print(f"path {path_to_totalseg} does not exist")
                continue
            
            anchor, image = load(path_to_image)
            anchor, totalseg = load(path_to_totalseg)
            partial_totalseg = parser.totalseg2mask(totalseg, [OrganTypeBase(name=ds.name.split('.')[0], label=1)])
            if partial_totalseg.sum() == 0:
                print(f"{path_to_totalseg} has no foreground")
                continue
            
            tumor_mask = SynthesisTumor(partial_totalseg, tumor_types[random.randint(0, len(tumor_types)-1)])
            # tumor_mask_tumor, _ = label(tumor_mask == 2)
            # tumor_mask_bincount = np.bincount(tumor_mask_tumor.flatten())[1:]
            subject = tio.Subject(
                image=tio.ScalarImage(tensor=image.astype(np.float32)[None]),
                totalseg=tio.ScalarImage(tensor=totalseg.astype(np.float32)[None]),
                tumorseg=tio.ScalarImage(tensor=tumor_mask.astype(np.float32)[None]),
                anchor=tio.ScalarImage(tensor=(tumor_mask == 2).astype(np.float32)[None])
            )
            subject = cropper(subject)
            # assert subject.image.data.shape == (1, 280, 280, 280) and subject.tumorseg.data.max() == 2, f"{subject.image.data.shape}, {subject.tumorseg.data.max()}"
            assert subject.image.data.shape == (1, 280, 280, 280), f"shape {subject.image.data.shape}"
            if subject.tumorseg.data.max() == 1: 
                print("not cropping tumor foreground")
                continue
            
            sample = {k: sitk.GetImageFromArray(getattr(subject, k).data[0]) for k in ['image', 'totalseg', 'tumorseg']}
            for k, v in sample.items():
                sample[k].SetOrigin(anchor.GetOrigin())
                sample[k].SetDirection(anchor.GetDirection())
                sample[k].SetSpacing(anchor.GetSpacing())

            for k, v in sample.items(): sitk.WriteImage(v, base / k / f"{itr:05d}.nii.gz")
            obj[str(path_to_image)] = {k: str(base / k / f"{itr:05d}.nii.gz") for k in sample.keys()} |\
                {"raw_image": str(path_to_image), "raw_totalseg": str(path_to_totalseg)}
            
            itr -= 1
            if itr < 0: break
            
        with open(inputs / f"syntumor_{base.name}.json", 'w') as f:
            json.dump(obj, f, indent=4)
        

if __name__ == '__main__':
    # syn()
    lesion_syn()