import numpy as np
import torch, math, os
from tqdm import tqdm
import torch.nn.functional as F
# https://github.com/JunyaoHu/common_metrics_on_video_quality/blob/main/fvd/styleganv/fvd.py


def load_i3d_pretrained(device=torch.device('cpu')):
    i3D_WEIGHTS_URL = "https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt"
    # filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'i3d_torchscript.pt')
    filepath = "/ailab/user/dailinrui/code/latentdiffusion/dependency/i3d_torchscript.pt"
    # print(filepath)
    if not os.path.exists(filepath):
        print(f"preparing for download {i3D_WEIGHTS_URL}, you can download it by yourself.")
        os.system(f"wget {i3D_WEIGHTS_URL} -O {filepath}")
    i3d = torch.jit.load(filepath).eval().to(device)
    i3d = torch.nn.DataParallel(i3d)
    return i3d
    

def get_feats(videos, detector, device, bs=10):
    # videos : torch.tensor BCTHW [0, 1]
    detector_kwargs = dict(rescale=False, resize=False, return_features=True) # Return raw features before the softmax layer.
    feats = np.empty((0, 400))
    with torch.no_grad():
        for i in range((len(videos)-1)//bs + 1):
            feats = np.vstack([feats, detector(torch.stack([preprocess_single(video) for video in videos[i*bs:(i+1)*bs]]).to(device), **detector_kwargs).detach().cpu().numpy()])
    return feats


def get_fvd_feats(videos, i3d, device, bs=10):
    # videos in [0, 1] as torch tensor BCTHW
    # videos = [preprocess_single(video) for video in videos]
    embeddings = get_feats(videos, i3d, device, bs)
    return embeddings


def preprocess_single(video, resolution=224, sequence_length=None):
    # video: CTHW, [0, 1]
    c, t, h, w = video.shape

    # temporal crop
    if sequence_length is not None:
        assert sequence_length <= t
        video = video[:, :sequence_length]

    # scale shorter side to resolution
    scale = resolution / min(h, w)
    if h < w:
        target_size = (resolution, math.ceil(w * scale))
    else:
        target_size = (math.ceil(h * scale), resolution)
    video = F.interpolate(video, size=target_size, mode='bilinear', align_corners=False)

    # center crop
    c, t, h, w = video.shape
    w_start = (w - resolution) // 2
    h_start = (h - resolution) // 2
    video = video[:, :, h_start:h_start + resolution, w_start:w_start + resolution]

    # [0, 1] -> [-1, 1]
    video = (video - 0.5) * 2

    return video.contiguous()


"""
Copy-pasted from https://github.com/cvpr2022-stylegan-v/stylegan-v/blob/main/src/metrics/frechet_video_distance.py
"""
from typing import Tuple
from scipy.linalg import sqrtm
import numpy as np


def compute_stats(feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = feats.mean(axis=0) # [d]
    sigma = np.cov(feats, rowvar=False) # [d, d]
    return mu, sigma


def frechet_distance(feats_fake: np.ndarray, feats_real: np.ndarray) -> float:
    mu_gen, sigma_gen = compute_stats(feats_fake)
    mu_real, sigma_real = compute_stats(feats_real)
    m = np.square(mu_gen - mu_real).sum()
    if feats_fake.shape[0]>1:
        s, _ = sqrtm(np.dot(sigma_gen, sigma_real), disp=False) # pylint: disable=no-member
        fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
    else:
        fid = np.real(m)
    return float(fid)


def trans(x):
    # if greyscale images add channel
    if x.shape[-3] == 1:
        x = x.repeat(1, 1, 3, 1, 1)

    # permute BTCHW -> BCTHW
    x = x.permute(0, 2, 1, 3, 4) 

    return x

class FrechetVideoDistance(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        self.metric = 0.
        self.total = 0
        self.videos1 = None
        self.videos2 = None
        
    def update(self, videos1, videos2):
        assert videos1.shape == videos2.shape
        self.total += 1
        self.videos1 = videos1
        self.videos2 = videos2
        
    def reset(self):
        self.total = 0
        self.metric = 0.

    def compute(self):
        videos1 = self.videos1
        videos2 = self.videos2
        device = videos1.device

        # videos [batch_size, timestamps, channel, h, w]
        
        assert videos1.shape == videos2.shape

        i3d = load_i3d_pretrained(device=device)
        fvd_results = {}

        # for calculate FVD, each clip_timestamp must >= 10
        for clip_timestamp in range(10, videos1.shape[-3]+1):
        
            # get a video clip
            # videos_clip [batch_size, channel, timestamps[:clip], h, w]
            videos_clip1 = videos1[:, :, : clip_timestamp]
            videos_clip2 = videos2[:, :, : clip_timestamp]

            # get FVD features
            feats1 = get_fvd_feats(videos_clip1, i3d=i3d, device=device)
            feats2 = get_fvd_feats(videos_clip2, i3d=i3d, device=device)
        
            # calculate FVD when timestamps[:clip]
            fvd_results[clip_timestamp] = frechet_distance(feats1, feats2)

        result = {
            "value": fvd_results,
            "video_setting": videos1.shape,
            "video_setting_name": "batch_size, channel, time, heigth, width",
            "final": fvd_results[videos1.shape[-3]]
        }
        
        self.metric += result['final']
        return self.metric / self.total

# test code / using example

# def main():
#     NUMBER_OF_VIDEOS = 8
#     VIDEO_LENGTH = 50
#     CHANNEL = 3
#     SIZE = 64
#     videos1 = torch.zeros(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)
#     videos2 = torch.ones(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)
#     device = torch.device("cuda")
#     # device = torch.device("cpu")

#     # import json
#     # result = calculate_fvd(videos1, videos2, device, method='videogpt')
#     # print(json.dumps(result, indent=4))

#     result = calculate_fvd(videos1, videos2, device, method='styleganv')