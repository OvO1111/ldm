import torch
import random
from torch import nn
import numpy as np
import torch.nn.functional as F
from einops import repeat, rearrange

from taming.modules.discriminator.model import weights_init
from taming.modules.util import ActNorm
from ldm.modules.losses.lpips import LPIPS


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss


def hinge_d_loss_with_exemplar_weights(logits_real, logits_fake, weights):
    assert weights.shape[0] == logits_real.shape[0] == logits_fake.shape[0]
    loss_real = torch.mean(F.relu(1. - logits_real), dim=[1,2,3])
    loss_fake = torch.mean(F.relu(1. + logits_fake), dim=[1,2,3])
    loss_real = (weights * loss_real).sum() / weights.sum()
    loss_fake = (weights * loss_fake).sum() / weights.sum()
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss

def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight


def measure_perplexity(predicted_indices, n_embed):
    # src: https://github.com/karpathy/deep-vector-quantization/blob/main/model.py
    # eval cluster perplexity. when perplexity == num_embeddings then all clusters are used exactly equally
    encodings = F.one_hot(predicted_indices, n_embed).float().reshape(-1, n_embed)
    avg_probs = encodings.mean(0)
    perplexity = (-(avg_probs * torch.log(avg_probs + 1e-10)).sum()).exp()
    cluster_use = torch.sum(avg_probs > 0)
    return perplexity, cluster_use

def l1(x, y):
    return torch.abs(x-y)


def l2(x, y):
    return torch.pow((x-y), 2)


class LPIPSWithDiscriminator(nn.Module):
    def __init__(self, 
                 disc_start, 
                 pixelloss_weight=1.0,
                 encoder_specific_weight=1.0, 
                 disc_num_layers=3, 
                 disc_in_channels=1, 
                 disc_factor=1.0, 
                 disc_weight=1.0,
                 logvar_init=1.0,
                 perceptual_weight=1.0, 
                 use_actnorm=False, 
                 disc_conditional=False,
                 disc_ndf=64, 
                 disc_loss="hinge", 
                 n_classes=None, 
                 perceptual_loss="lpips",
                 pixel_loss="l1", 
                 image_gan_weight=0.5, 
                 ct_gan_weight=0.5, 
                 gan_feat_weight=1.0, 
                 dims=3, 
                 n_frames_to_select=10, 
                 encodertype='kl'
                ):
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        assert perceptual_loss in ["lpips", "clips", "dists"]
        assert pixel_loss in ["l1", "l2"]
        self.dims = dims
        if perceptual_loss == "lpips":
            print(f"{self.__class__.__name__}: Running with LPIPS.")
            self.perceptual_loss = LPIPS().eval()
        else:
            raise ValueError(f"Unknown perceptual loss: >> {perceptual_loss} <<")
        self.encoder_type = encodertype
        self.pixel_loss = l1 if pixel_loss == 'l1' else l2

        self.n_frames = n_frames_to_select
        self.frame_discriminator = NLayerDiscriminator(
            input_nc=disc_in_channels,
            n_layers=disc_num_layers,
            use_actnorm=use_actnorm,
            ndf=disc_ndf,
        ).apply(weights_init)
        self.volume_discriminator = NLayerDiscriminator3D(
            input_nc=disc_in_channels,
            n_layers=disc_num_layers,
            use_actnorm=use_actnorm,
            ndf=disc_ndf
        ).apply(weights_init)
        
        self.volume_gan_weight = ct_gan_weight
        self.pixel_weight = pixelloss_weight
        self.frame_gan_weight = image_gan_weight
        self.perceptual_weight = perceptual_weight
        self.gan_feat_weight = gan_feat_weight
        self.discriminator_iter_start = disc_start
        self.encoder_specific_weight = encoder_specific_weight
        
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        else:
            raise ValueError(f"Unknown GAN loss '{disc_loss}'.")
        
        self.n_classes = n_classes
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight
    
    def forward(self, inputs, reconstructions, aux=None, *args, **kwargs):
        b, c, *shp = inputs.shape
        assert inputs.shape == reconstructions.shape, f"{inputs.shape} and {reconstructions.shape} not match"
        if c > 1:
            inputs = rearrange(inputs, "b c ... -> (b c) 1 ...")
            reconstructions = rearrange(reconstructions, "b c ... -> (b c) 1 ...")
        # backward compatible
        if aux is None:
            aux = kwargs.pop("posterior" if self.encoder_type == 'kl' else 'codebook_loss')
        out = self._forward(inputs, reconstructions, aux, *args, **kwargs)
        return out
    
    def _generator_update(self, inputs: torch.Tensor,
                          reconstructions: torch.Tensor,
                          input_frames: torch.Tensor,
                          reconstruction_frames: torch.Tensor,
                          loss_dict: dict,
                          cond=None, cond_frames=None, global_step=None, split='train', last_layer=None):
        p_loss = loss_dict.get("p_loss")
        rec_loss = loss_dict.get("rec_loss")
        nll_loss = loss_dict.get("nll_loss")
        encoder_loss = loss_dict.get("encoder_loss")
        use_3d = len(reconstructions.shape) == 5
        
        if cond is not None:
            reconstructions = torch.cat([reconstructions, cond], dim=1)
            reconstruction_frames = torch.cat([reconstruction_frames, cond_frames], dim=1)
        if self.frame_gan_weight > 0: 
            logits_fake_frame, pred_fake_frame = self.frame_discriminator(reconstruction_frames.contiguous())
        if self.volume_gan_weight > 0 and use_3d: 
            logits_fake_volume, pred_fake_volume = self.volume_discriminator(reconstructions.contiguous())
        g_loss = -logits_fake_frame.mean() if not use_3d else -(logits_fake_frame.mean() + logits_fake_volume.mean()) / 2

        try:
            d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
        except RuntimeError:
            assert not self.training
            d_weight = torch.tensor(0.0)

        disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
        frame_gan_feat_loss = 0
        volume_gan_feat_loss = 0
        if self.gan_feat_weight > 0:
            if self.frame_gan_weight > 0:
                _, pred_real_frame = self.frame_discriminator(input_frames)
                frame_gan_feat_loss = sum(F.l1_loss(pred_fake_frame[i], pred_real_frame[i].detach()) for i in range(len(pred_fake_frame)))
            if self.volume_gan_weight > 0 and use_3d:
                _, pred_real_volume = self.volume_discriminator(inputs)
                volume_gan_feat_loss = sum(F.l1_loss(pred_fake_volume[i], pred_real_volume[i].detach()) for i in range(len(pred_fake_volume)))
            gan_feat_loss = disc_factor * self.gan_feat_weight * (frame_gan_feat_loss + volume_gan_feat_loss)
        else:
            gan_feat_loss = torch.tensor(0.).to(g_loss.device)
                
        loss =  nll_loss +\
                d_weight * disc_factor * g_loss +\
                self.gan_feat_weight * gan_feat_loss +\
                self.encoder_specific_weight * encoder_loss.mean()

        log = {
            f"{split}/nll_loss": nll_loss.detach().mean(),
            f"{split}/reconstruction_loss": rec_loss.detach().mean(),
            f"{split}/perceptual_loss": p_loss.detach().mean(),
            f"{split}/generator_loss": g_loss.detach().mean(),
            f"{split}/encoder_loss": encoder_loss.detach().mean(),
            f"{split}/total_loss": loss.clone().detach().mean(),
            f"{split}/adaptive_weight": d_weight.detach(),
            f"{split}/disc_factor": torch.tensor(disc_factor),
        }

        return loss, log
    
    def _discriminator_update(self, inputs: torch.Tensor, 
                              reconstructions: torch.Tensor, 
                              input_frames: torch.Tensor,
                              reconstruction_frames: torch.Tensor,
                              cond=None, cond_frames=None, global_step=None, split='train'):
        use_3d = len(reconstructions.shape) == 5
    
        if cond is not None:
            inputs = torch.cat([inputs, cond], dim=1)
            input_frames = torch.cat([input_frames, cond_frames], dim=1)
            reconstructions = torch.cat([reconstructions, cond], dim=1)
            reconstruction_frames = torch.cat([reconstruction_frames, cond_frames], dim=1)
        if self.frame_gan_weight > 0: 
            logits_real_frame, _ = self.frame_discriminator(input_frames.contiguous().detach())
            logits_fake_frame, _ = self.frame_discriminator(reconstruction_frames.contiguous().detach())
        if self.volume_gan_weight > 0 and use_3d:
            logits_real_volume, _ = self.volume_discriminator(inputs.contiguous().detach())
            logits_fake_volume, _ = self.volume_discriminator(reconstructions.contiguous().detach())

        disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
        d_loss = disc_factor * self.disc_loss(logits_real_frame, logits_fake_frame)
        if use_3d: d_loss = (d_loss + disc_factor * self.disc_loss(logits_real_volume, logits_fake_volume)) / 2

        log = {
            f"{split}/disc_loss": d_loss.clone().detach().mean(),
            f"{split}/logits_real_frame": logits_real_frame.detach().mean(),
            f"{split}/logits_fake_frame": logits_fake_frame.detach().mean(),
        } | ({
            f"{split}/logits_real_volume": logits_real_volume.detach().mean(),
            f"{split}/logits_fake_volume": logits_fake_volume.detach().mean(),
        } if use_3d else {})
        return d_loss, log
    
    def _forward(self, inputs: torch.Tensor, 
                 reconstructions: torch.Tensor, 
                 aux: torch.Tensor, 
                 optimizer_idx: int, 
                 global_step: int, 
                 last_layer=None, cond=None, split="train", predicted_indices=None, weights=None):
        # encoder-specific loss
        log = {}
        if self.encoder_type == 'vq': 
            codebook_loss = aux
            if not hasattr(codebook_loss, "shape") and len(codebook_loss.shape) != 1:
                encoder_loss = torch.tensor([0.]).to(inputs.device)
            if predicted_indices is not None:
                assert self.n_classes is not None
                with torch.no_grad():
                    perplexity, cluster_usage = measure_perplexity(predicted_indices, self.n_classes)
                log[f"{split}/perplexity"] = perplexity
                log[f"{split}/cluster_usage"] = cluster_usage
                
        elif self.encoder_type == 'kl': 
            posteriors = aux
            kl_loss = posteriors.kl()
            encoder_loss = torch.sum(kl_loss) / kl_loss.shape[0]
            
        if len(reconstructions.shape) == 5:
            axis_to_select = random.randint(2, 4)
            axes = ["h", "w", "d"][axis_to_select - 2], " ".join([j for i, j in enumerate(["h", "w", "d"]) if i != axis_to_select - 2])
            indices_to_select = torch.randperm(reconstructions.shape[axis_to_select])[:self.n_frames].to(inputs.device)
            input_frames, reconstruction_frames, cond_frames = map(lambda i: rearrange(
                torch.index_select(i, axis_to_select, indices_to_select), f'b c h w d -> (b {axes[0]}) c {axes[1]}') if i is not None else None,
                [inputs, reconstructions, cond])
        else:
            cond_frames = cond
            input_frames = inputs
            reconstruction_frames = reconstructions
        
        rec_loss = self.pixel_loss(inputs.contiguous(), reconstructions.contiguous())
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(input_frames.contiguous(), reconstruction_frames.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss
        else: p_loss = torch.tensor([0.0])

        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        if weights is not None: nll_loss *= weights
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]

        # GAN
        if optimizer_idx == 0:
            # generator update
            loss, logs = self._generator_update(inputs, reconstructions, input_frames, reconstruction_frames,
                                                {"nll_loss": nll_loss,
                                                 "rec_loss": rec_loss,
                                                 "p_loss": p_loss,
                                                 "encoder_loss": encoder_loss},
                                                cond=cond, cond_frames=cond_frames, global_step=global_step, split=split, last_layer=last_layer)

        if optimizer_idx == 1:
            # discriminator update
            loss, logs = self._discriminator_update(inputs, reconstructions, input_frames, reconstruction_frames,
                                                    cond=cond, cond_frames=cond_frames, global_step=global_step, split=split)
        logs = log | logs
        return loss, logs


class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.SyncBatchNorm, use_sigmoid=False, getIntermFeat=True, use_actnorm=False):
        super(NLayerDiscriminator, self).__init__()
        if use_actnorm:
            norm_layer = ActNorm
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw,
                               stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw,
                                stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[-1], res[1:]
        else:
            return self.model(input)


class NLayerDiscriminator3D(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.SyncBatchNorm, use_sigmoid=False, getIntermFeat=True, use_actnorm=False):
        super(NLayerDiscriminator3D, self).__init__()
        if use_actnorm:
            norm_layer = ActNorm
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv3d(input_nc, ndf, kernel_size=kw,
                               stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv3d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv3d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv3d(nf, 1, kernel_size=kw,
                                stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[-1], res[1:]
        else:
            return self.model(input)
