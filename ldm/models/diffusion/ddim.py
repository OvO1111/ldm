"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm
from functools import partial

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like
from ldm.models.diffusion.measurements import FMCIBMeasurement


class DDIMStepSolver:
    def __init__(self, 
                 ddim_sampler,
                 ddim_use_original_steps=False, 
                 quantize_denoised=False, 
                 temperature=1., 
                 noise_dropout=0., 
                 score_corrector=None, 
                 corrector_kwargs={}, 
                 unconditional_guidance_scale=1., 
                 unconditional_conditioning=None,
                 repeat_noise=False,
                 timesteps=None
                ):
        self.ddim_sampler = ddim_sampler
        self.use_original_steps = ddim_use_original_steps
        self.quantize_denoised = quantize_denoised
        self.temperature = temperature
        self.noise_dropout = noise_dropout
        self.score_corrector = score_corrector
        self.corrector_kwargs = corrector_kwargs
        self.unconditional_guidance_scale = unconditional_guidance_scale
        self.unconditional_conditioning = unconditional_conditioning
        self.repeat_noise = repeat_noise
        
        # alias
        self.model = ddim_sampler.model
        self.ddim_alphas = ddim_sampler.ddim_alphas
        self.ddim_alphas_prev = ddim_sampler.ddim_alphas_prev
        self.ddim_sqrt_one_minus_alphas = ddim_sampler.ddim_sqrt_one_minus_alphas
        self.ddim_sigmas = ddim_sampler.ddim_sigmas
        self.ddim_sigmas_for_original_num_steps = ddim_sampler.ddim_sigmas_for_original_num_steps
        
        self.timesteps = timesteps
        self.step_counter = 0
        
    def parse_step(self, b, device):
        t = torch.full((b,), self.timesteps[self.step_counter], device=device, dtype=torch.long)
        index = len(self.timesteps) - self.step_counter - 1
        return t, index
    
    def _retrieve_score(self, x, c):
        b, *_, device = *x.shape, x.device
        t, index = self.parse_step(b, device)
        if self.unconditional_conditioning is None or self.unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t, c)[0]
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([self.unconditional_conditioning, c])
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
            e_t = e_t_uncond + self.unconditional_guidance_scale * (e_t - e_t_uncond)
        
        if self.score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = self.score_corrector.modify_score(self.ddim_sampler, e_t, x, t, c, index=index, **self.corrector_kwargs)
        return e_t
    
    def _retrieve_xprev(self, x, e_t):
        b, *_, device = *x.shape, x.device
        t, index = self.parse_step(b, device)
        alphas = self.model.alphas_cumprod if self.use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if self.use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if self.use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if self.use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1,) + (1,) * self.model.dims, alphas[index], device=device)
        a_prev = torch.full((b, 1,) + (1,) * self.model.dims, alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1,) + (1,) * self.model.dims, sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1,) + (1,) * self.model.dims, sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if self.quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, self.repeat_noise) * self.temperature
        if self.noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=self.noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0

    def step(self, x, c):
        e_t = self._retrieve_score(x, c)
        x_prev, pred_x0 = self._retrieve_xprev(x, e_t)
        self.step_counter += 1
        return x_prev.type(x.dtype), pred_x0.type(x.dtype) 


class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs={},
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        # C, H, W = shape
        size = (batch_size,) + shape
        if verbose: print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    verbose=verbose)
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs={},
                      unconditional_guidance_scale=1., unconditional_conditioning=None, verbose=False):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device, dtype=cond.dtype)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        if verbose: print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps) if verbose else time_range
        self.ddim_solver = DDIMStepSolver(
            self,
            ddim_use_original_steps=ddim_use_original_steps, 
            quantize_denoised=quantize_denoised, 
            temperature=temperature, 
            noise_dropout=noise_dropout, 
            score_corrector=score_corrector, 
            corrector_kwargs=corrector_kwargs, 
            unconditional_guidance_scale=unconditional_guidance_scale, 
            unconditional_conditioning=unconditional_conditioning,
            timesteps=list(reversed(range(0,timesteps))) if ddim_use_original_steps else np.flip(timesteps)
        )
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)
            
            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img
                
            img, pred_x0 = self.ddim_solver.step(img, cond)
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        # for i, step in enumerate(iterator):
        #     index = total_steps - i - 1
        #     ts = torch.full((b,), step, device=device, dtype=torch.long)

        #     if mask is not None:
        #         assert x0 is not None
        #         img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
        #         img = img_orig * mask + (1. - mask) * img

        #     outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
        #                               quantize_denoised=quantize_denoised, temperature=temperature,
        #                               noise_dropout=noise_dropout, score_corrector=score_corrector,
        #                               corrector_kwargs=corrector_kwargs,
        #                               unconditional_guidance_scale=unconditional_guidance_scale,
        #                               unconditional_conditioning=unconditional_conditioning)
        #     img, pred_x0 = outs
        #     if callback: callback(i)
        #     if img_callback: img_callback(pred_x0, i)

        #     if index % log_every_t == 0 or index == total_steps - 1:
        #         intermediates['x_inter'].append(img)
        #         intermediates['pred_x0'].append(pred_x0)

        return img, intermediates

    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None):
        b, *_, device = *x.shape, x.device

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t, c)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([unconditional_conditioning, c])
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1,) + (1,) * self.model.dims, alphas[index], device=device)
        a_prev = torch.full((b, 1,) + (1,) * self.model.dims, alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1,) + (1,) * self.model.dims, sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1,) + (1,) * self.model.dims, sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0

    
class FMCIBScoreCorrector(torch.nn.Module):
    def __init__(self, 
                 classifier_scale=1.,
                 weight_decay=.999,
                 betas=(0.1, 0.01),
                 lr=1e-2,
                 use_optimizer_on_classifier_scale=True):
        super().__init__()
        self.classifier_scale = torch.ones(1) * classifier_scale
        
        self.m_t, self.v_t = 0, 0
        self.weight_decay = weight_decay
        self.betas, self.lr = betas, lr
        self.use_optimizer_on_classifier_scale = use_optimizer_on_classifier_scale
        self.measurement = FMCIBMeasurement(use_as_measurement=True, use_regressor=True)
        
    def to(self, device):
        self.classifier_scale.to(device)
        self.measurement.to(device)
    
    def modify_score(self, ddim_sampler: DDIMSampler, score, x, t, c, index, **kw):
        assert "batch" in kw
        batch = kw["batch"]
        y = batch.get("survival")[:, 0].long()  # eventime
        
        b, *_, device = *x.shape, x.device
        self.classifier_scale = self.classifier_scale.to(device)
        a_t = torch.full((b, 1,) + (1,) * ddim_sampler.model.dims, ddim_sampler.alphas_cumprod[index], device=device)
        tm1 = torch.full((b,), ddim_sampler.ddim_solver.timesteps[-index], device=device)
        with torch.enable_grad():
            x.requires_grad = True
            classifier_scale = self.classifier_scale.requires_grad_(True)
            log_probs = torch.log_softmax(self.measurement.measure(x, c, t / ddim_sampler.ddpm_num_timesteps), dim=-1)
            y_log_probs = log_probs[range(len(log_probs)), y.view(-1)]
            grad = torch.autograd.grad(y_log_probs.sum(), x)[0]
            
            if self.use_optimizer_on_classifier_scale: 
                # 1 MCS
                x_prev, _ = ddim_sampler.ddim_solver._retrieve_xprev(x, score - 1. / a_t * classifier_scale * grad)
                log_probs2 = torch.log_softmax(self.measurement.measure(x_prev, c, tm1 / ddim_sampler.ddpm_num_timesteps), dim=-1)
                y_log_probs2 = log_probs2[range(len(log_probs2)), y.view(-1)]
                grad2 = torch.autograd.grad((y_log_probs2 - y_log_probs).sum(), classifier_scale)[0]
                # update classifier scale
                grad2 = grad2 + self.weight_decay * classifier_scale
                self.m_t = self.betas[0] * self.m_t + (1 - self.betas[0]) * grad2
                self.v_t = self.betas[1] * self.v_t + (1 - self.betas[1]) * grad2**2
                m_hat_t = self.m_t / (1 - self.betas[0])
                v_hat_t = self.v_t / (1 - self.betas[1])
                self.classifier_scale = classifier_scale - self.lr * m_hat_t / (v_hat_t.sqrt() + 1e-8)
            
            modified_score = score - 1. / a_t * self.classifier_scale * grad
            
        return modified_score
    
    
DefaultDDIMSampler = DDIMSampler