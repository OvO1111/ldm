import torch
from collections.abc import Iterable
from ldm.modules.diffusionmodules.util import noise_like


class DDIMStepSolver:
    def __init__(self, 
                 ddim_sampler,
                 ddim_use_original_steps=False, 
                 quantize_denoised=False, 
                 temperature=1., 
                 noise_dropout=0., 
                 score_corrector=None, 
                 unconditional_guidance_scale=1., 
                 unconditional_conditioning=None,
                 repeat_noise=False,
                 timesteps=None,
                 **kw
                ):
        self.ddim_sampler = ddim_sampler
        self.use_original_steps = ddim_use_original_steps
        self.quantize_denoised = quantize_denoised
        self.temperature = temperature
        self.noise_dropout = noise_dropout
        self.score_corrector = score_corrector
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
    
    def _retrieve_score(self, x, c, **kw):
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
            e_t = self.score_corrector.modify_score(ddim_sampler=self.ddim_sampler,
                                                    score=e_t, 
                                                    x=x, 
                                                    t=t, 
                                                    c=c, 
                                                    index=index, **kw)
        return e_t
    
    def _retrieve_xprev(self, x, e_t, **kw):
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

    def step(self, x, c, **kw):
        e_t = self._retrieve_score(x, c, **kw)
        x_prev, pred_x0 = self._retrieve_xprev(x, e_t, **kw)
        self.step_counter += 1
        return x_prev.type(x.dtype), pred_x0.type(x.dtype) 


class MultiDiffStepSolver(DDIMStepSolver):
    def __init__(self, 
                 ddim_sampler,
                 base_image_size,
                 final_image_size,
                 crop_intersect=.5,
                 max_infer_batch_size=4,
                 preserve_last=True,
                 **kw
                ):
        self.base_image_size = base_image_size
        self.final_image_size = final_image_size
        self.crop_intersect = crop_intersect
        self.max_infer_batch_size = max_infer_batch_size
        super().__init__(ddim_sampler=ddim_sampler, **kw)
        
        # compute crop centers
        self.crop_centers = []
        if not isinstance(self.crop_intersect, Iterable):
            self.crop_intersect = [self.crop_intersect] * len(self.base_image_size)
        if len(self.crop_intersect) == 1: self.crop_intersect = [self.crop_intersect[0] for _ in range(len(self.base_image_size))]
        self.crop_intersect = [round(self.base_image_size[_] * self.crop_intersect[_]) for _ in range(len(self.base_image_size))]
        
        xranges = [list(range(self.base_image_size[_] // 2, self.final_image_size[_] - self.crop_intersect[_], self.crop_intersect[_])) +\
            [self.final_image_size[_] - self.crop_intersect[_]] for _ in range(len(self.base_image_size))]
        if not preserve_last:
            [xranges[_].pop(-1) for _ in range(len(self.base_image_size))]
        for x in xranges[0]:
            for y in xranges[1]:
                if len(self.base_image_size) == 2:
                    self.crop_centers.append([x, y])
                elif len(self.base_image_size) == 3:
                    for z in xranges[2]:
                        self.crop_centers.append([x, y, z])
        # print(f"using {len(self.crop_centers)} crop centers")
    
    def cropper(self, final_x, final_c):
        centers = zip(*([iter(self.crop_centers)] * self.max_infer_batch_size))
        for center_group in centers:
            base_x, base_c = [], []
            for center in center_group:
                slices = [slice(None), slice(None)] + [slice(center[_] - self.base_image_size[_] // 2, center[_] + self.base_image_size[_] // 2) for _ in range(len(self.base_image_size))]
                base_x.append(final_x[*slices])
                base_c.append(final_c[*slices])
            base_x, base_c = torch.cat(base_x, dim=0), torch.cat(base_c, dim=0)
            yield base_x, base_c, center_group

    def step(self, x, c, **kw):
        b = x.shape[0]
        x_prev_final, pred_x0_final, counts = torch.zeros_like(x), torch.zeros_like(x), torch.zeros_like(x)
        for xx, cc, center_group in self.cropper(x, c):
            e_t = self._retrieve_score(xx, cc, **kw)
            x_prev, pred_x0 = self._retrieve_xprev(xx, e_t, **kw)
            
            for ib, center in enumerate(center_group):
                slices = [slice(None), slice(None)] + [slice(center[_] - self.base_image_size[_] // 2, center[_] + self.base_image_size[_] // 2 + 1) for _ in range(len(self.base_image_size))]
                x_prev_final[*slices] = x_prev[slice(ib, None, b)]
                pred_x0_final[*slices] = pred_x0[slice(ib, None, b)]
                counts[*slices] += 1
                
        x_prev_final /= counts
        pred_x0_final /= counts
        self.step_counter += 1
        return x_prev_final.type(x.dtype), pred_x0_final.type(x.dtype) 