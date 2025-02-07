import torch
from ldm.models.diffusion.sampling.ddim import DDIMSampler
from ldm.modules.diffusionmodules.util import instantiate_from_config

    
class SimpleScoreCorrector(torch.nn.Module):
    def __init__(self, 
                 classifier_config,
                 class_key,
                 classifier_scale=1.,
                 weight_decay=.999,
                 betas=(0.1, 0.01),
                 lr=1e-2,
                 use_optimizer_on_classifier_scale=True):
        super().__init__()
        self.class_key = class_key
        self.classifier_scale = torch.ones(1) * classifier_scale
        
        self.m_t, self.v_t = 0, 0
        self.weight_decay = weight_decay
        self.betas, self.lr = betas, lr
        self.use_optimizer = use_optimizer_on_classifier_scale
        self.classifier = instantiate_from_config(classifier_config)
        
    def to(self, device):
        self.classifier_scale.to(device)
        self.classifier.to(device)
        
    def get_logits(self, x, t):
        return self.classifier(x)
    
    def modify_score(self, ddim_sampler: DDIMSampler, score, x, t, index, c=None, **kw):
        batch = kw["batch"]
        y = batch.get(self.class_key)[:, 0].long()
        
        b, *_, device = *x.shape, x.device
        self.classifier_scale = self.classifier_scale.to(device)
        a_t = torch.full((b, 1,) + (1,) * ddim_sampler.model.dims, ddim_sampler.alphas_cumprod[index], device=device)
        tm1 = torch.full((b,), ddim_sampler.ddim_solver.timesteps[-index], device=device)
        with torch.enable_grad():
            x.requires_grad = True
            classifier_scale = self.classifier_scale.requires_grad_(True)
            log_probs = torch.log_softmax(self.get_logits(x, t / ddim_sampler.ddpm_num_timesteps), dim=-1)
            y_log_probs = log_probs[range(len(log_probs)), y.view(-1)]
            grad = torch.autograd.grad(y_log_probs.sum(), x)[0]
            
            if self.use_optimizer: 
                # 1 MCS
                x_prev, _ = ddim_sampler.ddim_solver._retrieve_xprev(x, score - 1. / a_t * classifier_scale * grad)
                log_probs2 = torch.log_softmax(self.get_logits(x_prev, tm1 / ddim_sampler.ddpm_num_timesteps), dim=-1)
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