import torch
import torch.nn as nn
from fmcib.models import fmcib_model
from ldm.models.template import BasePytorchLightningTrainer
from ldm.modules.diffusionmodules.util import timestep_embedding


class Classifier(nn.Module):
    def __init__(self, timesteps=1000, num_classes=2, model_channels=128, use_timesteps=False):
        super().__init__()
        self.timesteps = timesteps
        self.model_channels = model_channels
        self.fn = torch.nn.Sequential(
                torch.nn.SiLU(),
                torch.nn.Linear(4096, num_classes)  # eventtime
            )
        self.use_timesteps = use_timesteps
        if use_timesteps:       
            time_embed_dim = 4096
            self.time_embed = nn.Sequential(
                nn.Linear(model_channels, time_embed_dim),
                nn.SiLU(),
                nn.Linear(time_embed_dim, time_embed_dim),
            )

    def forward(self, x, t=None):
        if self.use_timesteps:  
            t_emb = timestep_embedding(t, self.model_channels, repeat_only=False)
            x = x + self.time_embed(t_emb)
        return self.fn(x)


class FMCIBMeasurement(BasePytorchLightningTrainer):
    def __init__(self, 
                 use_regressor=False, 
                 use_as_measurement=False, 
                 num_timesteps=1000,
                 use_timesteps=False,
                 risk_strat=12, 
                 **kwargs):
        super().__init__(**kwargs)
        self.model = fmcib_model(eval_mode=True)
        self.use_regressor = use_regressor
        if self.use_regressor:
            self.regressor = Classifier(num_timesteps, risk_strat, use_timesteps=use_timesteps)
        
        if self.ckpt_path is not None:
            self.init_from_ckpt(self.ckpt_path)
            
        if use_as_measurement:
            self.model.eval()
            self.regressor.eval()
        self.num_timesteps = num_timesteps

    def to(self, device):
        self.model.to(device)
        if self.use_regressor: self.regressor.to(device)
        
    def measure(self, x, c, t=None):
        # crop 50 ** 3 patch from x
        # c = torch.nonzero(c)
        # cc = torch.tensor(c).float().mean(dim=0).round().long()
        # indices = [slice(max(0, cc[0] - 25), min(x.shape[2], cc[0] + 25)),
        #            slice(max(0, cc[1] - 25), min(x.shape[3], cc[1] + 25)),
        #            slice(max(0, cc[2] - 25), min(x.shape[4], cc[2] + 25))]
        # x = x[:, :, *indices]
        # pad_x = [[50 - x.shape[i] // 2, 50 - x.shape[i] - x.shape[i] // 2] for i in range(2, 5)]
        # x = torch.nn.functional.pad(x, [*pad_x[2], *pad_x[1], *pad_x[0]], mode='constant', value=0)
        
        feat = self.model(x[:, 0: 1])
        if self.use_regressor: feat = self.regressor(feat, t)
        return feat
    
    def shared_step(self, batch):
        x, event = batch.get("image_and_seg"), batch.get("survival")[:, 0]
        event[event == -1] = 255
        prefix = "train" if self.training else "val"
        
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=x.device).long()
        feat = self.measure(x.float(), None, t)
        loss = torch.nn.functional.cross_entropy(feat, event.view(-1).long(), ignore_index=255, reduction="none").mean()
        return loss, {f"{prefix}_loss": loss, f"{prefix}_acc": (feat.argmax(1) == event).float().mean()}
    
    def log_images(self, batch, N=8, *args, **kwargs):
        x = batch['image_and_seg']
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=x.device).long()
        feat = self.measure(x.float(), None, t)
        logs = {"image_and_seg": x,
                "gt": str({"eventime": batch['survival'][:, 0].data.cpu().numpy(),
                           "event": batch['survival'][:, 1].data.cpu().numpy()}),
                "pred": str({"eventime": feat.argmax(1).data.cpu().numpy()})}
        return logs