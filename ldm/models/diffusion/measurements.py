import torch
import torch.nn as nn

from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from fmcib.models import fmcib_model
from lifelines.utils.concordance import concordance_index
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
                 max_eventime_yrs=12, 
                 **kwargs):
        super().__init__(**kwargs)
        self.model = fmcib_model(eval_mode=True)
        self.use_regressor = use_regressor
        self.max_eventime_yrs = max_eventime_yrs
        if self.use_regressor:
            self.regressor = Classifier(num_timesteps, max_eventime_yrs, use_timesteps=use_timesteps)
        
        if self.ckpt_path is not None:
            self.init_from_ckpt(self.ckpt_path)
            
        if use_as_measurement:
            self.model.eval()
            self.regressor.eval()
        self.num_timesteps = num_timesteps

    def to(self, device):
        self.model.to(device)
        if self.use_regressor: self.regressor.to(device)
        
    def get_loss_and_risk(self, model_outputs, events, censors, mode="nll", ignore_index=255):
        events = torch.clamp(events // 365, 0, self.max_eventime_yrs)
        model_outputs = model_outputs[events != ignore_index]
        events = events[events != ignore_index]
        censors = censors[censors != ignore_index]
        
        h = torch.sigmoid(model_outputs)  # harzards
        s = torch.cumprod(1 - h, dim=1)  # survival
        r = s.sum(-1)
        bs = s.shape[0]
        if mode == 'nll':
            l = censors * s[range(bs), events.long()].log() +\
                (1 - censors) * (h[range(bs), events.long()] * s[range(bs), events.long()]).log()
        return -l.sum(), r
        
    def measure(self, x, c, t=None):
        # resize x.shape to 50**3
        im = x[:, 0:1]
        if im.shape[0] != 50 or im.shape[1] != 50 or im.shape[2] != 50:
            im = nn.functional.interpolate(im, size=(50, 50, 50), mode="trilinear")
        
        feat = self.model(im)
        if self.use_regressor: feat = self.regressor(feat, t)
        return feat
    
    def on_train_epoch_start(self):
        self.train_risks = []
        self.train_events = []
        self.train_censors = []
    
    def training_step(self, batch, batch_idx):
        x, survival = batch.get("image_and_seg"), batch.get("survival")
        events, censors = survival[:, 0].view(-1), survival[:, 1].view(-1)
        prefix = "train" if self.training else "val"
        
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=x.device).long()
        feat = self.measure(x.float(), None, t)
        loss, risk = self.get_loss_and_risk(feat, events, censors)
        self.train_events.extend(survival[:, 0].view(-1).cpu().data.numpy().tolist())
        self.train_censors.extend(survival[:, 1].view(-1).cpu().data.numpy().tolist())
        self.train_risks.extend(risk.view(-1).cpu().data.numpy().tolist())
        
        self.log_dict({f"{prefix}_loss": loss,},
                      prog_bar=True, on_step=True, on_epoch=True, logger=True)
        return loss
    
    def on_train_epoch_end(self):
        if self.trainer.global_step > 0:
            c_index = concordance_index(self.train_events, self.train_risks, self.train_censors)
            self.log("train_c_index", c_index, prog_bar=True, logger=True, on_step=False, on_epoch=True)
    
    def on_validation_epoch_start(self):
        self.val_risks = []
        self.val_events = []
        self.val_censors = []
        
    def validation_step(self, batch, batch_idx):
        x, survival = batch.get("image_and_seg"), batch.get("survival")
        events, censors = survival[:, 0].view(-1), survival[:, 1].view(-1)
        prefix = "train" if self.training else "val"
        
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=x.device).long()
        feat = self.measure(x.float(), None, t)
        loss, risk = self.get_loss_and_risk(feat, events, censors)
        self.val_events.extend(survival[:, 0].view(-1).cpu().data.numpy().tolist())
        self.val_censors.extend(survival[:, 1].view(-1).cpu().data.numpy().tolist())
        self.val_risks.extend(risk.view(-1).cpu().data.numpy().tolist())

        self.log_dict({f"{prefix}_loss": loss}, prog_bar=True, on_step=True, on_epoch=True, logger=True)
        
    def on_validation_epoch_end(self):
        if self.trainer.global_step > 0:
            c_index = concordance_index(self.val_events, self.val_risks, self.val_censors)
            self.log("val_c_index", c_index, prog_bar=True, logger=True, on_step=False, on_epoch=True)
    
    def log_images(self, batch, *args, **kwargs):
        x, survival = batch.get("image_and_seg"), batch.get("survival")
        events, censors = survival[:, 0].view(-1), survival[:, 1].view(-1)
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=x.device).long()
        feat = self.measure(x.float(), None, t)
        loss, risk = self.get_loss_and_risk(feat, events, censors)
        logs = {"image_and_seg": x,
                "gt": str({"eventime": events.data.cpu().numpy().tolist(),
                           "censor": censors.data.cpu().numpy().tolist()}),
                "pred": str({"eventime": feat.argmax(1).data.cpu().numpy().tolist(),
                             "risk": risk.cpu().data.numpy().tolist()},)}
        return logs
    
    def configure_optimizers(self):
        param = list(self.regressor.parameters())
        optimizer = AdamW(param, lr=self.learning_rate)
        scheduler = LambdaLR(optimizer, lambda e: (1 - e / self.trainer.max_epochs) ** 0.8)
        return [optimizer], [scheduler]