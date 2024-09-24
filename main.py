import argparse, os, sys, datetime, glob, re
import numpy as np
import time
import torch
import wandb
import shutil
import pytorch_lightning as pl

from packaging import version
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset, _utils
from functools import partial
from itertools import cycle
from queue import Queue

from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.utilities import rank_zero_info

from ldm.util import instantiate_from_config, get_obj_from_str
from inference.utils import TwoStreamBatchSampler, DistributedTwoStreamBatchSampler, image_logger, visualize, combine_mask_and_im_v2


def exists(x):
    return x is not None


def default(x, defval=None):
    return x if x is not None else defval


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="test",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "--no-test",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="disable test",
    )
    parser.add_argument(
        "-p",
        "--project",
        help="name of new or path to existing project"
    )
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="/ailab/user/dailinrui/data/ldm",
        help="directory for logging dat shit",
    )
    parser.add_argument(
        "--scale_lr",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="scale base-lr by ngpu * batch_size * n_accumulate",
    )
    return parser


def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))


class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()

    dataset = worker_info.dataset
    worker_id = worker_info.id

    return np.random.seed(np.random.get_state()[1][0] + worker_id)


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, train=None, validation=None,
                 wrap=False, num_workers=None, use_worker_init_fn=False,
                 shuffle_val_dataloader=False, batch_sampler=None):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.use_worker_init_fn = use_worker_init_fn
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = partial(self._val_dataloader, shuffle=shuffle_val_dataloader)
        self.wrap = wrap
        self.has_batch_sampler = batch_sampler is not None
        self.batch_sampler_config = batch_sampler

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])
        if self.has_batch_sampler: self._get_batch_sampler(self.batch_sampler_config)
                
    def _get_batch_sampler(self, batch_sampler):
        dataset = self.datasets['train'] if 'train' in self.datasets else self.datasets['test']
        sampler = get_obj_from_str(batch_sampler["target"])
        if sampler == TwoStreamBatchSampler or sampler == DistributedTwoStreamBatchSampler:
            try:
                primary_batch_size = batch_sampler["params"].get("primary_batch_size", 1)
            except Exception: primary_batch_size = 1
            self.batch_sampler = sampler(primary_indices=dataset.fine_labeled_indices, 
                                         secondary_indices=dataset.coarse_labeled_indices,
                                         batch_size=self.batch_size,
                                         secondary_batch_size=self.batch_size-primary_batch_size, **batch_sampler["params"])
        else:
            raise NotImplementedError()

    def _train_dataloader(self):                
        if self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        if not self.has_batch_sampler:
            return DataLoader(self.datasets["train"],
                              batch_size=self.batch_size,
                              num_workers=self.num_workers,
                              worker_init_fn=init_fn,
                              shuffle=True, 
                              collate_fn=getattr(self.datasets["train"], "collate", _utils.collate.default_collate))
        else:
            return DataLoader(self.datasets["train"],
                              batch_sampler=self.batch_sampler,
                              num_workers=self.num_workers,
                              worker_init_fn=init_fn,
                              collate_fn=getattr(self.datasets["train"], "collate", _utils.collate.default_collate))

    def _val_dataloader(self, shuffle=False, batch_size=None):
        if self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["validation"],
                            batch_size=self.batch_size if batch_size is None else batch_size,
                            num_workers=self.num_workers,
                            worker_init_fn=init_fn,
                            shuffle=shuffle, 
                            collate_fn=getattr(self.datasets["validation"], "collate", _utils.collate.default_collate))
        
    def _test_dataloader(self, shuffle=False):
        if self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        if not self.has_batch_sampler:
            return cycle(DataLoader(self.datasets["test"],
                                    batch_size=self.batch_size,
                                    num_workers=self.num_workers,
                                    worker_init_fn=init_fn,
                                    shuffle=shuffle, 
                                    collate_fn=getattr(self.datasets["test"], "collate", _utils.collate.default_collate)))
        else:
            return cycle(DataLoader(self.datasets["test"],
                                    batch_sampler=self.batch_sampler,
                                    num_workers=self.num_workers,
                                    worker_init_fn=init_fn,
                                    collate_fn=getattr(self.datasets["test"], "collate", _utils.collate.default_collate)))
    
    def _predict_dataloader(self, shuffle=False):
        return self._test_dataloader(shuffle)


class SetupCallback(Callback):
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config

    def on_keyboard_interrupt(self, trainer, pl_module):
        if trainer.global_rank == 0:
            print("Summoning checkpoint.")
            ckpt_path = os.path.join(self.ckptdir, "last.ckpt")
            trainer.save_checkpoint(ckpt_path)

    def on_pretrain_routine_start(self, trainer, pl_module):
        os.makedirs(self.logdir, exist_ok=True)
        os.makedirs(self.ckptdir, exist_ok=True)
        os.makedirs(self.cfgdir, exist_ok=True)
        if trainer.global_rank == 0:
            # Create logdirs and save configs

            if "callbacks" in self.lightning_config:
                if 'metrics_over_trainsteps_checkpoint' in self.lightning_config['callbacks']:
                    os.makedirs(os.path.join(self.ckptdir, 'trainstep_checkpoints'), exist_ok=True)
            print("Project config")
            print(OmegaConf.to_yaml(self.config))
            OmegaConf.save(self.config,
                           os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)))

            print("Lightning config")
            print(OmegaConf.to_yaml(self.lightning_config))
            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}),
                           os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)))

        else:
            # ModelCheckpoint callback created log directory --- remove it
            if not self.resume and os.path.exists(self.logdir):
                dst, name = os.path.split(self.logdir)
                dst = os.path.join(dst, "child_runs", name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                try:
                    shutil.copytree(self.logdir, dst)
                    shutil.rmtree(self.logdir, ignore_errors=1)
                    ckptdir = os.path.join(self.logdir, "checkpoints")
                    cfgdir = os.path.join(self.logdir, "configs")
                    os.makedirs(ckptdir, exist_ok=True)
                    os.makedirs(cfgdir, exist_ok=True)
                except FileNotFoundError:
                    pass
                except FileExistsError:
                    pass
            pass


class ImageLogger(Callback):
    def __init__(self, 
                 train_batch_frequency=None,        # train log frequency
                 test_batch_frequency=None,         # test log frequency
                 val_batch_frequency=None,          # validation log frequency
                 max_images=-1,                     # max images to perserve in each image folder
                 is_training=True,                  # is training / inferencing model
                 log_on_batch_idx=True,             # log image on batch idx / gs idx
                 log_images_kwargs=None,            # kwargs to be passed to pl_module.log_images
                 logger={},                         # logger for each returned image
                 log_separate=False,                # whether to log all images on the same canvas
                 log_local_only=True):              # only log to local folders (not in tensorboard / wandb)
        super().__init__()
        self.batch_freq_tr = train_batch_frequency
        self.batch_freq_te = test_batch_frequency
        self.batch_freq_val = default(val_batch_frequency, train_batch_frequency)
        
        self.is_training = is_training
        assert exists(train_batch_frequency) or exists(test_batch_frequency)
        if self.batch_freq_tr is None and self.batch_freq_val is None and self.batch_freq_te is not None:
            self.is_training = False
        assert (is_training and exists(train_batch_frequency)) or (not is_training and exists(test_batch_frequency))
        self.max_images = max_images
        self.logger_log_images = {
            WandbLogger: self._wandb,
            TensorBoardLogger: self._board
        } if not log_local_only else {}
        
        # log some more at the start of each epoch
        self.log_steps_tr = [10 ** n for n in range(int(np.log10(self.batch_freq_tr)) + 1)]
        self.log_steps_val = [10 ** n for n in range(int(np.log10(self.batch_freq_val)) + 1)]
        self.log_steps_te = [10 ** n for n in range(int(np.log10(self.batch_freq_te)) + 1)]
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_separate = log_separate
        
        def _get_logger(target, params):
            if not isinstance(params, dict): params = OmegaConf.to_container(params)
            if target == "mask_rescale":
                return lambda x: visualize(x.long(), **(params | {"is_mask": True}))
            if target == "image_rescale":
                return lambda x: visualize((x - x.min()) / (x.max() - x.min()), **params)
            if target == "image_and_mask":
                return lambda x: combine_mask_and_im_v2(x, **params)
        
        self.keep_queue_tr = Queue(self.max_images)
        self.keep_queue_val = Queue(self.max_images)
        self.keep_queue_te = Queue(self.max_images)
        
        self.logger = {}
        for name, val in logger.items():
            self.logger[name] = _get_logger(val["target"], val.get("params", {}))
            
    @rank_zero_only
    def _wandb(self, pl_module, images, batch_idx, split):
        for k in images:
            _im = images[k]
            pl_module.logger.experiment.log({f"{split}_{k}": [wandb.Image(_im)]})
            
    @rank_zero_only
    def _board(self, pl_module, images, batch_idx, split):
        for k in images:
            _im = images[k]
            pl_module.logger.experiment.add_image(f"{split}_{k}", _im.transpose(2, 0, 1), batch_idx)
            
    @staticmethod
    def _maybe_mkdir(path):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        return path
    
    def _enqueue_and_dequeue(self, entry, split="train"):
        keep_q = self.choose_on_split(split, self.keep_queue_tr, self.keep_queue_val, self.keep_queue_te)
        if keep_q.full():
            to_remove = keep_q.get_nowait()
            os.remove(to_remove)
        keep_q.put_nowait(entry)

    @rank_zero_only
    def log_local(self, save_dir, split, images, global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "images", split)
        self._maybe_mkdir(root)
        filename = "gs-{:06}_e-{:06}_b-{:06}.png".format(
                global_step,
                current_epoch,
                batch_idx)
        path = os.path.join(root, filename)
        if self.log_separate:
            path = lambda x: os.path.join(self._maybe_mkdir(os.path.join(root, x)), filename)
            local_images = image_logger(images, path, n_grid_images=16, log_separate=True, **self.logger)
            for k in local_images.keys(): self._enqueue_and_dequeue(path(k), split)
        else:
            local_images = image_logger(images, path, n_grid_images=16, log_separate=False, **self.logger)
            self._enqueue_and_dequeue(path, split)
        return local_images

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step
        if (self.check_frequency(check_idx, split) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images)):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()
            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)

            for k in images:
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()

            local_images = self.log_local(pl_module.logger.save_dir, split, images,
                                          pl_module.global_step, pl_module.current_epoch, batch_idx)
            logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)
            logger_log_images(pl_module, local_images, batch_idx, split)

            if is_train:
                pl_module.train()
                
    def log_gradients(self, trainer, pl_module, batch_idx, split):
        # log cam
        raise NotImplementedError("CAM not yet implemented")
    
    def choose_on_split(self, split, tr, val, te):
        if split == 'train': return tr
        elif split == 'val': return val
        elif split == 'validation': return val
        elif split == 'test': return te
        else: raise RuntimeError("split should be one of train / val / test")

    def check_frequency(self, check_idx, split="train"):
        log_steps = self.choose_on_split(split, self.log_steps_tr, self.log_steps_val, self.log_steps_te)
        batch_freq = self.choose_on_split(split, self.batch_freq_tr, self.batch_freq_val, self.batch_freq_te)
        if ((check_idx % batch_freq) == 0 or (check_idx in log_steps)):
            # try:
            #     log_steps.pop(0)
            # except IndexError as e:
            #     print(e)
            #     pass
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.is_training and (pl_module.global_step > 0 or self.log_first_step):
            self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.is_training and pl_module.global_step > 0:
            self.log_img(pl_module, batch, batch_idx, split="val")

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled:
            self.log_img(pl_module, batch, batch_idx, split="test")
            
    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled:
            self.log_img(pl_module, batch, batch_idx, split="test")


class CUDACallback(Callback):
    # see https://github.com/SeanNaren/minGPT/blob/master/mingpt/callback.py
    def on_train_epoch_start(self, trainer, pl_module):
        # Reset the memory use counter
        torch.cuda.reset_peak_memory_stats(trainer.root_gpu)
        torch.cuda.synchronize(trainer.root_gpu)
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
    def on_train_epoch_end(self, trainer, pl_module):
        torch.cuda.synchronize(trainer.root_gpu)
        max_memory = torch.cuda.max_memory_allocated(trainer.root_gpu) / 2 ** 20
        epoch_time = time.time() - self.start_time

        try:
            max_memory = trainer.training_type_plugin.reduce(max_memory)
            epoch_time = trainer.training_type_plugin.reduce(epoch_time)

            rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
            rank_zero_info(f"Average Peak memory {max_memory:.2f}MiB")
        except AttributeError:
            pass


if __name__ == "__main__":
    sys.path.append(os.getcwd())

    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)

    opt, unknown = parser.parse_known_args()
    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            # idx = len(paths)-paths[::-1].index("logs")+1
            # logdir = "/".join(paths[:idx])
            logdir = "/".join(paths[:-2])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

        opt.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs + opt.base
        _tmp = logdir.split("/")
        nowname = _tmp[-1]
    else:
        if opt.name:
            name = opt.name
        elif opt.base:
            cfg_fname = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = cfg_name
        else:
            name = ""
        nowname = name + opt.postfix
        logdir = os.path.join(opt.logdir, nowname)

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    seed_everything(opt.seed)

    try:
        # init and save configs
        configs = [OmegaConf.load(cfg) for cfg in opt.base]
        cli = OmegaConf.from_dotlist(unknown)
        config = OmegaConf.merge(*configs, cli)
        lightning_config = config.pop("lightning", OmegaConf.create())
        # merge trainer cli with config
        trainer_config = lightning_config.get("trainer", OmegaConf.create())
        # default to ddp
        trainer_config["strategy"] = "ddp"
        trainer_config["accelerator"] = "gpu"
        for k in nondefault_trainer_args(opt):
            trainer_config[k] = getattr(opt, k)
        if not "gpus" in trainer_config:
            del trainer_config["strategy"], trainer_config["accelerator"]
            cpu = True
        else:
            gpuinfo = trainer_config["gpus"]
            print(f"Running on GPUs {gpuinfo}")
            trainer_config["devices"] = len(re.sub(r"[^0-9]+", "", gpuinfo))
            cpu = False
        trainer_opt = argparse.Namespace(**trainer_config)
        lightning_config.trainer = trainer_config

        # model
        model = instantiate_from_config(config.model)

        # trainer and callbacks
        trainer_kwargs = dict()

        # default logger configs
        default_logger_cfgs = {
            "wandb": {
                "target": "pytorch_lightning.loggers.WandbLogger",
                "params": {
                    'project': "ldm",
                    "name": nowname.replace("/", "_"),
                    "save_dir": logdir,
                    "offline": opt.debug,
                    "id": datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
                }
            },
            "tensorboard": {
                "target": "pytorch_lightning.loggers.TensorBoardLogger",
                "params": {
                    "save_dir": logdir,
                    "name": nowname,
                    "version": "tensorboard",
                }
            }
        }
        default_logger_cfg = default_logger_cfgs["wandb"] if not opt.debug else default_logger_cfgs["tensorboard"]
        if "logger" in lightning_config:
            logger_cfg = lightning_config.logger
        else:
            logger_cfg = OmegaConf.create()
        logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
        trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

        # modelcheckpoint - use TrainResult/EvalResult(checkpoint_on=metric) to
        # specify which metric is used to determine best models
        default_modelckpt_cfg = {
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": ckptdir,
                "filename": "{epoch:06}",
                "verbose": True,
                "save_last": True,
            }
        }
        if hasattr(model, "monitor"):
            print(f"Monitoring {model.monitor} as checkpoint metric.")
            default_modelckpt_cfg["params"]["monitor"] = model.monitor
            default_modelckpt_cfg["params"]["save_top_k"] = 3

        if "modelcheckpoint" in lightning_config:
            modelckpt_cfg = lightning_config.modelcheckpoint
        else:
            modelckpt_cfg =  OmegaConf.create()
        modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)
        print(f"Merged modelckpt-cfg: \n{modelckpt_cfg}")
        if version.parse(pl.__version__) < version.parse('1.4.0'):
            trainer_kwargs["checkpoint_callback"] = instantiate_from_config(modelckpt_cfg)

        # add callback which sets up log directory
        default_callbacks_cfg = {
            "setup_callback": {
                "target": "main.SetupCallback",
                "params": {
                    "resume": opt.resume,
                    "now": datetime.datetime.now().strftime("%Y%m%d"),
                    "logdir": logdir,
                    "ckptdir": ckptdir,
                    "cfgdir": cfgdir,
                    "config": config,
                    "lightning_config": lightning_config,
                }
            },
            "learning_rate_logger": {
                "target": "main.LearningRateMonitor",
                "params": {
                    "logging_interval": "step",
                    # "log_momentum": True
                }
            },
            "cuda_callback": {
                "target": "main.CUDACallback"
            },
        }
        if version.parse(pl.__version__) >= version.parse('1.4.0'):
            default_callbacks_cfg.update({'checkpoint_callback': modelckpt_cfg})

        if "callbacks" in lightning_config:
            callbacks_cfg = lightning_config.callbacks
        else:
            callbacks_cfg = OmegaConf.create()

        if 'metrics_over_trainsteps_checkpoint' in callbacks_cfg:
            print(
                'Caution: Saving checkpoints every n train steps without deleting. This might require some free space.')
            default_metrics_over_trainsteps_ckpt_dict = {
                'metrics_over_trainsteps_checkpoint':
                    {"target": 'pytorch_lightning.callbacks.ModelCheckpoint',
                     'params': {
                         "dirpath": os.path.join(ckptdir, 'trainstep_checkpoints'),
                         "filename": "{epoch:06}-{step:09}",
                         "verbose": True,
                         'save_top_k': -1,
                         'every_n_train_steps': 10000,
                         'save_weights_only': True
                     }
                     }
            }
            default_callbacks_cfg.update(default_metrics_over_trainsteps_ckpt_dict)

        callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
        if 'ignore_keys_callback' in callbacks_cfg and hasattr(trainer_opt, 'resume_from_checkpoint'):
            callbacks_cfg.ignore_keys_callback.params['ckpt_path'] = trainer_opt.resume_from_checkpoint
        elif 'ignore_keys_callback' in callbacks_cfg:
            del callbacks_cfg['ignore_keys_callback']

        trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]

        trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)
        trainer.logdir = logdir  ###

        # data
        data = instantiate_from_config(config.data)
        # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
        # calling these ourselves should not be necessary but it is.
        # lightning still takes care of proper multiprocessing though
        data.prepare_data()
        data.setup()
        print("#### Data #####")
        # for k in data.datasets:
        #     print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")

        # configure learning rate
        bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
        if not cpu:
            ngpu = len(lightning_config.trainer.gpus.strip(",").split(','))
        else:
            ngpu = 1
        if 'accumulate_grad_batches' in lightning_config.trainer:
            accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches
        else:
            accumulate_grad_batches = 1
        print(f"accumulate_grad_batches = {accumulate_grad_batches}")
        lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
        if opt.scale_lr:
            model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
            print(
                "Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
                    model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr))
        else:
            model.learning_rate = base_lr
            print("++++ NOT USING LR SCALING ++++")
            print(f"Setting learning rate to {model.learning_rate:.2e}")


        # allow checkpointing via USR1
        def melk(*args, **kwargs):
            # run all checkpoint hooks
            if trainer.global_rank == 0:
                print("Summoning checkpoint.")
                ckpt_path = os.path.join(ckptdir, "last.ckpt")
                trainer.save_checkpoint(ckpt_path)


        def divein(*args, **kwargs):
            if trainer.global_rank == 0:
                import pdb;
                pdb.set_trace()


        import signal

        signal.signal(signal.SIGUSR1, melk)
        signal.signal(signal.SIGUSR2, divein)
        
        # create necessary folders
        os.makedirs(ckptdir, exist_ok=True)
        os.makedirs(cfgdir, exist_ok=True)
        os.makedirs(os.path.join(logdir, "wandb"), exist_ok=True)

        # run
        if opt.train:
            try:
                trainer.fit(model, data)
            except Exception:
                melk()
                raise
        else:
            trainer.test(model, data)
        # if not opt.no_test and not trainer.interrupted:
        #     trainer.test(model, data)
    except Exception:
        # if opt.debug and trainer.global_rank == 0:
        #     try:
        #         import pudb as debugger
        #     except ImportError:
        #         import pdb as debugger
        #     debugger.post_mortem()
        raise
    finally:
        # move newly created debug project to debug_runs
        # if opt.debug and not opt.resume and trainer.global_rank == 0:
        #     dst, name = os.path.split(logdir)
        #     dst = os.path.join(dst, "debug_runs", name)
        #     os.makedirs(os.path.split(dst)[0], exist_ok=True)
        #     os.rename(logdir, dst)
        if trainer.global_rank == 0:
            print(trainer.profiler.summary())
