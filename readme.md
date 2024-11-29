3D version of "High-Resolution Image Synthesis with Latent Diffusion Models" or "Stable Diffusion"
- prepare env:
`conda env create -f environment.yml`
- train command:
`torchrun --nproc-per-node=<ngpus> main.py --base <config file path> -t --name <experiment name>`
- test command:
`python main.py --base <config file path> --name <experiment name>`, which should be just the train command minus `-t` flag

 before train/inference process, compile a config file, which consists of three major parts: `model`, `data` and other lightning-related modules.

 Firstly, in `model`, there is
```
base_learning_rate: 1.5e-4    
```
specifying the BASE learning rate for training model, which may be changed by pytorch-lightning  following specific batch size and GPU number of choice
```
train_target: TARGET_TRAIN
test_target:  TARGET_TEST
params:       **PARAM_DICT
```
where each target name should be the module import path relative to the root directory (`./`), *e.g.* a train target of `ldm.models.ldm.ddpm.DDPM` will instantiate `DDPM` module in `./ldm/models/ldm/ddpm.py` for training. Pass the parameters in the params part in a key: value format
```
test_only_params: **PARAM_DICT
```
this param dict is used to assign extra parameters in the TARGET_TEST module (which should inherit the TRAIN_TARGET), mostly the checkpoint path to trained model and others related to saving the generated images

Secondly, in `data`, there generally is not much to change and the parameters should be self-explanatory
```
target: main.DataModuleFromConfig
params:
  batch_size: $bs
  num_workers: $nw
  train: 
    target: ldm.data.base.SimpleDataset
    params: **PARAM_DICT
  validation:
    ...
  test:
    ...
```
which uses the module `SimpleDataset` in `./ldm/data/base.py`, a wrapper class of `CacheDataset` from `monai`

Finally, you can pass whatever arguments supported by pytorch-lightning trainer either as command flags or in the `trainer` section of the config file
```
accumulate_grad_batches: 1
```
specifies that pl should accumulate model gradient per batch
```
max_epochs: 500
```
specifies that the training process lasts for 500 epochs. 
```
limit_test_batches: 100
```
Remember to add this line to force the number (100 here) of samples generated at inference. Demo config files can be found under `config/` folder


# References
Refer to the following directories for more details
- [Latent Diffusion](https://github.com/CompVis/latent-diffusion)
- [taming-transformers](https://github.com/CompVis/taming-transformers)