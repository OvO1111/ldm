# Train command
`torchrun --nproc_per_node $N_GPU main.py --base $CFG_FILE -t --name $EXP_NAME --gpus 0,1...`

## Stage -1: prepare environment
```
# fork this repository to your github
git clone $GIT_ADDR_TO_YOUR_REPO ./latentdiffusion
cd ./diffusion
pip install -r requirements.txt
```

## Stage 0: write config file
config files resides under `configs/`
- autoencoder config files:
```
model:
** take KL-AE as an example **
  base_learning_rate: $LR
  target: ldm.models.autoencoder.AutoencoderKL
  params:
    ** ldm.models.autoencoder.AutoencoderKL **'s keyword arguments
    lossconfig:
      target: ldm.modules.losses.LPIPSWithDiscriminator
      params:
        ...

    ddconfig:
      ** ldm.modules.diffusionmodules.model.Encoder/Decoder **'s keyword arguments
      ...
    use_checkpoint: true          # use checkpoint to save GPU memory

data:
** take BraTS21 as an example **
  target: main.DataModuleFromConfig
  params:
    batch_size: 1
    num_workers: 2
    wrap: False
    train:
      target: ldm.data.brats2021.BraTS2021_3D
      params:
        split: train
        crop_to: [64, 64, 64]
    validation:
      target: ldm.data.brats2021.BraTS2021_3D
      params:
        split: val
        crop_to: [64, 64, 64]

-- logging arguments and pytorchlightning callbacks --
...
```
- latentdiffusion config files:
```
model:
  base_learning_rate: $LR
  target: ldm.models.diffusion.ddpm.LatentDiffusion
  params:
    ...
    first_stage_key: image      # data key
    cond_stage_key: mask        # condition key
    conditioning_key: concat    # conditional type
    image_size: [8, 16, 16]     # after first-stage encoding
    channels: 4                 # after first-stage encoding
    ...

    unet_config:
      ** diffusion UNet config **
      target: ldm.modules.diffusionmodules.openaimodel.UNetModel
      params:
        image_size: 64  # not used
        ...
        use_checkpoint: True    # always use checkpoint to save GPU memory

    first_stage_config:
      ** your autoencoder setting, copy from autoencoder's config, with loss=nn.Identity **
      ckpt_path: /path/to/your/pretrained/ae
      ...

    cond_stage_config: 
      ** use __is_unconditional__ if running without conditions, else follow the same format as above **
      ...

data:
  ** same as in autoencoder config **
  ...

-- logging arguments and pytorchlightning callbacks --
...
```

## Stage 1: train autoencoder
 ##### For training autoencoder using KL regularization, the code workflow is: 
- `main.py` ( trainer function ) -> 
- `ldm/models/autoencoder.py` ( autoencoder wrapper ) -> 
- `ldm/modules/losses/contperceptual.py` ( LPIPS and GAN loss ) & `ldm/modules/diffusionmodules/model.py` ( autoencoder model class )

 ##### For training autoencoder using Vector Quantization, the code workflow is: 
 - `main.py` ( trainer function ) -> 
- `ldm/models/autoencoder.py` ( autoencoder wrapper ) -> 
- `ldm/modules/losses/vqperceptual.py` ( LPIPS and codebook loss ) & `ldm/modules/diffusionmodules/model.py` ( autoencoder model class )

## Stage 2: train diffusion model
##### For training latentdiffusion model, the code workflow is:
- `main.py` ( trainer function ) -> 
- `ldm/models/diffusion/ddpm.py` ( diffusion wrapper ) ->
- `ldm/modules/diffusionmodules/openaimodel.py` ( diffusion UNet )
- `ldm/models/diffusion/ddim.py` is for fast reverse sampling using DDIM


# Inference command
`python inference/sample.py -r $MODEL_CKPT --batch_size 1` 
- the `batch_size 1` is mandatory as per my use cases

# References
Refer to the following directories for more details
- [Latent Diffusion](https://github.com/CompVis/latent-diffusion)
- [taming-transformers](https://github.com/CompVis/taming-transformers)