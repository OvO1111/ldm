3D version of "High-Resolution Image Synthesis with Latent Diffusion Models" or "Stable Diffusion"

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
```
# instantiate training module
model:
  train_target: ldm.models.ddpm.LatentDiffusion
  test_target: inference.models.InferLatentDiffusion
  test_only_params:
    save_dataset: true
    save_dataset_path: ...
    suffix_keys: 
      samples: .nii.gz
  params:
    ...

# instantiate dataset
data:
  target: main.DataModuleFromConfig
  params:
    batch_size: 1
    train:
      target: ldm.data.brats2021.BraTS2021_3D
      params:
        split: train
        ...

# instantiate lightning logger
lightning:
  callbacks:
    image_logger:
      target: main.ImageLogger
      params:
        train_batch_frequency: 200
        max_images: 20

# instantiate trainer
trainer:
  benchmark: true
  max_epochs: 1000
  limit_test_batches: 100
  resume_from_checkpoint: null
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
`python main.py --base $CFG_FILE --name $EXP_NAME --gpus 0,` 
- CFG_FILE is specified like that in training cmd

# References
Refer to the following directories for more details
- [Latent Diffusion](https://github.com/CompVis/latent-diffusion)
- [taming-transformers](https://github.com/CompVis/taming-transformers)