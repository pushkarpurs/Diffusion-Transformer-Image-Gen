# CelebHQ Diffusion Transformer Image Generator

This project implements a **Diffusion Transformer (DiT)**-based image generator trained on the [CelebHQ Resized 256x256 dataset](https://www.kaggle.com/datasets/badasstechie/celebahq-resized-256x256). Unlike the original DiT paper, which incorporates Variational Autoencoders (VAEs), this project adopts a **U-Net-style encoder-decoder** architecture with convolutional upsampling and downsampling layers and skip connections for mapping to and from the latent space.

## Key Features

- **Diffusion-based Generation**: Uses denoising diffusion probabilistic models (DDPM) to iteratively generate images.
- **Transformer Blocks**: Employs DiT blocks inspired by [DiT: Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748) to model global interactions in the latent space.
- **No VAEs**: Replaces VAE-based encoders/decoders with convolutional networks to simplify the architecture and enable end-to-end training.
- **Skip Connections**: Similar to U-Net, skip connections allow effective gradient flow and preserve spatial detail in the latent-to-image reconstruction.

## Dataset

- **Dataset**: [CelebAHQ Resized 256x256](https://www.kaggle.com/datasets/badasstechie/celebahq-resized-256x256)
- **Description**: High-quality aligned and cropped celebrity face images at 256x256 resolution.
- **Preprocessing**: Normalized to `[-1, 1]`, augmented optionally with random horizontal flips.

## Model Architecture

### 1. **Encoder and Decoder**

- Convolutional downsampling layers encode the input image into a latent representation.
- The decoder upsamples this latent representation back into the image space.
- Skip connections connect the encoder and decoder across layers to preserve spatial features.

### 2. **DiT Blocks**

- Transformer blocks operate on the latent representation at each timestep.
- Positional encodings and timestep embeddings are used to condition the model on diffusion steps.

### 3. **Diffusion Process**

- The forward process adds noise to the image across `T` steps.
- The reverse process denoises step-by-step, guided by the trained model's predictions.

## Training Details

- **Framework**: PyTorch
- **Loss Function**: Mean Squared Error (MSE) between predicted and actual noise
- **Optimizer**: AdamW
- **Scheduler**: Linear beta schedule
- **Timesteps**: 1000
- **Latent Space**: The images are converted from (B,3,256,256) --> (B,4,32,32) (Dimentions of each channel downsampled by a factor of 8 to reduce VRAM usage) 


### Download the Model
```Bash
wget https://huggingface.co/PushkarUrs/Diffusion-Transformer-Image-Gen/resolve/main/dit_final_12_f_250.pth
