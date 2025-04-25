# image-restoration

A simple model to restore images from noise - and in the future from "out of focus" effects.

## Denoising example

### Noise Level σ = 50

#### Landscape Model

Noisy

![Noisy Image](images/image_noised.jpg)

Denoised

![Denoised Image](images/image_denoised.jpg)

Original

![Original Image](images/image_original.jpeg)

#### BSDS300 model

| Noisy | Denoised | Original |
|-------|----------|----------|
| ![Noisy Image](images/image_bsds300_noised.jpg) | ![Denoised Image](images/image_bsds300_denoised.jpg) | ![Original Image](images/image_bsds300_original.jpg) |

## Benchmark Results

RED30 comes from the paper [Xiao-Jiao Mao et al., 2016](https://arxiv.org/pdf/1606.08921).

### SSIM

| Noise Level | RED30 | DA_v1 |
|-------------|-------|-------|
| σ = 30      | 27.95 | 33.77 |
| σ = 50      | 25.75 | 31.47 |

### PSNR

| Noise Level | RED30  | DA_v1 |
|-------------|--------|-------|
| σ = 30      | 0.8019 | 0.9287 |
| σ = 50      | 0.7167 | 0.8911 |

## Training on noise level σ = 50

Green (1745518710) is trained on BSDS300 and pink (1745509286) is trained on a local dataset of landscape images.

#### SSIM

![train/ssim](graphs/50_train_ssim.png)
![val/ssim](graphs/50_val_ssim.png)

#### PSNR

![train/psnr](graphs/50_train_psnr.png)
![val/psnr](graphs/50_val_psnr.png)

#### Loss

![train/loss](graphs/50_train_loss.png)
![val/loss](graphs/50_val_loss.png)
