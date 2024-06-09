import os
import random
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


random.seed(42)

training_dataset_index = 0
epoch = 0

transform_input = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

def load_image(image_path):
    image = Image.open(image_path)
    image = transform_input(image)
    return image


def introduce_noise(input_image, device, noise_factor=0.3):
    noisy_input_image = input_image + noise_factor * torch.randn(input_image.size()).to(device)
    noisy_input_image = torch.clamp(noisy_input_image, 0, 1).to(device)
    return noisy_input_image


def calculate_psnr(base_image, modified_image):
    """Peak Signal-to-Noise Ratio"""
    mse = np.mean((base_image - modified_image) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr
