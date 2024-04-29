import os
import random
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


transform_input = transforms.Compose([
    transforms.Resize((16, 16)),
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

def prepare_dataset(device, base_path):
    dataset = os.listdir(base_path)
    training_dataset = []

    with tqdm(total=len(dataset), desc=f"loading files") as pbar:
        for file_name in dataset:
            if file_name.split(".")[-1] in ("JPG", "jpeg"):
                image_path = f"{base_path}/{file_name}"
                input_image = load_image(image_path).to(device)
                noisy_input_image = introduce_noise(input_image, device)
                training_dataset.append((input_image, noisy_input_image))
            pbar.update(1)

    print(f"{len(training_dataset)=}")
    return training_dataset

def get_batch(batch_size, training_dataset):
    original_images = []
    noisy_images = []
    for _ in range(batch_size):
        image_name = random.choice(training_dataset)
        input_image, noisy_input_image = image_name

        original_images.append(input_image)
        noisy_images.append(noisy_input_image)
    return original_images, noisy_images
