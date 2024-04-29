import torch
from PIL import Image
from torchvision import transforms


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
