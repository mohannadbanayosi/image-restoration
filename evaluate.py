import torch
from dataset import calculate_psnr, introduce_noise, load_image
from model import DenoisingAutoencoder
from torchvision import transforms


# device = "mps" if torch.backends.mps.is_available() else "cpu"
device = "cpu"

base_path = "IMAGES_PATH"
image_path = f"{base_path}/IMAGE_NAME.JPG"
input_image, original_size = load_image(image_path)
original_width, original_height = original_size
input_image = input_image.unsqueeze(0).to(device)

noisy_input_image = introduce_noise(input_image, device)

transform_output = transforms.Compose([
    transforms.Resize((original_height, original_width)),
    transforms.ToPILImage()
])

output_path = f'{base_path}/output_noised.jpg'
noisy_image_resized = noisy_input_image.squeeze(0)
transform_output(noisy_image_resized).save(output_path)

model = DenoisingAutoencoder().to(device)

print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

model_directory = "/Users/mohannadalbanayosy/Projects/mohannadbanayosi/image-restoration/model_resources"
model_path = "1718556848.pth"
print(f"Loading model from {model_directory}/{model_path}")
model.load_state_dict(torch.load(f"{model_directory}/{model_path}"))
model.eval()

with torch.no_grad():
    model.eval()
    output_image = model(noisy_input_image)

output_image_resized = output_image.squeeze(0)

input_img_np = input_image.squeeze(0).numpy().transpose((1, 2, 0))
output_img_np = output_image.squeeze(0).numpy().transpose((1, 2, 0))
psnr = calculate_psnr(input_img_np, output_img_np)
print(f"Image: PSNR = {psnr:.2f}")

output_path = f'{base_path}/output_denoised.jpg'
transform_output(output_image_resized).save(output_path)

print("Denoised output image saved at:", output_path)
