import torch
from dataset import calculate_psnr, introduce_noise, load_image
from model import DenoisingAutoencoder
from torchvision import transforms
from torchmetrics.image import StructuralSimilarityIndexMeasure

torch.manual_seed(43)

# device = "mps" if torch.backends.mps.is_available() else "cpu"
device = "mps"

base_path = "IMAGES_PATH"
file_name = "FILE_NAME"
image_path = f"{base_path}/{file_name}.jpeg"
input_image, original_size = load_image(image_path)
original_width, original_height = original_size
input_image = input_image.unsqueeze(0).to(device)

noisy_input_image = introduce_noise(input_image, device, 50)

transform_output = transforms.Compose([
    transforms.Resize((original_height, original_width)),
    transforms.ToPILImage()
])

model_directory = "PATH"
model_path = "MODEL"
output_path = f'{base_path}/{file_name}_noised_{model_path.split(".")[0]}.jpg'
noisy_image_resized = noisy_input_image.squeeze(0)
transform_output(noisy_image_resized).save(output_path)

model = DenoisingAutoencoder().to(device)

print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

print(f"Loading model from {model_directory}/{model_path}")
model.load_state_dict(torch.load(f"{model_directory}/{model_path}"))
model.eval()

with torch.no_grad():
    model.eval()
    output_image = model(noisy_input_image)

output_image_resized = output_image.squeeze(0)

input_img_np = input_image.cpu().squeeze(0).numpy().transpose((1, 2, 0))
output_img_np = output_image.cpu().squeeze(0).numpy().transpose((1, 2, 0))
ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
psnr = calculate_psnr(input_img_np, output_img_np)
print(f"Image: PSNR = {psnr:.2f}")
ssim_value = ssim(input_image, output_image)
print(f"Image: SSIM = {ssim_value.item():.2f}")

output_path = f'{base_path}/{file_name}_denoised_{model_path.split(".")[0]}.jpg'
transform_output(output_image_resized).save(output_path)

print("Denoised output image saved at:", output_path)
