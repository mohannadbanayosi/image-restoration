# TODO: fix ordering of the imports
import time
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from tqdm import tqdm
from dataset import calculate_psnr, get_batch, prepare_dataset, transform_input
from model import DenoisingAutoencoder
from torchmetrics.image import StructuralSimilarityIndexMeasure

# TODO: add to HP: degree of noise
# Hyperparameters
batch_size = 64
learning_rate = 0.00001
max_iters = 20000
eval_interval = int(max_iters/20)
eval_iters = 20

torch.manual_seed(1337)

device = "mps" if torch.backends.mps.is_available() else "cpu"
model = DenoisingAutoencoder().to(device)
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

# TODO: check possibility to switch to perceptual or adversarial loss
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

# image_dataset = datasets.ImageFolder(root="IMAGES_PATH", transform=transform_input)
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_input)
val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_input)

# train_size = int(0.9 * len(image_dataset))
# val_size = len(image_dataset) - train_size
# train_dataset, val_dataset = random_split(image_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
print(f"{len(train_loader.dataset)=}", f"{len(test_loader.dataset)=}")


@torch.no_grad()
def calculate_batch_psnr(batch_input_images, output_image):
    psnrs = torch.zeros(batch_input_images.size(0))
    for i in range(len(batch_input_images)):
        input_img_np = batch_input_images[i].cpu().numpy().transpose((1, 2, 0))
        output_img_np = output_image[i].cpu().numpy().transpose((1, 2, 0))
        psnrs[i] = calculate_psnr(input_img_np, output_img_np)
    return psnrs.mean()


plots = {
        "loss": {
            "training": [],
            "validation": []
        },
        "psnr": {
            "training": [],
            "validation": []
        },
        "ssim": {
            "training": [],
            "validation": []
        }
    }

def train_model(model, dataloader, valloader, criterion, optimizer, num_epochs=25):
    # TODO: use already implemented function to add noise instead of duplicating the code here
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_loss_val = 0.0
        running_psnrs = 0.0
        running_psnrs_val = 0.0
        running_ssim = 0.0
        running_ssim_val = 0.0
        for inputs, _ in tqdm(dataloader):
            inputs = inputs.to(device)
            noisy_inputs = inputs + 0.1 * torch.randn_like(inputs)
            noisy_inputs = torch.clamp(noisy_inputs, 0., 1.)

            optimizer.zero_grad()
            outputs = model(noisy_inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()

            ssim_value = ssim(outputs, inputs)
            
            running_loss += loss.item() * inputs.size(0)
            psnrs = calculate_batch_psnr(inputs, outputs)
            running_psnrs += psnrs * inputs.size(0)
            running_ssim += ssim_value.item() * inputs.size(0)

        
        for inputs, _ in tqdm(valloader):
            model.eval()
            inputs = inputs.to(device)
            noisy_inputs = inputs + 0.1 * torch.randn_like(inputs)
            noisy_inputs = torch.clamp(noisy_inputs, 0., 1.)

            outputs = model(noisy_inputs)
            loss = criterion(outputs, inputs)

            ssim_value = ssim(outputs, inputs)
            
            running_loss_val += loss.item() * inputs.size(0)
            psnrs = calculate_batch_psnr(inputs, outputs)
            running_psnrs_val += psnrs * inputs.size(0)
            running_ssim_val += ssim_value.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_loss_val = running_loss_val / len(valloader.dataset)
        epoch_psnr = running_psnrs / len(dataloader.dataset)
        epoch_psnr_val = running_psnrs_val / len(valloader.dataset)
        epoch_ssim = running_ssim / len(dataloader.dataset)
        epoch_ssim_val = running_ssim_val / len(valloader.dataset)
        print(f'Epoch {epoch}/{num_epochs - 1}, Loss: {epoch_loss:.4f}, PSNR: {epoch_psnr:.4f}, SSIM: {epoch_ssim:.4f}')
        print(f'Epoch val {epoch}/{num_epochs - 1}, Loss: {epoch_loss_val:.4f}, PSNR: {epoch_psnr_val:.4f}, SSIM: {epoch_ssim_val:.4f}')
        plots["loss"]["training"].append(epoch_loss)
        plots["loss"]["validation"].append(epoch_loss_val)
        plots["psnr"]["training"].append(epoch_psnr)
        plots["psnr"]["validation"].append(epoch_psnr_val)
        plots["ssim"]["training"].append(epoch_ssim)
        plots["ssim"]["validation"].append(epoch_ssim_val)
    return epoch_loss_val

if __name__ == '__main__':
    final_val_loss = train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=42)

    print("Final Training Loss:", final_val_loss)

    timestamp = int(time.time())

    for plot_type in ("loss", "psnr", "ssim"):
        y_training = plots[plot_type]["training"]
        y_validation = plots[plot_type]["validation"]
        plt.plot(y_training, label = "Training")
        plt.plot(y_validation, label = "Validation")
        plt.ylabel(plot_type)
        plt.legend()
        plt.savefig(f"graphs/{timestamp}_{plot_type}.png")
        plt.clf()

    final_loss = "{:.6f}".format(final_val_loss).replace(".", "")
    # TODO: add to model name: loss and degree of noise
    # TODO: write metadata file with HPs
    model_resource_path = f"model_resources/model_image_{timestamp}_{0}.pth"
    print(f"Saving model under {model_resource_path}")
    torch.save(model.state_dict(), model_resource_path)
