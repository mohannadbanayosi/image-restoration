# TODO: fix ordering of the imports
import json
import time
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from tqdm import tqdm
import wandb
from dataset import calculate_psnr, transform_input
from model import DenoisingAutoencoder
from torchmetrics.image import StructuralSimilarityIndexMeasure
from loss import CombinedLoss

# Hyperparameters and config
batch_size = 32
learning_rate = 2.5e-4
num_workers = 1
num_epochs = 60
noise_level = 0.3
weight_decay = 2.5e-4
patience = 10
min_lr = 1e-6
alpha = 0.3
wandb_logging = False

# Initialize wandb config
wandb_config = {
    "batch_size": batch_size,
    "learning_rate": learning_rate,
    "num_workers": num_workers,
    "num_epochs": num_epochs,
    "noise_level": noise_level,
    "weight_decay": weight_decay,
    "patience": patience,
    "min_lr": min_lr,
    "alpha": alpha,
    "architecture": "DenoisingAutoencoder",
    "dataset": "local/mountain",
    "optimizer": "AdamW",
    "loss_function": "CombinedLoss",
    "scheduler": "ReduceLROnPlateau"
}

@torch.no_grad()
def calculate_batch_metrics(batch_input_images, output_images, ssim_metric):
    """Calculate PSNR and SSIM for a batch of images"""
    batch_size = batch_input_images.size(0)
    psnrs = torch.zeros(batch_size)
    ssims = torch.zeros(batch_size)
    
    for i in range(batch_size):
        input_img_np = batch_input_images[i].cpu().numpy().transpose((1, 2, 0))
        output_img_np = output_images[i].cpu().numpy().transpose((1, 2, 0))
        psnrs[i] = calculate_psnr(input_img_np, output_img_np)
        ssims[i] = ssim_metric(
            batch_input_images[i:i+1], 
            output_images[i:i+1]
        ).item()
    
    return psnrs.mean().item(), ssims.mean().item()

def save_metadata(architecture_info, filename="config_metadata.json"):
    # TODO: improve and get rid of hard-coded values where possible
    config = {
        "dataset": {
            "name": "local/mountain",
            "root_dir": "/data",
            "split": "train",
            "transform": {}
        },
        "dataloader": {
            "batch_size": batch_size,
            "shuffle": True,
            "num_workers": num_workers
        },
        "model": {
            "architecture": "DenoisingAutoencoder",
            "noise_level": noise_level,
            "architecture_info": architecture_info
        },
        "training": {
            "num_epochs": num_epochs,
            "loss_function": "MSELoss",
            "optimizer": "Adam",
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "patience": patience,
            "min_lr": min_lr,
            "alpha": alpha
        },
        "device": str(device)
    }
    with open(f"model_resources/{filename}", 'w') as f:
        json.dump(config, f, indent=4)


def save_final(final_res, filename="config_metrics.json"):
    with open(f"model_resources/{filename}", 'w') as f:
        json.dump(final_res, f, indent=4)

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

def train_model(model, dataloader, valloader, criterion, optimizer, scheduler, num_epochs=25):
    # TODO: use already implemented function to add noise instead of duplicating the code here
    if wandb_logging:
        wandb_config["architecture_info"] = model.get_model_architecture()
        wandb.init(
            project="image-restoration",
            config=wandb_config,
            name=f"denoising_autoencoder_{int(time.time())}"
        )
        wandb.watch(model, log="all")

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_loss_val = 0.0
        running_psnr = 0.0
        running_psnr_val = 0.0
        running_ssim = 0.0
        running_ssim_val = 0.0
        
        for inputs, _ in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            inputs = inputs.to(device)
            noisy_inputs = inputs + noise_level * torch.randn_like(inputs)
            noisy_inputs = torch.clamp(noisy_inputs, 0., 1.)

            optimizer.zero_grad()
            outputs = model(noisy_inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()

            batch_psnr, batch_ssim = calculate_batch_metrics(inputs, outputs, ssim)
            running_loss += loss.item() * inputs.size(0)
            running_psnr += batch_psnr * inputs.size(0)
            running_ssim += batch_ssim * inputs.size(0)

        model.eval()
        with torch.no_grad():
            for inputs, _ in tqdm(valloader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                inputs = inputs.to(device)
                noisy_inputs = inputs + noise_level * torch.randn_like(inputs)
                noisy_inputs = torch.clamp(noisy_inputs, 0., 1.)

                outputs = model(noisy_inputs)
                loss = criterion(outputs, inputs)

                batch_psnr, batch_ssim = calculate_batch_metrics(inputs, outputs, ssim)
                running_loss_val += loss.item() * inputs.size(0)
                running_psnr_val += batch_psnr * inputs.size(0)
                running_ssim_val += batch_ssim * inputs.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_loss_val = running_loss_val / len(valloader.dataset)
        epoch_psnr = running_psnr / len(dataloader.dataset)
        epoch_psnr_val = running_psnr_val / len(valloader.dataset)
        epoch_ssim = running_ssim / len(dataloader.dataset)
        epoch_ssim_val = running_ssim_val / len(valloader.dataset)

        if wandb_logging:
            wandb.log({
                "train/loss": epoch_loss,
                "train/psnr": epoch_psnr,
                "train/ssim": epoch_ssim,
                "val/loss": epoch_loss_val,
                "val/psnr": epoch_psnr_val,
                "val/ssim": epoch_ssim_val,
                "learning_rate": optimizer.param_groups[0]['lr'],
                "epoch": epoch
            })
        
        scheduler.step(epoch_loss_val)
        
        # Early stopping check
        if epoch_loss_val < best_val_loss:
            best_val_loss = epoch_loss_val
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
            if wandb_logging:
                wandb.save("best_model.pth")
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            model.load_state_dict(best_model_state)
            break
            
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, PSNR: {epoch_psnr:.4f}, SSIM: {epoch_ssim:.4f}')
        print(f'Validation - Loss: {epoch_loss_val:.4f}, PSNR: {epoch_psnr_val:.4f}, SSIM: {epoch_ssim_val:.4f}')
        

        plots["loss"]["training"].append(epoch_loss)
        plots["loss"]["validation"].append(epoch_loss_val)
        plots["psnr"]["training"].append(epoch_psnr)
        plots["psnr"]["validation"].append(epoch_psnr_val)
        plots["ssim"]["training"].append(epoch_ssim)
        plots["ssim"]["validation"].append(epoch_ssim_val)
        
        if wandb_logging and (epoch + 1) % 10 == 0:
            checkpoint_path = f"model_resources/checkpoint_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, checkpoint_path)
            wandb.save(checkpoint_path)

    if wandb_logging:
        wandb.finish()
    
    return best_model_state

if __name__ == '__main__':
    timestamp = int(time.time())
    print(f"model will be saved under {timestamp}")

    torch.manual_seed(1337)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = DenoisingAutoencoder().to(device)
    print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

    criterion = CombinedLoss(alpha=alpha).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=learning_rate,
        weight_decay=weight_decay
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True,
        min_lr=min_lr
    )
    
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    image_dataset = datasets.ImageFolder(root="mountain", transform=transform_input)
    # train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_input)
    # val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_input)

    train_size = int(0.9 * len(image_dataset))
    val_size = len(image_dataset) - train_size
    train_dataset, val_dataset = random_split(image_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    print(f"{len(train_loader.dataset)=}", f"{len(test_loader.dataset)=}")
    
    save_metadata(model.get_model_architecture(),f"{timestamp}_metadata.json")
    
    best_model_state = train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs=num_epochs)

    model.load_state_dict(best_model_state)

    save_final(plots, f"{timestamp}_metrics.json")

    for plot_type in ("loss", "psnr", "ssim"):
        y_training = plots[plot_type]["training"]
        y_validation = plots[plot_type]["validation"]
        plt.plot(y_training, label = "Training")
        plt.plot(y_validation, label = "Validation")
        plt.ylabel(plot_type)
        plt.legend()
        plt.savefig(f"graphs/{timestamp}_{plot_type}.png")
        plt.clf()

    model_resource_path = f"model_resources/{timestamp}.pth"
    print(f"Saving model under {model_resource_path}")
    torch.save(model.state_dict(), model_resource_path)
