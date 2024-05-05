import time
import torch
import torch.nn as nn
from tqdm import tqdm
from dataset import calculate_psnr, get_batch, prepare_dataset
from model import DenoisingAutoencoder

# Hyperparameters
batch_size = 256
learning_rate = 0.0005
max_iters = 20000
eval_interval = 1000
eval_iters = 20

device = "mps" if torch.backends.mps.is_available() else "cpu"
model = DenoisingAutoencoder().to(device)
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

# TODO: check possibility to switch to perceptual or adversarial loss
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

dataset = prepare_dataset(device, "IMAGES_PATH")
n = int(0.9*len(dataset))
training_dataset = dataset[:n]
validation_dataset = dataset[n:]
print(f"{len(training_dataset)=}", f"{len(validation_dataset)=}")

@torch.no_grad()
def calculate_batch_psnr(batch_input_images, output_image):
    psnrs = torch.zeros(batch_size)
    for i in range(len(batch_input_images)):
        input_img_np = batch_input_images[i].cpu().numpy().transpose((1, 2, 0))
        output_img_np = output_image[i].cpu().numpy().transpose((1, 2, 0))
        psnrs[i] = calculate_psnr(input_img_np, output_img_np)
    return psnrs.mean()


@torch.no_grad()
def estimate_loss():
    final_losses = {}
    final_psnrs = {}
    model.eval()

    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        psnrs = torch.zeros(eval_iters)
        for k in range(eval_iters):
            original_images, noisy_images = get_batch(batch_size, training_dataset if split == "train" else validation_dataset, selection_mode="random")
            batch_input_images = torch.stack(original_images)
            batch_noisy_images = torch.stack(noisy_images)
            output_image = model(batch_noisy_images)
            loss = criterion(output_image, batch_input_images)
            psnr = calculate_batch_psnr(batch_input_images, output_image)
            losses[k] = loss.item()
            psnrs[k] = psnr
        final_losses[split] = losses.mean()
        final_psnrs[split] = psnrs.mean()

    model.train()
    return final_losses, final_psnrs

with tqdm(total=max_iters, desc=f"{batch_size=}") as pbar:
    for iter in range(max_iters):
        original_images, noisy_images = get_batch(batch_size, training_dataset, selection_mode="iterative")

        batch_input_images = torch.stack(original_images)
        batch_noisy_images = torch.stack(noisy_images)

        output_image = model(batch_noisy_images)

        loss = criterion(output_image, batch_input_images)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iter % eval_interval == 0:
            print(f"Step {iter}: Training Loss:", loss.item())
            calculate_batch_psnr(batch_input_images, output_image)
            losses, psnrs = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.5f}, val loss {losses['val']:.5f}")
            print(f"step {iter}: train psnr  {psnrs['train']:.2f}, val psnr {psnrs['val']:.2f}")
        pbar.update(1)

print("Final Training Loss:", loss.item())

final_loss = "{:.6f}".format(loss.item()).replace(".", "")
model_resource_path = f"model_resources/model_image_{int(time.time())}_{final_loss}.pth"
print(f"Saving model under {model_resource_path}")
torch.save(model.state_dict(), model_resource_path)
