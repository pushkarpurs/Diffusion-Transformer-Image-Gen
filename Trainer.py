import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision.utils import make_grid
from DataLoader import ImageDataset
from TimestepEmbedding import LinearNoiseScheduler
from DiTCore import DiT

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size=16
lr=2e-4
num_epochs=250
num_timesteps=1000  # Number of diffusion steps
beta_start, beta_end=1e-4, 0.02
no_of_dit_blocks=12
checkpoint_path="dit_checkpoint_12_f.pth"  # Single checkpoint file to store latest state

def save(imgs, tstep,n):
    ims = torch.clamp(imgs, -1., 1.).detach().cpu()
    ims = (ims + 1) / 2
    grid = make_grid(ims, nrow=1)
    img = torchvision.transforms.ToPILImage()(grid)
    img.save(os.path.join('generated_images', 'samples', '{}_x{}.png'.format(tstep, n)))
    img.close()


def train():
    # Load Dataset
    dataset = ImageDataset("data")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # Initialize Model, Noise Scheduler, and Optimizer
    model = DiT(no_of_dit_blocks).to(device)
    scheduler = LinearNoiseScheduler(num_timesteps, beta_start, beta_end)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()  # Noise prediction loss

    start_epoch = 0  # Default starting epoch

    # Load checkpoint if it exists
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1  # Resume from next epoch
        print(f"Resumed training from epoch {start_epoch}")

    # Training Loop
    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0
        for images in dataloader:
            images = images.to(device)
            # Sample a random timestep for each image
            t = torch.randint(0, num_timesteps, (batch_size,), device=device).long()  # (B,1)
            # Generate random noise
            noise = torch.randn_like(images).to(device)
            # Forward process: Add noise to images
            noisy_images = scheduler.add_noise(images, noise, t)

            # Model prediction (predicts noise)
            predicted_img = model(noisy_images, t)

             # Save intermediate images for visualization
            save(images[0], t[0], 0)
            save(noisy_images[0], t[0], 1)
            save(predicted_img[0], t[0], 2)
            save(images[1], t[1], 0)
            save(noisy_images[1], t[1], 1)
            save(predicted_img[1], t[1], 2)
            save(images[2], t[2], 0)
            save(noisy_images[2], t[2], 1)
            save(predicted_img[2], t[2], 2)

            # Compute loss (MSE between predicted and actual noise)
            loss = loss_fn(predicted_img, images)
            epoch_loss += loss.item()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] | Loss: {avg_loss:.6f}")

        # Save model checkpoint every 100 epochs
        if (epoch + 1) % 1 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch+1}")

    print("Training complete.")
    torch.save(model.state_dict(), "dit_final_12_f_250.pth")

if __name__ == "__main__":
    print("Training DiT model...")
    if not os.path.exists(os.path.join('generated_images', 'samples')):
        os.mkdir(os.path.join('generated_images', 'samples'))
    train()
