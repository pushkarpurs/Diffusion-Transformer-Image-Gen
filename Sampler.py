import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import torchvision
from torchvision.utils import make_grid
from DiTCore import DiT
from TimestepEmbedding import LinearNoiseScheduler

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters (Same as training)
image_size = (3, 256, 256)  # (Channels, Height, Width) — Adjust if needed
num_timesteps = 1000
beta_start, beta_end = 1e-4, 0.02
no_of_dit_blocks = 12  # Match your trained model

# Load trained model
model = DiT(no_of_dit_blocks).to(device)
model.load_state_dict(torch.load("dit_final_12_f_250.pth", map_location=device))
model.eval()  # Set to evaluation mode

# Initialize noise scheduler
scheduler = LinearNoiseScheduler(num_timesteps, beta_start, beta_end)

@torch.no_grad()
def generate_images():
    """ Generate images from pure noise using the trained diffusion model """
    xt = torch.randn((1, *image_size)).to(device)  # Start with Gaussian noise
    
    for t in reversed(range(num_timesteps)):  # Reverse diffusion process

        t_tensor = torch.full((1,), t, device=device, dtype=torch.long)

        # ims = torch.clamp(xt, -1., 1.).detach().cpu()
        # ims = (ims + 1) / 2
        # grid = make_grid(ims, nrow=1)
        # img = torchvision.transforms.ToPILImage()(grid)
        # if not os.path.exists(os.path.join('generated_images', 'samples3')):
        #     os.mkdir(os.path.join('generated_images', 'samples3'))
        # img.save(os.path.join('generated_images', 'samples3', '{}_x1.png'.format(t)))
        # img.close()


        image_pred = model(xt, t_tensor)  # Predict image


        # ims = torch.clamp(image_pred, -1., 1.).detach().cpu()
        # ims = (ims + 1) / 2
        # grid = make_grid(ims, nrow=1)
        # img = torchvision.transforms.ToPILImage()(grid)
        # img.save(os.path.join('generated_images', 'samples3', '{}_x2.png'.format(t)))
        # img.close()

        xt,x0 = scheduler.sample_prev_timestep(image_pred, xt, t)  # Get previous timestep


        # ims = torch.clamp(xt, -1., 1.).detach().cpu()
        # ims = (ims + 1) / 2
        # grid = make_grid(ims, nrow=1)
        # img = torchvision.transforms.ToPILImage()(grid)
        # img.save(os.path.join('generated_images', 'samples3', '{}_x3.png'.format(t)))
        # img.close()


    return xt  # Final denoised image

generated_images = generate_images()

# Convert images to CPU and normalize
generated_images = (generated_images.clamp(-1, 1) + 1) / 2  # Rescale to [0,1]
generated_images = generated_images.permute(0, 2, 3, 1).cpu().numpy()  # (B, H, W, C)

# Save and Display
os.makedirs("generated_images", exist_ok=True)
for i, img in enumerate(generated_images):
    plt.imsave(f"generated_images/samples4/sample_0.png", img)
    plt.subplot(1,1, i + 1)
    plt.imshow(img)
    plt.axis("off")
plt.show()