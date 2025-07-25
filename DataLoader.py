import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.image_paths = [os.path.join(data_dir, fname) 
                            for fname in os.listdir(data_dir) 
                            if fname.lower().endswith(('png', 'jpg', 'jpeg'))]
        
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),  # Resize all images to 256x256
            transforms.ToTensor(),  # Convert to tensor (C, H, W) format
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize [-1, 1]
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")  # Ensure 3-channel RGB format
        image = self.transform(image)
        return image

if __name__ == "__main__":
    print("Testing Image Dataset")
    dataset = ImageDataset("data")
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    for images in dataloader:
        print(images.shape)
        break