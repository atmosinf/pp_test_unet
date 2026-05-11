import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import OxfordIIITPet
from torchvision import transforms
import numpy as np
from tqdm import tqdm
from unet import UNet
import os

class TargetTransform:
    def __call__(self, mask):
        mask = transforms.functional.resize(mask, (256, 256), interpolation=transforms.functional.InterpolationMode.NEAREST)
        mask = torch.from_numpy(np.array(mask)).long()
        mask = mask - 1 # Oxford-IIIT Pet mask values are 1, 2, 3. We need 0, 1, 2.
        return mask

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    target_transform = TargetTransform()

    # Dataset & DataLoader
    data_dir = './data'
    os.makedirs(data_dir, exist_ok=True)
    
    print("Loading dataset...")
    train_dataset = OxfordIIITPet(root=data_dir, split="trainval", target_types="segmentation", download=True,
                                  transform=transform, target_transform=target_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)

    # Model
    model = UNet(n_channels=3, n_classes=3).to(device)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    epochs = 5
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for images, masks in pbar:
                images = images.to(device)
                masks = masks.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                pbar.set_postfix(loss=loss.item())
        
        print(f"Epoch {epoch+1} average loss: {epoch_loss/len(train_loader):.4f}")

    # Save the model
    torch.save(model.state_dict(), "unet.pth")
    print("Model saved to unet.pth")

if __name__ == "__main__":
    main()
