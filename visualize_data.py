import matplotlib.pyplot as plt
from torchvision.datasets import OxfordIIITPet
import random
import numpy as np
import os

def visualize_sample():
    # Load dataset without any transforms to see original images and masks
    data_dir = './data'
    dataset = OxfordIIITPet(root=data_dir, split="trainval", target_types="segmentation")
    
    # Pick a random index
    idx = random.randint(0, len(dataset) - 1)
    image, mask = dataset[idx]
    
    # Convert mask to numpy array for visualization
    mask_np = np.array(mask)
    
    # Mask values in Oxford-IIIT Pet:
    # 1: Foreground (Pet)
    # 2: Background
    # 3: Not classified / Boundary
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Show original image
    axes[0].imshow(image)
    axes[0].set_title(f"Original Image (Index: {idx})")
    axes[0].axis('off')
    
    # Show mask
    # We use vmin=1 and vmax=3 to keep colors consistent
    mask_plot = axes[1].imshow(mask_np, cmap='viridis', vmin=1, vmax=3)
    axes[1].set_title("Annotation Mask")
    axes[1].axis('off')
    
    # Add a colorbar for the mask
    cbar = fig.colorbar(mask_plot, ax=axes[1], fraction=0.046, pad=0.04, ticks=[1, 2, 3])
    cbar.ax.set_yticklabels(['Foreground (1)', 'Background (2)', 'Boundary (3)'])
    
    output_path = "sample_visualization.png"
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Saved visualization to {output_path}")

if __name__ == "__main__":
    visualize_sample()
