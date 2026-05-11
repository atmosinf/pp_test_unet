import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import argparse
from unet import UNet
import numpy as np

def run_inference(image_path, model_path, output_path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load the model
    model = UNet(n_channels=3, n_classes=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # Image transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        output = model(input_tensor)
        # The output has shape (1, 3, H, W). Get the argmax over the class dimension
        prediction = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

    # The prediction is 256x256. We could resize it back to original size if needed,
    # but for visualization 256x256 is fine.
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image.resize((256, 256)))
    axes[0].set_title("Input Image")
    axes[0].axis('off')

    # Mask classes: 0 = Foreground, 1 = Background, 2 = Not classified
    axes[1].imshow(prediction, cmap='viridis')
    axes[1].set_title("Predicted Segmentation")
    axes[1].axis('off')

    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Saved inference result to {output_path}")
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run UNet Inference")
    parser.add_argument("image_path", type=str, help="Path to input image")
    parser.add_argument("--model_path", type=str, default="unet.pth", help="Path to trained model")
    parser.add_argument("--output_path", type=str, default="output.png", help="Path to save output visualization")
    args = parser.parse_args()

    run_inference(args.image_path, args.model_path, args.output_path)
