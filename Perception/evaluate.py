import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from Perception.model.unet import UNet
from Perception.utils.utils import find_edge_channel

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

device = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate(model, image):
    model.eval()
    model = model.to(device)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges, edges_inv = find_edge_channel(image)
    
    output_image = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
    output_image[:, :, 0] = gray
    output_image[:, :, 1] = edges
    output_image[:, :, 2] = edges_inv
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((180, 330)),
        transforms.ToTensor()
    ])
    
    input_tensor = transform(output_image)
    input_tensor = input_tensor.unsqueeze(0)
    input_tensor = input_tensor.to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.sigmoid(output)
        binary_mask = (pred > 0.5).float()
    
    mask_np = binary_mask.squeeze().cpu().numpy()
    mask_resized = cv2.resize(mask_np, (image.shape[1], image.shape[0]), 
                              interpolation=cv2.INTER_NEAREST)
    
    segmentation_mask = (mask_resized * 255).astype(np.uint8)
    
    overlay = image.copy()
    overlay[segmentation_mask > 127] = [0, 255, 0]
    visualization = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
    
    return segmentation_mask, visualization


def evaluate_on_custom_images(weight_path, custom_images_dir="./data/custom/inputs"):
    print(f"Loading model from {weight_path}...")
    model = UNet()
    
    save_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(save_dict["model"])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Model was trained for {save_dict['epochs'][-1] if save_dict['epochs'] else 0} epochs")
    print(f"Best validation loss: {save_dict.get('min_loss', 'N/A')}")
    
    if not os.path.exists(custom_images_dir):
        print(f"Error: Directory {custom_images_dir} does not exist!")
        return
    
    image_files = [f for f in os.listdir(custom_images_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if len(image_files) == 0:
        print(f"No images found in {custom_images_dir}")
        return
    
    print(f"\nFound {len(image_files)} test images")
    print("-" * 50)
    
    output_dir = "./results"
    os.makedirs(output_dir, exist_ok=True)
    
    for idx, img_file in enumerate(image_files):
        img_path = os.path.join(custom_images_dir, img_file)
        print(f"\nProcessing ({idx+1}/{len(image_files)}): {img_file}")
        
        image = cv2.imread(img_path)
        if image is None:
            print(f"  Error: Could not read image {img_file}")
            continue
        
        print(f"  Image shape: {image.shape}")
        
        mask, visualization = evaluate(model, image)
        
        lane_pixels = np.sum(mask > 127)
        total_pixels = mask.shape[0] * mask.shape[1]
        lane_percentage = (lane_pixels / total_pixels) * 100
        
        print(f"  Lane coverage: {lane_percentage:.2f}%")
        print(f"  Lane pixels detected: {lane_pixels}")
        
        output_path_mask = os.path.join(output_dir, f"{os.path.splitext(img_file)[0]}_mask.png")
        output_path_viz = os.path.join(output_dir, f"{os.path.splitext(img_file)[0]}_visualization.png")
        
        cv2.imwrite(output_path_mask, mask)
        cv2.imwrite(output_path_viz, visualization)
        
        print(f"  Saved mask to: {output_path_mask}")
        print(f"  Saved visualization to: {output_path_viz}")
        
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(f"Original Image\n{img_file}")
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(mask, cmap='gray')
        plt.title(f"Segmentation Mask\nLane: {lane_percentage:.1f}%")
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(visualization, cv2.COLOR_BGR2RGB))
        plt.title("Overlay Visualization")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{os.path.splitext(img_file)[0]}_combined.png"), 
                    dpi=150, bbox_inches='tight')
        plt.close()
    
    print("\n" + "="*50)
    print("Evaluation complete!")
    print(f"Results saved to: {output_dir}")
    print("="*50)


if __name__ == "__main__":
    weight_path = os.path.join(THIS_DIR, "model.pt")
    
    if not os.path.exists(weight_path):
        print("=" * 60)
        print("ERROR: Weight file not found!")
        print("=" * 60)
        print(f"Expected at: {weight_path}")
        print("\nAvailable .pt files:")
        pt_files = [f for f in os.listdir('.') if f.endswith('.pt')]
        if pt_files:
            for f in pt_files:
                print(f"  - {f}")
        else:
            print("  (None found - train the model first)")
    else:
        evaluate_on_custom_images(weight_path)