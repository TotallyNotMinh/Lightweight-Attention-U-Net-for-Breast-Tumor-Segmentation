import os
import random
import cv2
import numpy as np
import torch
from data.load_data import BreastUltrasoundDataset

# Create output folder for the visualisations
output_dir = "augmented_samples"
os.makedirs(output_dir, exist_ok=True)

# Set seed for reproducible/clear visualisations
random.seed(42)
np.random.seed(42)

print("Initializing BreastUltrasoundDataset (without native loader augmentation)...")
dataset = BreastUltrasoundDataset(augment=False)
print(f"Dataset successfully loaded. Total samples available: {len(dataset)}")

# Get the first sample
image_tensor, mask_tensor = dataset[0]

# Convert tensors back to standard uint8 grayscale images (H, W) for OpenCV processing
# Loader outputs (1, H, W) normalized to [0.0, 1.0]
orig_img = (image_tensor[0].numpy() * 255.0).clip(0, 255).astype(np.uint8)
orig_mask = (mask_tensor[0].numpy() * 255.0).clip(0, 255).astype(np.uint8)

# Helper function to create a side-by-side image and mask comparison
def create_combined(img, msk):
    return np.hstack([img, msk])

# Save the original baseline preprocessed sample
cv2.imwrite(os.path.join(output_dir, "00_original_image.png"), orig_img)
cv2.imwrite(os.path.join(output_dir, "00_original_mask.png"), orig_mask)
cv2.imwrite(os.path.join(output_dir, "00_original_combined.png"), create_combined(orig_img, orig_mask))

print("Saved original preprocessed image and mask.")

# Dictionary to hold the results for grid/mosaic creation
augmentation_results = {
    "Original": (orig_img, orig_mask)
}

# Define each individual augmentation function matching the load_data.py logic exactly

# 1. Horizontal Flip
def apply_horizontal_flip(image, mask):
    return cv2.flip(image, 1), cv2.flip(mask, 1)

# 2. Vertical Flip
def apply_vertical_flip(image, mask):
    return cv2.flip(image, 0), cv2.flip(mask, 0)

# 3. Random Rotation (forcing 20 degrees for clear visibility)
def apply_rotation(image, mask, angle=20):
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
    img_rot = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)
    msk_rot = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST)
    return img_rot, msk_rot

# 4. Brightness & Contrast Jitter (using explicit visible scale and offset)
def apply_brightness_contrast(image, mask, alpha=1.3, beta=20):
    img_jitter = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return img_jitter, mask

# 5. Gaussian Noise (using standard deviation of 15 for clear visibility)
def apply_gaussian_noise(image, mask, sigma=15):
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    img_noise = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img_noise, mask

# 6. Gaussian Blur (using a 5x5 kernel)
def apply_gaussian_blur(image, mask):
    img_blur = cv2.GaussianBlur(image, (5, 5), 0)
    return img_blur, mask

# 7. Scale Jitter (forcing a visible zoom out of 0.8)
def apply_scale_jitter(image, mask, scale=0.8):
    h, w = image.shape[:2]
    new_h, new_w = int(h * scale), int(w * scale)
    img_res = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    msk_res = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    
    # Pad back to original size since scale < 1.0
    pad_y = (h - new_h) // 2
    pad_x = (w - new_w) // 2
    img_out = cv2.copyMakeBorder(img_res, pad_y, h-new_h-pad_y, pad_x, w-new_w-pad_x,
                                cv2.BORDER_CONSTANT, value=0)
    msk_out = cv2.copyMakeBorder(msk_res, pad_y, h-new_h-pad_y, pad_x, w-new_w-pad_x,
                                cv2.BORDER_CONSTANT, value=0)
    return img_out, msk_out

# 8. Elastic Deformation (probe compression simulation)
def apply_elastic_deformation(image, mask, sigma=8, alpha=45):
    h, w = image.shape[:2]
    dx = cv2.GaussianBlur(
        np.random.randn(h, w).astype(np.float32), (0, 0), sigma
    ) * alpha
    dy = cv2.GaussianBlur(
        np.random.randn(h, w).astype(np.float32), (0, 0), sigma
    ) * alpha
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = np.clip(x + dx, 0, w - 1).astype(np.float32)
    map_y = np.clip(y + dy, 0, h - 1).astype(np.float32)
    img_deform = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
    msk_deform = cv2.remap(mask, map_x, map_y, cv2.INTER_NEAREST)
    return img_deform, msk_deform

# 9. Coarse Dropout / Cutout (ultrasound shadow simulation)
def apply_coarse_dropout(image, mask):
    img_drop = image.copy()
    h, w = image.shape[:2]
    # Draw 3 visible blocks of size 30x30
    n_holes = 3
    hole_size = 30
    for _ in range(n_holes):
        x1 = random.randint(0, w - hole_size)
        y1 = random.randint(0, h - hole_size)
        img_drop[y1:y1+hole_size, x1:x1+hole_size] = 0
    return img_drop, mask


# Run and save each augmentation
augmentations = [
    ("01_horizontal_flip", apply_horizontal_flip, "Horizontal Flip"),
    ("02_vertical_flip", apply_vertical_flip, "Vertical Flip"),
    ("03_rotation", apply_rotation, "Random Rotation"),
    ("04_brightness_contrast", apply_brightness_contrast, "Brightness & Contrast"),
    ("05_gaussian_noise", apply_gaussian_noise, "Gaussian Noise"),
    ("06_gaussian_blur", apply_gaussian_blur, "Gaussian Blur"),
    ("07_scale_jitter", apply_scale_jitter, "Scale Jitter (Zoom-Out)"),
    ("08_elastic_deformation", apply_elastic_deformation, "Elastic Deformation"),
    ("09_coarse_dropout", apply_coarse_dropout, "Coarse Dropout (Cutout)")
]

for filename_prefix, func, label in augmentations:
    # Run augmentation
    img_aug, msk_aug = func(orig_img.copy(), orig_mask.copy())
    
    # Save individual files
    cv2.imwrite(os.path.join(output_dir, f"{filename_prefix}_image.png"), img_aug)
    cv2.imwrite(os.path.join(output_dir, f"{filename_prefix}_mask.png"), msk_aug)
    cv2.imwrite(os.path.join(output_dir, f"{filename_prefix}_combined.png"), create_combined(img_aug, msk_aug))
    
    # Store for grid
    augmentation_results[label] = (img_aug, msk_aug)
    print(f"Applied and saved: {label}")


# Create a single, beautifully organized comparison grid (3x4 grid)
# Grid layout: 12 cells total (1 original + 9 augmentations + 2 blank cells)
print("Creating comprehensive grid comparison...")

cell_w, cell_h = 256, 256
grid_cols = 4
grid_rows = 3

# We will create two separate grids: one for images and one for masks
grid_img = np.zeros((grid_rows * cell_h, grid_cols * cell_w), dtype=np.uint8)
grid_msk = np.zeros((grid_rows * cell_h, grid_cols * cell_w), dtype=np.uint8)

labels_keys = [
    "Original", "Horizontal Flip", "Vertical Flip", "Random Rotation",
    "Brightness & Contrast", "Gaussian Noise", "Gaussian Blur", "Scale Jitter (Zoom-Out)",
    "Elastic Deformation", "Coarse Dropout (Cutout)"
]

for idx, label in enumerate(labels_keys):
    r = idx // grid_cols
    c = idx % grid_cols
    
    img_data, msk_data = augmentation_results[label]
    
    # Write images to grid and draw labels on them
    img_cell = img_data.copy()
    cv2.putText(img_cell, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,), 1, cv2.LINE_AA)
    grid_img[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w] = img_cell
    
    # Write masks to grid and draw labels on them
    msk_cell = msk_data.copy()
    cv2.putText(msk_cell, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,), 1, cv2.LINE_AA)
    grid_msk[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w] = msk_cell

# Save grids
cv2.imwrite(os.path.join(output_dir, "augmentations_grid_images.png"), grid_img)
cv2.imwrite(os.path.join(output_dir, "augmentations_grid_masks.png"), grid_msk)

# Concatenate the images and masks grids horizontally for a master comparison sheet
master_grid = np.hstack([grid_img, grid_msk])
cv2.imwrite(os.path.join(output_dir, "augmentations_grid_comparison.png"), master_grid)

print(f"\nSUCCESS! All augmentations saved inside directory '{output_dir}/'")
print("Files generated:")
print(" - Individual PNG files (00_original_* up to 09_coarse_dropout_*)")
print(" - Single-view horizontal comparisons (*_combined.png)")
print(" - Master grid comparisons ('augmentations_grid_images.png', 'augmentations_grid_masks.png')")
print(" - Single master comparative visual canvas ('augmentations_grid_comparison.png')")
