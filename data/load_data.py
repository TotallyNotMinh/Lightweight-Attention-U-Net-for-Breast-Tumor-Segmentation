import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import kagglehub
import random


import random
import cv2
import numpy as np

def augment(image, mask):

    # Horizontal flip
    if random.random() > 0.3:
        image = cv2.flip(image, 1)
        mask  = cv2.flip(mask, 1)

    # Vertical flip
    if random.random() > 0.3:
        image = cv2.flip(image, 0)
        mask  = cv2.flip(mask, 0)

    # Rotation
    if random.random() > 0.3:
        angle = random.uniform(-30, 30)
        h, w  = image.shape[:2]
        M     = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
        image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)
        mask  = cv2.warpAffine(mask,  M, (w, h), flags=cv2.INTER_NEAREST)

    # Brightness / contrast
    if random.random() > 0.3:
        alpha = 0.7 + random.random() * 0.6   # wider range: 0.7–1.3
        beta  = random.randint(-20, 20)         # stronger shift
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    # Gaussian noise
    if random.random() > 0.3:
        noise = np.random.normal(0, random.uniform(2, 25), image.shape).astype(np.float32)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Gaussian blur
    if random.random() > 0.3:
        ksize = random.choice([3, 5])
        image = cv2.GaussianBlur(image, (ksize, ksize), 0)

    # Scale jitter — single resize via crop/pad instead of double resize
    if random.random() > 0.3:
        scale = random.uniform(0.8, 1.2)
        h, w  = image.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        mask  = cv2.resize(mask,  (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        # Crop or pad back to original size without a second resize pass
        if scale > 1.0:
            start_y = (new_h - h) // 2
            start_x = (new_w - w) // 2
            image = image[start_y:start_y+h, start_x:start_x+w]
            mask  = mask [start_y:start_y+h, start_x:start_x+w]
        else:
            pad_y = (h - new_h) // 2
            pad_x = (w - new_w) // 2
            image = cv2.copyMakeBorder(image, pad_y, h-new_h-pad_y, pad_x, w-new_w-pad_x,
                                       cv2.BORDER_CONSTANT, value=0)
            mask  = cv2.copyMakeBorder(mask,  pad_y, h-new_h-pad_y, pad_x, w-new_w-pad_x,
                                       cv2.BORDER_CONSTANT, value=0)

    # Elastic deformation — critical for small medical datasets
    if random.random() > 0.5:
        h, w   = image.shape[:2]
        sigma  = random.uniform(6, 10)
        alpha  = random.uniform(30, 60)
        dx = cv2.GaussianBlur(
            np.random.randn(h, w).astype(np.float32), (0, 0), sigma
        ) * alpha
        dy = cv2.GaussianBlur(
            np.random.randn(h, w).astype(np.float32), (0, 0), sigma
        ) * alpha
        x, y   = np.meshgrid(np.arange(w), np.arange(h))
        map_x  = np.clip(x + dx, 0, w - 1).astype(np.float32)
        map_y  = np.clip(y + dy, 0, h - 1).astype(np.float32)
        image  = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
        mask   = cv2.remap(mask,  map_x, map_y, cv2.INTER_NEAREST)

    # Coarse dropout — randomly blank out patches (simulates ultrasound artefacts)
    if random.random() > 0.5:
        h, w     = image.shape[:2]
        n_holes  = random.randint(1, 4)
        hole_size = random.randint(16, 40)
        for _ in range(n_holes):
            x1 = random.randint(0, w - hole_size)
            y1 = random.randint(0, h - hole_size)
            image[y1:y1+hole_size, x1:x1+hole_size] = 0

    return image, mask
    

class BreastUltrasoundDataset(Dataset):

    def __init__(self, image_size=256, augment=True):

        self.root_dir = kagglehub.dataset_download(
            "aryashah2k/breast-ultrasound-images-dataset"
        )

        self.image_size = image_size
        self.samples = []
        self.do_augment = augment
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        classes = ["benign", "malignant"]

        for cls in classes:

            class_dir = os.path.join(
                self.root_dir,
                "Dataset_BUSI_with_GT",
                cls
            )

            files = os.listdir(class_dir)

            image_files = [
                f for f in files
                if "_mask" not in f and f.endswith(".png")
            ]

            for img_file in image_files:

                img_path = os.path.join(class_dir, img_file)

                base_name = img_file.replace(".png", "")

                mask_files = [
                    f for f in files
                    if f.startswith(base_name + "_mask")
                    and f.endswith(".png")
                ]

                mask_paths = [
                    os.path.join(class_dir, f)
                    for f in mask_files
                ]

                self.samples.append({
                    "image": img_path,
                    "masks": mask_paths,
                    "label": cls
                })

    def __len__(self):

        return len(self.samples)

    def pad_to_square(self, image, mask=None):

        h, w = image.shape[:2]

        size = max(h, w)

        pad_h = size - h
        pad_w = size - w

        top = pad_h // 2
        bottom = pad_h - top

        left = pad_w // 2
        right = pad_w - left

        image = cv2.copyMakeBorder(
            image,
            top,
            bottom,
            left,
            right,
            borderType=cv2.BORDER_CONSTANT,
            value=0
        )

        if mask is not None:

            mask = cv2.copyMakeBorder(
                mask,
                top,
                bottom,
                left,
                right,
                borderType=cv2.BORDER_CONSTANT,
                value=0
            )

            return image, mask

        return image

    def combine_masks(self, mask_paths, image_shape): 

        if len(mask_paths) == 0:

            return np.zeros(image_shape, dtype=np.uint8)

        combined_mask = np.zeros(image_shape, dtype=np.uint8)

        for path in mask_paths:

            mask = cv2.imread(
                path,
                cv2.IMREAD_GRAYSCALE
            )

            if mask is None:
                continue

            mask = (mask > 0).astype(np.uint8)

            combined_mask = np.maximum(
                combined_mask,
                mask
            )

        return combined_mask


    def __getitem__(self, idx):

        sample = self.samples[idx]

        image = cv2.imread(sample["image"], cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise ValueError(f"Failed to load image: {sample['image']}")

        mask = self.combine_masks(sample["masks"], image.shape)

        image, mask = self.pad_to_square(image, mask)

        image = cv2.resize(
            image,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_LINEAR
        )

        mask = cv2.resize(
            mask,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_NEAREST
        )

        
        #CLAHE Preprocessing
        image = self.clahe.apply(image)
        
        if self.do_augment:
            image, mask = augment(image, mask)
            
        # normalize image
        image = image.astype(np.float32) / 255.0
        mask = (mask > 0).astype(np.float32)


        mask = mask[None, :, :]   # (1, H, W)

        image = image[None, :, :] # (1, H, W)

        return torch.from_numpy(image).float(), torch.from_numpy(mask).float()