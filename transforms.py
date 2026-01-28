# transforms.py
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

def crop_black_borders(image):
    """Remove black borders around the fundus image"""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Threshold to find non-black area
    _, thresh = cv2.threshold(gray, 7, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        # Add small margin
        margin = int(min(w, h) * 0.03)
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(image.shape[1] - x, w + 2*margin)
        h = min(image.shape[0] - y, h + 2*margin)
        return image[y:y+h, x:x+w]
    return image


def get_transforms(mode="val", input_size=512):
    """
    Returns Albumentations pipeline
    mode: "train" or "val"
    """
    base_transforms = [
        A.Lambda(image=crop_black_borders, always_apply=True),
        A.Resize(input_size, input_size),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8,8), p=0.75),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ]

    if mode == "train":
        augmentations = [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.4),
            A.ShiftScaleRotate(
                shift_limit=0.12,
                scale_limit=0.15,
                rotate_limit=40,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.7
            ),
            A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.35, p=0.7),
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=20, p=0.4),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.25),
        ]
        # Insert augmentations before normalization
        return A.Compose(augmentations + base_transforms)
    
    return A.Compose(base_transforms)