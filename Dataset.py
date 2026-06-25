import os
import cv2
import random
import numpy as np
from tqdm import tqdm
import albumentations as A

INPUT_DIR = "urdu_alphabets"
OUTPUT_DIR = "urdu_dataset_augmented"
TARGET_IMAGES_PER_CLASS = 500
IMAGE_SIZE = 64 


def replace_background(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    bg_type = random.choice(['solid', 'gradient', 'noise'])
    background = np.zeros_like(img)

    if bg_type == 'solid':
        color = [random.randint(0, 255) for _ in range(3)]
        background[:] = color
    elif bg_type == 'gradient':
        c1 = np.array([random.randint(0, 255) for _ in range(3)])
        c2 = np.array([random.randint(0, 255) for _ in range(3)])
        for i in range(img.shape[0]):
            t = i / img.shape[0]
            background[i, :] = ((1 - t) * c1 + t * c2).astype(np.uint8)
    else:
        background = np.random.randint(180, 255, img.shape, dtype=np.uint8)

    ink_color = np.array([
        random.randint(0, 80),
        random.randint(0, 80),
        random.randint(0, 150)
    ], dtype=np.uint8)

    result = background.copy()

    result[mask > 0] = ink_color

    return result


transform = A.Compose([
    A.Rotate(limit=20, p=0.8),
    A.ShiftScaleRotate(shift_limit=0.08, scale_limit=0.15, rotate_limit=0, p=0.7),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
    A.GaussianBlur(blur_limit=(3, 5), p=0.3),
    A.GaussNoise(std_range=(0.02, 0.08), p=0.3),
    A.ElasticTransform(alpha=1, sigma=50, p=0.3),
    A.Perspective(scale=(0.03, 0.08), p=0.3),
    # ↓ NEW: random hue/saturation shifts
    A.HueSaturationValue(
        hue_shift_limit=30,
        sat_shift_limit=40,
        val_shift_limit=30,
        p=0.5
    ),
])

os.makedirs(OUTPUT_DIR, exist_ok=True)

for class_name in os.listdir(INPUT_DIR):
    class_input_path = os.path.join(INPUT_DIR, class_name)
    if not os.path.isdir(class_input_path):
        continue

    class_output_path = os.path.join(OUTPUT_DIR, class_name)
    os.makedirs(class_output_path, exist_ok=True)

    images = []
    for file in os.listdir(class_input_path):
        img = cv2.imread(os.path.join(class_input_path, file))
        if img is None:
            continue
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        images.append(img)

    if not images:
        print(f"No images found in {class_name}")
        continue

    print(f"\nGenerating: {class_name}")
    counter = 0

    # Save originals
    for img in images:
        cv2.imwrite(os.path.join(class_output_path, f"{counter}.jpg"), img)
        counter += 1


    while counter < TARGET_IMAGES_PER_CLASS:
        img = random.choice(images)

        if random.random() < 0.5:
            img = replace_background(img)

        augmented = transform(image=img)["image"]
        cv2.imwrite(os.path.join(class_output_path, f"{counter}.jpg"), augmented)
        counter += 1

    print(f"Done: {class_name} → {counter} images")

print("\nDataset generation complete!")