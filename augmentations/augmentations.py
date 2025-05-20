<<<<<<< HEAD
import os
import albumentations as A
from utils import read_image, label_flip_horizontal, label_flip_vertically, label_feature_augmentation, save_augmented_pair

async def apply_random_features(image_path, label_path, output_folder):
    image = await read_image(image_path)
    transform = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.RandomGamma(gamma_limit=(80, 120), p=0.5),
        A.RandomShadow(p=0.5),
        A.MotionBlur(blur_limit=3, p=0.3),
        A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5),
        A.GaussianBlur(p=0.3),
        A.ColorJitter(p=0.4),
    ])
    image_aug = transform(image=image)['image']
    label = await label_feature_augmentation(label_path, image_aug)
    return await save_augmented_pair(image_aug, label, image_path, label_path, output_folder, prefix="feat")

async def apply_flip_horizontal(image_path, label_path, output_folder):
    image = await read_image(image_path)
    width = image.shape[1]
    transform = A.HorizontalFlip(p=1)
    image_aug = transform(image=image)['image']
    label = await label_flip_horizontal(label_path, width, image_aug)
    return await save_augmented_pair(image_aug, label, image_path, label_path, output_folder, prefix="hflip")

async def apply_flip_vertical(image_path, label_path, output_folder):
    image = await read_image(image_path)
    height = image.shape[0]
    transform = A.VerticalFlip(p=1)
    image_aug = transform(image=image)['image']
    label = await label_flip_vertically(label_path, height, image_aug)
    return await save_augmented_pair(image_aug, label, image_path, label_path, output_folder, prefix="vflip")
=======
import os
import albumentations as A
from utils import read_image, label_flip_horizontal, label_flip_vertically, label_feature_augmentation, save_augmented_pair

async def apply_random_features(image_path, label_path, output_folder):
    image = await read_image(image_path)
    transform = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.RandomGamma(gamma_limit=(80, 120), p=0.5),
        A.RandomShadow(p=0.5),
        A.MotionBlur(blur_limit=3, p=0.3),
        A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5),
        A.GaussianBlur(p=0.3),
        A.ColorJitter(p=0.4),
    ])
    image_aug = transform(image=image)['image']
    label = await label_feature_augmentation(label_path, image_aug)
    return await save_augmented_pair(image_aug, label, image_path, label_path, output_folder, prefix="feat")

async def apply_flip_horizontal(image_path, label_path, output_folder):
    image = await read_image(image_path)
    width = image.shape[1]
    transform = A.HorizontalFlip(p=1)
    image_aug = transform(image=image)['image']
    label = await label_flip_horizontal(label_path, width, image_aug)
    return await save_augmented_pair(image_aug, label, image_path, label_path, output_folder, prefix="hflip")

async def apply_flip_vertical(image_path, label_path, output_folder):
    image = await read_image(image_path)
    height = image.shape[0]
    transform = A.VerticalFlip(p=1)
    image_aug = transform(image=image)['image']
    label = await label_flip_vertically(label_path, height, image_aug)
    return await save_augmented_pair(image_aug, label, image_path, label_path, output_folder, prefix="vflip")
>>>>>>> 70f2afaad93a7194b5e7131c2056d060e0901a74
