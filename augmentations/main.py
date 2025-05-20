<<<<<<< HEAD
import asyncio
import random
from utils import list_all_files
from augmentations import apply_flip_horizontal, apply_flip_vertical, apply_random_features

folder = "../frames/"
output_folder = "../augmented/"

async def process_image(img_path, lbl_path):
    augmentations = []

    # 25% chance to skip
    if random.random() < 0.25:
        return "skip", img_path, lbl_path

    # Randomly choose to apply each augmentation
    if random.random() < 0.5:
        augmentations.append(apply_flip_horizontal)
    if random.random() < 0.5:
        augmentations.append(apply_flip_vertical)
    if random.random() < 0.5:
        augmentations.append(apply_random_features)

    # Apply each selected augmentation and update paths
    for aug in augmentations:
        img_path, lbl_path = await aug(img_path, lbl_path, output_folder)

    return "+".join([aug.__name__ for aug in augmentations]) or "none", img_path, lbl_path

if __name__ == '__main__':
    images, labels = list_all_files(folder)

    count = {
        'skip': 0,
        'apply_flip_horizontal': 0,
        'apply_flip_vertical': 0,
        'apply_random_features': 0,
        'none': 0
    }

    for img_name in images:
        label_name = img_name.replace('.png', '.json')
        img_path = folder + img_name
        lbl_path = folder + label_name

        result, img_path, lbl_path = asyncio.run(process_image(img_path, lbl_path))
        if result in count:
            count[result] += 1
        elif "+" in result:
            for r in result.split("+"):
                count[r] += 1

    print("Skipped:", count['skip'])
    print("Horizontal flips:", count['apply_flip_horizontal'])
    print("Vertical flips:", count['apply_flip_vertical'])
    print("Feature aug:", count['apply_random_features'])
=======
import asyncio
import random
from utils import list_all_files
from augmentations import apply_flip_horizontal, apply_flip_vertical, apply_random_features

folder = "../frames/"
output_folder = "../augmented/"

async def process_image(img_path, lbl_path):
    augmentations = []

    # 25% chance to skip
    if random.random() < 0.25:
        return "skip", img_path, lbl_path

    # Randomly choose to apply each augmentation
    if random.random() < 0.5:
        augmentations.append(apply_flip_horizontal)
    if random.random() < 0.5:
        augmentations.append(apply_flip_vertical)
    if random.random() < 0.5:
        augmentations.append(apply_random_features)

    # Apply each selected augmentation and update paths
    for aug in augmentations:
        img_path, lbl_path = await aug(img_path, lbl_path, output_folder)

    return "+".join([aug.__name__ for aug in augmentations]) or "none", img_path, lbl_path

if __name__ == '__main__':
    images, labels = list_all_files(folder)

    count = {
        'skip': 0,
        'apply_flip_horizontal': 0,
        'apply_flip_vertical': 0,
        'apply_random_features': 0,
        'none': 0
    }

    for img_name in images:
        label_name = img_name.replace('.png', '.json')
        img_path = folder + img_name
        lbl_path = folder + label_name

        result, img_path, lbl_path = asyncio.run(process_image(img_path, lbl_path))
        if result in count:
            count[result] += 1
        elif "+" in result:
            for r in result.split("+"):
                count[r] += 1

    print("Skipped:", count['skip'])
    print("Horizontal flips:", count['apply_flip_horizontal'])
    print("Vertical flips:", count['apply_flip_vertical'])
    print("Feature aug:", count['apply_random_features'])
>>>>>>> 70f2afaad93a7194b5e7131c2056d060e0901a74
