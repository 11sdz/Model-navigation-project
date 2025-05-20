import os
import os
import json
import base64
import cv2
import cv2

output_dir = "..\\augmented\\"
output_dir = "..\\augmented\\"

def list_all_files(directory):
    image_ext = ["png"]
    label_ext = ["json"]
    files = os.listdir(directory)
    labels = [f for f in files if any(f.endswith(ext) for ext in label_ext)]
    images = [f for f in files if any(f.endswith(ext) for ext in image_ext) and os.path.splitext(f)[0]+".json" in labels]
    return images, labels

async def read_image(image_path):
    image = cv2.imread(image_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

async def label_feature_augmentation(label_path, image):
    with open(label_path, 'r') as f:
        label_data = json.load(f)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    _, img_encoded = cv2.imencode('.png', image)
    label_data['imageData'] = base64.b64encode(img_encoded).decode('utf-8')
    return label_data

async def label_flip_horizontal(label_path, width, image):
    with open(label_path, 'r') as f:
        label_data = json.load(f)
    for shape in label_data['shapes']:
async def read_image(image_path):
    image = cv2.imread(image_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

async def label_feature_augmentation(label_path, image):
    with open(label_path, 'r') as f:
        label_data = json.load(f)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    _, img_encoded = cv2.imencode('.png', image)
    label_data['imageData'] = base64.b64encode(img_encoded).decode('utf-8')
    return label_data

async def label_flip_horizontal(label_path, width, image):
    with open(label_path, 'r') as f:
        label_data = json.load(f)
    for shape in label_data['shapes']:
        if shape['shape_type'] == 'polygon':
            shape['points'] = [[width - x, y] for x, y in shape['points']]
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            shape['points'] = [[width - x, y] for x, y in shape['points']]
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    _, img_encoded = cv2.imencode('.png', image)
    label_data['imageData'] = base64.b64encode(img_encoded).decode('utf-8')
    return label_data
    label_data['imageData'] = base64.b64encode(img_encoded).decode('utf-8')
    return label_data

async def label_flip_vertically(label_path, height, image):
    with open(label_path, 'r') as f:
        label_data = json.load(f)
    for shape in label_data['shapes']:
async def label_flip_vertically(label_path, height, image):
    with open(label_path, 'r') as f:
        label_data = json.load(f)
    for shape in label_data['shapes']:
        if shape['shape_type'] == 'polygon':
            shape['points'] = [[x, height - y] for x, y in shape['points']]
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            shape['points'] = [[x, height - y] for x, y in shape['points']]
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    _, img_encoded = cv2.imencode('.png', image)
    label_data['imageData'] = base64.b64encode(img_encoded).decode('utf-8')
    return label_data
    label_data['imageData'] = base64.b64encode(img_encoded).decode('utf-8')
    return label_data

async def save_augmented_pair(image, label, image_path, label_path, output_folder, prefix):
    image_name = os.path.basename(image_path)
    label_name = os.path.basename(label_path)

    save_image_path = os.path.join(output_folder, f"{prefix}_{image_name}")
    save_label_path = os.path.join(output_folder, f"{prefix}_{label_name}")

    label['imagePath'] = save_image_path

    cv2.imwrite(save_image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    with open(save_label_path, 'w') as f:
        json.dump(label, f, indent=4)
async def save_augmented_pair(image, label, image_path, label_path, output_folder, prefix):
    image_name = os.path.basename(image_path)
    label_name = os.path.basename(label_path)

    save_image_path = os.path.join(output_folder, f"{prefix}_{image_name}")
    save_label_path = os.path.join(output_folder, f"{prefix}_{label_name}")

    label['imagePath'] = save_image_path

    cv2.imwrite(save_image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    with open(save_label_path, 'w') as f:
        json.dump(label, f, indent=4)

    return save_image_path, save_label_path
    return save_image_path, save_label_path

