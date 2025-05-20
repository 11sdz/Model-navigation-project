import torch
import yaml
from ultralytics import YOLO
from pathlib import Path
from sklearn.model_selection import KFold
import shutil


# 🛠 Configuration
EPOCHS = 100
IMG_SIZE = 640
NUM_FOLDS = 5
MODEL_PATH = 'yolo11n-seg.pt'
PROJECT_DIR = 'runs/kfold'
ALLOWED_EXTENSIONS = [".png", ".jpg", ".jpeg"]


def get_class_names():
    return [
        "court", "building_52", "building_5153", "building_54", "building_55", "building_56",
        "building_58", "building_60", "building_1", "building_2", "building_3", "building_4",
        "building_5", "building_6", "building_7", "building_8", "building_9", "building_10",
        "building_11", "building_3a", "building_ac", "building_21", "building_20", "soccer_field",
        "building_police", "gate_uppercampus", "building_29", "building_30", "gate_lowercampus",
        "101_doorm", "102_doorm", "103_doorm", "111_doorm", "112_doorm", "113_doorm",
        "114_doorm", "115_doorm", "116_doorm", "east_road", "moriya_road", "ramatgolan_road",
        "eastborder_road", "shufersal_road", "metzil_road", "roundabout_building30",
        "roundabout_andarta", "roundabout_ramatgolan", "roundabout_moriya", "parking_uppercampus"
    ]


def prepare_directories(fold_dir):
    for split in ['train', 'val']:
        (fold_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (fold_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def copy_data(indices, image_files, labels_dir, img_dst, lbl_dst):
    for idx in indices:
        img_path = image_files[idx]
        lbl_path = labels_dir / f"{img_path.stem}.txt"
        shutil.copy(img_path, img_dst / img_path.name)
        if lbl_path.exists():
            shutil.copy(lbl_path, lbl_dst / lbl_path.name)
        else:
            print(f"⚠️ Warning: Missing label for {img_path.name}")


def write_data_yaml(fold_dir, class_names):
    yaml_path = fold_dir / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump({
            'train': str(fold_dir / 'images' / 'train'),
            'val': str(fold_dir / 'images' / 'val'),
            'nc': len(class_names),
            'names': class_names
        }, f)


def get_next_run_name(project_dir, base_name):
    i = 0
    while (Path(project_dir) / f"{base_name}_{i}").exists():
        i += 1
    return f"{base_name}_{i}"


def kfold_split(images_dir, labels_dir, num_folds=NUM_FOLDS, seed=42):
    class_names = get_class_names()
    image_files = sorted([f for f in images_dir.glob("*") if f.suffix.lower() in ALLOWED_EXTENSIONS])
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)

    for fold, (train_idx, val_idx) in enumerate(kf.split(image_files)):
        print(f"\n🟡 Starting Fold {fold + 1}/{num_folds}")
        fold_dir = images_dir.parent / f"fold{fold}"
        prepare_directories(fold_dir)

        copy_data(train_idx, image_files, labels_dir, fold_dir / "images" / "train", fold_dir / "labels" / "train")
        copy_data(val_idx, image_files, labels_dir, fold_dir / "images" / "val", fold_dir / "labels" / "val")
        write_data_yaml(fold_dir, class_names)

        run_name = get_next_run_name(PROJECT_DIR, f"fold{fold}")
        model = YOLO(MODEL_PATH)
        results = model.train(
            data=str(fold_dir / 'data.yaml'),
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            project=PROJECT_DIR,
            name=run_name,
            exist_ok=True
        )

        # Optional: log mAP50 if available
        try:
            metrics = results.metrics
            print(f"📊 Fold {fold} mAP50: {metrics.map50:.4f}")
        except Exception as e:
            print(f"⚠️ Could not retrieve mAP50 for fold {fold}: {e}")


def main():
    print(f"Torch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    base_path = Path("C:/Users/danie/PycharmProjects/mymodel2/dataset_split")
    images_dir = base_path / "images" / "all"
    labels_dir = base_path / "labels" / "all"

    print(f"Images path: {images_dir}")
    print(f"Labels path: {labels_dir}")

    kfold_split(images_dir, labels_dir)


if __name__ == '__main__':
    main()
