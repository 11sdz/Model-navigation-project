# YOLOv11 Segmentation Dataset Preparation

This repository contains scripts for preparing, augmenting, and training image datasets for YOLOv11 segmentation tasks. The workflow is as follows:

1. **Download Videos**: Use `script.py` to download videos from YouTube.
2. **Label Images**: Label the images in the `frames/` folder using LabelMe.
3. **Augment Dataset**: Use the augmentation scripts in the `augmentations/` directory to expand your dataset.
4. **Convert Annotations**: Run `prepare.py` to convert JSON annotations to YOLO-compatible TXT files.
5. **Copy Matching Images**: Use `copy-matching-png.py` to copy images that match the generated TXT label files.
6. **Train Model**: Use `train-kfold.py` to train the model using k-fold cross-validation.
7. **Predict**: Use `predict.py` to run predictions on new videos.

## Project Structure

- **prepare.py**: Converts JSON annotations from the `frames/` directory into YOLOv11 segmentation format, saving the labels in `dataset/labels/`.
- **copy-matching-png.py**: Copies matching PNG files based on JSON annotations.
- **copy-frames.py**: Copies frame data for further processing.
- **script.py**: Downloads videos from YouTube.
- **augmentations/**: Contains scripts for data augmentation:
  - `main.py`: Main augmentation script.
  - `augmentations.py`: Defines augmentation functions.
  - `utils.py`: Utility functions for augmentation.
- **frames/**: Contains JSON files with frame annotations.
- **runs/**: Contains training and prediction outputs:
  - `segment/`: Outputs from segmentation training.
  - `kfold/`: Outputs from k-fold cross-validation training.

## Prerequisites

- Python 3.x
- Required Python packages (install via `pip install -r requirements.txt`):
  - json
  - os
  - cv2
  - torch
  - ultralytics
  - yt_dlp
  - tqdm
  - yaml
  - sklearn

## Usage

1. **Download Videos**:
   - Use `script.py` to download videos from YouTube.

2. **Label Images**:
   - Label the images in the `frames/` folder using LabelMe.

3. **Augment Dataset**:
   - Navigate to the `augmentations/` directory.
   - Run the main augmentation script:
     ```bash
     python main.py
     ```

4. **Convert Annotations**:
   - Run `prepare.py` to convert JSON annotations to YOLO format:
     ```bash
     python prepare.py
     ```
   - The converted labels will be saved in `dataset/labels/`.

5. **Copy Matching Images**:
   - Use `copy-matching-png.py` to copy PNG files matching your JSON annotations:
     ```bash
     python copy-matching-png.py
     ```

6. **Train Model**:
   - Use `train-kfold.py` to train the model using k-fold cross-validation:
     ```bash
     python train-kfold.py
     ```

7. **Predict**:
   - Use `predict.py` to run predictions on new videos:
     ```bash
     python predict.py
     ```

## Class Mapping

The `prepare.py` script uses a predefined class mapping to convert class names to numeric IDs. Modify the `class_map` dictionary in `prepare.py` if your classes differ.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.
