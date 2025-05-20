import json
import os

# Define class mapping (modify if needed)
class_map = {
    "court": 0,
    "building_52": 1,
    "building_5153": 2,
    "building_54": 3,
    "building_55": 4,
    "building_56": 5,
    "building_58": 6,
    "building_60": 7,
    "building_1": 8,
    "building_2": 9,
    "building_3": 10,
    "building_4": 11,
    "building_5": 12,
    "building_6": 13,
    "building_7": 14,
    "building_8": 15,
    "building_9": 16,
    "building_10": 17,
    "building_11": 18,
    "building_3a": 19,
    "building_ac": 20,
    "building_21": 21,
    "building_20": 22,
    "soccer_field": 23,
    "building_police": 24,
    "gate_uppercampus": 25,
    "building_29":26,
    "building_30":27,
    "gate_lowercampus": 28,
    "101_doorm":29,
    "102_doorm": 30,
    "103_doorm": 31,
    "111_doorm": 32,
    "112_doorm": 33,
    "113_doorm": 34,
    "114_doorm": 35,
    "115_doorm": 36,
    "116_doorm": 37,
    "east_road":38,
    "moriya_road":39,
    "ramatgolan_road":40,
    "eastborder_road":41,
    "shufersal_road":42,
    "metzil_road":43,
    "roundabout_building30":44,
    "roundabout_andarta":45,
    "roundabout_ramatgolan":46,
    "roundabout_moriya":47,
    "parking_uppercampus":48,

}

# Define paths
json_dir = "frames"  # Update to your JSON folder
output_dir = "dataset/labels"
os.makedirs(output_dir, exist_ok=True)

# Convert JSON annotations to YOLOv11 segmentation format
for file in os.listdir(json_dir):
    if file.endswith(".json"):
        with open(os.path.join(json_dir, file)) as f:
            data = json.load(f)

        image_filename = os.path.splitext(os.path.basename(data["imagePath"]))[0]
        label_file = os.path.join(output_dir, f"{image_filename}.txt")

        with open(label_file, "w") as lf:
            for shape in data["shapes"]:
                class_name = shape["label"]
                if class_name not in class_map:
                    continue  # Skip unknown classes

                class_id = class_map[class_name]
                points = shape["points"]

                # Normalize polygon points
                img_w, img_h = data["imageWidth"], data["imageHeight"]
                normalized_points = []
                for x, y in points:
                    normalized_x = x / img_w
                    normalized_y = y / img_h
                    normalized_points.extend([normalized_x, normalized_y])

                # Convert list to space-separated string
                points_str = " ".join(map(str, normalized_points))
                lf.write(f"{class_id} {points_str}\n")

print("✅ JSON annotations converted to YOLOv11 segmentation format.")

