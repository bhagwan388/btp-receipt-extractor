import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import random
import shutil

def convert_coords_to_yolo(img_w, img_h, coords):
    """Converts SROIE coordinates (x1,y1,x2,y2,x3,y3,x4,y4) to YOLO format."""
    x_coords = [float(coords[i]) for i in [0, 2, 4, 6]]
    y_coords = [float(coords[i]) for i in [1, 3, 5, 7]]
    x_min, y_min = min(x_coords), min(y_coords)
    x_max, y_max = max(x_coords), max(y_coords)
    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    width = x_max - x_min
    height = y_max - y_min
    return x_center / img_w, y_center / img_h, width / img_w, height / img_h

def process_sroie_for_text_detection(base_path):
    print("Starting SROIE to YOLO format conversion (for single 'text' class)...")
    raw_data_path = base_path / "data" / "raw_sroie" / "SROIE2019"
    yolo_output_path = base_path / "data" / "yolo_format"

    if yolo_output_path.exists():
        shutil.rmtree(yolo_output_path)

    all_images = sorted(list((raw_data_path / "train" / "img").glob("*.jpg")))
    random.shuffle(all_images)

    split_index = int(len(all_images) * 0.9)
    train_images = all_images[:split_index]
    val_images = all_images[split_index:]

    splits = {"train": train_images, "val": val_images}

    for split_name, image_paths in splits.items():
        print(f"\nProcessing {split_name} split...")
        image_out_dir = yolo_output_path / "images" / split_name
        label_out_dir = yolo_output_path / "labels" / split_name
        image_out_dir.mkdir(parents=True, exist_ok=True)
        label_out_dir.mkdir(parents=True, exist_ok=True)

        for img_path in tqdm(image_paths):
            shutil.copy(img_path, image_out_dir)
            with Image.open(img_path) as img:
                img_w, img_h = img.size

            # Use the 'box' folder which contains coordinates for EVERY word.
            label_path = raw_data_path / "train" / "box" / (img_path.stem + ".txt")
            if not label_path.exists():
                continue

            yolo_annotations = []
            with open(label_path, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    parts = line.strip().split(',', 8)
                    if len(parts) < 9: continue

                    coords = parts[:8]
                    # Every valid line is now class 0 ('text')
                    class_id = 0
                    yolo_coords = convert_coords_to_yolo(img_w, img_h, coords)
                    yolo_annotations.append(f"{class_id} {' '.join(map(str, yolo_coords))}")

            with open(label_out_dir / (img_path.stem + ".txt"), 'w') as f:
                f.write("\n".join(yolo_annotations))

    print("\nSROIE data processing for text detection is complete!")

if __name__ == '__main__':
    process_sroie_for_text_detection(Path.cwd())