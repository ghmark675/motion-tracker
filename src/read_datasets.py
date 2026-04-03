import os
import numpy as np
from glob import glob


def parse_line(line):
    parts = np.array(list(map(float, line.strip().split())))

    class_id = int(parts[0])
    bbox = parts[1:5]

    kp = parts[5:].reshape(-1, 3)  # (17, 3)

    return {
        "class_id": class_id,
        "bbox": bbox,
        "keypoints": kp[:, :2],  # (17, 2)
        "visible": kp[:, 2],  # (17,)
    }


def load_dataset_numpy(base_dir):
    dataset = []

    for split in ["train", "val"]:
        label_dir = os.path.join(base_dir, "labels", split)
        image_dir = os.path.join(base_dir, "images", split)

        label_files = glob(os.path.join(label_dir, "*.txt"))

        for label_path in label_files:
            file_name = os.path.splitext(os.path.basename(label_path))[0]

            image_path = None
            for ext in [".jpg", ".png", ".jpeg"]:
                p = os.path.join(image_dir, file_name + ext)
                if os.path.exists(p):
                    image_path = p
                    break

            with open(label_path, "r") as f:
                line = f.readline()

            ann = parse_line(line)

            dataset.append(
                {
                    "image_path": image_path,
                    "keypoints": ann["keypoints"],
                    "visible": ann["visible"],
                    "bbox": ann["bbox"],
                    "split": split,
                }
            )

    return dataset


if __name__ == "__main__":
    base_dir = r"H:\\golf_data\\keyframes_yolo2"

    data = load_dataset_numpy(base_dir)

    print(f"总样本数: {len(data)}")
    print("示例数据：")
    print(data[0])
