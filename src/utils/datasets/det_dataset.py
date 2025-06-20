import random
import numpy as np
import torch
from torch.utils.data import Dataset
import pathlib
import cv2
import yaml


class DET_DS(Dataset):
    def __init__(
        self,
        root_dir,
        type,
        image_dir,
        label_dir,
        config_file,
        transforms=None,
        input_size=224,
    ):
        self.root_dir = pathlib.Path("data", root_dir)
        self.type = type
        self.image_dir = self.root_dir / self.type / image_dir
        self.label_dir = self.root_dir / self.type / label_dir
        self.transforms = transforms
        self.input_size = input_size
        self.config_dir = self.root_dir / config_file
        raw_paths = sorted(list(pathlib.Path(self.image_dir).glob("*.jpg")))

        valid = []
        for p in raw_paths:
            if cv2.imread(str(p)) is not None:
                valid.append(p)
            else:
                print(f"[WARN] dropping bad image {p.name}")
        self.image_paths = valid
        with open(self.config_dir, "r") as f:
            config = yaml.safe_load(f)
        self.labels = [name for name in config["names"]]
        self.id2lbl = dict(enumerate(self.labels))
        self.lbl2id = {v: k for k, v in self.id2lbl.items()}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label_path = pathlib.Path(self.label_dir) / (image_path.stem + ".txt")
        img = cv2.imread(image_path)
        if img is None:
            new_idx = random.randrange(len(self.image_paths))
            print("img is empty")
            return self.__getitem__(new_idx)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        boxes = []
        labels = []

        height, width = img.shape[:2]
        with open(label_path, "r") as f:
            for line in f.readlines():
                cls, cx, cy, w, h = map(float, line.strip().split())

                x1 = (cx - w / 2) * width
                y1 = (cy - h / 2) * height
                box_w = w * width
                box_h = h * height
                boxes.append([x1, y1, box_w, box_h])
                labels.append(int(cls))

        if self.transforms:
            augmented = self.transforms(
                image=img, bboxes=np.array(boxes), labels=np.array(labels)
            )
            img = augmented["image"]
            boxes = augmented["bboxes"]
            labels = augmented["labels"]

        target = {
            "image_id": idx,
            "annotations": [
                {
                    "bbox": box,
                    "category_id": label,
                    "area": box[2] * box[3],
                    "iscrowd": 0,
                }
                for box, label in zip(boxes, labels)
            ],
            "orig_size": torch.tensor([height, width]),
        }
        return img, target
