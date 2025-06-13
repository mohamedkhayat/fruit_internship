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
        self.image_paths = sorted(list(pathlib.Path(self.image_dir).glob("*.jpg")))
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
        img = cv2.imread(str(image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        boxes = []
        labels = []

        height, width = img.shape[:2]
        with open(label_path, "r") as f:
            for line in f.readlines():
                cls, cx, cy, w, h = map(float, line.strip().split())

                x1 = (cx - w / 2) * width
                y1 = (cy - h / 2) * height
                x2 = (cx + w / 2) * width
                y2 = (cy + h / 2) * height

                boxes.append([x1, y1, x2, y2])
                labels.append(int(cls))

        if self.transforms:
            augmented = self.transforms(
                image=img, bboxes=np.array(boxes), labels=np.array(labels)
            )
            img = augmented["image"]
            boxes = augmented["bboxes"]
            labels = augmented["labels"]

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        return img, target


if __name__ == "__main__":
    ds = DET_DS(
        "data/big-Fruits-detection",
        "train",
        "images",
        "labels",
        config_file="data.yaml",
    )
    img, target = ds[0]
    print(img.shape, target)
