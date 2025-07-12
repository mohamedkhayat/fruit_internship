import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .det_dataset import DET_DS


def mosaic_augmentation(dataset: DET_DS, target_size: int, num_to_mosaic: int = 4):
    """
    Creates a mosaic image by combining multiple images from the dataset.
    This implementation is inspired by the YOLOv5 mosaic augmentation logic.

    Args:
        dataset: The base dataset object (your DET_DS instance) to pull images from.
        target_size: The final output size of the mosaic image (e.g., 480).
        num_to_mosaic: The number of images to combine (almost always 4).

    Returns:
        Tuple of (mosaic_image, combined_boxes, combined_labels) in numpy format.
        Boxes are in COCO format [x_min, y_min, width, height].
    """
    combined_boxes = []
    combined_labels = []
    
    center_x = int(np.random.uniform(target_size // 2, target_size * 1.5))
    center_y = int(np.random.uniform(target_size // 2, target_size * 1.5))

    indices = [np.random.randint(0, len(dataset)) for _ in range(num_to_mosaic)]
    
    mosaic_image = np.full((target_size * 2, target_size * 2, 3), 114, dtype=np.uint8)

    for i, index in enumerate(indices):
        img, boxes, labels = dataset.get_raw_item(index)
        h, w, _ = img.shape

        if i == 0:  # Top-left quadrant
            x1a, y1a, x2a, y2a = max(center_x - w, 0), max(center_y - h, 0), center_x, center_y
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
        elif i == 1:  # Top-right quadrant
            x1a, y1a, x2a, y2a = center_x, max(center_y - h, 0), min(center_x + w, target_size * 2), center_y
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # Bottom-left quadrant
            x1a, y1a, x2a, y2a = max(center_x - w, 0), center_y, center_x, min(center_y + h, target_size * 2)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        else:  # Bottom-right quadrant
            x1a, y1a, x2a, y2a = center_x, center_y, min(center_x + w, target_size * 2), min(center_y + h, target_size * 2)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        mosaic_image[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
        
        pad_x = x1a - x1b
        pad_y = y1a - y1b

        if boxes.size > 0:
            boxes[:, 0] += pad_x  # Update x_min
            boxes[:, 1] += pad_y  # Update y_min
            combined_boxes.append(boxes)
            combined_labels.append(labels)

    if combined_boxes:
        combined_boxes = np.concatenate(combined_boxes, 0)
        combined_labels = np.concatenate(combined_labels, 0)

        np.clip(combined_boxes[:, 0], 0, 2 * target_size, out=combined_boxes[:, 0])
        np.clip(combined_boxes[:, 1], 0, 2 * target_size, out=combined_boxes[:, 1])

    final_image = mosaic_image[
        center_y - target_size // 2 : center_y + target_size // 2,
        center_x - target_size // 2 : center_x + target_size // 2,
    ]

    if combined_boxes:
        combined_boxes[:, 0] -= (center_x - target_size // 2)
        combined_boxes[:, 1] -= (center_y - target_size // 2)

        i = (combined_boxes[:, 2] > 2) & (combined_boxes[:, 3] > 2)
        combined_boxes = combined_boxes[i]
        combined_labels = combined_labels[i]

    return final_image, combined_boxes, combined_labels


class MosaicDataset(Dataset):
    """
    A Dataset wrapper that applies Mosaic augmentation.
    With a given probability, it fetches 4 samples and combines them into one.
    Otherwise, it returns a single sample. It then passes the result through
    the standard albumentations pipeline.
    """
    def __init__(self, dataset: DET_DS, target_size: int, mosaic_prob: float = 0.8):
        self.dataset = dataset
        self.target_size = target_size
        self.mosaic_prob = mosaic_prob

        self.transforms = self.dataset.transforms
        self.processor = self.dataset.processor
        self.id2lbl = self.dataset.id2lbl
        self.labels = self.dataset.labels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if np.random.rand() < self.mosaic_prob:
            img, boxes, labels = mosaic_augmentation(self.dataset, self.target_size)
        else:
            img, boxes, labels = self.dataset.get_raw_item(idx)

        if self.transforms:
            try:
                augmented = self.transforms(image=img, bboxes=boxes, labels=labels)
                img = augmented['image']
                boxes = np.array(augmented['bboxes'])
                labels = np.array(augmented['labels'])
            except ValueError:
                return self.__getitem__(np.random.randint(0, len(self)))


        if len(labels) == 0:
            return self.__getitem__(np.random.randint(0, len(self)))
            
        target = {
            "image_id": torch.tensor(idx),
            "annotations": [
                {"bbox": box.tolist(), "category_id": label, "area": box[2]*box[3], "iscrowd": 0}
                for box, label in zip(boxes, labels)
            ]
        }

        if self.processor:
            result = self.processor(images=img, annotations=target, return_tensors="pt")
            return {k: v.squeeze(0) for k, v in result.items()}
        else:
            raise AttributeError("HuggingFace Processor not found in MosaicDataset")