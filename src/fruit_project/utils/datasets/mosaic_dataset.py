from typing import Dict, Tuple
import numpy as np
from torch.utils.data import Dataset
from .det_dataset import DET_DS, format_for_hf_processor


class UltralyticsStyleMosaic:
    """
    Ultralytics-style mosaic augmentation implementation.

    Key differences from standard implementations:
    - Uses 2x image size canvas for better context
    - Smart center point selection with bias
    - Efficient bbox transformation
    - Minimal memory footprint
    - Battle-tested coordinate transforms
    """

    def __init__(
        self,
        target_size: int = 640,
        center_range: float = 0.5,
        pad_val: int = 114,
    ):
        """
        Args:
            target_size: Final output image size
            center_range: Range for center point selection (0.5 = ±50% from center)
            pad_val: Padding value for empty areas
        """
        self.target_size = target_size
        self.center_range = center_range
        self.pad_val = pad_val
        self.n_images = 4
        # Pre-compute mosaic positions for efficiency
        self.positions = [(0, 0), (0, 1), (1, 0), (1, 1)]  # 2x2 grid

    def __call__(self, dataset: DET_DS) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply mosaic augmentation following Ultralytics methodology.

        Returns:
            Tuple of (mosaic_image, boxes, labels) where boxes are in COCO format
        """
        # Sample random images - allow replacement for small datasets
        indices = np.random.choice(len(dataset), self.n_images, replace=True)

        # Create 2x target_size canvas (Ultralytics signature approach)
        canvas_size = self.target_size * 2
        canvas = np.full((canvas_size, canvas_size, 3), self.pad_val, dtype=np.uint8)

        # Generate center point with controlled randomness
        center_x = int(
            np.random.uniform(
                self.target_size * (1 - self.center_range),
                self.target_size * (1 + self.center_range),
            )
        )
        center_y = int(
            np.random.uniform(
                self.target_size * (1 - self.center_range),
                self.target_size * (1 + self.center_range),
            )
        )

        all_boxes = []
        all_labels = []

        for i, idx in enumerate(indices):
            img, boxes, labels = dataset.get_raw_item(idx)
            h, w = img.shape[:2]

            # Calculate placement coordinates (Fixed Ultralytics approach)
            # 2x2 mosaic - proper quadrant assignment
            if i == 0:  # Top-left
                x1a, y1a = max(center_x - w, 0), max(center_y - h, 0)
                x2a, y2a = center_x, center_y
            elif i == 1:  # Top-right
                x1a, y1a = center_x, max(center_y - h, 0)
                x2a, y2a = min(center_x + w, canvas_size), center_y
            elif i == 2:  # Bottom-left
                x1a, y1a = max(center_x - w, 0), center_y
                x2a, y2a = center_x, min(center_y + h, canvas_size)
            else:  # Bottom-right (i == 3)
                x1a, y1a = center_x, center_y
                x2a, y2a = (
                    min(center_x + w, canvas_size),
                    min(center_y + h, canvas_size),
                )

            # Calculate source image coordinates properly
            x1b = max(0, w - (x2a - x1a))
            y1b = max(0, h - (y2a - y1a))
            x2b = w
            y2b = h

            # Handle edge cases for right and bottom quadrants
            if i == 1 or i == 3:  # Right side quadrants
                x1b = 0
                x2b = min(w, x2a - x1a)
            if i == 2 or i == 3:  # Bottom side quadrants
                y1b = 0
                y2b = min(h, y2a - y1a)

            # Ensure coordinates are valid
            x1a, y1a = max(0, x1a), max(0, y1a)
            x2a, y2a = min(canvas_size, x2a), min(canvas_size, y2a)
            x1b, y1b = max(0, x1b), max(0, y1b)
            x2b, y2b = min(w, x2b), min(h, y2b)

            # Place image on canvas
            if x2a > x1a and y2a > y1a and x2b > x1b and y2b > y1b:
                canvas[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]

                # Transform bounding boxes
                if len(boxes) > 0:
                    boxes_transformed = boxes.copy().astype(np.float32)

                    # Apply offset transformation
                    offset_x = x1a - x1b
                    offset_y = y1a - y1b

                    boxes_transformed[:, 0] += offset_x  # x
                    boxes_transformed[:, 1] += offset_y  # y

                    # Clip boxes to canvas bounds
                    boxes_transformed[:, 0] = np.clip(
                        boxes_transformed[:, 0], 0, canvas_size
                    )
                    boxes_transformed[:, 1] = np.clip(
                        boxes_transformed[:, 1], 0, canvas_size
                    )

                    # Adjust width and height after clipping
                    boxes_transformed[:, 2] = np.minimum(
                        boxes_transformed[:, 2], canvas_size - boxes_transformed[:, 0]
                    )
                    boxes_transformed[:, 3] = np.minimum(
                        boxes_transformed[:, 3], canvas_size - boxes_transformed[:, 1]
                    )

                    # Only keep boxes with meaningful area (stricter filtering)
                    areas = boxes_transformed[:, 2] * boxes_transformed[:, 3]
                    # Use minimum area of 16 pixels (4x4) and ensure box dimensions > 2 pixels
                    min_dim = 2.0
                    valid_mask = (
                        (areas > 16.0)
                        & (boxes_transformed[:, 2] > min_dim)
                        & (boxes_transformed[:, 3] > min_dim)
                    )

                    if np.any(valid_mask):
                        all_boxes.append(boxes_transformed[valid_mask])
                        all_labels.append(labels[valid_mask])

        # Combine all annotations
        if len(all_boxes) > 0:
            final_boxes = np.concatenate(all_boxes, axis=0)
            final_labels = np.concatenate(all_labels, axis=0)
        else:
            final_boxes = np.array([]).reshape(0, 4)
            final_labels = np.array([])

        # Final crop to target size (Ultralytics approach)
        crop_x1 = center_x - self.target_size // 2
        crop_y1 = center_y - self.target_size // 2
        crop_x2 = crop_x1 + self.target_size
        crop_y2 = crop_y1 + self.target_size

        # Ensure crop bounds are valid
        crop_x1 = max(0, min(crop_x1, canvas_size - self.target_size))
        crop_y1 = max(0, min(crop_y1, canvas_size - self.target_size))
        crop_x2 = crop_x1 + self.target_size
        crop_y2 = crop_y1 + self.target_size

        final_image = canvas[crop_y1:crop_y2, crop_x1:crop_x2]

        # Transform boxes to final coordinate system
        if len(final_boxes) > 0:
            final_boxes[:, 0] -= crop_x1
            final_boxes[:, 1] -= crop_y1

            # Final clipping
            final_boxes[:, 0] = np.clip(final_boxes[:, 0], 0, self.target_size)
            final_boxes[:, 1] = np.clip(final_boxes[:, 1], 0, self.target_size)
            final_boxes[:, 2] = np.clip(
                final_boxes[:, 2], 0, self.target_size - final_boxes[:, 0]
            )
            final_boxes[:, 3] = np.clip(
                final_boxes[:, 3], 0, self.target_size - final_boxes[:, 1]
            )

            # Final area filter (more conservative)
            areas = final_boxes[:, 2] * final_boxes[:, 3]
            min_dim = 2.0
            valid_mask = (
                (areas > 16.0)
                & (final_boxes[:, 2] > min_dim)
                & (final_boxes[:, 3] > min_dim)
            )

            final_boxes = final_boxes[valid_mask]
            final_labels = final_labels[valid_mask]

        return final_image, final_boxes, final_labels


class UltralyticsStyleMosaicDataset(Dataset):
    """
    Dataset wrapper that applies Ultralytics-style mosaic augmentation.

    Features:
    - Configurable mosaic probability
    - Automatic epoch-based mosaic disable (like YOLOv8)
    - Efficient memory usage
    - COCO format compatibility
    """

    def __init__(
        self,
        dataset: DET_DS,
        target_size: int = 640,
        mosaic_prob: float = 0.8,
        disable_mosaic_epochs: int = 10,
        current_epoch: int = 0,
        total_epochs: int = 100,
    ):
        """
        Args:
            dataset: Base dataset (your DET_DS instance)
            target_size: Final image size
            mosaic_prob: Probability of applying mosaic (0.0-1.0)
            n_images: Number of images to mosaic (4 or 9)
            disable_mosaic_epochs: Disable mosaic in last N epochs
            current_epoch: Current training epoch
            total_epochs: Total training epochs
        """
        self.dataset = dataset
        self.easy_transforms = None
        self.target_size = target_size
        self.mosaic_prob = mosaic_prob
        self.disable_mosaic_epochs = disable_mosaic_epochs
        self.current_epoch = current_epoch
        self.total_epochs = total_epochs

        # Initialize mosaic augmentation
        self.mosaic_aug = UltralyticsStyleMosaic(target_size=target_size)

        # Copy dataset attributes
        self.transforms = dataset.transforms
        self.processor = dataset.processor
        self.id2lbl = dataset.id2lbl
        self.lbl2id = dataset.lbl2id
        self.labels = dataset.labels
        self.image_paths = dataset.image_paths
        self.label_paths = dataset.label_paths
        self.config_dir = dataset.config_dir
        self.input_size = dataset.input_size

    def update_epoch(self, epoch: int):
        """Update current epoch for mosaic scheduling."""
        self.current_epoch = epoch

    def should_apply_mosaic(self) -> bool:
        """
        Determine if mosaic should be applied based on epoch and probability.
        Following YOLOv8 pattern: disable mosaic in last few epochs.
        """
        # Disable mosaic in last N epochs
        if self.current_epoch >= (self.total_epochs - self.disable_mosaic_epochs):
            self.dataset.transforms = self.easy_transforms
            return False

        # Apply based on probability
        return np.random.rand() < self.mosaic_prob

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """Get item with optional mosaic augmentation."""

        if self.should_apply_mosaic():
            # Apply mosaic augmentation
            img, boxes, labels = self.mosaic_aug(self.dataset)

            # Convert to list format for albumentations
            boxes = boxes.tolist() if len(boxes) > 0 else []
            labels = labels.tolist() if len(labels) > 0 else []
        else:
            # Use single image
            img, boxes, labels = self.dataset.get_raw_item(idx)
            boxes = boxes.tolist() if len(boxes) > 0 else []
            labels = labels.tolist() if len(labels) > 0 else []

        # Apply other augmentations
        if self.transforms:
            try:
                augmented = self.transforms(image=img, bboxes=boxes, labels=labels)
                img = augmented["image"]
                boxes = augmented["bboxes"]
                labels = augmented["labels"]
            except Exception as e:
                # Fallback to original if augmentation fails
                print(f"Augmentation failed for idx {idx}: {e}")
                img, boxes, labels = self.dataset.get_raw_item(idx)
                boxes = boxes.tolist() if len(boxes) > 0 else []
                labels = labels.tolist() if len(labels) > 0 else []

        # Prepare target in COCO format
        target = format_for_hf_processor(boxes, labels, idx)

        # Apply processor if available
        if hasattr(self, "processor") and self.processor:
            try:
                result = self.processor(
                    images=img,
                    annotations=target,
                    return_tensors="pt",
                    do_pad=True,
                )
                result = {k: v[0] for k, v in result.items()}
                return result
            except Exception as e:
                print(f"Processor failed for idx {idx}: {e}")
                raise AttributeError("HuggingFace Processor failed")
        else:
            raise AttributeError("no processor found")


def create_ultralytics_mosaic_dataset(
    dataset: DET_DS,
    target_size: int = 640,
    mosaic_prob: float = 0.8,
    disable_mosaic_epochs: int = 10,
    total_epochs: int = 100,
) -> UltralyticsStyleMosaicDataset:
    """
    Convenience function to create Ultralytics-style mosaic dataset.

    Usage:
        # Create mosaic dataset
        mosaic_dataset = create_ultralytics_mosaic_dataset(
            dataset=your_dataset,
            target_size=640,
            mosaic_prob=0.8,
            n_images=4,
            total_epochs=100
        )

        # Update epoch during training
        for epoch in range(100):
            mosaic_dataset.update_epoch(epoch)
            # ... training loop
    """
    return UltralyticsStyleMosaicDataset(
        dataset=dataset,
        target_size=target_size,
        mosaic_prob=mosaic_prob,
        disable_mosaic_epochs=disable_mosaic_epochs,
        current_epoch=0,
        total_epochs=total_epochs,
    )
