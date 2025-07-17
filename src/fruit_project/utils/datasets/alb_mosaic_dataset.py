# SPDX-FileCopyrightText: 2025 Mohamed Khayat
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import Tuple, Dict, Any, List
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations import Compose
from tqdm import tqdm

from fruit_project.models.transforms_factory import get_bbox_params
from .det_dataset import DET_DS, format_for_hf_processor


class AlbumentationsMosaicDataset(Dataset):
    """
    Dataset wrapper that applies Albumentations' native Mosaic augmentation,
    following the correct API based on official documentation.
    """

    has_warned_mosaic = False

    def __init__(
        self,
        dataset: DET_DS,
        current_epoch: int = 0,
        hard_transforms: Compose = None,
        easy_transforms: Compose = None,
        cfg=None,
    ):
        self.dataset = dataset
        self.target_size = cfg.model.input_size
        self.mosaic_prob = cfg.mosaic.prob
        self.disable_mosaic_epochs = cfg.mosaic.disable_epoch
        self.current_epoch = current_epoch
        self.total_epochs = cfg.epochs
        # Copy dataset attributes
        self.processor = dataset.processor
        self.id2lbl = dataset.id2lbl
        self.lbl2id = dataset.lbl2id
        self.labels = dataset.labels
        self.image_paths = dataset.image_paths
        self.label_paths = dataset.label_paths
        self.config_dir = dataset.config_dir
        self.input_size = dataset.input_size

        self.mosaic_transform = A.Mosaic(
            grid_yx=(2, 2),
            target_size=(self.target_size, self.target_size),
            cell_shape=(self.target_size // 2, self.target_size // 2),
            fill=114,
            center_range=(0.4, 0.6),
            metadata_key="mosaic_metadata",
            p=1.0,
        )

        self.bbox_params = get_bbox_params(cfg)
        self.hard_transforms = hard_transforms
        self.easy_transforms = easy_transforms

        mosaic_pipeline = [
            self.mosaic_transform,
            A.Resize(self.target_size, self.target_size),
        ]
        if self.hard_transforms:
            mosaic_pipeline.extend(self.hard_transforms.transforms)

        self.mosaic_compose = A.Compose(
            mosaic_pipeline,
            bbox_params=self.bbox_params,
        )

        easy_pipeline = [A.Resize(self.target_size, self.target_size)]
        if self.easy_transforms:
            easy_pipeline.extend(self.easy_transforms.transforms)

        self.easy_compose = A.Compose(
            easy_pipeline,
            bbox_params=self.bbox_params,
        )

    def update_epoch(self, epoch: int):
        """Update current epoch for mosaic scheduling."""
        self.current_epoch = epoch

    def should_apply_mosaic(self) -> bool:
        """Determine if mosaic should be applied based on epoch and probability."""
        if self.current_epoch >= (self.total_epochs - self.disable_mosaic_epochs):
            if not AlbumentationsMosaicDataset.has_warned_mosaic and (
                (torch.utils.data.get_worker_info() is None)
                or (torch.utils.data.get_worker_info().id == 0)
            ):
                AlbumentationsMosaicDataset.has_warned_mosaic = True
                tqdm.write("switched off mosaic")
            return False
        return np.random.rand() < self.mosaic_prob

    def _validate_and_clip_bbox(
        self, bbox: List[float], img_width: int, img_height: int
    ) -> List[float] | None:
        """Validate and clip bounding box coordinates."""
        x, y, w, h = bbox
        x = max(0, min(x, img_width - 1))
        y = max(0, min(y, img_height - 1))
        w = max(1, min(w, img_width - x))
        h = max(1, min(h, img_height - y))
        return [float(x), float(y), float(w), float(h)]

    def _prepare_mosaic_metadata(self, primary_idx: int) -> List[Dict[str, Any]]:
        """
        Prepare metadata for Albumentations Mosaic transform.
        This now returns a LIST OF DICTIONARIES, as required by the docs.
        """
        additional_indices = np.random.choice(
            [i for i in range(len(self.dataset)) if i != primary_idx],
            size=3,
            replace=True,
        )

        mosaic_items = []

        for idx in additional_indices:
            img, boxes, labels = self.dataset.get_raw_item(idx)
            img_height, img_width = img.shape[:2]

            coco_boxes = []
            valid_labels = []

            if len(boxes) > 0:
                for box, label in zip(boxes, labels):
                    clipped_box = self._validate_and_clip_bbox(
                        box, img_width, img_height
                    )
                    if clipped_box is not None:
                        coco_boxes.append(clipped_box)
                        valid_labels.append(int(label))

            mosaic_items.append(
                {"image": img, "bboxes": coco_boxes, "labels": valid_labels}
            )

        return mosaic_items

    def _apply_mosaic_augmentation(self, idx: int) -> Tuple[np.ndarray, List, List]:
        """Apply Albumentations Mosaic transform."""
        primary_img, primary_boxes, primary_labels = self.dataset.get_raw_item(idx)
        img_height, img_width = primary_img.shape[:2]

        primary_coco_boxes = []
        valid_primary_labels = []

        if len(primary_boxes) > 0:
            for box, label in zip(primary_boxes, primary_labels):
                clipped_box = self._validate_and_clip_bbox(box, img_width, img_height)
                if clipped_box is not None:
                    primary_coco_boxes.append(clipped_box)
                    valid_primary_labels.append(int(label))

        metadata_list = self._prepare_mosaic_metadata(idx)
        try:
            # Use the pre-composed mosaic+hard transform pipeline
            augmented = self.mosaic_compose(
                image=primary_img,
                bboxes=primary_coco_boxes,
                labels=valid_primary_labels,
                mosaic_metadata=metadata_list,
            )
            return augmented["image"], augmented["bboxes"], augmented["labels"]
        except Exception as e:
            tqdm.write(f"Mosaic augmentation failed for idx {idx}: {e}. Falling back.")
            # Fallback should now use the easy transform
            return self._apply_fallback_transform(idx, use_easy_transforms=True)

    def _apply_fallback_transform(
        self, idx: int, use_easy_transforms: bool = False
    ) -> Tuple[np.ndarray, List, List]:
        """Apply fallback transforms when mosaic fails or is disabled."""
        img, boxes, labels = self.dataset.get_raw_item(idx)
        img_height, img_width = img.shape[:2]

        coco_boxes = []
        valid_labels = []

        if len(boxes) > 0:
            for box, label in zip(boxes, labels):
                clipped_box = self._validate_and_clip_bbox(box, img_width, img_height)
                if clipped_box is not None:
                    coco_boxes.append(clipped_box)
                    valid_labels.append(int(label))

        is_final_epochs = not self.should_apply_mosaic()

        if is_final_epochs or use_easy_transforms:
            transform_pipeline = self.easy_compose

        else:
            fallback_hard_pipeline = [A.Resize(self.target_size, self.target_size)]
            if self.hard_transforms:
                fallback_hard_pipeline.extend(self.hard_transforms.transforms)

            transform_pipeline = A.Compose(
                fallback_hard_pipeline, bbox_params=self.bbox_params
            )

        try:
            augmented = transform_pipeline(
                image=img, bboxes=coco_boxes, labels=valid_labels
            )
            return augmented["image"], augmented["bboxes"], augmented["labels"]
        except Exception as e:
            tqdm.write(f"Fallback transform failed for idx {idx}: {e}")
            resized = A.Resize(self.target_size, self.target_size)(image=img)
            return resized["image"], [], []

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """Get item with optional mosaic augmentation."""
        if self.should_apply_mosaic():
            img, boxes, labels = self._apply_mosaic_augmentation(idx)
        else:
            img, boxes, labels = self._apply_fallback_transform(idx)

        target = format_for_hf_processor(boxes, labels, idx)

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
                tqdm.write(f"Processor failed for idx {idx}: {e}")
                raise AttributeError("HuggingFace Processor failed")
        else:
            raise AttributeError("No processor found")


def create_albumentations_mosaic_dataset(
    dataset: DET_DS,
    hard_transforms: Compose = None,
    easy_transforms: Compose = None,
    cfg=None,
) -> AlbumentationsMosaicDataset:
    return AlbumentationsMosaicDataset(
        dataset=dataset,
        current_epoch=0,
        hard_transforms=hard_transforms,
        easy_transforms=easy_transforms,
        cfg=cfg,
    )
