import albumentations as A
import os
import cv2

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"


def get_transforms(cfg):
    transforms = {
        "train": A.Compose(
            [
                A.RandomSizedBBoxSafeCrop(
                    height=cfg.model.input_size,
                    width=cfg.model.input_size,
                    erosion_rate=0.2,
                    p=0.8,
                ),
                A.HorizontalFlip(p=0.5),
                A.RGBShift(15, 15, 15, p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.HueSaturationValue(10, 25, 10, p=0.3),
                A.CLAHE(clip_limit=2.0, p=0.2),
            ],
            bbox_params=A.BboxParams(
                format="coco",
                label_fields=["labels"],
                clip=True,
                min_visibility=cfg.min_viz,
                min_area=cfg.min_area,
                min_width=cfg.min_width,
                min_height=cfg.min_height,
            ),
        ),
        "test": A.Compose(
            [
                A.LongestMaxSize(max_size=cfg.model.input_size, p=1.0),
                A.PadIfNeeded(
                    min_height=cfg.model.input_size,
                    min_width=cfg.model.input_size,
                    border_mode=cv2.BORDER_CONSTANT,
                    fill=0,
                    p=1.0,
                ),
            ],
            bbox_params=A.BboxParams(
                format="coco",
                label_fields=["labels"],
                clip=True,
            ),
        ),
    }
    return transforms
