import albumentations as A
import os
import cv2

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"


def get_transforms():
    transforms = {
        "train": A.Compose(
            [
                A.Perspective(p=0.1),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.HueSaturationValue(p=0.1),
            ],
            bbox_params=A.BboxParams(
                format="pascal_voc",
                label_fields=["labels"],
                clip=True,
                min_area=25,
                min_width=1,
                min_height=1,
            ),
        ),
        "test": A.Compose(
            [A.NoOp()],
            bbox_params=A.BboxParams(
                format="pascal_voc",
                label_fields=["labels"],
                clip=True,
                min_area=1,
                min_width=1,
                min_height=1,
            ),
        ),
    }
    return transforms
