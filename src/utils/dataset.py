from torch.utils.data import Dataset
import cv2
from typing import List, Tuple
import torchvision
from PIL import Image
import albumentations


class DS(Dataset):
    def __init__(
        self,
        samples: List[Tuple[str, str]],
        labels: List,
        id2lbl,
        lbl2id,
        transforms=None,
    ):
        self.samples = samples
        self.labels = labels

        self.id2lbl = id2lbl
        self.lbl2id = lbl2id

        self.transforms = transforms

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transforms:
            if isinstance(self.transforms, albumentations.Compose):
                aug = self.transforms(image=img)
                img = aug["image"]
            elif isinstance(self.transforms, torchvision.transforms.Compose):
                img = Image.fromarray(img)
                img = self.transforms(img)

        lbl = self.lbl2id[label]

        return img, lbl
