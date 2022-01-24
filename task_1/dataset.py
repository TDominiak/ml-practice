import dataclasses
import os
from typing import Dict, List

import torch
from torchvision import datasets
from torchvision.transforms import transforms


def get_data_transformation() -> Dict:
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    return data_transforms


def get_image_dataset(data_transforms: Dict, data_dir: str, sets: List[str]):
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in sets}

    return image_datasets


def _get_data_loaders(image_datasets: Dict, sets: List[str]):
    data_loaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16, shuffle=True, num_workers=4)
                    for x in sets}

    return data_loaders


@dataclasses.dataclass
class DatasetConfig:
    sets: List[str]
    data_transformation: Dict
    image_datasets: Dict

    def __post_init__(self):
        self.data_loaders = _get_data_loaders(self.image_datasets, self.sets)
