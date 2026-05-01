from typing import Callable
import os

import numpy as np
from PIL import Image, ImageSequence

import torch
from torch.utils.data import Dataset

from src.datasets.transforms import TrainTransform, TestTransform
from src.datasets.prepro import compute_weight_map


class ISBI2012SegmentationDataset(Dataset):
    """
    Dataset wrapper for the ISBI 2012 neuronal structure segmentation dataset.

    Reference:
    ISBI 2012 challange: Segmentation of neuronal structures in EM stacks.
    https://imagej.net/events/isbi-2012-segmentation-challenge 
    
    U-Net: Convolutional Networks for Biomedical Image Segmentation
    https://arxiv.org/abs/1505.04597
    """

    def __init__(
            self,
            root: str='./ISBI-2012-challenge/',
            features: str='train-volume.tif',
            labels: str='train-labels.tif',
            transform: Callable | None = None,
        ) -> None:
        super().__init__()

        assert os.path.exists(root), f"{root} does not exist."
        self.root = root
        self.features = features
        self.labels = labels
        self.transform = transform
        self._load_data()

    def _load_data(self) -> None:
        path_features = os.path.join(self.root, self.features)
        self.data = [
            image.convert("L").copy()
            for image in ImageSequence.Iterator(Image.open(path_features))
        ]
        
        path_targets = os.path.join(self.root, self.labels)
        self.targets = [
            Image.fromarray((np.array(image) > 0).astype(np.uint8))
            for image in ImageSequence.Iterator(Image.open(path_targets))
        ]

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(
            self, 
            idx: int | torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if torch.is_tensor(idx):
            idx = idx.item()

        img, mask = self.data[idx], self.targets[idx]   # [512, 512], [512, 512]
        if self.transform:
            img, mask = self.transform(img, mask)       # [1, 512, 512], [512, 512]

        return img, mask


class ISBI2012UNetDataset(Dataset):
    """
    Dataset for the ISBI 2012 neuronal structure segmentation task ready for UNet inference.

    Reference:
    ISBI 2012 challange: Segmentation of neuronal structures in EM stacks.
    https://imagej.net/events/isbi-2012-segmentation-challenge 
    
    U-Net: Convolutional Networks for Biomedical Image Segmentation
    https://arxiv.org/abs/1505.04597
    """

    def __init__(
            self,
            root: str='./ISBI-2012-challenge/',
            features: str='train-volume.tif',
            labels: str='train-labels.tif',
            w0: float=10.0,
            sigma: float=5.0,
            transform: TrainTransform | TestTransform = TrainTransform(),
        ) -> None:
        super().__init__()

        self.base_dataset = ISBI2012SegmentationDataset(
            root=root,
            features=features,
            labels=labels,
            transform=None,
        )
        
        self.transform = transform
        self.w0 = w0
        self.sigma = sigma

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img, mask = self.base_dataset[idx]                      # [512, 512], [512, 512]
        img, mask = self.transform(img, mask)                   # [1, 572, 572], [388, 388]
        weight = compute_weight_map(mask, self.w0, self.sigma)  # [388, 388]
        return img, mask, weight