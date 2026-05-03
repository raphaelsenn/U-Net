import torch
from torch.utils.data import DataLoader

import torchvision.transforms.functional as TF
from torchmetrics.segmentation import MeanIoU, DiceScore

from src.config import TrainConfig
from src.unet.unet import UNet
from src.loss.unet_loss import UNetLoss
from src.datasets.dataset import ISBI2012UNetDataset
from src.datasets.transforms import TestTransform


ROOT = "./ISBI-2012-challenge"

TRAIN_VOLUME = "train-volume.tif"
TRAIN_LABELS = "train-labels.tif"

TEST_VOLUME = "test-volume.tif"
TEST_LABELS = "test-labels.tif"


class UNetEvaluator:
    """Designed for the ISBI cell-tracking challange."""  
    def __init__(self, model: UNet, cfg: TrainConfig) -> None:
        self.model = model 
        self.device = cfg.device 
        self.n_classes = cfg.n_classes

        self.miou = MeanIoU(
            num_classes=cfg.n_classes,
            include_background=True,
            input_format="index",
        ).to(self.device)

        self.dice = DiceScore(
            num_classes=cfg.n_classes,
            include_background=True,
            average="macro",
            input_format="index",
        ).to(self.device)

        self.criterion = UNetLoss()

        train_set = ISBI2012UNetDataset(
            ROOT,
            TRAIN_VOLUME,
            TRAIN_LABELS,
            transform=TestTransform()
        )
        
        val_set = ISBI2012UNetDataset(
            ROOT,
            TEST_VOLUME,
            TEST_LABELS, 
            transform=TestTransform()
        )

        self.train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=False)
        self.val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False)

    @torch.no_grad()
    def evaluate(self, train_set: bool=True) -> dict[str, float]:
        dataloader = self.train_loader if train_set else self.val_loader

        self.model.eval() 
        self.miou.reset()
        self.dice.reset()
        
        n_samples = len(dataloader.dataset) 
        total_loss = 0.0
        for img, target, _ in dataloader:
            B, H, W = target.shape
            
            img = img.to(self.device)               # [B, 1, 512, 512]
            target = target.to(self.device)         # [B, 512, 512]
            
            logits = self.model.predict_tiled(img)  # [B, 2, 512, 512]
            preds = torch.argmax(logits, dim=1)     # [B, 512, 512]

            loss = self.criterion(logits, target)

            self.miou.update(preds, target)
            self.dice.update(preds, target)
            
            total_loss += loss * B

        mean_loss = total_loss / n_samples
        mean_miou = self.miou.compute()
        mean_dice = self.dice.compute()

        return {
            "loss": float(mean_loss.item()), 
            "miou": float(mean_miou.item()), 
            "dice": float(mean_dice.item())
        }