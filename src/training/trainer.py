import torch
from torch.utils.data import DataLoader
import pandas as pd

from src.config import TrainConfig
from src.unet.unet import UNet
from src.loss.unet_loss import UNetLoss
from src.evaluation.evaluator import UNetEvaluator
from src.evaluation.visualizer import UNetVisualizer


class UNetTrainer:
    def __init__(
            self,
            cfg: TrainConfig,
            model: UNet, 
    ) -> None:
        self.cfg = cfg
        self.device = cfg.device

        self.model = model.to(self.device)
        self.criterion = UNetLoss()
        self.optimizer = torch.optim.SGD(
            model.parameters(), cfg.learning_rate, cfg.momentum
        )
        
        self.evaluator = UNetEvaluator(self.model, cfg) 
        self.visualizer = UNetVisualizer(self.model, cfg) 
        self.stats = {
            "Epoch": [],
            "Train-Loss": [],
            "Val-Loss": [],
            "Train-miou": [],
            "Val-miou": [],
            "Train-dice": [],
            "Val-dice": [],
        }

    def train_one_epoch(self, dataloader: DataLoader) -> None:
        self.model.train() 
        for img, mask, weight in dataloader:
            img = img.to(self.device)           # [B, C, 512, 512]
            mask = mask.to(self.device)         # [B, 1, 388, 388]
            weight = weight.to(self.device)     # [B, 1, 388, 388]

            logits = self.model(img)
            loss = self.criterion(logits, mask, weight)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def train(self, train_loader: DataLoader) -> None:
        for epoch in range(1, self.cfg.n_epochs + 1):
            self.train_one_epoch(train_loader)
            self.visualizer.visualize(epoch)

            if epoch % self.cfg.eval_every == 0:
                self.checkpoint(epoch)

                if self.cfg.verbose:
                    self.print_stats()

            if epoch % self.cfg.visualize_every == 0:
                self.visualizer.visualize(epoch)

        self.checkpoint(self.cfg.n_epochs+1)

    def checkpoint(self, epoch) -> None:
        train_scores = self.evaluator.evaluate(train_set=True)
        val_scores = self.evaluator.evaluate(train_set=False)
        self.update_stats(epoch, train_scores, val_scores)     
        self.save()

    def save(self) -> None:
        params = sum(p.numel() for p in self.model.parameters()) 
        model_name = (
            f"unet-p{params}"
            f"-in_c{self.model.in_channels}"
            f"-out_c{self.model.out_channels}"
            f"-feat_c{self.model.feature_channels}.pt"
        )
        torch.save(self.model.state_dict(), model_name)
        
        report_name = (
            f"unet-p{params}"
            f"-in_c{self.model.in_channels}"
            f"-out_c{self.model.out_channels}"
            f"-feat_c{self.model.feature_channels}.csv"
        )
        pd.DataFrame().from_dict(self.stats).to_csv(report_name, index=False)

    def update_stats(self, epoch: int, train_scores: dict, val_scores: dict) -> None:
        self.stats["Epoch"].append(epoch)

        self.stats["Train-Loss"].append(train_scores["loss"])
        self.stats["Val-Loss"].append(val_scores["loss"])

        self.stats["Train-miou"].append(train_scores["miou"])
        self.stats["Val-miou"].append(val_scores["miou"])

        self.stats["Train-dice"].append(train_scores["dice"])
        self.stats["Val-dice"].append(val_scores["dice"])

    def print_stats(self) -> None:
        report = (f"Epoch: {self.stats["Epoch"][-1]}\t"
                  f"Train-Loss: {self.stats["Train-Loss"][-1]:.4f}\t"
                  f"Val-Loss: {self.stats["Val-Loss"][-1]:.4f}\t"
                  f"Train-miou: {self.stats["Train-miou"][-1]:.4f}\t"
                  f"Val-miou: {self.stats["Val-miou"][-1]:.4f}\t"
                  f"Train-dice: {self.stats["Train-dice"][-1]:.4f}\t"
                  f"Val-dice: {self.stats["Val-dice"][-1]:.4f}\t"

        )
        print(report)