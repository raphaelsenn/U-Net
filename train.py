import argparse
from argparse import Namespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import TrainConfig
from src.unet.unet import UNet
from src.training.trainer import UNetTrainer
from src.datasets.dataset import ISBI2012UNetDataset


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(
        prog='UNet Training: ISBI-2012 Neuronal Structure Segmentation',
        description='Train a UNet model on the ISBI-2012 dataset for neuronal structure segmentation.'
    )
    
    parser.add_argument('--in_channels', type=int, default=1)
    parser.add_argument('--n_classes', type=int, default=2)
    
    parser.add_argument('--n_epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--momentum', type=float, default=0.99)

    parser.add_argument('--root', type=str, default='./ISBI-2012-challenge')
    parser.add_argument('--w0', type=float, default=10.0)
    parser.add_argument('--sigma', type=float, default=5.0)

    parser.add_argument('--device', type=str, default='mps')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--pin_memory', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--eval_every', type=int, default=1)
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--verbose', type=bool, default=True)

    return parser.parse_args()


def build_config(args: Namespace) -> TrainConfig:
    cfg = TrainConfig(
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        in_channels=args.in_channels,
        n_classes=args.n_classes,
        batch_size=args.batch_size,
        device=torch.device(args.device),
        seed=args.seed,
        eval_every=args.eval_every,
        save_every=args.save_every
    )
    return cfg


def build_dataloaders(args: Namespace) -> DataLoader:
    train_set = ISBI2012UNetDataset(args.root, w0=args.w0, sigma=args.sigma)
    train_loader = DataLoader(
        train_set, 
        args.batch_size, 
        True, 
        num_workers=args.num_workers
    )
    
    return train_loader


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == '__main__':
    args = parse_args() 
    cfg = build_config(args)
    set_seeds(cfg.seed) 
    train_loader = build_dataloaders(args)

    unet = UNet(cfg.in_channels, cfg.n_classes)
    trainer = UNetTrainer(cfg, unet)
    trainer.train(train_loader)