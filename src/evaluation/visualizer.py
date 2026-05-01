import os

import torch
import torchvision.transforms.functional as TF

# import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import TrainConfig
from src.unet.unet import UNet
from src.datasets.dataset import ISBI2012SegmentationDataset
from src.datasets.transforms import TestTransform


ROOT = "./ISBI-2012-challenge"
TEST_VOLUME = "test-volume.tif"
TEST_LABELS = "test-labels.tif"
VISUALIZE_DIR = "pred-labels"

DPI = 150
PANEL_PX = 384
N_COLS = 6
N_ROWS = 3
FIGSIZE = (N_COLS * PANEL_PX / DPI, N_ROWS * PANEL_PX / DPI)


class UNetVisualizer:
    """Designed for the ISBI cell-tracking challange.""" 
    def __init__(self, model: UNet, cfg: TrainConfig) -> None:
        self.model = model 
        self.cfg = cfg
        
        self.dataset = ISBI2012SegmentationDataset(
            root=ROOT, 
            features=TEST_VOLUME, 
            labels=TEST_LABELS, 
            transform=TestTransform()
        )

        if not os.path.exists(VISUALIZE_DIR):
            os.mkdir(VISUALIZE_DIR)

    @torch.no_grad() 
    def visualize(self, epoch: int) -> None:
        self.model.eval()
        
        fig, ax = plt.subplots(
            nrows=N_ROWS, 
            ncols=N_COLS, 
            figsize=FIGSIZE, 
            dpi=DPI, 
            constrained_layout=True
        )
        
        titles = ["Input", "Target", "Prediction"]
        for r in range(N_ROWS):
            ax[r, 0].set_ylabel(titles[r])

        for c in range(min(N_COLS, len(self.dataset))):
            x, y = self.dataset[c]                              # [1, 512, 704], [512, 512]
            if x.ndim == 2:
                x = x.unsqueeze(0)                              # [1, 1, 512, 512]
            x_in = x.unsqueeze(0).to(self.cfg.device)           # [1, 1, 512, 512]

            y_hat = self.model.predict_tiled(x_in)              # [1, 2, 512, 512]
            y_hat = torch.argmax(y_hat, dim=1)                  # [1, 512, 512] 

            y_hat = y_hat.cpu().squeeze(0).numpy()              # [512, 512]
            img_np = x_in.cpu().squeeze((0, 1)).numpy()         # [512, 512]
            y_np = y.cpu().numpy() if torch.is_tensor(y) else y # [512, 512]

            ax[0, c].imshow(img_np, cmap="gray", interpolation="nearest")
            ax[1, c].imshow(y_np, cmap="gray", interpolation="nearest")
            ax[2, c].imshow(y_hat, cmap="gray", interpolation="nearest")

            for r in range(N_ROWS):
                ax[r, c].axis("off")

        out = os.path.join(VISUALIZE_DIR, f"pred-labels-{epoch}.png")
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)