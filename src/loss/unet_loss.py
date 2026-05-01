import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetLoss(nn.Module):
    """
    Simple implementation of the weighted per-pixel cross-entropy loss.
    Described in the paper:
    "U-Net: Convolutional Networks for Biomedical Image Segmentation".

    Reference:
    U-Net: Convolutional Networks for Biomedical Image Segmentation 
    By Ronneberger, Brox and Fischer (2015) 
    https://lmb.informatik.uni-freiburg.de/Publications/2015/RFB15a/
    """ 
    def __init__(self) -> None:
        super().__init__() 

    def forward(
            self, 
            logits: torch.Tensor, 
            targets: torch.Tensor, 
            weight: torch.Tensor | None = None
    ) -> torch.Tensor:
        # logits: [1, 2, 388, 388], target: [1, 388, 388]
        cross_entropy = F.cross_entropy(logits, targets, reduction='none')
        if weight is not None: 
            cross_entropy = weight * cross_entropy
        return cross_entropy.mean()