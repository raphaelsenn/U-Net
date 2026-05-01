import numpy as np
from scipy import ndimage
from scipy.ndimage import distance_transform_edt

import torch


def compute_class_weight_map(target: np.ndarray) -> np.ndarray:
    """
    Compute wc(x), the class-balancing part of the U-Net weight map.

    Input:
    ------
    target: Binary mask [H, W], background = 0, foreground = 1, dtype=np.uint8
    """
    target = np.asarray(target)
    cells = target == 1
    background = target == 0

    n_cells = np.sum(cells)
    n_background = np.sum(background)
    n_pixels = target.size

    weight = np.zeros_like(target, dtype=np.float32)

    if n_cells > 0:
        cell_weight = n_pixels / (2.0 * n_cells)
        weight[cells] = cell_weight

    if n_background > 0:
        background_weight = n_pixels / (2.0 * n_background)
        weight[background] = background_weight

    return weight


def compute_border_weight_map(
    target: np.ndarray,
    w0: float = 10.0,
    sigma: float = 5.0,
) -> np.ndarray:
    """
    Compute the original U-Net border weight term:

        w0 * exp(-((d1 + d2)^2) / (2 * sigma^2))

    where d1 and d2 are distances to the nearest and second-nearest cell instance.

    Input:
    ----
    target: Binary mask [H, W], background = 0, foreground = 1, dtype=np.uint8
    """
    unique_cells, n_unique_cells = ndimage.label(target)        # [512, 512]
    
    if n_unique_cells < 2:
        return np.zeros_like(target, dtype=np.float32)          # [512, 512]

    distances = []
    for cell_id in range(1, n_unique_cells + 1):
        curr_cell = unique_cells == cell_id                     # [512, 512]

        # Distance from every pixel to this instance
        distance = distance_transform_edt(np.invert(curr_cell)) # [512, 512]
        distances.append(distance)
    
    distances = np.stack(distances, axis=0)                     # [n_unique_cells, 512, 512]
    distances = np.sort(distances, axis=0)                      # [n_unique_cells, 512, 512]

    # Query distance d1 and d2
    d1 = distances[0]                                           # [512, 512]
    d2 = distances[1]                                           # [512, 512]

    # Compute final boarder map
    # Read more here: https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/
    border_weight_map = w0 * np.exp(-((d1 + d2) ** 2) / (2.0 * sigma ** 2))

    return border_weight_map.astype(np.float32)


def compute_weight_map(
    target: np.ndarray | torch.Tensor,
    w0: float = 10.0,
    sigma: float = 5.0,
) -> torch.Tensor:
    """
    Compute the original U-Net weight map:

        w(x) = wc(x) + w0 * exp(-((d1(x) + d2(x))^2) / (2 * sigma^2))

    Input:
    ----
    target: Binary mask [512, 512], background = 0, foreground = 1
    """
    if isinstance(target, torch.Tensor): 
        target = target.numpy()
    assert isinstance(target, np.ndarray), f"target needs to be np.ndarray or torch.Tensor"
    class_weight = compute_class_weight_map(target)
    border_weight = compute_border_weight_map(target, w0=w0, sigma=sigma)
    weight_map = class_weight + border_weight
    return torch.as_tensor(weight_map, dtype=torch.float32)