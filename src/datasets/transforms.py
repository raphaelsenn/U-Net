import cv2
import torch
from PIL import Image

import torchvision.transforms.functional as TF
import albumentations as A
import numpy as np


class TrainTransform:
    """
    Applies data augmentation to input and target data.

    Reference:
    U-Net: Convolutional Networks for Biomedical Image Segmentation
    https://arxiv.org/abs/1505.04597
    """ 
    def __init__(self) -> None:
        super().__init__()
        
        # Heavy data augmentation
        # Read more here: https://lmb.informatik.uni-freiburg.de/Publications/2015/RFB15a/
        self.transform = A.Compose([
            A.OneOf([
                A.HorizontalFlip(p=1.0),
                A.VerticalFlip(p=1.0),
                A.Transpose(p=1.0),
                A.RandomRotate90(p=1.0),
                A.Affine(
                    translate_percent=(-0.0625, 0.0625),
                    scale=(0.9, 1.1),
                    rotate=(-45, 45),
                    border_mode=cv2.BORDER_REFLECT,
                    interpolation=cv2.INTER_LINEAR,
                    mask_interpolation=cv2.INTER_NEAREST,
                    p=1.0,
            )], p=0.8),
            A.ElasticTransform(
                alpha=35,
                sigma=5,
                approximate=True,
                border_mode=cv2.BORDER_REFLECT,
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                p=0.3,
            ),

            A.RandomBrightnessContrast(
                brightness_limit=0.15,
                contrast_limit=0.15,
                p=0.3,
            ),

            A.GaussNoise(
                std_range=(0.0, 0.08),
                mean_range=(0.0, 0.0),
                per_channel=False,
                p=0.2,
            ),

            A.MultiplicativeNoise(
                multiplier=(0.9, 1.1),
                elementwise=True,
                p=0.2,
            ),
        ])

    def __call__(
            self, 
            img: Image.Image,
            mask: Image.Image,
            img_out_shape: list=[572, 572], 
            mask_out_shape: list=[388, 388]
        ) -> tuple[torch.Tensor, torch.Tensor]:

        # Pad before agumentation (so rotations/shifts do not destroy boarder context) 
        margin = (img_out_shape[0] - mask_out_shape[0]) // 2            # 92
        img = TF.pad(img, padding=4*[margin], padding_mode="reflect")   # [664, 664] 
        mask = TF.pad(mask, padding=4*[margin], padding_mode="reflect") # [664, 664]
        
        img_np = np.asarray(img).astype(np.uint8)                       # [644, 644]
        mask_np = np.asarray(mask).astype(np.uint8)                     # [644, 644]

        aug = self.transform(image=img_np, mask=mask_np) 
        img_np, mask_np = aug["image"], aug["mask"]                     # [644, 644] both

        img_tensor = torch.as_tensor(img_np, dtype=torch.float32)       # [644, 644]
        mask_tensor = torch.as_tensor(mask_np, dtype=torch.long)        # [644, 644]

        img_tensor = TF.center_crop(img_tensor, img_out_shape)          # [572, 572]
        mask_tensor = TF.center_crop(mask_tensor, mask_out_shape)       # [388, 388]

        img_tensor = img_tensor.unsqueeze(0)                            # [1, 572, 572]
        img_tensor = img_tensor.div(255.0)                              # [1, 572, 572]

        return img_tensor, mask_tensor


class TestTransform:
    """
    Applies data transformation to test data (input and target).
    
    Reference:
    U-Net: Convolutional Networks for Biomedical Image Segmentation
    https://arxiv.org/abs/1505.04597
    """ 
    def __call__(
            self, 
            img: Image.Image, 
            mask: Image.Image,
        ) -> tuple[torch.Tensor, torch.Tensor]:

        img_tensor = TF.pil_to_tensor(img).to(torch.float32)                # [1, 512, 512]
        img_tensor = img_tensor.div(255.0)                                  # [1, 512, 512]

        mask_tensor = TF.pil_to_tensor(mask).to(torch.long)           # [1, 512, 512]
        mask_tensor = mask_tensor.squeeze(0)                                # [512, 512]

        return img_tensor, mask_tensor