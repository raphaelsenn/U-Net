import torch
import torch.nn as nn

import torchvision.transforms.functional as TF


class DoubleConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()        
        self.double_conv2d = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 0),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 0),
            nn.ReLU(True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv2d(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()        
        self.double_conv2d = DoubleConv2d(in_channels, out_channels)
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.double_conv2d(x) 
        return self.max_pool2d(x), x


class Up(nn.Module):
    def __init__(
            self, 
            in_channels: int,
            out_channels: int
        ) -> None:
        super().__init__()        
        self.conv2d_up = nn.ConvTranspose2d(in_channels, out_channels, 2, 2, 0)
        self.double_conv2d = DoubleConv2d(2*out_channels, out_channels)

    def forward(self, x_residual: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        x = self.conv2d_up(x)        
        _, _, H, W = x.shape 
        x_residual = TF.center_crop(x_residual, output_size=(H, W))
        x = torch.cat([x_residual, x], dim=1) 
        return self.double_conv2d(x)        


class UNet(nn.Module):
    """
    Implementation of the U-Net architecture for image segmentation.

    Reference:
    U-Net: Convolutional Networks for Biomedical Image Segmentation; Brox et al., 2015
    https://lmb.informatik.uni-freiburg.de/Publications/2015/RFB15a/
    """
    def __init__(self, in_channels: int, out_channels: int, feature_channels: int=64) -> None:
        super().__init__()        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feature_channels = feature_channels

        # Contracting path
        self.down1 = Down(in_channels, feature_channels)
        self.down2 = Down(feature_channels, 2 * feature_channels)
        self.down3 = Down(2 * feature_channels, 4 * feature_channels)
        self.down4 = Down(4 * feature_channels, 8 * feature_channels)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            DoubleConv2d(8 * feature_channels, 16 * feature_channels),
            nn.Dropout(0.5)
        )

        # Expanding path
        self.up1 = Up(16 * feature_channels, 8 * feature_channels)
        self.up2 = Up(8 * feature_channels, 4 * feature_channels)
        self.up3 = Up(4 * feature_channels, 2 * feature_channels)
        self.up4 = Up(2 * feature_channels, feature_channels)
        
        # Output layer 
        self.final_conv2d = nn.Conv2d(feature_channels, out_channels, 1, 1, 0)

        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Downwards
        x1_out, x1 = self.down1(x)
        x2_out, x2 = self.down2(x1_out)
        x3_out, x3 = self.down3(x2_out)
        x4_out, x4 = self.down4(x3_out)

        # Bottleneck
        bottleneck = self.bottleneck(x4_out)

        # Upwards
        up = self.up1(x4, bottleneck)
        up = self.up2(x3, up)
        up = self.up3(x2, up)
        up = self.up4(x1, up)
        return self.final_conv2d(up)

    def predict_tiled(
            self, 
            image: torch.Tensor, 
            patch_size: int=256,
            unet_input_shape: tuple[int, int] = (572, 572),
    ) -> torch.Tensor:
        """
        Predict a full-size segmentation map using tiled prediction.

        image: [B, C, H, W]

        For each patch:
            256 x 256 image patch
            -> reflect pad to 572 x 572
            -> U-Net output 388 x 388
            -> center crop to 256 x 256
            -> paste into full prediction
        """ 
        self.eval() 

        B, _, H, W = image.shape 
        C_out = self.out_channels
        pred = torch.zeros(
            size=(B, C_out, H, W), dtype=torch.float32, device=image.device
        )
        
        margin_h = (unet_input_shape[0] - patch_size) // 2
        margin_w = (unet_input_shape[1] - patch_size) // 2
        for i in range(0, H, patch_size):
            for j in range(0, W, patch_size):
                img_patch = image[:, :, i:i+patch_size, j:j+patch_size] # [B, C=1, 256, 256]
                img_patch = TF.pad(
                    img_patch,
                    padding=[margin_w, margin_h, margin_w, margin_h],
                    padding_mode="reflect",
                )                                                       # [B, C=1, 572, 572]
                pred_patch = self(img_patch)                            # [B, C=2, 388, 388]
                pred_patch = TF.center_crop(pred_patch, 2*[patch_size]) # [B, C=2, 256, 256]
                pred[:, :, i:i+patch_size, j:j+patch_size] = pred_patch
        return pred

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

                if m.bias is not None:
                    nn.init.zeros_(m.bias)