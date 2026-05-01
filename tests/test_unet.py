import pytest
import torch

from src import UNet
from src.unet.unet import DoubleConv2d


class TestDoubleConv2d:
    def test_shape(self) -> None:
        layer = DoubleConv2d(in_channels=1, out_channels=4)

        x = torch.randn(2, 1, 64, 64)
        y = layer(x)

        assert y.shape == (2, 4, 60, 60)

    def test_output_is_finite(self) -> None:
        layer = DoubleConv2d(1, 4)

        x = torch.randn(2, 1, 64, 64)
        y = layer(x)

        assert torch.isfinite(y).all()


class TestUNet:
    def test_original_unet_shape(self) -> None:
        unet = UNet(in_channels=1, out_channels=2, feature_channels=64)

        img = torch.randn(1, 1, 572, 572)
        out = unet(img)

        assert out.shape == (1, 2, 388, 388)

    def test_shape_with_smaller_feature_channels(self) -> None:
        unet = UNet(in_channels=1, out_channels=2, feature_channels=4)

        img = torch.randn(1, 1, 572, 572)
        out = unet(img)

        assert out.shape == (1, 2, 388, 388)

    def test_batch_shape(self) -> None:
        unet = UNet(in_channels=1, out_channels=3, feature_channels=4)

        img = torch.randn(4, 1, 572, 572)
        out = unet(img)

        assert out.shape == (4, 3, 388, 388)

    def test_rgb_input_shape(self) -> None:
        unet = UNet(in_channels=3, out_channels=2, feature_channels=4)

        img = torch.randn(2, 3, 572, 572)
        out = unet(img)

        assert out.shape == (2, 2, 388, 388)

    def test_output_is_finite(self) -> None:
        unet = UNet(in_channels=1, out_channels=2, feature_channels=4)

        img = torch.randn(1, 1, 572, 572)
        out = unet(img)

        assert torch.isfinite(out).all()

    def test_backward_pass(self) -> None:
        unet = UNet(in_channels=1, out_channels=2, feature_channels=4)

        img = torch.randn(1, 1, 572, 572)
        out = unet(img)

        loss = out.mean()
        loss.backward()

        for name, param in unet.named_parameters():
            assert param.grad is not None, f"No gradient for parameter: {name}"
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient in: {name}"