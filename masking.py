import torch.nn as nn
import torch.nn.functional as F


def maskedMseLoss(pred, target, mask):
    """Compute the masked mean squared error loss.

    Args:
        "mask" should be of shape (batch_size, channel, time).

        "pred" and "target" should be of shape (batch_size, channel, time, features).

    Returns:
        float: the mean squared error loss without accounting for masked regions.
    """

    mask = mask.unsqueeze(-1).expand_as(pred)
    loss = F.mse_loss(pred * mask, target * mask)

    return loss


class MaskedConv2d(nn.Conv2d):
    """
    Custom masked convolution layer.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        *args,
        **kwargs
    ):
        if isinstance(padding, str):
            raise ValueError("Only numeric padding is supported")

        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, *args, **kwargs
        )

    def forward(self, x, mask):
        out = super().forward(x)

        try:
            maskOut = F.max_pool2d(mask, self.kernel_size, self.stride, self.padding)  # type: ignore # type ensured by __init__
        except RuntimeError as e:
            if "Output size is too small" in str(e):
                # it is okay for last layer
                maskOut = None
            else:
                raise e
        return out, maskOut
