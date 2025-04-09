from torch.nn import functional as F
from torch.nn import (
    Module,
    Sequential,
    Hardswish,
    Conv1d,
    ConvTranspose1d,
    BatchNorm1d,
    Linear,
    Unflatten,
)

from model_options.mobilenet.mobilenet import (
    BNECKS_OF_SMALL,
    SE_REDUCTION,
    SqueezeExcite,
    createConv1dBlock,
    BN_EPS,
    BN_MOMENTUM,
)


class InvertedBottleneck(Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        expand_channels,
        kernel_size,
        activation,
        use_se,
        stride,
        upsample=False,
        bn_eps=BN_EPS,
        bn_momentum=BN_MOMENTUM,
    ) -> None:
        super().__init__()
        self.use_se = use_se
        self.upsample = upsample
        self.stride = stride
        self.connectFlag = in_channels == out_channels and stride == 1

        # Pointwise expansion
        self.conv1 = createConv1dBlock(
            in_channels,
            expand_channels,
            1,
            activation,
            bn_eps=bn_eps,
            bn_momentum=bn_momentum,
        )

        # Depthwise conv (with optional transposed conv for upsampling)
        if upsample:
            self.dconv1 = Sequential(
                ConvTranspose1d(
                    expand_channels,
                    expand_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=kernel_size // 2,
                    output_padding=stride - 1,
                    groups=expand_channels,  # Depthwise
                ),
                BatchNorm1d(expand_channels, eps=bn_eps, momentum=bn_momentum),
                activation,
            )
        else:
            self.dconv1 = createConv1dBlock(
                expand_channels,
                expand_channels,
                kernel_size,
                activation,
                stride=stride,
                groups=expand_channels,
                bn_eps=bn_eps,
                bn_momentum=bn_momentum,
            )

        # Squeeze-and-excite
        if use_se:
            self.squeeze = SqueezeExcite(expand_channels)

        # Pointwise projection
        self.conv2 = createConv1dBlock(
            expand_channels,
            out_channels,
            1,
            None,
            bn_eps=bn_eps,
            bn_momentum=bn_momentum,
        )

        # Skip connection (with optional upsampling)
        if not self.connectFlag:
            self.shortcut = Sequential(
                Conv1d(in_channels, out_channels, 1, stride=1),
                BatchNorm1d(out_channels, eps=bn_eps, momentum=bn_momentum),
            )

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.dconv1(x)
        if hasattr(self, "squeeze"):
            x = self.squeeze(x)
        x = self.conv2(x)

        if not self.connectFlag:
            if self.upsample:
                # For upsampling, we need to handle identity differently
                identity = F.interpolate(
                    identity, scale_factor=self.stride, mode="nearest"
                )
            identity = self.shortcut(identity)

        return x + identity


class MobileNetDecoder(Module):
    def __init__(
        self,
        in_channels,  # Input channels (should match encoder output)
        out_size,  # Output size (channels, time)
        bnecks=None,
        dropout=0.2,
        bn_eps=BN_EPS,
        bn_momentum=BN_MOMENTUM,
        se_reduction=SE_REDUCTION,
    ) -> None:
        super().__init__()
        self.out_channels, self.out_time = out_size

        # Reverse the bottleneck configuration for decoder
        self.bnecks = bnecks or BNECKS_OF_SMALL[::-1]  # Flip order

        # Initial projection (mirrors encoder's final layers)
        self.projection = Sequential(
            Linear(in_channels, 1024),
            Hardswish(),
            Linear(1024, 576),
            Hardswish(),
        )

        # Initial conv to expand to spatial dimension
        self.conv1 = Sequential(
            Unflatten(1, (576, 1)),  # Reshape to (batch, channels, 1)
            createConv1dBlock(
                576, 96, 1, Hardswish(), bn_eps=bn_eps, bn_momentum=bn_momentum
            ),
        )

        # Bottleneck layers (upsampling)
        self.invRes = Sequential()
        for idx, (out, inp, exp, k, act, se, s) in enumerate(self.bnecks):
            # Note: We swap in/out channels since we're going backwards
            self.invRes.append(
                InvertedBottleneck(
                    inp,  # Note the order is reversed
                    out,  # from encoder's bottlenecks
                    exp,
                    k,
                    act,
                    se,
                    stride=s,
                    upsample=(s > 1),  # Upsample if stride > 1
                    bn_eps=bn_eps,
                    bn_momentum=bn_momentum,
                )
            )

        # Final upsample to original resolution
        self.upsample = Sequential(
            ConvTranspose1d(
                16, 16, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            BatchNorm1d(16, eps=bn_eps, momentum=bn_momentum),
            Hardswish(),
            Conv1d(16, self.out_channels, kernel_size=1),
        )

    def forward(self, x):
        x = self.projection(x)
        x = self.conv1(x)
        x = self.invRes(x)
        x = self.upsample(x)

        # Handle final time dimension if needed
        if x.size(2) != self.out_time:
            x = F.interpolate(x, size=self.out_time, mode="linear", align_corners=False)

        return x
