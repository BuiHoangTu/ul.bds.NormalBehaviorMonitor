import torch
import torch.nn as nn
import torch.nn.functional as F


# From Chen, Y., Zhang, H., Wang, Y., Yang, Y., Zhou, X., & Wu, Q. M. J. (2021). MAMA Net: Multi-Scale Attention Memory Autoencoder Network for Anomaly Detection. IEEE Transactions on Medical Imaging, 40(3), 1032–1041. https://doi.org/10.1109/TMI.2020.3045295
class MultiScaleAttentionBlock2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        patch_channels=None,
        block_type="SA",
    ):
        super(MultiScaleAttentionBlock2d, self).__init__()
        self.block_type = block_type  # 'SA', 'DA', 'UA'

        patch_channels = patch_channels if patch_channels is not None else out_channels

        # --- (1) Pixel Patch Attention ---
        # Query, Key, Value projections (1x1 convs)
        self.conv_q = nn.Conv2d(in_channels, patch_channels, kernel_size=1, stride=1)
        self.conv_k = nn.Conv2d(in_channels, patch_channels, kernel_size=1, stride=1)
        self.conv_v = nn.Conv2d(in_channels, patch_channels, kernel_size=1, stride=1)

        # --- (2) Channel Attention ---
        if self.block_type == "DA":
            self.conv_cq = nn.Conv2d(
                patch_channels, out_channels, kernel_size=3, stride=2, padding=1
            )
        elif self.block_type == "UA":
            self.conv_cq = nn.ConvTranspose2d(
                patch_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            )
        else:  # 'SA'
            self.conv_cq = nn.Conv2d(
                patch_channels, out_channels, kernel_size=3, stride=1, padding=1
            )

        self.conv_ck = nn.Conv2d(patch_channels, out_channels, kernel_size=1)
        self.conv_cv = nn.Conv2d(patch_channels, out_channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.size()

        # --- (1) Pixel Patch Attention ---
        q = self.conv_q(x)  # (B, C', H, W)
        k = self.conv_k(x)
        v = self.conv_v(x)

        q_flat = q.view(B, -1, H * W).permute(0, 2, 1)  # (B, HW, C')
        k_flat = k.view(B, -1, H * W)  # (B, C', HW)
        attention_map = torch.bmm(q_flat, k_flat) / (q.size(1) ** 0.5)  # (B, HW, HW)
        attention_map = F.softmax(attention_map, dim=-1)

        v_flat = v.view(B, -1, H * W).permute(0, 2, 1)  # (B, HW, C')
        out_p = torch.bmm(attention_map, v_flat)  # (B, HW, C')
        out_p = out_p.permute(0, 2, 1).contiguous().view(B, -1, H, W)

        # --- (2) Channel Attention ---
        cq = self.conv_cq(out_p)
        ck = self.conv_ck(out_p)
        cv = self.conv_cv(out_p)

        B2, C2, H2, W2 = cq.size()
        cq_flat = cq.view(B2, C2, -1)  # (B, C'', H2*W2)
        ck_flat = ck.view(B2, C2, -1)
        cv_flat = cv.view(B2, C2, -1)

        channel_attention = torch.bmm(cq_flat, ck_flat.transpose(1, 2)) / (C2**0.5)
        channel_attention = F.softmax(channel_attention, dim=-1)

        out_c = torch.bmm(channel_attention, cv_flat)  # (B, C'', H2*W2)
        out_c = out_c.view(B2, C2, H2, W2)

        return out_c


def _cnn1dBlk(*args, **kwargs):
    """
    Placeholder for 1D CNN block.
    Replace with actual implementation if needed.
    """
    return nn.Sequential(
        nn.Conv1d(*args, **kwargs),
        nn.BatchNorm1d(kwargs.get('out_channels', args[1])),
        nn.ReLU(),
    )
def _dcnn1dBlk(*args, **kwargs):
    """
    Placeholder for 1D Deconvolution block.
    Replace with actual implementation if needed.
    """
    return nn.Sequential(
        nn.ConvTranspose1d(*args, **kwargs),
        nn.BatchNorm1d(kwargs.get('out_channels', args[1])),
    )
class MultiScaleAttentionBlock1D(nn.Module):

    def __init__(self, in_channels, out_channels, block_type="SA"):
        """
        Optimized 1D multi-scale attention block (non-patch version)

        Args:
            in_channels: Input channels (C)
            out_channels: Output channels (C')
            block_type: Type of attention block ('SA', 'DA', 'UA')
        """
        super().__init__()
        self.block_type = block_type  # 'SA', 'DA', 'UA'

        # Position-wise attention (full sequence)
        self.conv_t_qkv = _cnn1dBlk(in_channels, 3 * out_channels, kernel_size=1)

        # Channel attention
        if block_type == "DA":
            self.conv_cq = _cnn1dBlk(
                out_channels, out_channels, kernel_size=3, stride=2, padding=1
            )
        elif block_type == "UA":
            self.conv_cq = _dcnn1dBlk(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            )
        else:  # 'SA'
            self.conv_cq = _cnn1dBlk(
                out_channels, out_channels, kernel_size=3, stride=1, padding=1
            )

        self.conv_ck = _cnn1dBlk(out_channels, out_channels, kernel_size=1, stride=1)
        self.conv_cv = _cnn1dBlk(out_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        # Temporal attention
        tq, tk, tv = self.conv_t_qkv(x).chunk(3, dim=1)  # 3 [B, C, L]

        t_attn = torch.bmm(tq, tk.transpose(1, 2))  # [B, C, C]
        t_attn = F.softmax(t_attn, dim=-1)

        t_out = torch.bmm(t_attn, tv)  # [B, C, L]

        # Channel attention
        cq = self.conv_cq(t_out)  # [B, C', L']
        ck = self.conv_ck(t_out)  # [B, C', L']
        cv = self.conv_cv(t_out)  # [B, C', L']

        c_attn = torch.bmm(ck.transpose(1, 2), cq)  # [B, L', L'']
        c_attn = F.softmax(c_attn, dim=-1)

        c_out = torch.bmm(cv, c_attn)

        return c_out
