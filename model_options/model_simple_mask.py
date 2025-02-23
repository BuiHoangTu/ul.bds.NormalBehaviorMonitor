import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, hidden_dims):
        super(Autoencoder, self).__init__()
        self.hidden_dims = hidden_dims

        # Learnable representation for invalid values
        self.mask_embedding = nn.Parameter(torch.zeros(1, 1, 1, 1))  # [C, H, W] format

        # Encoder
        self.encoder = nn.Sequential()
        for i in range(len(hidden_dims)):
            in_dim = 1 if i == 0 else hidden_dims[i - 1]
            out_dim = hidden_dims[i]

            self.encoder.add_module(
                f"enc_conv{i}",
                nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
            )
            if i < len(hidden_dims) - 1:
                self.encoder.add_module(f"enc_relu{i}", nn.ReLU())

        # Decoder
        self.decoder = nn.Sequential()
        for i in reversed(range(len(hidden_dims))):
            in_dim = hidden_dims[i]
            out_dim = 1 if i == 0 else hidden_dims[i - 1]

            self.decoder.add_module(
                f"dec_conv{i}",
                nn.ConvTranspose2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
            )
            if i > 0:
                self.decoder.add_module(f"dec_relu{i}", nn.ReLU())
            else:
                self.decoder.add_module("dec_sigmoid", nn.Sigmoid())

    def forward(self, x, mask=None):
        # Replace invalid values with learnable embedding
        if mask is not None:
            x = x * mask + (1 - mask) * self.mask_embedding

        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed
