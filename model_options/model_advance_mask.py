import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from masking import MaskedConv2d, maskedMseLoss as masked_mse_loss


# Autoencoder Model
class Autoencoder(nn.Module):
    def __init__(self, hidden_dims):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            MaskedConv2d(1, hidden_dims, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_dims, 1, kernel_size=3, stride=2, output_padding=1
            ),
            nn.ReLU(),
        )
        self.mask_embedding = nn.Parameter(
            torch.zeros(1, 1, 1, 1)
        )  # 1 Learnable placeholder

    def forward(self, x, mask=None):
        if mask is not None:
            x = x * mask + (1 - mask) * self.mask_embedding  # input masking
        latent, mask = self.encoder(x, mask)
        reconstructed = self.decoder(latent)
        return reconstructed


# Training Loop
model = Autoencoder(hidden_dims=16)
optimizer = optim.Adam(model.parameters(), lr=0.001)
dataloader = ...  # Assume this is defined

for batch in dataloader:
    x, mask = batch
    x_reconstructed = model(x, mask)
    loss = masked_mse_loss(x_reconstructed, x, mask)
    loss.backward()
    optimizer.step()
