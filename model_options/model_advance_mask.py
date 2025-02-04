import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# Preprocessing Function
def preprocess_batch(batch):
    data = batch["feature"]  # Shape: (batch_size, channels, timesteps, features)
    mask = batch["valid"].float()  # 1 for valid, 0 for invalid
    data[~mask.bool()] = 0  # Replace invalid values with 0
    return data, mask


# Masked Loss Function
def masked_mse_loss(output, target, mask):
    loss = (output - target) ** 2
    masked_loss = loss * mask  # Zero out invalid timesteps
    return masked_loss.sum() / mask.sum()  # Normalize by valid timesteps


# Custom Masked Convolution Layer
class MaskedConv2d(nn.Conv2d):
    def forward(self, x, mask):
        out = super().forward(x)
        
        if isinstance(self.padding, str):
            raise ValueError("Only numeric padding is supported")
        
        mask_out = F.max_pool2d(mask, self.kernel_size, self.stride, self.padding)
        return out, mask_out


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
        )  # Learnable placeholder

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
