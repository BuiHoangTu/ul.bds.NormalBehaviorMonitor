from torch import Module

from model_options.mobilenet.invert_mobilenet import MobileNetDecoder
from model_options.mobilenet.mobilenet import MobileNetEncoder


class Autoencoder(Module):
    def __init__(self, sampleShape, latent_dim: int):
        super().__init__()

        n_feat, _ = sampleShape

        self.encoder = MobileNetEncoder(n_feat, latent_dim)
        self.decoder = MobileNetDecoder(latent_dim, sampleShape)

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed
