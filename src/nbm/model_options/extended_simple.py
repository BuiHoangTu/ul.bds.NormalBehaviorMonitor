import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, sampleShape, latent_dim: int):
        super().__init__()

        sampleShape = torch.tensor(sampleShape)
        n_feat, _ = sampleShape

        self.encoder = Encoder(n_feat, latent_dim)
        self.decoder = Decoder(sampleShape, latent_dim)

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed


class Encoder(nn.Module):
    def _cnnBlock(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2),
        )

    def __init__(
        self,
        n_feature,
        latentDim: int,
    ):
        super(Encoder, self).__init__()
        self.latentDim = latentDim

        # layers
        self.featExtractLayers = nn.Sequential()
        self.latentLayers = nn.Sequential()

        # populate layers
        ## feature extraction layers
        out_channels = -1
        for i in range(4):
            # use n_feature for the first layer, otherwise use out_channels from the previous layer
            in_channels = n_feature if i == 0 else out_channels
            out_channels = 16 * (2**i)
            self.featExtractLayers.append(self._cnnBlock(in_channels, out_channels))

        ## hidden latents
        self.latentLayers.append(nn.LazyLinear(latentDim * 4))
        self.latentLayers.append(nn.ReLU())
        self.latentLayers.append(nn.Linear(latentDim * 4, latentDim))

    def forward(self, x):
        x = self.featExtractLayers(x)
        x = x.flatten(start_dim=1)
        x = self.latentLayers(x)
        x = torch.tanh(x)
        return x


class Decoder(nn.Module):
    def _deconvBlock(self, in_channels, out_channels):
        return nn.Sequential(
            ### Use stride = 2 at convT to improve accuracy
            nn.Upsample(scale_factor=2, mode="linear"),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size=3, padding=1),
        )

    def __init__(
        self,
        reconstrShape,
        latentDim: int,
    ):
        super(Decoder, self).__init__()
        self.n_reconstrFeat, self.reconstrDim = reconstrShape

        self.n_convTChannels = 16 * (2**3)
        # the size is 16*2^3 * 96/2^4 {pooling}
        deconvInput = int(self.n_convTChannels * self.reconstrDim / 2**4)

        # layers
        self.latentLayers = nn.Sequential()
        self.decryptLayers = nn.Sequential()

        # populate layers
        ## de-latent layers
        self.latentLayers.append(nn.Linear(latentDim, latentDim * 4))
        self.latentLayers.append(nn.ReLU())
        self.latentLayers.append(nn.Linear(latentDim * 4, deconvInput))

        ## feature decryption layers
        for i in range(4):
            self.decryptLayers.append(
                self._deconvBlock(
                    self.n_convTChannels // (2**i),
                    self.n_convTChannels // (2 ** (i + 1)) if i < 3 else self.n_reconstrFeat,
                )
            )

    def forward(self, x):
        x = self.latentLayers(x)
        x = x.view(x.size(0), self.n_convTChannels, -1)
        x = self.decryptLayers(x)
        return x
