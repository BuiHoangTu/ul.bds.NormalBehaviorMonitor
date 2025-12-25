import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, sampleShape, latent_dim: int):
        super().__init__()

        n_feat, _ = sampleShape

        self.encoder = SimpleEncoder(n_feat, latent_dim)
        self.decoder = SimpleDecoder(sampleShape)

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed


class SimpleEncoder(nn.Module):
    def __init__(
        self,
        n_feature,
        latentDim: int,
        n_hiddenLatent=0,
        hiddenLatentReducingFactor=0.7,
    ):
        super(SimpleEncoder, self).__init__()
        self.latentDim = latentDim
        self.n_hiddenLatent = n_hiddenLatent
        self.hiddenLatentReducingFactor = hiddenLatentReducingFactor

        # layers
        self.featExtractLayers = nn.Sequential()

        self.hiddenLatents = nn.Sequential()
        self.fc = nn.LazyLinear(latentDim)

        ## feature extraction layers
        self.featExtractLayers.append(nn.Conv1d(n_feature, 32, kernel_size=3))
        self.featExtractLayers.append(nn.ReLU())

        self.featExtractLayers.append(nn.Conv1d(32, 64, kernel_size=3))

        ## hidden latents
        currDim = int(latentDim / (hiddenLatentReducingFactor**n_hiddenLatent))
        for i in range(n_hiddenLatent):
            self.hiddenLatents.add_module(f"fc{i}", nn.LazyLinear(currDim))
            self.hiddenLatents.add_module(f"relu{i}", nn.ReLU())
            currDim = int(currDim * hiddenLatentReducingFactor)
        self.hiddenLatents.add_module("fc_fin", nn.LazyLinear(currDim))
        self.hiddenLatents.add_module("relu_fin", nn.ReLU())

    def forward(self, x):
        x = self.featExtractLayers(x)
        x = x.view(x.size(0), -1)
        x = self.hiddenLatents(x)
        x = self.fc(x)
        return x


class SimpleDecoder(nn.Module):
    def __init__(
        self,
        reconstrShape,
        n_hiddenLatent=0,
        hiddenLatentReducingFactor=0.7,
    ):
        super(SimpleDecoder, self).__init__()
        self.n_reconstrFeat, self.reconstrDim = reconstrShape
        self.n_hiddenLatent = n_hiddenLatent
        self.hiddenLatentReducingFactor = hiddenLatentReducingFactor
        deconvInput = (self.reconstrDim) * 64

        # layers
        self.hiddenLatents = nn.Sequential()
        self.fc = nn.LazyLinear(deconvInput)
        self.reverseExtractLayers = nn.Sequential()

        ## hidden latents
        currDim = int(self.reconstrDim * (hiddenLatentReducingFactor**n_hiddenLatent))
        for i in range(n_hiddenLatent):
            self.hiddenLatents.add_module(f"fc{i}", nn.LazyLinear(currDim))
            self.hiddenLatents.add_module(f"relu{i}", nn.ReLU())
            currDim = int(currDim / hiddenLatentReducingFactor)
        self.hiddenLatents.add_module("fc_fin", nn.LazyLinear(currDim))
        self.hiddenLatents.add_module("relu_fin", nn.ReLU())

        # deconvolution layers
        self.reverseExtractLayers.append(
            nn.ConvTranspose1d(64, 32, kernel_size=3, padding=1)
        )
        self.reverseExtractLayers.append(nn.ReLU())

        self.reverseExtractLayers.append(
            nn.ConvTranspose1d(32, self.n_reconstrFeat, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.hiddenLatents(x)
        x = self.fc(x)
        x = x.view(x.size(0), 64, -1)
        x = self.reverseExtractLayers(x)
        return x
