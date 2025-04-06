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

        self.conv1 = nn.Conv1d(n_feature, 32, kernel_size=3)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3)
        self.hiddenLatents = nn.Sequential()
        self.fc = nn.LazyLinear(latentDim)

        ## hidden latents
        currDim = int(latentDim / (hiddenLatentReducingFactor**n_hiddenLatent))
        for i in range(n_hiddenLatent):
            self.hiddenLatents.add_module(f"fc{i}", nn.LazyLinear(currDim))
            self.hiddenLatents.add_module(f"relu{i}", nn.ReLU())
            currDim = int(currDim * hiddenLatentReducingFactor)
        self.hiddenLatents.add_module("fc_fin", nn.LazyLinear(currDim))
        self.hiddenLatents.add_module("relu_fin", nn.ReLU())

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
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

        self.hiddenLatents = nn.Sequential()
        self.fc = nn.LazyLinear(deconvInput)
        self.deconv1 = nn.ConvTranspose1d(64, 32, kernel_size=3, padding=1)
        self.deconv2 = nn.ConvTranspose1d(
            32, self.n_reconstrFeat, kernel_size=3, padding=1
        )

        ## hidden latents
        currDim = int(self.reconstrDim * (hiddenLatentReducingFactor**n_hiddenLatent))
        for i in range(n_hiddenLatent):
            self.hiddenLatents.add_module(f"fc{i}", nn.LazyLinear(currDim))
            self.hiddenLatents.add_module(f"relu{i}", nn.ReLU())
            currDim = int(currDim / hiddenLatentReducingFactor)
        self.hiddenLatents.add_module("fc_fin", nn.LazyLinear(currDim))
        self.hiddenLatents.add_module("relu_fin", nn.ReLU())

    def forward(self, x):
        x = self.hiddenLatents(x)
        x = self.fc(x)
        x = x.view(x.size(0), 64, -1)
        x = self.deconv1(x)
        x = self.deconv2(x)
        return x
