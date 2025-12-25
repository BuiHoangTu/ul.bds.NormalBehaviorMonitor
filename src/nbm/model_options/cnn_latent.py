import torch
import torch.nn as nn


def _convBlock(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=3, padding="same"),
        nn.ReLU(),
        nn.AvgPool1d(kernel_size=2),
    )


def _deconvBlock(in_channels, out_channels):
    return nn.Sequential(
        ### Use stride = 2 at convT to improve accuracy
        nn.Upsample(scale_factor=2, mode="linear"),
        nn.ReLU(),
        nn.ConvTranspose1d(in_channels, out_channels, kernel_size=3, padding=1),
    )


class Encoder(nn.Module):
    def __init__(self, inChannels, outChannels, hiddenDim):
        super().__init__()

        self.extractor = nn.Sequential(
            _convBlock(inChannels, hiddenDim),
        )
        self.compressor = nn.Conv1d(hiddenDim, outChannels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.extractor(x)
        x = self.compressor(x)
        return x


class Decoder(nn.Module):
    def __init__(self, inChannels, outChannels, hiddenDim):

        super().__init__()

        self.decompressor = nn.ConvTranspose1d(
            inChannels, hiddenDim, kernel_size=3, padding=1
        )

        self.reconstructor = nn.Sequential(
            _deconvBlock(hiddenDim, outChannels),
        )

    def forward(self, latent):
        x = self.decompressor(latent)
        x = self.reconstructor(x)
        return x


class Autoencoder(nn.Module):
    def __init__(
        self,
        modelMatrix=[
            (7, 8, 16),
            (8, 4, 32),
            (4, 4, 64),
        ],
        *args,
        **kwargs,
    ):
        super().__init__()

        self.modelMatrix = modelMatrix

        self.n_stages = len(self.modelMatrix)

        # model definition
        self.encoders = nn.Sequential()
        self.latentActivation = nn.Tanh()
        self.decoders = nn.Sequential()

        # populate the encoders
        for layer in self.modelMatrix:
            inChannels, outChannels, hiddenDim = layer
            self.encoders.append(Encoder(inChannels, outChannels, hiddenDim))

        # populate the decoders
        for layer in reversed(self.modelMatrix):
            outChannels, inChannels, hiddenDim = layer
            self.decoders.append(Decoder(inChannels, outChannels, hiddenDim))

    def encode(self, x):
        return self.latentActivation(self.encoders(x))

    def decode(self, latent):
        return self.decoders(latent)

    def forward(self, x):
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed

    def getNStages(self) -> int:
        """Returns the number of stages in the model."""
        return self.n_stages

    def getStage(self, stage: int):
        assert (
            0 <= stage < self.n_stages
        ), f"Invalid stage index {stage}, must be in range [0, {self.n_stages - 1}]"

        if stage == self.n_stages - 1:
            return self

        model = nn.Sequential()

        for j in range(0, stage):
            # get encoder blocks and disable training
            block = self.encoders[j]
            for param in block.parameters():
                param.requires_grad = False
            model.append(block)

        # add trainable encoder block
        block = self.encoders[stage]
        for param in block.parameters():
            param.requires_grad = True
        model.append(block)
        # add latent activation
        model.append(self.latentActivation)
        # add trainable decoder blocks
        block = self.decoders[self.n_stages - 1 - stage]
        for param in block.parameters():
            param.requires_grad = True
        model.append(block)

        for j in range(self.n_stages - stage, self.n_stages):
            # get decoder blocks and disable training
            block = self.decoders[j]
            for param in block.parameters():
                param.requires_grad = False
            model.append(block)

        return model
