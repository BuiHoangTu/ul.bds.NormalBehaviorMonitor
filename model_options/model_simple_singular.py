import torch.nn as nn


class SimpleEncoder(nn.Module):
    def __init__(
        self,
        latentDim: int,
        n_hiddenLatent=0,
        hiddenLatentReducingFactor=0.7,
    ):
        super(SimpleEncoder, self).__init__()
        self.latentDim = latentDim
        self.n_hiddenLatent = n_hiddenLatent
        self.hiddenLatentReducingFactor = hiddenLatentReducingFactor

        ## Trainable layers
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3)
        self.hiddenLatents = nn.Sequential()
        self.fc = nn.LazyLinear(latentDim)

        ## hidden latents
        for i in range(n_hiddenLatent):
            dim = int(latentDim / (hiddenLatentReducingFactor ** (n_hiddenLatent - i)))
            self.hiddenLatents.add_module(f"fc{i}", nn.LazyLinear(dim))
            self.hiddenLatents.add_module(f"relu{i}", nn.ReLU())

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
        reconstructDim: int,
        n_hiddenLatent=0,
        hiddenLatentReducingFactor=0.7,
    ):
        super(SimpleDecoder, self).__init__()
        self.reconstructDim = reconstructDim
        self.n_hiddenLatent = n_hiddenLatent
        self.hiddenLatentReducingFactor = hiddenLatentReducingFactor

        ## Trainable layers
        self.hiddenLatents = nn.Sequential()
        # make sure the output can be sent to the deconv layers
        deconvInput = (reconstructDim-4) * 64
        self.fc = nn.LazyLinear(deconvInput)
        self.deconv1 = nn.ConvTranspose1d(64, 32, kernel_size=3)
        self.deconv2 = nn.ConvTranspose1d(32, 1, kernel_size=3)

        ## hidden latents
        for i in range(n_hiddenLatent):
            dim = int(reconstructDim * (hiddenLatentReducingFactor ** (n_hiddenLatent - i)))
            self.hiddenLatents.add_module(f"fc{i}", nn.LazyLinear(dim))
            self.hiddenLatents.add_module(f"relu{i}", nn.ReLU())

    def forward(self, x):
        x = self.hiddenLatents(x)
        x = self.fc(x)
        x = x.view(x.size(0), 64, -1)
        x = self.deconv1(x)
        x = self.deconv2(x)
        return x
