import torch.nn as nn
from model_options import IAutoencoder


class SimpleRegressionModel(nn.Module):
    """
    A simple regression model that can be used in conjunction with the MultiHeadAEWrapper.
    This model takes encoded data and produces regression outputs.
    """

    def __init__(self, outDim):
        super(SimpleRegressionModel, self).__init__()

        self.fc = nn.LazyLinear(32)
        self.out = nn.Linear(32, outDim)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.fc(x)
        x = nn.ReLU()(x)
        x = self.out(x)
        return x

class MultiHeadAEWrapper(nn.Module):
    """
    A wrapper for a multi-head regressive autoencoder model.

    The wrapper flattens the encoded output and passes it through a regression model.
    Returns both the reconstructed input and the regression output.
    """

    def __init__(self, aeModel: IAutoencoder, regressionModel=None):
        super(MultiHeadAEWrapper, self).__init__()

        self.aeModel = aeModel
        
        if regressionModel is None:
            regressionModel = SimpleRegressionModel(outDim=1)
        self.regressionModel = regressionModel

    def forward(self, x):
        encoded = self.aeModel.encode(x)
        decoded = self.aeModel.decode(encoded)

        regressionOut = self.regressionModel(encoded)

        return decoded, regressionOut
