from model_options.multihead.regressive_wrapper import SimpleRegressionModel
import torch.nn as nn


class BottleneckHead(nn.Module):
    def __init__(self, model, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.ae = model
        self.predictor = SimpleRegressionModel(1)

    def forward(self, x):
        reconst = self.ae(x)
        err = reconst - x

        flattenErr = err.reshape(err.size(0), -1)
        pred = self.predictor(flattenErr)
        return reconst, pred