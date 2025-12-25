from typing import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F


class TargetedLoss(nn.Module):
    """
    This class help choose which features of the output to use for loss calculation
    """

    def __init__(self, loss, target) -> None:
        super().__init__()

        self.loss = loss
        self.target = target

    def forward(self, pred, actual):
        pred = pred[:, self.target, :]
        actual = actual[:, self.target, :]
        return self.loss(pred, actual)


class SampleWeightedLoss(nn.Module):
    """
    This class allows calculating a weight for the loss based on sample pred and actual values.
    """

    def __init__(
        self,
        loss,
        weightCal: Callable,
        reduction: str = "mean",
        flipNegative: bool = True,
    ) -> None:
        """
        Args:
            loss (Callable): The loss function to be used
            weightCal (Callable): A function that takes in pred and actual and returns a weight tensor
        """
        super().__init__()

        if not callable(weightCal):
            raise ValueError("weightCal must be a callable function")
        if not isinstance(loss, Callable):
            raise ValueError("loss must be a callable function")

        self.loss = loss
        self.weightCal = weightCal
        self.flipNegative = flipNegative
        if reduction not in ["mean", "sum", "none"]:
            raise ValueError("reduction must be one of 'mean', 'sum', or 'none'")
        self.reduction = reduction

    def forward(self, pred, actual):
        weight = self.weightCal(pred, actual)
        loss = self.loss(pred, actual)
        if self.flipNegative:
            # resolve for negative weights
            # flip the loss to 1 / loss and flip it to positive
            loss = torch.where(weight < 0, 1 / (1 + loss), loss)
            weight = torch.abs(weight)

        weightedLoss = loss * weight
        if self.reduction == "mean":
            return weightedLoss.mean()
        elif self.reduction == "sum":
            return weightedLoss.sum()
        else:
            return weightedLoss


class AutoWeightedLoss(nn.Module):
    """
    Auto decide the weight for each loss components
    """

    def __init__(self, **loss_fns):
        """
        loss_fns: keyword arguments of loss_name=loss_function,
                  each loss_function must take (pred, actual) as input.
        """
        super().__init__()

        self.loss_fns = nn.ModuleDict(loss_fns)

        # Create one learnable log-variance per loss
        self.logvars = nn.ParameterDict(
            {name: nn.Parameter(torch.zeros(1)) for name in loss_fns}
        )

    def forward(self, pred, actual):
        total_loss = 0.0

        for name, loss_fn in self.loss_fns.items():
            logvar = self.logvars[name]
            loss = loss_fn(pred, actual)
            weighted_loss = 0.5 * torch.exp(-logvar) * loss + 0.5 * logvar
            total_loss += weighted_loss

        return total_loss


class FourierLoss(nn.Module):
    """
    Fourier loss that calculates the MSE of the Fourier transform of the predictions and actual values.
    """

    def __init__(self, reduction="mean"):
        super(FourierLoss, self).__init__()
        self.reduction = reduction

    def forward(self, pred, actual):
        # calculate the Fourier transform
        amp_pred = torch.abs(torch.fft.fft(pred, dim=-1))
        amp_actual = torch.abs(torch.fft.fft(actual, dim=-1))

        # calculate the MSE loss
        return F.mse_loss(amp_pred, amp_actual, reduction=self.reduction)
