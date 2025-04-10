from typing import Callable


class TargetedLoss:
    """
    This class help choose which features of the output to use for loss calculation
    """

    def __init__(self, loss, target) -> None:
        self.loss = loss
        self.target = target
        pass

    def __call__(self, pred, actual):
        pred = pred[:, self.target, :]
        actual = actual[:, self.target, :]
        return self.loss(pred, actual)


class WeightedLoss:
    """
    This class allows choosing a feature as weight for the loss calculation
    """

    def __init__(self, loss, weightCal: Callable) -> None:
        """
        Args:
            loss (Callable): The loss function to be used
            weightCal (Callable): A function that takes in pred and actual and returns a weight tensor
        """
        if not callable(weightCal):
            raise ValueError("weightCal must be a callable function")
        if not isinstance(loss, Callable):
            raise ValueError("loss must be a callable function")
        
        self.loss = loss
        self.weightCal = weightCal

    def __call__(self, pred, actual):
        weight = self.weightCal(pred, actual)
        return (self.loss(pred, actual) * weight).mean()
