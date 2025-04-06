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
