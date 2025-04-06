from typing import Any, Callable
import torch
from torch.nn import Module
from torch.optim.lr_scheduler import ReduceLROnPlateau


def infer(model, device, testLoader, inferBatch):
    model.to(device)

    model.eval()

    predProbas = []
    actuals = []

    with torch.no_grad():
        for batch in testLoader:
            try:
                batch = batch.to(device)
            except AttributeError:
                batch = [b.to(device) for b in batch]

            outputs, batchY = inferBatch(batch, model)

            predProbas.append(outputs)
            actuals.append(batchY)

    return torch.cat(predProbas), torch.cat(actuals)


def train(
    model,
    device,
    trainLoader,
    valLoader,
    criterion,
    optimizer,
    epochs,
    earlyStopping,
    inferBatch: Callable[[Any, Module], Any],
):
    """
    Train the model with early stopping and learning rate scheduling.
    Args:
        model: The model to train.
        device: The device to use for training (CPU or GPU).
        trainLoader: DataLoader for the training data.
        valLoader: DataLoader for the validation data.
        criterion: Loss function.
        optimizer: Optimizer.
        epochs: Number of epochs to train for.
        earlyStopping: Number of epochs with no improvement after which training will be stopped.
        batchToInfer: Function to process the batch before inference. It should take a batch and the model as input and return the model outputs and the target values.
    Returns:
        model: The trained model.
        trainLosses: List of training losses.
        valLosses: List of validation losses.
    """

    model.to(device)

    optimScheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=4)

    # early stopping
    bestValLoss = float("inf")
    bestModelState = None
    patience = 0

    trainLosses = []
    valLosses = []

    for epoch in range(epochs):
        model.train()
        trainLoss = 0

        for batch in trainLoader:
            try:
                batch = batch.to(device)
            except AttributeError:
                batch = [b.to(device) for b in batch]

            outputs, batchY = inferBatch(batch, model)
            loss = criterion(outputs, batchY)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            trainLoss += loss.item() * len(batchY)

        trainLoss /= len(trainLoader.dataset)
        trainLosses.append(trainLoss)

        # validation
        valPred, valActual = infer(model, device, valLoader, inferBatch)
        valLoss = criterion(valPred, valActual)
        valLosses.append(valLoss.item())

        optimScheduler.step(valLoss)

        if valLoss < bestValLoss:
            bestValLoss = valLoss
            bestModelState = model.state_dict()
            patience = 0
        else:
            patience += 1

        print(
            f"Epoch {epoch+1}/{epochs} Train Loss: {trainLoss:.4f} Val Loss: {valLoss:.4f}"
        )

        if patience > earlyStopping:
            break

    model.load_state_dict(bestModelState)
    return model, trainLosses, valLosses
