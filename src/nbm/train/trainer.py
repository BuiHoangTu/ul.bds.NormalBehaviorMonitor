from typing import Any, Callable
from preprocess.cls_dataset import IMMUTE_GROUP, TARGET_GROUP
import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model_options import ITrainableByLayer


def infer(model, device, testLoader, inferBatch):
    """
    Run inference
    Args:
        model: The model to use for inference.
        device: The device to use for inference (CPU or GPU).
        testLoader: DataLoader for the test data.
        inferBatch: Function to process the batch before inference. It should take a batch and the model as input and return (outputs, actual values).
    Returns:
        (pred, actuals, ...): List of predicted probabilities; List of actual values; ....
    """

    model.to(device)

    model.eval()

    outputs = []

    with torch.no_grad():
        for batch in testLoader:
            try:
                batch = batch.to(device)
            except AttributeError:
                batch = [b.to(device) for b in batch]

            output = inferBatch(batch, model)

            outputs.append(output)

    return tuple(torch.cat(elem) for elem in zip(*outputs))


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
    trackVars: dict[str, Callable[[Any, Any], Any]] = {},
    verbose=True,
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
    
    try:
        criterion.to(device)
    except AttributeError:
        # criterion is not a torch.nn.Module, so we don't need to move it to device
        pass

    optimScheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=4)

    # early stopping
    bestValLoss = float("inf")
    bestModelState = None
    patience = 0

    trainLosses = []
    valLosses = []
    trackVarValues = {k: [] for k in trackVars.keys()}

    for epoch in range(epochs):
        model.train()
        trainLoss = 0

        for batch in trainLoader:
            try:
                batch = batch.to(device)
            except AttributeError:
                batch = [b.to(device) for b in batch]

            outputs = inferBatch(batch, model)
            loss = criterion(*outputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            trainLoss += loss.item()

        trainLoss /= len(trainLoader)
        trainLosses.append(trainLoss)

        # validation
        with torch.no_grad():
            valOuts = infer(model, device, valLoader, inferBatch)
            valLoss = criterion(*valOuts)
            valLosses.append(valLoss.item())

        optimScheduler.step(valLoss)

        for varName, varFunc in trackVars.items():
            try:
                trackVal = varFunc(model, valOuts)
            except Exception as e:
                trackVal = e
            trackVarValues[varName].append(trackVal)

        if valLoss < bestValLoss:
            bestValLoss = valLoss
            bestModelState = model.state_dict()
            patience = 0
        else:
            patience += 1

        if verbose:
            print(
                f"Epoch {epoch+1}/{epochs} Train Loss: {trainLoss:.4f} Val Loss: {valLoss:.4f}"
            )

        if patience > earlyStopping:
            break

    model.load_state_dict(bestModelState)
    return model, trainLosses, valLosses, trackVarValues


# TODO: fix the optimizer re-initialization. Should be an exact copy with the same training parameters.
def trainLayerByLayer(
    model: ITrainableByLayer,
    device,
    trainLoader,
    valLoader,
    criterion,
    optimizer,
    epochsPerLayer: int | list[int],
    earlyStopping,
    inferBatch: Callable[[Any, Module], Any],
    trackVars: dict[str, Callable[[Module, DataLoader], Any]] = {},
    verbose=True,
):
    """
    Train the model layer by layer.
    This function trains each stage of the model separately.
    Args:
        model: The model to train.
        device: The device to use for training (CPU or GPU).
        trainLoader: DataLoader for the training data.
        valLoader: DataLoader for the validation data.
        criterion: Loss function.
        optimizer: Optimizer.
        epochsPerLayer: Number of epochs to train for each layer.
        earlyStopping: Number of epochs with no improvement after which training will be stopped.
        batchToInfer: Function to process the batch before inference. It should take a batch and the model as input and return the model outputs and the target values.
    Returns:
        model: The trained model.
        trainLosses: List of training losses.
        valLosses: List of validation losses.
    """

    n_stages = model.getNStages()

    if isinstance(epochsPerLayer, int):
        epochsPerLayer = [epochsPerLayer] * n_stages

    trainLosseses = []
    valLosseses = []
    infos = []

    for i in range(n_stages):
        if verbose:
            print(f"Training stage {i+1}/{n_stages}")

        submodel = model.getStage(i)

        optimizer = torch.optim.Adam(
            submodel.parameters(), lr=optimizer.defaults["lr"]
        )

        # Train the model for the current stage
        _, trainLosses, valLosses, info = train(
            submodel,
            device,
            trainLoader,
            valLoader,
            criterion,
            optimizer,
            epochsPerLayer[i],  # type: ignore
            earlyStopping,
            inferBatch,
            trackVars,
            verbose=verbose,
        )

        trainLosseses.append(trainLosses)
        valLosseses.append(valLosses)
        infos.append(info)

    return model, trainLosseses, valLosseses, infos


def weightCalcNormal(self, pred, actual) -> Any:
    """Calculates weights using normal label only."""
    normalness = self.indexer.slice(actual, ["normalbehaviour"], dim=1)
    # batch, 1, timesteps

    # halfN_timeSteps = normalness.shape[2] / 2
    totalNorm = normalness.sum(dim=2, keepdim=True)

    # convert to std range of -1 -> 1
    weight = (totalNorm - 127) / 1
    # convert to 0 -> 1 range
    weight = (weight + 1) / 2
    # clip weight
    weight = torch.nan_to_num(weight, nan=0, posinf=1, neginf=0.0)
    weight = torch.clamp(weight, min=0.0, max=1.0)

    return weight


def weightCalcAbnormal(self, pred, actual) -> Any:
    """Calculates weights with neg weight for abnormal behavior."""
    normalness = self.indexer.slice(actual, ["normalbehaviour"], dim=1)

    totalNorm = normalness.sum(dim=2, keepdim=True)

    # convert to std range of -1 -> 1
    weight = (totalNorm - 127) / 1
    # convert to 0 -> 1 range
    weight = (weight + 1) / 2
    # clip weight
    weight = torch.nan_to_num(weight, nan=0, posinf=1, neginf=-1)
    weight = torch.clamp(weight, min=-1.0, max=1.0)

    return weight


def inferBatchM_head(self, batch, model) -> Any:
    indexer = self.indexer
    immuteFeats = indexer.getGroupTags(IMMUTE_GROUP)
    targetFeats = indexer.getGroupTags(TARGET_GROUP)

    imm = indexer.slice(batch, immuteFeats, dim=1)
    inp = indexer.slice(batch, targetFeats, dim=1)

    reconst, pred = model(inp)

    label = indexer.slice(batch, ["normalbehaviour"], dim=1)
    # rm feat dim (d=1)  # (batch, 1)
    label = label.squeeze(1).sum(dim=1, keepdim=True) >= 127
    # check the mean and deviation for why 127
    label = label.float()

    return torch.cat((reconst, imm), dim=1), torch.cat((inp, imm), dim=1), pred, label


def inferBatchReconst(self, batch, model) -> Any:
    indexer = self.indexer
    immuteFeats = indexer.getGroupTags(IMMUTE_GROUP)
    targetFeats = indexer.getGroupTags(TARGET_GROUP)

    imm = indexer.slice(batch, immuteFeats, dim=1)
    inp = indexer.slice(batch, targetFeats, dim=1)

    reconst = model(inp)

    return torch.cat((reconst, imm), dim=1), torch.cat((inp, imm), dim=1)


class TrainInjection:
    def __init__(self, indexer: Any):
        self.indexer = indexer

    def inferBatch(self, batch: Any, model) -> Any:
        """Inference function for a batch of data."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def weightCal(self, pred, actual) -> Any:
        """Calculates weights based on predictions."""
        underPerfProba = self.indexer.slice(actual, ["normalbehaviour"], dim=1)

        underPerfProba = torch.clamp(underPerfProba, min=0.0, max=1.0)  # type: ignore
        underPerfProba = torch.nan_to_num(underPerfProba, nan=0.0, posinf=1, neginf=0.0)
        weight = 1 - underPerfProba

        return weight
