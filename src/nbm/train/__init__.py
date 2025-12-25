from datetime import datetime
from pathlib import Path
import types

import numpy as np
from preprocess.cls_dataset import IMMUTE_GROUP, TARGET_GROUP, TurbineDataset
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.data import TensorDataset
from ignite.engine import Engine
from ignite.metrics import MaximumMeanDiscrepancy
from scipy.stats import wasserstein_distance


from train.loss import SampleWeightedLoss, TargetedLoss
from train.trainer import TrainInjection, infer, inferBatchM_head, inferBatchReconst, train, weightCalcAbnormal, weightCalcNormal


def evalReconstErr(model, device, testLoader, trainInjection, lossFunc, config):
    reconst, actual, *_ = infer(model, device, testLoader, trainInjection.inferBatch)

    # slice the target features only
    indexer = trainInjection.indexer
    targetFeats = indexer.getGroupTags(TARGET_GROUP)

    weight = trainInjection.weightCal(reconst, actual)
    normMask = torch.where(weight > 0, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device))

    reconst = indexer.slice(reconst, targetFeats, dim=1)
    actual = indexer.slice(actual, targetFeats, dim=1)
    
    # mae
    reconstErr = torch.mean(normMask * torch.abs(reconst - actual))

    # psnr
    mse = torch.mean(normMask * (reconst - actual) ** 2)
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = (10 * torch.log10(1.0 / mse)).item()

    # mmd (maximum mean discrepancy)
    def compute_mmd(x, y, sigma=1.0):
        dset = TensorDataset(x, y)
        loader = DataLoader(dset, batch_size=config.get("batch_size", 96), shuffle=False)

        def eval_f(engine, batch):
            return batch

        eval_tor = Engine(eval_f)

        mmd = MaximumMeanDiscrepancy()
        mmd.attach(eval_tor, "mmd")

        state = eval_tor.run(loader)

        return state.metrics["mmd"]

    mmd = compute_mmd(reconst, actual)

    # wasserstein distance
    x = reconst.detach().cpu().numpy()
    y = actual.detach().cpu().numpy()

    x_flat = x.reshape(x.shape[0], -1)
    y_flat = y.reshape(y.shape[0], -1)
    dists = [
        wasserstein_distance(x_flat[:, i], y_flat[:, i]) for i in range(x_flat.shape[1])
    ]
    wass_dist = np.mean(dists)

    ### choose threshold
    reconstErr = lossFunc(reconst, actual)
    precisions, recalls, thresholds = precision_recall_curve(
        normMask.flatten(), 1 - reconstErr.cpu().detach().numpy().mean(axis=2).mean(axis=1)
    )
    j = 2 * precisions * recalls / (precisions + recalls + 1e-12)
    bestThreshold = thresholds[np.argmax(j)]
    
    ## predictive threshold 
    predLabels = (1 - reconstErr.cpu().detach().numpy().mean(axis=2).mean(axis=1)) >= bestThreshold
    # accuracy
    acc = accuracy_score(normMask.flatten().cpu().numpy(), predLabels)
    # precision
    prec = precision_score(normMask.flatten().cpu().numpy(), predLabels)
    # recall
    rec = recall_score(normMask.flatten().cpu().numpy(), predLabels)
    # f1 score
    f1 = f1_score(normMask.flatten().cpu().numpy(), predLabels)

    auc = roc_auc_score(normMask.flatten().cpu().numpy(), 1 - reconstErr.cpu().detach().numpy().mean(axis=2).mean(axis=1))

    return (
        reconstErr.mean().item(),
        psnr,
        mmd,
        wass_dist,
        bestThreshold,
        acc,
        prec,
        rec,
        f1,
        auc,
    )

def loadConfig(configPath: Path) -> dict:
    import yaml
    if configPath is None:
        return {}

    if not configPath.exists():
        raise FileNotFoundError(f"Config file not found: {configPath}")

    with open(configPath, "r") as f:
        return yaml.safe_load(f)["training"] or {}


def buildTrainModel(
    trainSet: TurbineDataset,
    valSet: TurbineDataset,
    testSet: TurbineDataset,
    modelType: str = "vanilla",
    **config,
):

    batchSize = config.get("batch_size")
    lr = config.get("lr")
    epochs = config.get("epochs")
    earlyStopping = config.get("early_stopping")
    device = config.get("device")
    verbose = config.get("verbose")
    
    # assert configs type
    assert isinstance(batchSize, int), "batch_size must be an integer"
    assert isinstance(lr, float), "lr must be a float"
    assert isinstance(epochs, int), "epochs must be an integer"
    assert isinstance(earlyStopping, int), "early_stopping must be an integer"
    assert isinstance(device, str), "device must be a string"
    assert isinstance(verbose, bool), "verbose must be a boolean"

    trainLoader = DataLoader(trainSet, batch_size=batchSize, shuffle=True, pin_memory=True)
    valLoader = DataLoader(valSet, batch_size=batchSize, shuffle=False, pin_memory=True)
    testLoader = DataLoader(testSet, batch_size=batchSize, shuffle=False, pin_memory=True)

    indexer = trainSet.indexer
    targetFeats = indexer.getGroupTags(TARGET_GROUP)
    immuteFeats = indexer.getGroupTags(IMMUTE_GROUP)

    trainInjection = TrainInjection(indexer)

    def buildTrainMlpClassify(self, batch, model):
        """Inference function for a batch of data."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    # try vanilla model first
    if modelType == "vanilla":
        from model_options.cnn_latent import Autoencoder

        model = Autoencoder()

        # select the inference reconstruction only
        trainInjection.inferBatch = types.MethodType(inferBatchReconst, trainInjection)
        # select the weight calculation for normal data only
        trainInjection.weightCal = types.MethodType(weightCalcNormal, trainInjection)

        rawLoss = torch.nn.MSELoss(reduction="none")
        targetedLoss = TargetedLoss(rawLoss, indexer.getIdx(targetFeats))
        criterion = SampleWeightedLoss(
            targetedLoss,
            weightCal=trainInjection.weightCal,
        )
        optimizer = optim.Adam(model.parameters(), lr=lr)

        model, trainLosses, valLosses, info = train(
            model,
            device,
            trainLoader,
            valLoader,
            criterion,
            optimizer,
            epochs=epochs,
            earlyStopping=earlyStopping,
            inferBatch=trainInjection.inferBatch,
            verbose=verbose,
        )

        rawReconstMetrics = evalReconstErr(
            model, device, testLoader, trainInjection, rawLoss, config
        )

        return model, trainLosses, valLosses, info, *rawReconstMetrics

    elif modelType == "m_head-classify":
        from model_options.cnn_latent import Autoencoder
        from model_options.multihead.regressive_wrapper import MultiHeadAEWrapper

        aeModel = Autoencoder()
        model = MultiHeadAEWrapper(aeModel)

        # select the inference reconstruction only
        trainInjection.inferBatch = types.MethodType(inferBatchM_head, trainInjection)
        # select the weight calculation for normal data only
        trainInjection.weightCal = types.MethodType(weightCalcNormal, trainInjection)

        rawLoss = torch.nn.MSELoss(reduction="none")
        targetedLoss = TargetedLoss(rawLoss, indexer.getIdx(targetFeats))
        weightedLoss = SampleWeightedLoss(
            targetedLoss,
            weightCal=trainInjection.weightCal,
        )
        class MultiHeadLoss(torch.nn.Module): # type: ignore
            def __init__(self):
                super(MultiHeadLoss, self).__init__() # type: ignore
                self.loss1 = weightedLoss
                self.loss2 = torch.nn.BCEWithLogitsLoss()
            def forward(self, reconst, org, pred, actual):
                reconstLoss = self.loss1(reconst, org)
                predLoss = self.loss2(pred, actual)
                return reconstLoss + predLoss

        criterion = MultiHeadLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        model, trainLosses, valLosses, info = train(
            model,
            device,
            trainLoader,
            valLoader,
            criterion,
            optimizer,
            epochs=epochs,
            earlyStopping=earlyStopping,
            inferBatch=trainInjection.inferBatch,
            verbose=verbose,
        )

        rawReconstMetrics = evalReconstErr(
            model, device, testLoader, trainInjection, rawLoss, config
        )

        return model, trainLosses, valLosses, info, *rawReconstMetrics

    elif modelType == "classify-only":
        from model_options.cnn_latent import Autoencoder
        from model_options.multihead.regressive_wrapper import MultiHeadAEWrapper

        aeModel = Autoencoder()
        model = MultiHeadAEWrapper(aeModel)

        # select the inference reconstruction only
        trainInjection.inferBatch = types.MethodType(inferBatchM_head, trainInjection)
        # select the weight calculation for normal data only
        trainInjection.weightCal = types.MethodType(weightCalcNormal, trainInjection)

        rawLoss = torch.nn.MSELoss(reduction="none")
        targetedLoss = TargetedLoss(rawLoss, indexer.getIdx(targetFeats))
        weightedLoss = SampleWeightedLoss(
            targetedLoss,
            weightCal=trainInjection.weightCal,
        )

        class MultiHeadLoss(torch.nn.Module): # type: ignore
            def __init__(self):
                super(MultiHeadLoss, self).__init__() # type: ignore
                self.loss1 = weightedLoss
                self.loss2 = torch.nn.BCEWithLogitsLoss()

            def forward(self, reconst, org, pred, actual):
                reconstLoss = self.loss1(reconst, org)
                predLoss = self.loss2(pred, actual)
                return reconstLoss * 0 + predLoss

        criterion = MultiHeadLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        model, trainLosses, valLosses, info = train(
            model,
            device,
            trainLoader,
            valLoader,
            criterion,
            optimizer,
            epochs=epochs,
            earlyStopping=earlyStopping,
            inferBatch=trainInjection.inferBatch,
            verbose=verbose,
        )

        rawReconstMetrics = evalReconstErr(
            model, device, testLoader, trainInjection, rawLoss, config
        )

        return model, trainLosses, valLosses, info, *rawReconstMetrics

    elif modelType == "bottleneck-ae":
        from model_options.cnn_latent import Autoencoder
        from model_options.bottleneck.wrappers import BottleneckHead

        aeModel = Autoencoder()
        model = BottleneckHead(aeModel)

        # select the inference reconstruction only
        trainInjection.inferBatch = types.MethodType(inferBatchM_head, trainInjection)
        # select the weight calculation for normal data only
        trainInjection.weightCal = types.MethodType(weightCalcNormal, trainInjection)

        rawLoss = torch.nn.MSELoss(reduction="none")
        targetedLoss = TargetedLoss(rawLoss, indexer.getIdx(targetFeats))
        weightedLoss = SampleWeightedLoss(
            targetedLoss,
            weightCal=trainInjection.weightCal,
        )

        class MultiHeadLoss(torch.nn.Module): # type: ignore
            def __init__(self):
                super(MultiHeadLoss, self).__init__() # type: ignore
                self.loss1 = weightedLoss
                self.loss2 = torch.nn.BCEWithLogitsLoss()

            def forward(self, reconst, org, pred, actual):
                reconstLoss = self.loss1(reconst, org)
                predLoss = self.loss2(pred, actual)
                return reconstLoss + predLoss

        criterion = MultiHeadLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        model, trainLosses, valLosses, info = train(
            model,
            device,
            trainLoader,
            valLoader,
            criterion,
            optimizer,
            epochs=epochs,
            earlyStopping=earlyStopping,
            inferBatch=trainInjection.inferBatch,
            verbose=verbose,
        )

        rawReconstMetrics = evalReconstErr(
            model, device, testLoader, trainInjection, rawLoss, config
        )

        return model, trainLosses, valLosses, info, *rawReconstMetrics

    elif modelType == "bottleneck-only":
        from model_options.cnn_latent import Autoencoder
        from model_options.bottleneck.wrappers import BottleneckHead

        aeModel = Autoencoder()
        model = BottleneckHead(aeModel)

        # select the inference reconstruction only
        trainInjection.inferBatch = types.MethodType(inferBatchM_head, trainInjection)
        # select the weight calculation for normal data only
        trainInjection.weightCal = types.MethodType(weightCalcNormal, trainInjection)

        rawLoss = torch.nn.MSELoss(reduction="none")
        targetedLoss = TargetedLoss(rawLoss, indexer.getIdx(targetFeats))
        weightedLoss = SampleWeightedLoss(
            targetedLoss,
            weightCal=trainInjection.weightCal,
        )

        class MultiHeadLoss(torch.nn.Module):
            def __init__(self):
                super(MultiHeadLoss, self).__init__()
                self.loss1 = weightedLoss
                self.loss2 = torch.nn.BCEWithLogitsLoss()

            def forward(self, reconst, org, pred, actual):
                reconstLoss = self.loss1(reconst, org)
                predLoss = self.loss2(pred, actual)
                return reconstLoss * 0 + predLoss

        criterion = MultiHeadLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        model, trainLosses, valLosses, info = train(
            model,
            device,
            trainLoader,
            valLoader,
            criterion,
            optimizer,
            epochs=epochs,
            earlyStopping=earlyStopping,
            inferBatch=trainInjection.inferBatch,
            verbose=verbose,
        )

        rawReconstMetrics = evalReconstErr(
            model, device, testLoader, trainInjection, rawLoss, config
        )

        return model, trainLosses, valLosses, info, *rawReconstMetrics

    elif modelType == "abn-reinforce":
        from model_options.cnn_latent import Autoencoder

        model = Autoencoder()

        # select the inference reconstruction only
        trainInjection.inferBatch = types.MethodType(inferBatchReconst, trainInjection)
        # select the weight calculation for normal data only
        trainInjection.weightCal = types.MethodType(weightCalcAbnormal, trainInjection)

        rawLoss = torch.nn.MSELoss(reduction="none")
        targetedLoss = TargetedLoss(rawLoss, indexer.getIdx(targetFeats))
        criterion = SampleWeightedLoss(
            targetedLoss,
            weightCal=trainInjection.weightCal,
        )
        optimizer = optim.Adam(model.parameters(), lr=lr)

        model, trainLosses, valLosses, info = train(
            model,
            device,
            trainLoader,
            valLoader,
            criterion,
            optimizer,
            epochs=epochs,
            earlyStopping=earlyStopping,
            inferBatch=trainInjection.inferBatch,
            verbose=verbose,
        )

        rawReconstMetrics = evalReconstErr(
            model, device, testLoader, trainInjection, rawLoss, config
        )

        return model, trainLosses, valLosses, info, *rawReconstMetrics

    raise ValueError(f"Unknown model type: {modelType}")


if __name__ == "__main__":
    # read the cmd args
    import argparse

    parser = argparse.ArgumentParser()
    # read the process type
    parser.add_argument(
        "--config-path",
        type=str,
        default="train-config.yaml",
        help="Path to the config file",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=False,
        help="Input directory containing the preprocessed data",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="vanilla",
        choices=["vanilla", "m_head-classify", "classify-only", "bottleneck-ae", "bottleneck-only", "abn-reinforce"],
        help="Type of model to train.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=False,
        help="Output directory to save the trained model",
    )
    args = parser.parse_args()

    configs = loadConfig(Path(args.config_path))

    # override configs with cmd args
    configs.update({k: v for k, v in vars(args).items() if v is not None})

    # read the data from inputDir
    inputDir = Path(configs["input_dir"])

    trainSet = TurbineDataset.load(inputDir / "trainSet")
    valSet = TurbineDataset.load(inputDir / "valSet")
    testSet = TurbineDataset.load(inputDir / "testSet")

    model, trainLosses, valLosses, info, *rawReconstMetrics = buildTrainModel(
        trainSet,
        valSet,
        testSet,
        modelType=configs["model_type"],
        **configs
    )

    # save model to outputDir
    outputDir = Path(configs["output_dir"])
    outputDir.mkdir(parents=True, exist_ok=True)

    timeNow = datetime.now().strftime("%Y%m%d_%H%M%S")
    modelName = f"{configs['model_type']}_{timeNow}.pth"

    torch.save(model.state_dict(), outputDir / modelName)
