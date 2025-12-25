#!/usr/bin/env python
# coding: utf-8

# # Configs

# In[ ]:


import random
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score
import torch


RANDOM_SEED = 17


random.seed(RANDOM_SEED)

np.random.seed(RANDOM_SEED)

torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)


aeStruct = [
    (7, 8, 16),
    (8, 4, 32),
    (4, 4, 64),
]


# # Auto encoder

# ## Read data

# In[2]:


from data_reader.data import listTurbines


turbines = listTurbines()
turbines


# In[3]:


# from preprocess.transform_to_3d import TurbineData


# turbines = listTurbines()
# turbineDatas = [TurbineData(turbine, verbose=False) for turbine in turbines]
# commonCols = set(turbineDatas[0].columns).intersection(
#     *[turbineData.columns for turbineData in turbineDatas]
# )
# print(f"Common columns: {sorted(commonCols)}")


# ## Split train test

# In[4]:


from data_reader.data import getDataRange
from preprocess.normalize import createFeatureTransformer


targetFeatRange = getDataRange()
# targetFeats = ["avgwindspeed"]
targetFeats = list(targetFeatRange.keys())

immuteFeats = [
    "datetime",
    "underperformanceprobability",
    "normalbehaviour",
]

angleFeats = ["avgwinddirection"]  # TODO: ["avgwinddirection"]

rangedFeatRanges = {k: targetFeatRange[k] for k in targetFeats if k not in angleFeats}

transformer = createFeatureTransformer(
    rangedFeatRanges=rangedFeatRanges,
    angleFeats=angleFeats,
    immuteFeats=[],  # immuteFeats are handled below
)


# In[5]:


import pandas as pd
from preprocess.cls_dataset import TurbineDataset
from preprocess.split_data import splitIndices
from preprocess.transform_to_3d import generateStackedTurbineData


N_STEPS_PER_SAMPLE = 128
TEST_RATIO = 0.2
VAL_RATIO = 0.2

trainSets = []
valSets = []
testSets = []

abnTrainSets = []
abnTestSets = []


def evalUnderperformValid(
    data2dDf: pd.DataFrame,
):
    maxConsecutiveInvalid = 0
    maxInvalid = 0.5 * len(data2dDf)

    invalidMask = (
        (data2dDf["underperformanceprobabilityvalid"] == 0)
        | (data2dDf["underperformanceprobabilityvalid"].isna())
        | (data2dDf["original_feature_count"] == 0)
    )

    nContinousInvalid = 0
    nInvalid = 0
    for invalidNess in invalidMask:
        if invalidNess is True:
            nInvalid += 1
            nContinousInvalid += 1

            if nContinousInvalid > maxConsecutiveInvalid:
                return False
            if nInvalid > maxInvalid:
                return False

        else:
            nContinousInvalid = 0

    # check if last step is valid
    return not invalidMask.iloc[-1]


def evalNorm_Valid(
    data2dDf: pd.DataFrame,
):
    if evalUnderperformValid(data2dDf) is False:
        return False

    normalness = data2dDf["normalbehaviour"].astype(bool)
    # normal if at least 99% of the data is normal
    return normalness.sum() >= np.floor(len(normalness) * 0.99)


def evalAbn_Valid(
    data2dDf: pd.DataFrame,
):
    if evalUnderperformValid(data2dDf) is False:
        return False

    normalness = data2dDf["normalbehaviour"].astype(bool)
    # abnormal if at least 1% of the data is abnormal
    return ~(normalness.sum() >= np.floor(len(normalness) * 0.99))


lastIndexer = None
for turbine in turbines:
    print(f"Processing turbine: {turbine}")

    indexer, (dataNorm, dataAbn) = generateStackedTurbineData(
        turbine,
        n_timesteps=N_STEPS_PER_SAMPLE,
        conditions=[evalNorm_Valid, evalAbn_Valid],
    )

    print(f"Normal valid data: {dataNorm.shape[0]}")
    print(f"Abnormal valid data: {dataAbn.shape[0]}")

    # split normal to train, val, test
    trainIndices, valIndices, testIndices = splitIndices(
        list(range(dataNorm.shape[0])),
        testRatio=TEST_RATIO,
        valRatio=VAL_RATIO,
    )
    trainSet = TurbineDataset.from3dNumpy(
        dataNorm, indexer, targetFeats, transformer, immuteFeats, trainIndices
    )
    valSet = TurbineDataset.from3dNumpy(
        dataNorm, indexer, targetFeats, transformer, immuteFeats, valIndices
    )
    testSet = TurbineDataset.from3dNumpy(
        dataNorm, indexer, targetFeats, transformer, immuteFeats, testIndices
    )

    abnTrainIndices, abnValIndices, abnTestIndices = splitIndices(
        list(range(dataAbn.shape[0])),
        testRatio=TEST_RATIO,
        valRatio=VAL_RATIO,
    )
    abnTrainSet = TurbineDataset.from3dNumpy(
        dataAbn, indexer, targetFeats, transformer, immuteFeats, abnTrainIndices
    )
    abnValSet = TurbineDataset.from3dNumpy(
        dataAbn, indexer, targetFeats, transformer, immuteFeats, abnValIndices
    )
    abnTestSet = TurbineDataset.from3dNumpy(
        dataAbn, indexer, targetFeats, transformer, immuteFeats, abnTestIndices
    )

    trainSets.append(trainSet)
    valSets.append(valSet)
    testSets.append(testSet)

    abnTrainSets.append(abnTrainSet)
    abnTestSets.append(abnTestSet)

    if lastIndexer is not None and lastIndexer != indexer:
        raise ValueError(
            f"At turbine {turbine}, indexer changed from {lastIndexer} to {indexer}. "
        )
    lastIndexer = indexer

    print("=" * 50)


# ## Create data loader

# In[ ]:


from torch.utils.data import DataLoader

from preprocess.cls_dataset import IMMUTE_GROUP, TARGET_GROUP, TurbineDataset


BATCH_SIZE = 96

# check if trainSets is unbound
try:
    trainSet = TurbineDataset.merge(trainSets)
    valSet = TurbineDataset.merge(valSets)
    testSet = TurbineDataset.merge(testSets)

    abnTrainSet = TurbineDataset.merge(abnTrainSets)
    abnTestSet = TurbineDataset.merge(abnTestSets)

    # # save the datasets
    # trainSet.save("tmp/full_train_set")
    # valSet.save("tmp/full_val_set")
    # testSet.save("tmp/full_test_set")
    # abnormalTestSet.save("tmp/full_abnormal_test_set")
    # print("Merged and saved datasets successfully.")

except NameError as e:
    raise e
    # print("The composite dataset is not defined. Try loading the datasets from files.")
    # trainSet = TurbineDataset.load("tmp/full_train_set")
    # valSet = TurbineDataset.load("tmp/full_val_set")
    # testSet = TurbineDataset.load("tmp/full_test_set")
    # abnTestSet = TurbineDataset.load("tmp/full_abnormal_test_set")

print(
    f"Normal/Abnormal ratio: {(len(trainSet) + len(valSet) + len(testSet)) / (len(abnTrainSet) + len(abnValSet) + len(abnTestSet))}"
)


indexer = trainSet.indexer

trainLoader = DataLoader(
    TurbineDataset.merge([trainSet, abnTrainSet]),
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=True,
)
valLoader = DataLoader(
    TurbineDataset.merge([valSet, abnValSet]),
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=True,
)
testLoader = DataLoader(
    TurbineDataset.merge([testSet, abnTestSet]),
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=True,
)

targetFeats = indexer.getGroupTags(TARGET_GROUP)
print(f"Target features: {targetFeats}")

immuteFeats = indexer.getGroupTags(IMMUTE_GROUP)
print(f"Immute features: {immuteFeats}")

first_batch = next(iter(trainLoader))
inputShape = np.array(first_batch[0].size())
inputShape[0] = len(targetFeats)

# model latent space
n_feat = len(targetFeats)
## reduce the n_feat by n_decompressed feats
sin_feats = [f[:-4] for f in targetFeats if f.endswith("_sin")]  # type: ignore
cos_feats = [f[:-4] for f in targetFeats if f.endswith("_cos")]  # type: ignore
n_decompressed = len(set(sin_feats + cos_feats))


print(f"inputShape: {inputShape}")


# ## Validate model

# In[ ]:


import torch
import torch.nn as nn
from torchinfo import summary
from model_options.cnn_latent import Autoencoder
from model_options.multihead.regressive_wrapper import MultiHeadAEWrapper


# test if the model is working
tmp = Autoencoder(aeStruct)
testModel = MultiHeadAEWrapper(tmp)

summary(
    testModel, torch.Size((96, inputShape[0], inputShape[1])), depth=7, device="cpu"
)


# ## Training

# ### Train Injections

# In[8]:


def inferBatch(batch, model):

    # reshapre to (batch, feats as channels, timesteps)
    imm = indexer.slice(batch, immuteFeats, dim=1)
    inp = indexer.slice(batch, targetFeats, dim=1)

    reconst, pred = model(inp)
    label = indexer.slice(batch, ["normalbehaviour"], dim=1)
    # rm feat dim (d=1)  # (batch, 1)
    label = label.squeeze(1).sum(dim=1, keepdim=True) >= 127
    # check next cell for explanation of 127
    label = label.float()

    return torch.cat((reconst, imm), dim=1), torch.cat((inp, imm), dim=1), pred, label


# In[ ]:


n_norm = indexer.slice(trainSet.items, ["normalbehaviour"], dim=1).sum(2).squeeze(1)
minN_norm = n_norm.min()
maxN_norm = n_norm.max()
stdN_norm = n_norm.std()
meanN_norm = n_norm.mean()

print(
    f"Normalness: min={minN_norm}, max={maxN_norm}, std={stdN_norm}, mean={meanN_norm}"
)

# weighting by std normalizing  will lead to bad loss function
# manually change std and mean
stdN_norm = 1
meanN_norm = 127
# after weighting, 127 -> 0.5; 128+ -> 1.0; others -> 0


# cal weight for reconstruction loss
def weightCal(pred, actual):
    normalness = indexer.slice(actual, ["normalbehaviour"], dim=1)
    # batch, 1, timesteps

    # halfN_timeSteps = normalness.shape[2] / 2
    totalNorm = normalness.sum(dim=2, keepdim=True)

    # convert to std range of -1 -> 1
    weight = (totalNorm - meanN_norm) / stdN_norm
    # convert to 0 -> 1 range
    weight = (weight + 1) / 2
    # clip weight
    weight = torch.nan_to_num(weight, nan=0, posinf=1, neginf=0.0)
    weight = torch.clamp(weight, min=0.0, max=1.0)

    return weight


def calRawAUC(aeStruct):
    # ### Loop

    # In[ ]:

    import torch.optim as optim

    from train.loss import SampleWeightedLoss, TargetedLoss
    from train.trainer import infer, train

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = MultiHeadAEWrapper(Autoencoder(aeStruct))

    # use mean average error
    rawLoss = nn.MSELoss(reduction="none")
    targetedLoss = TargetedLoss(rawLoss, indexer.getIdx(targetFeats))
    weightedLoss = SampleWeightedLoss(targetedLoss, weightCal)

    class MultiHeadLoss(nn.Module):
        def __init__(self):
            super(MultiHeadLoss, self).__init__()

            # for reconstruction loss
            self.loss1 = weightedLoss

            # for prediction loss
            self.loss2 = nn.BCEWithLogitsLoss()

            # TODO: auto weighting 2 losses

        def forward(self, reconst, org, pred, actual):
            reconstLoss = self.loss1(reconst, org)

            predLoss = self.loss2(pred, actual)

            return reconstLoss + predLoss

    criterion = MultiHeadLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 100
    earlyStopping = 10

    trainLosses = []
    valLosses = []

    def abnormErrCal(model, valOuts, *_, **__):
        pred, actual, *_ = valOuts
        weight = weightCal(pred, actual)

        # retain only samples with weight <= 0
        weight = weight.squeeze()
        pred = pred[weight <= 0]
        actual = actual[weight <= 0]

        pred = indexer.slice(pred, targetFeats, dim=1)
        actual = indexer.slice(actual, targetFeats, dim=1)

        return torch.square((pred - actual)).mean()

    def normErrCal(model, valOuts, *_, **__):
        pred, actual, *_ = valOuts
        weight = weightCal(pred, actual)

        # retain only samples with weight > 0
        weight = weight.squeeze()
        pred = pred[weight > 0.1]
        actual = actual[weight > 0.1]

        pred = indexer.slice(pred, targetFeats, dim=1)
        actual = indexer.slice(actual, targetFeats, dim=1)

        return torch.square((pred - actual)).mean()

    model, trainLosses, valLosses, info = train(
        model,
        device,
        trainLoader,
        valLoader,
        criterion,
        optimizer,
        epochs=num_epochs,
        earlyStopping=earlyStopping,
        inferBatch=inferBatch,
        verbose=False,
        trackVars={
            "abnormalErr": abnormErrCal,
            "normalErr": normErrCal,
        },
    )

    recons, orgs, abnAndNormUnderPerfPred, abnAndNormUnderPerfActual = infer(
        model, device, testLoader, inferBatch
    )

    orgFeat = indexer.slice(orgs, targetFeats, dim=1)
    reconsFeat = indexer.slice(recons, targetFeats, dim=1)
    reconsError = torch.abs(orgFeat - reconsFeat)

    normalFlag = indexer.slice(orgs, ["normalbehaviour"], dim=1)

    sampleNorms = []
    for i0 in range(reconsError.size(0)):
        # if more than 127
        normalCount = normalFlag[i0, :, :].bool().sum().item()
        if normalCount >= 127:
            sampleNorms.append(True)
        else:
            sampleNorms.append(False)

    print("Abnorm ratio: ", 1 - np.mean(sampleNorms))

    auc = roc_auc_score(
        sampleNorms, 1 - reconsError.cpu().numpy().mean(axis=2).mean(axis=1)
    )  # max per feature, mean per day
    print(f"Overall AUC: {auc:.4f}")

    return auc


compression_rates = []
_30aucs_over_compression_rate = []
modelStructs = []
for i in range(56, 28, -2):
    aeStruct = [
        (7, 14, 16),
        (14, 28, 32),
        (28, i, 64),
    ]
    modelStructs.append(aeStruct)
for i in range(28, 14, -2):
    aeStruct = [
        (7, 14, 16),
        (14, i, 32),
        (i, i, 64),
    ]
    modelStructs.append(aeStruct)
for i in range(14, 2, -2):
    aeStruct = [
        (7, 8, 16),
        (8, i, 32),
        (i, i, 64),
    ]
    modelStructs.append(aeStruct)


for aeStruct in modelStructs:
    print(f"Testing model structure: {aeStruct}")

    _30aucs = []
    for i in range(30):
        auc = calRawAUC(aeStruct)
        _30aucs.append(auc)
    _30aucs_over_compression_rate.append(_30aucs)

    compression_rate = 56 / aeStruct[-1][1]
    compression_rates.append(compression_rate)

    print(
        f"Compression rate: {compression_rate:.2f}, AUC: {np.mean(_30aucs):.4f} ± {np.std(_30aucs):.4f}"
    )


# save results
np.savez(
    "output/search-compression-classify_model.npz",
    compression_rates=compression_rates,
    _30aucs_over_compression_rate=_30aucs_over_compression_rate,
)

# # load results
# import numpy as np
# data = np.load("output/search-compression-classify_model.npz")
# compression_rates = data["compression_rates"]
# _30aucs_over_compression_rate = data["_30aucs_over_compression_rate"]

def grouped_box_plot(compression_rates, _30aucs_over_compression_rate):
    logvar_compression_rates = np.log(compression_rates)
    
    df = pd.DataFrame({
        "logvar-compression-rate": np.repeat(logvar_compression_rates, 30),
        "auc": np.concatenate(_30aucs_over_compression_rate)})
    
    num_bins = len(compression_rates)
    
    df["compression-rate"] = pd.cut(
        df["logvar-compression-rate"],
        bins=num_bins,
        labels=[f"{cr:.2f}" for cr in compression_rates],
        include_lowest=True
    )
    
    grouped = df.groupby("compression-rate")["auc"]
    
    box_data = [group for name, group in grouped]

    plt.figure(figsize=(10, 6))
    plt.boxplot(box_data, showmeans=True)
    
    bin_labels = [f"{cr:.2f}" for cr in compression_rates]
    plt.xticks(ticks=np.arange(1, len(bin_labels) + 1),
               labels=bin_labels, rotation=45)

    plt.xlabel("Compression Rate (times)")
    plt.ylabel("AUC")    
    plt.title("AUC vs Compression Rate")
    plt.savefig("output/search-compression-classify_model.svg")
