#!/usr/bin/env python
# coding: utf-8

# # Configs

# [1]:


import random
import numpy as np
import torch
from model_options.cnn_latent import Autoencoder
from train.loss import AutoWeightedLoss


RANDOM_SEED = 17
COMPRESSION = 16  # 16 is the min compression for the extended simple model, 2^4 times at cnn layers


random.seed(RANDOM_SEED)

np.random.seed(RANDOM_SEED)

torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)


aeStruct = [
    (7, 8, 16),
    (8, 6, 32),
    (6, 6, 64),  # best threshold based prediction
]


# # Auto encoder

# ## Read data

# [2]:


from data_reader.data import listTurbines


turbines = listTurbines()
turbines


# [3]:


# from preprocess.transform_to_3d import TurbineData


# turbines = listTurbines()
# turbineDatas = [TurbineData(turbine, verbose=False) for turbine in turbines]
# commonCols = set(turbineDatas[0].columns).intersection(
#     *[turbineData.columns for turbineData in turbineDatas]
# )
# print(f"Common columns: {sorted(commonCols)}")


# ## Split train test

# [4]:


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


# [5]:


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
    # normal if at least 99% of the data is normal; for 128 steps, this means at least 127 steps are normal
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

# [6]:


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


indexer = trainSet.indexer


# control ratio of normal to abnormal samples
print(f"Normal/Abnormal ratio: {len(trainSet) / len(abnTrainSet)}")

trainLoader = DataLoader(
    TurbineDataset.merge([trainSet, abnTrainSet]),
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=True,
)

trainNormLoader = DataLoader(
    trainSet,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=True,
)
abnTrainLoader = DataLoader(
    abnTrainSet,
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

latentSpace = int((n_feat - n_decompressed) * inputShape[1] // COMPRESSION)

print(f"inputShape: {inputShape}")
print(f"latentSpace: {latentSpace}")


# ## Validate model

# [7]:


import torch
import torch.nn as nn
from torchinfo import summary

# test if the model is working
testModel = Autoencoder(aeStruct)

summary(
    testModel, torch.Size((96, inputShape[0], inputShape[1])), depth=7, device="cpu"
)


# ## Training

# ### Train Injections

# [8]:


def inferBatch(batch, model):

    # reshapre to (batch, feats as channels, timesteps)
    imm = indexer.slice(batch, immuteFeats, dim=1)
    inp = indexer.slice(batch, targetFeats, dim=1)

    out = model(inp)

    return torch.cat((out, imm), dim=1), torch.cat((inp, imm), dim=1)


# [9]:


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
# after weighting, 127 -> 0.5; 128+ -> 1.0; others -> [-1, 0]


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
    weight = torch.nan_to_num(weight, nan=0, posinf=1, neginf=-1)
    weight = torch.clamp(weight, min=-1.0, max=1.0)

    return weight


# ### Loop

# [10]:

def run(rawLoss):
    from matplotlib import pyplot as plt
    import torch.optim as optim

    from train.loss import SampleWeightedLoss, TargetedLoss
    from train.trainer import infer, train

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = Autoencoder(aeStruct)

    # use mean square error
    # rawLoss = nn.MSELoss(reduction="none")  # use none to get per sample loss
    targetedLoss = TargetedLoss(rawLoss, indexer.getIdx(targetFeats))
    criterion = SampleWeightedLoss(
        targetedLoss,
        weightCal=weightCal,
    )
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 100
    earlyStopping = 10

    def abnormErrCal(model, valOuts, *_, **__):
        # valOuts = infer(model, device, trainLoader, inferBatch)

        pred, actual = valOuts
        weight = weightCal(pred, actual)

        # retain only samples with weight <= 0
        weight = weight.squeeze()
        pred = pred[weight <= 0]
        actual = actual[weight <= 0]

        pred = indexer.slice(pred, targetFeats, dim=1)
        actual = indexer.slice(actual, targetFeats, dim=1)

        return torch.square((pred - actual)).mean()

    def normErrCal(model, valOuts, *_, **__):
        pred, actual = valOuts
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
        epochs=300,
        earlyStopping=earlyStopping,
        inferBatch=inferBatch,
        verbose=False,
        trackVars={
            "abnormalErr": abnormErrCal,
            "normalErr": normErrCal,
        },
    )

    # [11]:

    plt.plot(trainLosses[10:])
    plt.plot(valLosses[10:])

    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend(["Train", "Validation"])
    plt.title("Loss over iterations")

    plt.show()

    # [12]:

    abnErr = info["abnormalErr"]
    abnErr = [i.item() for i in abnErr]

    normErr = info["normalErr"]
    normErr = [i.item() for i in normErr]

    plt.plot(abnErr[10:])
    plt.plot(normErr[10:])

    plt.xlabel("Iterations")
    plt.ylabel("Error")
    plt.legend(["Abnormal", "Normal"])
    plt.title("Error over iterations")
    plt.show()

    # ## Evaluating

    # ### Reconst MAE

    # [13]:

    from preprocess.normalize import decodeAngles
    from train.trainer import infer

    reconsts, orgs, *_ = infer(model, device, testLoader, inferBatch)

    # filter the reconstructed and original data for "normalbehaviour" sum >= 127 only
    isNormArr = indexer.slice(orgs, ["normalbehaviour"], dim=1)
    isNormMask = isNormArr.squeeze(1).sum(dim=1) >= 127
    reconsts = reconsts[isNormMask]
    orgs = orgs[isNormMask]

    circularFeats = ["avgwinddirection"]
    skipFeats = []
    errs = []
    for feat in targetFeats:
        if feat in skipFeats:
            continue

        assert isinstance(feat, str), f"Feature {feat} is not a string"
        featIndex = indexer[feat]
        print(f"Feature {feat} :")

        if feat in circularFeats:
            featErr = torch.sub(
                reconsts[:, featIndex : (featIndex + 1), :],
                orgs[:, featIndex : (featIndex + 1), :],
            )
            featErr = torch.where(
                featErr > 0.5,
                featErr - 1,
                torch.where(featErr < -0.5, featErr + 1, featErr),
            )

        elif feat.endswith("_sin") or feat.endswith("_cos"):
            print(f"Feature {feat[:-4]} :")

            sinFeat = feat[:-4] + "_sin"
            cosFeat = feat[:-4] + "_cos"
            skipFeats.append(sinFeat)
            skipFeats.append(cosFeat)

            sinIndex = indexer[sinFeat]
            cosIndex = indexer[cosFeat]

            # transform sin and cos data to n * time_steps, 2
            sinRec = reconsts[:, sinIndex : (sinIndex + 1), :].reshape(-1, 1)
            cosRec = reconsts[:, cosIndex : (cosIndex + 1), :].reshape(-1, 1)

            sinOrg = orgs[:, sinIndex : (sinIndex + 1), :].reshape(-1, 1)
            cosOrg = orgs[:, cosIndex : (cosIndex + 1), :].reshape(-1, 1)

            angleRec = torch.cat((sinRec, cosRec), dim=1)
            angleOrg = torch.cat((sinOrg, cosOrg), dim=1)

            # run through decodeAngles (n * time_steps, 1)
            angleRec = decodeAngles(angleRec.cpu().numpy())
            angleOrg = decodeAngles(angleOrg.cpu().numpy())

            # transform back to n, 1, time_steps
            angleRec = angleRec.reshape(reconsts.size(0), 1, reconsts.size(2))
            angleOrg = angleOrg.reshape(orgs.size(0), 1, orgs.size(2))

            # calculate error like in circular feats but for 360 instead of 1
            featErr = angleRec - angleOrg
            featErr = (
                np.where(
                    featErr > 180,
                    featErr - 360,
                    np.where(featErr < -180, featErr + 360, featErr),
                )
                / 360.0
            )
            featErr = torch.tensor(featErr, device=device)
        else:
            featErr = torch.sub(
                reconsts[:, featIndex : (featIndex + 1), :],
                orgs[:, featIndex : (featIndex + 1), :],
            )

        errs.append(featErr)

        print(f"\tMean: {featErr.mean().item()}")
        print(f"\tStd: {featErr.std().item()}")
        print(f"\tMax: {featErr.max().item()}")
        print(f"\tMin: {featErr.min().item()}")
        print(f"\tMean abs: {featErr.abs().mean().item()}")

    errs = torch.stack(errs, dim=1)
    print(f"Test MAE: {errs.abs().mean().mean().item()}")

    # ### Other Reconst metrics

    # [14]:

    org = indexer.slice(orgs, targetFeats, dim=1)
    reconst = indexer.slice(reconsts, targetFeats, dim=1)

    # #### Peak Signal-to-Noise Ratio

    # [15]:

    MAX_PIXEL = 1

    mse = torch.mean((org - reconst) ** 2).item()  # type: ignore
    psnr = 10 * np.log10(MAX_PIXEL / mse)
    print(f"PSNR: {psnr:.4f} dB")

    # #### Maximum Mean Discrepancy

    # [16]:

    from torch.utils.data import TensorDataset
    from ignite.engine import Engine
    from ignite.metrics import MaximumMeanDiscrepancy

    def compute_mmd(x, y, sigma=1.0):
        dset = TensorDataset(x, y)
        loader = DataLoader(dset, batch_size=BATCH_SIZE, shuffle=False)

        def eval_f(engine, batch):
            return batch

        eval_tor = Engine(eval_f)

        mmd = MaximumMeanDiscrepancy()
        mmd.attach(eval_tor, "mmd")

        state = eval_tor.run(loader)

        return state.metrics["mmd"]

    compute_mmd(reconst, org)

    # #### Wasserstein

    # [17]:

    from scipy.stats import wasserstein_distance

    def compute_wasserstein(x, y):
        x_flat = x.reshape(x.shape[0], -1)
        y_flat = y.reshape(y.shape[0], -1)
        dists = [
            wasserstein_distance(x_flat[:, i], y_flat[:, i]) for i in range(x_flat.shape[1])
        ]
        return np.mean(dists)

    compute_wasserstein(org.cpu().numpy(), reconst.cpu().numpy())

    # #### t-NSE

    # [18]:

    from sklearn.manifold import TSNE

    def plot_tsne(z, labels=None, title="Latent t-SNE"):
        tsne = TSNE(n_components=2, perplexity=30, random_state=0)
        z_2d = tsne.fit_transform(z)
        plt.figure(figsize=(6, 5))
        if labels is not None:
            plt.scatter(z_2d[:, 0], z_2d[:, 1], c=labels, cmap="tab10", s=10)
        else:
            plt.scatter(z_2d[:, 0], z_2d[:, 1], s=10)
        plt.title(title)
        plt.show()

    def inferBatchEnc(batch, model):

        # reshapre to (batch, feats as channels, timesteps)
        imm = indexer.slice(batch, immuteFeats, dim=1)
        inp = indexer.slice(batch, targetFeats, dim=1)

        out = model(inp)

        return out, torch.cat((inp, imm), dim=1)  # type: ignore

    encoded, _ = infer(model.encoders, device, testLoader, inferBatchEnc)
    encoded = torch.tanh(encoded)  # add encoder activation
    z = encoded.detach().cpu().numpy()
    # plot_tsne(z)

    # ## Reconstruct graph

    # [19]:

    a = 5000
    b = a + 1000

    plt.figure(figsize=(15, 10))
    for i, feat in enumerate(targetFeats):
        featIdx = indexer[feat]
        org = orgs[:, featIdx, :].detach().cpu().numpy().reshape(-1)
        reconst = reconsts[:, featIdx, :].detach().cpu().numpy().reshape(-1)

        plt.subplot(len(targetFeats) // 2 + 1, 2, i + 1)
        plt.plot(org[a:b], label="Original")
        plt.plot(reconst[a:b], label="Reconstructed")
        plt.plot(org[a:b] - reconst[a:b], label="Diff")
        plt.legend()
        plt.title(str(feat))
    plt.tight_layout()
    plt.show()

    # [20]:

    plt.figure(figsize=(15, 10))
    for i, feat in enumerate(targetFeats):
        featIdx = indexer[feat]
        org = orgs[:, featIdx, :].detach().cpu().numpy().reshape(-1)
        reconst = reconsts[:, featIdx, :].detach().cpu().numpy().reshape(-1)

        plt.subplot(len(targetFeats), 1, i + 1)
        plt.plot(org, label="Original")
        plt.plot(reconst, label="Reconstructed")
        plt.plot(org - reconst, label="Diff")
        plt.legend()
        plt.title(str(feat))
    plt.tight_layout()
    plt.savefig("output/target_feats.svg")
    plt.close()

    # [21]:

    # plot for all targetFeats
    plt.figure(figsize=(15, 10))
    for i, feat in enumerate(targetFeats):
        featIdx = indexer[feat]
        org = orgs[:, featIdx, :].detach().cpu().numpy().reshape(-1)
        reconst = reconsts[:, featIdx, :].detach().cpu().numpy().reshape(-1)

        sortedOrg, sortedReconst = zip(*sorted(zip(org, reconst)))

        plt.subplot(len(targetFeats) // 2 + 1, 2, i + 1)
        plt.plot(sortedReconst, label="Reconstructed", color="#ff7f0e")
        plt.plot(sortedOrg, label="Original", color="#1f77b4")
        plt.title(str(feat))
        plt.legend()
    plt.tight_layout()
    plt.show()

    # ## [scatter] Real vs. predicted

    # [22]:

    # scatter plot of real vs predicted for all targetFeats
    plt.figure(figsize=(15, 10))
    for i, feat in enumerate(targetFeats):
        featIdx = indexer[feat]
        org = orgs[:, featIdx, :].detach().cpu().numpy().reshape(-1)
        reconst = reconsts[:, featIdx, :].detach().cpu().numpy().reshape(-1)

        plt.subplot(len(targetFeats) // 2 + 1, 2, i + 1)
        plt.scatter(org, reconst, s=0.5)
        plt.xlabel("Original")
        plt.ylabel("Reconstructed")
        plt.title(str(feat))
    plt.tight_layout()
    plt.show()

    # ## [line] Error through time
    # (either MAE or MSE) + (aggregate weekly or monthly to ease visualization)

    # [23]:

    import datetime

    timeFeat = orgs[:, indexer["datetime"], :].detach().cpu().numpy().reshape(-1)

    plt.figure(figsize=(15, 10))
    for i, feat in enumerate(targetFeats):
        featIdx = indexer[feat]
        org = orgs[:, featIdx, :].detach().cpu().numpy().reshape(-1)
        reconst = reconsts[:, featIdx, :].detach().cpu().numpy().reshape(-1)

        diff = org - reconst

        sortedTime, diffByTime = zip(*sorted(zip(timeFeat, diff)))

        map = {}
        unkTime = 0
        for value, time in zip(diffByTime, sortedTime):
            try:
                time = datetime.datetime.fromtimestamp(float(time))
            except ValueError:
                unkTime += 1
                continue

            if time not in map:
                map[time] = (value, 1)
            else:
                map[time] = (map[time][0] + value, map[time][1] + 1)

        diffByTime = [value / count for value, count in map.values()]
        timeUniq = list(map.keys())

        plt.subplot(len(targetFeats) // 2 + 1, 2, i + 1)
        plt.scatter(timeUniq, diffByTime, s=0.5)
        plt.title(str(feat))
        plt.xlabel("Time")
        plt.ylabel("Diff")
    plt.tight_layout()
    plt.show()

    # [24]:

    import pandas as pd

    timeFeat = orgs[:, indexer["datetime"], :].detach().cpu().numpy().reshape(-1)

    plt.figure(figsize=(15, 10))
    for i, feat in enumerate(targetFeats):
        featIdx = indexer[feat]
        org = orgs[:, featIdx, :].detach().cpu().numpy().reshape(-1)
        reconst = reconsts[:, featIdx, :].detach().cpu().numpy().reshape(-1)

        diff = org - reconst

        sortedTime, diffByTime = zip(*sorted(zip(timeFeat, diff)))

        map = {}
        unkTime = 0
        for value, time in zip(diffByTime, sortedTime):
            try:
                time = datetime.datetime.fromtimestamp(float(time))
            except ValueError:
                unkTime += 1
                continue

            if time not in map:
                map[time] = (value, 1)
            else:
                map[time] = (map[time][0] + value, map[time][1] + 1)

        diffByTime = [value / count for value, count in map.values()]
        timeUniq = list(map.keys())

        dfDiff = pd.DataFrame({"time": timeUniq, "diff": diffByTime})
        dfDiff.set_index("time", inplace=True)

        dfWeek = dfDiff.resample("W").mean()
        dfWeek.dropna(inplace=True)

        dfMonth = dfDiff.resample("ME").mean()
        dfMonth.dropna(inplace=True)

        plt.subplot(len(targetFeats) // 2 + 1, 2, i + 1)
        plt.plot(dfWeek.index, dfWeek["diff"], label="Weekly")
        plt.plot(dfMonth.index, dfMonth["diff"], label="Monthly")
        plt.title(str(feat))
        plt.xlabel("Time")
        plt.ylabel("Diff")
        plt.legend()
    plt.tight_layout()
    plt.show()

    # ## [scatter] Err vs UnderProba
    # Error vs. underperformance probability

    # [25]:

    org = orgs[:, featIdx].detach().cpu().numpy().reshape(-1)
    reconst = reconsts[:, featIdx].detach().cpu().numpy().reshape(-1)
    diff = org - reconst

    underProba = (
        orgs[:, indexer["underperformanceprobability"]].detach().cpu().numpy().reshape(-1)
    )

    plt.figure(figsize=(10, 5))
    plt.scatter(underProba, diff**2, s=0.5)
    plt.xlabel("Underperformance probability")
    plt.ylabel("Diff")
    plt.title(f"{0} diff vs underperformance probability")
    plt.savefig("output/diff_vs_underperformance.pdf")
    plt.show()

    # [26]:

    ### Average diff by underperformance probability
    df = pd.DataFrame({"underProba": underProba, "diff": np.abs(diff)})

    num_bins = 20
    df["binned"] = pd.cut(df["underProba"], bins=num_bins)

    grouped = df.groupby("binned")["diff"]

    box_data = [group for _, group in grouped]

    # Create boxplot
    plt.figure(figsize=(12, 6))
    plt.boxplot(box_data, showmeans=True)

    # Label x-axis with bin ranges
    bin_labels = [
        f"{interval.left:.2f}-{interval.right:.2f}" for interval in grouped.groups.keys()  # type: ignore
    ]
    plt.xticks(ticks=np.arange(1, len(bin_labels) + 1), labels=bin_labels, rotation=45)

    plt.xlabel("Underperformance Probability (binned)")
    plt.ylabel("Diff")
    plt.title(f"{0} diff vs underperformance probability (box plot)")
    plt.tight_layout()
    plt.show()

    # # Abnormal Detection

    # [27]:

    recons, orgs = infer(
        model, device, testLoader, inferBatch
    )

    orgFeat = indexer.slice(orgs, targetFeats, dim=1)
    reconsFeat = indexer.slice(recons, targetFeats, dim=1)

    # targetedLoss.reduction = "none"  # keep per sample loss
    reconsError = targetedLoss(orgFeat, reconsFeat)

    timeFlag = indexer.slice(orgs, ["datetime"], dim=1)
    normalFlag = indexer.slice(orgs, ["normalbehaviour"], dim=1)

    sampleNorms = []
    for i0 in range(reconsError.size(0)):
        # if more than half during time steps are normal, then the sample is normal
        normalCount = normalFlag[i0, :, :].bool().sum().item()
        if normalCount > 0.5 * reconsError.size(2):
            sampleNorms.append(True)
        else:
            sampleNorms.append(False)

    print("Abnorm ratio: ", 1 - np.mean(sampleNorms))

    # [28]:

    # auc for each feature, using std
    from sklearn.metrics import roc_auc_score

    for i in range(reconsError.size(1)):
        feature_errors = reconsError[:, i]
        auc = roc_auc_score(sampleNorms, 1 - feature_errors.cpu().numpy().mean(axis=1))
        print(f"Feature {targetFeats[i]}: AUC = {auc:.4f}")

    auc = roc_auc_score(
        sampleNorms, 1 - reconsError.cpu().numpy().mean(axis=2).mean(axis=1)
    )  # max per feature, mean per day
    print(f"Overall AUC: {auc:.4f}")

    # [29]:

    # visualize roc curve
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(
        sampleNorms, 1 - reconsError.cpu().numpy().mean(axis=2).mean(axis=1)
    )
    plt.figure(figsize=(10, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve for Day Error vs Normal Flag")
    plt.legend()
    plt.grid()
    plt.show()

    # ### Err histogram

    # [30]:

    plt.figure(figsize=(15, 10))
    for i in range(reconsError.size(1)):
        featDayErr = reconsError[:, i]  # (sample, timestep)
        # slice err to normal and abnormal days; dayNormal - (sample)
        normalDays = featDayErr[sampleNorms, :]
        abnormalDays = featDayErr[~np.array(sampleNorms), :]

        plt.subplot(len(targetFeats) // 2 + 1, 2, i + 1)

        plt.hist(
            normalDays.std(dim=1).cpu().numpy(),
            bins=50,
            alpha=0.5,
            histtype="step",
            label="Normal Days",
        )
        plt.hist(
            abnormalDays.std(dim=1).cpu().numpy(),
            bins=50,
            alpha=0.5,
            histtype="step",
            label="Abnormal Days",
        )

        plt.xlabel("Standard Deviation of Day Error")
        plt.ylabel("Frequency")
        plt.legend()
        plt.title(f"Histogram of {feat}")
    plt.tight_layout()
    plt.show()

    # ## AU-PR

    # [31]:

    from sklearn.metrics import average_precision_score, precision_recall_curve

    print(
        f"AUPR: {average_precision_score(sampleNorms, 1 - reconsError.cpu().numpy().mean(axis=2).mean(axis=1))}"
    )

    precisions, recalls, thresholds = precision_recall_curve(
        sampleNorms, 1 - reconsError.cpu().numpy().mean(axis=2).mean(axis=1)
    )
    plt.figure(figsize=(10, 5))
    plt.plot(
        recalls,
        precisions,
        label=f"AUPR = {average_precision_score(sampleNorms, 1 - reconsError.cpu().numpy().mean(axis=2).mean(axis=1)):.4f}",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve for Day Error vs Normal Flag")
    plt.legend()
    plt.grid()
    plt.show()

    # [32]:

    # choose the theshold
    # j = 2 * precisions * recalls / (precisions + recalls + 1e-12)
    # bestThreshold = thresholds[np.argmax(j)]
    # bestThreshold

    j = tpr + fpr
    bestThreshold = thresholds[np.argmax(j)]
    print(f"Best threshold: {bestThreshold:.4f}")

    # [33]:

    # compute other metrics and plot confusion matrix

    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
    import seaborn as sns

    y_pred = (1 - reconsError.cpu().numpy().mean(axis=2).mean(axis=1)) > bestThreshold

    acc = accuracy_score(sampleNorms, y_pred)
    print(f"Accuracy: {acc:.4f}")

    recall = recall_score(sampleNorms, y_pred)
    print(f"Recall: {recall:.4f}")

    precision = precision_score(sampleNorms, y_pred)
    print(f"Precision: {precision:.4f}")

    f1 = f1_score(sampleNorms, y_pred)
    print(f"F1 Score: {f1:.4f}")

    auc = roc_auc_score(sampleNorms, 1 - reconsError.cpu().numpy().mean(axis=2).mean(axis=1))
    print(f"AUC: {auc:.4f}")

    cf = confusion_matrix(sampleNorms, y_pred)

    plt.figure(figsize=(10, 7))
    sns.heatmap(
        cf,
        annot=True,
        fmt="d",
        cmap="Blues",
        annot_kws={"size": 16},
        xticklabels=["Abnormal", "Normal"],
        yticklabels=["Abnormal", "Normal"],
    )
    plt.xlabel("Predicted", fontsize=16)
    plt.ylabel("Actual", fontsize=16)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # plt.title("Confusion Matrix", fontsize=18)
    plt.show()

from ignite.metrics import MaximumMeanDiscrepancy

if __name__ == "__main__":
    lossFuncs = []

    # mmd
    tmpLoss1 = nn.MSELoss(reduction="none")
    class MMDLoss(nn.Module):
        def __init__(self):
            super(MMDLoss, self).__init__()
            self.mmd = MaximumMeanDiscrepancy()
        
        def forward(self, x, y):
            self.mmd.reset()
            x = x.view(x.shape[0], -1)
            y = y.view(y.shape[0], -1)
            self.mmd.update((x, y))
            return self.mmd.compute()


    # wasserstein
    tmpLoss1 = nn.MSELoss(reduction="none")
    def wassersteinLoss(x, y):
        x = x.view(x.shape[0], -1)
        y = y.view(y.shape[0], -1)

        x_sorted, _ = torch.sort(x, dim=1)
        y_sorted, _ = torch.sort(y, dim=1)

        return torch.mean(torch.abs(x_sorted - y_sorted), dim=1)

    tmpLoss2 = wassersteinLoss
    tmpLoss = AutoWeightedLoss(loss1=tmpLoss1, loss2=tmpLoss2)
