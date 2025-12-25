import numpy as np
from preprocess.cls_dataset import TARGET_GROUP
from torch.utils.data import TensorDataset, DataLoader
from ignite.engine import Engine
from ignite.metrics import MaximumMeanDiscrepancy
from scipy.stats import wasserstein_distance


def mse(reconst, actual, indexer):
    reconst = reconst[:, indexer.getIdxInGroup(TARGET_GROUP)]
    actual = actual[:, indexer.getIdxInGroup(TARGET_GROUP)]

    return ((reconst - actual) ** 2).mean().item()


def featMse(reconst, actual, indexer):
    featMses = {}
    for feat in indexer.getGroupTags(TARGET_GROUP):
        id = indexer[feat]
        mse = ((reconst[:, id] - actual[:, id]) ** 2).mean().item()
        featMses[feat] = mse
    return featMses


def mmd(reconst, actual, indexer):
    reconst = reconst[:, indexer.getIdxInGroup(TARGET_GROUP)]
    actual = actual[:, indexer.getIdxInGroup(TARGET_GROUP)]

    dset = TensorDataset(reconst, actual)
    loader = DataLoader(dset, batch_size=1024, shuffle=False)

    def eval_f(engine, batch):
        return batch

    eval_tor = Engine(eval_f)

    mmd = MaximumMeanDiscrepancy()
    mmd.attach(eval_tor, "mmd")

    state = eval_tor.run(loader)

    return state.metrics["mmd"]


def wasserstein(reconst, actual, indexer):
    x_flat = reconst[:, indexer.getIdxInGroup(TARGET_GROUP)]
    x_flat = x_flat.cpu().numpy()
    x_flat = x_flat.reshape(x_flat.shape[0], -1)

    y_flat = actual[:, indexer.getIdxInGroup(TARGET_GROUP)]
    y_flat = y_flat.cpu().numpy()
    y_flat = y_flat.reshape(y_flat.shape[0], -1)

    dists = [
        wasserstein_distance(x_flat[:, i], y_flat[:, i]) for i in range(x_flat.shape[1])
    ]
    return np.mean(dists)


# def tnse():
#     raise NotImplementedError("tnse not implemented yet")
