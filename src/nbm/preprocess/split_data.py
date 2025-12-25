from typing import Optional
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from data_reader.data import getDataRange
from preprocess.cls_dataset import toTurbineDatasets


def splitIndices(indices: list[int], testRatio, valRatio):

    np.random.shuffle(indices)

    n_test = int(len(indices) * testRatio)
    testIndices = indices[:n_test]
    trainIndices = indices[n_test:]

    n_val = int(len(trainIndices) * valRatio)
    valIndices = trainIndices[:n_val]
    trainIndices = trainIndices[n_val:]
    return trainIndices, valIndices, testIndices


def splitTurbineData(
    turbineData,
    indicesToUse,
    testRatio,
    valRatio,
    targetFeats: Optional[list[str]] = None,
):
    """DEPRECATED: Split manually instead."""
    
    trainIndices, valIndices, testIndices = splitIndices(
        indicesToUse, testRatio, valRatio
    )

    featRange = getDataRange()
    if targetFeats is None:
        targetFeats = list(featRange.keys())

    immuteFeats = [
        "datetime",
        "underperformanceprobability",
    ]

    scaler = MinMaxScaler()
    n_features = len(targetFeats)
    scaler.fit(np.zeros((1, n_features)))

    # populate the scaler with the min and max values
    scaler.data_min_ = np.array([featRange[k][0] for k in targetFeats])
    scaler.data_max_ = np.array([featRange[k][1] for k in targetFeats])
    scaler.data_range_ = scaler.data_max_ - scaler.data_min_
    scaler.feature_names_in__ = np.array(targetFeats)  # type: ignore
    scaler.scale_ = 1 / scaler.data_range_

    return toTurbineDatasets(
        turbineData,
        (trainIndices, valIndices, testIndices),
        targetFeats,
        scaler,
        immuteFeats,
    )
