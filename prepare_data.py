from functools import cache
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pandas as pd

from data import parquet


REMOVED_COLUMNS = [  # columns that are for sure not needed
    "turbineid",
    "datetime",
    "site",
    "phase",
    "manufacturer",
    "model",
]


@cache
def listTurbines() -> list[str]:
    dfData = parquet.read("./data/Carrickallen.parquet")
    return dfData["turbineid"].unique().tolist()


def readTurbine(name: Optional[str] = None) -> pd.DataFrame:
    dfData = parquet.read("./data/Carrickallen.parquet")
    dfData = dfData[
        [
            col
            for col in dfData.columns
            if col
            not in [
                "capacity",
                "cutinwindspeed",
                "dateinstalled",
            ]
        ]
    ]

    if name:
        dfTurbine = dfData[dfData["turbineid"] == name]
    else:
        dfTurbine = dfData

    return dfTurbine


@cache
def getColumns():
    return readTurbine().columns.difference(REMOVED_COLUMNS, False)


def getStackedTurbineData(
    name: str,
    n_days=5,
    shift=1,
    recompute=False,
) -> tuple[h5py.Dataset, h5py.File]:

    dataFile = Path("tmp") / f"dataset_{name}_{n_days}_{shift}.hdf5"
    if recompute:
        if dataFile.exists():
            dataFile.unlink()

    if dataFile.exists():
        f = h5py.File(dataFile, "r+")
        if name in f:
            dset = f[name]
            if isinstance(dset, h5py.Dataset):
                return dset, f
            else:
                raise KeyError(f"Dataset {name} exists but is not a h5py.Dataset")
        else:
            f.close()

    dfData = readTurbine(name)

    # fill missing index
    dfData = dfData.set_index("datetime")
    fullDateTimeRange = pd.date_range(
        start=dfData.index.min(), end=dfData.index.max(), freq="10min"
    )
    dfFilled = dfData.reindex(fullDateTimeRange)
    dfFilled = dfFilled.reset_index()
    dfFilled = dfFilled.rename(columns={"index": "datetime"})

    # stack data
    N_ROWS_PER_DAY = int(24 * 60 / 10)  # 10 minutes per row

    timeSteps = n_days * N_ROWS_PER_DAY

    dfReduced = dfFilled.drop(columns=REMOVED_COLUMNS)

    arrayReduced = dfReduced.to_numpy()
    arrayReduced = arrayReduced.astype(np.float32)

    n_stack = (len(arrayReduced) - timeSteps) // shift

    f = h5py.File(dataFile, "a")
    dset = f.create_dataset(
        name,
        (n_stack, timeSteps, arrayReduced.shape[1]),
        dtype=np.float32,
    )
    for i in range(n_stack):
        dset[i] = arrayReduced[i * shift : i * shift + timeSteps]

    return dset, f


def filterUnderperformValid(
    turbineData2d,
    idUnderperformanceprobabilityvalid,
    maxConsecutiveInvalid=3,
    maxInvalidRate=0.1,
):
    nContinuousInvalid = 0
    nInvalid = 0

    for timeStep in turbineData2d:
        if timeStep[idUnderperformanceprobabilityvalid] == 0:
            nContinuousInvalid += 1
            nInvalid += 1

            if nContinuousInvalid > maxConsecutiveInvalid:
                return False
            if nInvalid > maxInvalidRate * len(turbineData2d):
                return False

        else:
            nContinuousInvalid = 0
    return True


def filterUnderperformProba(
    turbineData2d,
    idUnderperformanceproba,
    underperformThreshold=0.7,
    maxConsecutiveUnderperform=3,
    maxUnderperformRate=0.1,
):
    if underperformThreshold >= 1:
        return True

    nContinuousUnderperform = 0
    nUnderperform = 0

    for timeStep in turbineData2d:
        if timeStep[idUnderperformanceproba] > underperformThreshold:
            nContinuousUnderperform += 1
            nUnderperform += 1

            if nContinuousUnderperform > maxConsecutiveUnderperform:
                return False
            if nUnderperform > maxUnderperformRate * len(turbineData2d):
                return False

        else:
            nContinuousUnderperform = 0
    return True


def getValidData(
    name: str,
    maxConsecutiveInvalid=3,
    maxInvalidRate=0.1,
    underperformThreshold=0.7,
    maxConsecutiveUnderperform=3,
    maxUnderperformRate=0.1,
    verbose=False,
):
    turbineData3d, f = getStackedTurbineData(name)

    columns = getColumns()
    
    idUnderperformValid = columns.get_loc("underperformanceprobabilityvalid")
    validness = []
    for i in range(turbineData3d.shape[0]):
        validness.append(
            filterUnderperformValid(
                turbineData3d[i],
                idUnderperformValid,
                maxConsecutiveInvalid,
                maxInvalidRate,
            )
        )
    validness = np.array(validness)
    if verbose:
        print(f"Valid: {sum(validness)}")

    idUnderperformProba = columns.get_loc("underperformanceprobability")
    notUnderperformness = []
    for i in range(turbineData3d.shape[0]):
        notUnderperformness.append(
            filterUnderperformProba(
                turbineData3d[i],
                idUnderperformProba,
                underperformThreshold,
                maxConsecutiveUnderperform,
                maxUnderperformRate,
            )
        )
    notUnderperformness = np.array(notUnderperformness)
    if verbose:
        print(f"Not underperform: {sum(notUnderperformness)}")
    
    normalIndices = np.where(validness & notUnderperformness)[0]
    if verbose:
        print(f"Normal: {len(normalIndices)}")

    return turbineData3d, normalIndices, f


if __name__ == "__main__":
    for turbine in listTurbines():
        print(f"Reading {turbine}")
        dset, f = getStackedTurbineData(turbine, recompute=True)
        print(dset.shape)
        f.close()
