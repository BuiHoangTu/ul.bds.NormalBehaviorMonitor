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
    sampleLen_s=10 * 60,  # default to 10 minutes
    recompute=False,
    verbose=False,
) -> tuple[h5py.Dataset, h5py.File]:

    ## >>> read cached data
    dataFile = Path("tmp") / f"dataset_{name}_{n_days}_{shift}_{sampleLen_s}.hdf5"
    if recompute:
        if dataFile.exists():
            dataFile.unlink()

    if dataFile.exists():
        f = h5py.File(dataFile, "r", swmr=True)
        if name in f:
            dset = f[name]
            if isinstance(dset, h5py.Dataset):
                return dset, f
            else:
                raise KeyError(f"Dataset {name} exists but is not a h5py.Dataset")
        else:
            f.close()
    ## <<< read cached data

    dfData = readTurbine(name)

    if verbose:
        originalLen = len(dfData)
        originalNans = dfData.isna().sum().to_dict()

    # resample data
    dfData.set_index("datetime", inplace=True)
    dfData.drop(
        columns=[
            col
            for col in REMOVED_COLUMNS
            if (col in dfData.columns) and (col != "datetime")
        ],
        inplace=True,
    )
    dfFilled = dfData.resample(f"{sampleLen_s}s").mean()

    dfFilled.reset_index(inplace=True)
    dfFilled.rename(columns={"index": "datetime"}, inplace=True)

    if verbose:
        n_nan = dfFilled.isna().sum().to_dict()

        print(f"Length of data: {len(dfFilled)} / {originalLen}")

        for col in n_nan:
            print(f"\n{col}: {n_nan[col]} / {originalNans[col]}")

    # stack data
    N_ROWS_PER_DAY = int(24 * 60 * 60 / sampleLen_s)  # 24h in seconds / sampleLen_s

    timeSteps = n_days * N_ROWS_PER_DAY

    dfFilled = dfFilled.drop(columns=["datetime"])
    arrayFilled = dfFilled.to_numpy()
    arrayFilled = arrayFilled.astype(np.float32)

    n_stack = (len(arrayFilled) - timeSteps) // shift

    f = h5py.File(dataFile, "a")
    dset = f.create_dataset(
        name,
        (n_stack, timeSteps, arrayFilled.shape[1]),
        dtype=np.float32,
    )
    for i in range(n_stack):
        dset[i] = arrayFilled[i * shift : i * shift + timeSteps]

    return dset, f


class TurbineData:
    USING_TURBINES = set()
    COL_UNDERPERFORMANCE_PROBA = "underperformanceprobability"
    COL_UNDERPERFORMANCE_PROBA_VALID = "underperformanceprobabilityvalid"

    def __init__(self, turbineName: str, verbose=False):
        self.turbineName = turbineName
        self.verbose = verbose

        if verbose:
            if turbineName in TurbineData.USING_TURBINES:
                print(f"Warning: {turbineName} is already in use")
        TurbineData.USING_TURBINES.add(turbineName)

        self.data3d, self.f = getStackedTurbineData(turbineName)
        self.columns = getColumns()

        self.idUnderperformProba = self.getIdOfColumn(self.COL_UNDERPERFORMANCE_PROBA)
        self.idUnderperformValid = self.getIdOfColumn(
            self.COL_UNDERPERFORMANCE_PROBA_VALID
        )

    def getIdOfColumn(self, columnName: str) -> int:
        """
        Get the index of a column in the dataset

        Args:
            columnName (str): name of the column

        Returns:
            int: index of the column

        Raises:
            KeyError: if the column does not exist
        """
        return self.columns.get_loc(columnName)  # type: ignore # overload's bug

    def _evalUnderperformValid(
        self, turbineData2d, maxConsecutiveInvalid, maxInvalidRate
    ):
        nContinuousInvalid = 0
        nInvalid = 0

        for timeStep in turbineData2d:
            if (timeStep[self.idUnderperformValid] == 0) or (
                timeStep[self.idUnderperformValid] == np.nan
            ):
                nContinuousInvalid += 1
                nInvalid += 1

                if nContinuousInvalid > maxConsecutiveInvalid:
                    return False
                if nInvalid > maxInvalidRate * len(turbineData2d):
                    return False

            else:
                nContinuousInvalid = 0

        # check if last value is valid
        if (timeStep[self.idUnderperformValid] == 0) or (
            timeStep[self.idUnderperformValid] == np.nan
        ):
            return False

        return True

    def _evalUnderperformProba(
        self,
        turbineData2d,
        underperformThreshold,
        maxConsecutiveUnderperform,
        maxUnderperformRate,
    ):
        if underperformThreshold >= 1:
            return True

        nContinuousUnderperform = 0
        nUnderperform = 0

        for timeStep in turbineData2d:
            if (timeStep[self.idUnderperformProba] > underperformThreshold) or (
                timeStep[self.idUnderperformProba] == np.nan
            ):
                nContinuousUnderperform += 1
                nUnderperform += 1

                if nContinuousUnderperform > maxConsecutiveUnderperform:
                    return False
                if nUnderperform > maxUnderperformRate * len(turbineData2d):
                    return False

            else:
                nContinuousUnderperform = 0
        return True

    def getNormalIndices(
        self,
        maxConsecutiveInvalid=3,
        maxInvalidRate=0.1,
        underperformThreshold=0.7,
        maxConsecutiveUnderperform=3,
        maxUnderperformRate=0.1,
    ):
        validness = []
        for i in range(self.data3d.shape[0]):
            validness.append(
                self._evalUnderperformValid(
                    self.data3d[i],
                    maxConsecutiveInvalid,
                    maxInvalidRate,
                )
            )
        validness = np.array(validness)
        if self.verbose:
            print(f"Valid: {sum(validness)}")

        notUnderperformness = []
        for i in range(self.data3d.shape[0]):
            notUnderperformness.append(
                self._evalUnderperformProba(
                    self.data3d[i],
                    underperformThreshold,
                    maxConsecutiveUnderperform,
                    maxUnderperformRate,
                )
            )
        notUnderperformness = np.array(notUnderperformness)
        if self.verbose:
            print(f"Not underperform: {sum(notUnderperformness)}")

        normalIndices = np.where(validness & notUnderperformness)[0]
        if self.verbose:
            print(f"Normal: {len(normalIndices)}")

        return normalIndices

    def close(self):
        self.f.close()
        try:
            TurbineData.USING_TURBINES.remove(self.turbineName)
        except KeyError:
            if self.verbose:
                print(
                    f"Warning: Data tracking is broken, {self.turbineName} is not tracked when closing"
                )

    def __del__(self):
        self.close()


if __name__ == "__main__":
    for turbine in listTurbines():
        print(f"Reading {turbine}")
        dset, f = getStackedTurbineData(turbine, recompute=True, verbose=True)
        print(dset.shape)
        f.close()
