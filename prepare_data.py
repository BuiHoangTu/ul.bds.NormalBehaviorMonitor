from functools import cache
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pandas as pd

from data import parquet


_REMOVED_COLS = [  # columns that are for sure not needed
    "turbineid",
    "site",
    "phase",
    "manufacturer",
    "model",
]

_FLAG_COLS = [
    "turbulentvalid",
    "underperformanceprobabilityvalid",
    "overperformanceprobabilityvalid",
]

ORG_FEAT_COUNT = "original_feature_count"

# default values for getStackedTurbineData
DEF_N_DAYS = 2
DEF_SHIFT = 1
DEF_SAMPLE_LEN_S = 30 * 60  # default to 30 minutes -> 3 rows per sample


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


def getStackedTurbineData(
    name: str,
    n_days=DEF_N_DAYS,
    shift=DEF_SHIFT,
    sampleLen_s=DEF_SAMPLE_LEN_S,
    recompute=False,
    verbose=False,
) -> tuple[pd.Index, h5py.Dataset, h5py.File]:
    DSET_ID = "data3d"
    COLS_ID = "columns"

    ## >>> read cached data
    dataFile = Path("tmp") / f"dataset_{name}_{n_days}_{shift}_{sampleLen_s}.hdf5"
    if recompute:
        if dataFile.exists():
            dataFile.unlink()

    if dataFile.exists():
        f = h5py.File(dataFile, "r", swmr=True)
        if DSET_ID in f and COLS_ID in f:
            colStore = f[COLS_ID]
            if isinstance(colStore, h5py.Dataset):
                cols = pd.Index(colStore.asstr()[()])
            else:
                raise KeyError(
                    f"Columns' storage ``{name}`` exists but is not a h5py.Dataset"
                )

            dset = f[DSET_ID]
            if isinstance(dset, h5py.Dataset):
                return cols, dset, f
            else:
                raise KeyError(f"Dataset ``{name}`` exists but is not a h5py.Dataset")
        else:
            f.close()
    ## <<< read cached data

    dfData = readTurbine(name)
    dfData["turbulent"] = dfData["turbulent"].astype(float)

    if verbose:
        originalLen = len(dfData)
        originalNans = dfData.isna().sum().to_dict()

    # resample data
    dfData.set_index("datetime", inplace=True)
    dfData.drop(
        columns=[col for col in _REMOVED_COLS if (col in dfData.columns)],
        inplace=True,
    )
    dfFilled = dfData.resample(f"{sampleLen_s}s").mean()

    # mark empty rows
    dfFilled[ORG_FEAT_COUNT] = dfFilled.notna().astype(int).sum(axis=1)

    # empty out invalid data, leave flags
    VALID_COL = "underperformanceprobabilityvalid"
    emptyingCols = dfFilled.columns.difference(_FLAG_COLS).difference([ORG_FEAT_COUNT])
    dfFilled.loc[dfFilled[VALID_COL] == 0, emptyingCols] = np.nan

    dfFilled.interpolate(method="time", inplace=True)

    dfFilled.reset_index(inplace=True)
    dfFilled.rename(columns={"index": "datetime"}, inplace=True)

    if verbose:
        n_nan = dfFilled.isna().sum().to_dict()

        print(f"Length of data: {len(dfFilled)} / {originalLen}")
        print("Nans in features:")
        for col in n_nan.keys() & originalNans.keys():
            print(f"\t{col}: {n_nan[col]} / {originalNans[col]}")

        print("=" * 50)

    # stack data
    N_ROWS_PER_DAY = int(24 * 60 * 60 / sampleLen_s)  # 24h in seconds / sampleLen_s

    timeSteps = n_days * N_ROWS_PER_DAY

    # convert date to numpy float
    dfFilled["datetime"] = (
        dfFilled["datetime"].astype(np.int64) // 10**9  # nanoseconds to seconds
    )
    arrayFilled = dfFilled.to_numpy()
    arrayFilled = arrayFilled.astype(np.float32)

    n_stack = (len(arrayFilled) - timeSteps) // shift

    f = h5py.File(dataFile, "a")
    dset = f.create_dataset(
        DSET_ID,
        (n_stack, timeSteps, arrayFilled.shape[1]),
        dtype=np.float32,
    )

    ID_OF_VALID_COL = dfFilled.columns.get_loc(VALID_COL)
    ID_OF_N_FEAT = dfFilled.columns.get_loc(ORG_FEAT_COUNT)

    for i in range(n_stack):
        sample = arrayFilled[i * shift : i * shift + timeSteps]

        # check if sample has at least one valid value
        if sample[:, ID_OF_N_FEAT].sum() > 0 and sample[:, ID_OF_VALID_COL].sum() > 0:
            dset[i] = sample

    f.create_dataset(COLS_ID, data=dfFilled.columns.to_numpy(dtype="S"))

    return dfFilled.columns, dset, f


class TurbineData:
    USING_TURBINES = set()
    COL_UNDERPERF_PROBA = "underperformanceprobability"
    COL_UNDERPERF_VALID = "underperformanceprobabilityvalid"

    def __init__(
        self,
        turbineName: str,
        n_days=DEF_N_DAYS,
        shift=DEF_SHIFT,
        sampleLen_s=DEF_SAMPLE_LEN_S,
        verbose=False,
    ):
        self.turbineName = turbineName
        self.verbose = verbose

        if verbose:
            if turbineName in TurbineData.USING_TURBINES:
                print(f"Warning: {turbineName} is already in use")
        TurbineData.USING_TURBINES.add(turbineName)

        self.columns, self.data3d, self.f = getStackedTurbineData(
            turbineName, n_days, shift, sampleLen_s, verbose=verbose
        )

        self._idUnderperfProba = self.getIdOfColumn(self.COL_UNDERPERF_PROBA)
        self._idUnderperfValid = self.getIdOfColumn(self.COL_UNDERPERF_VALID)
        self._idOrgNFeat = self.getIdOfColumn(ORG_FEAT_COUNT)

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
            if (
                (timeStep[self._idUnderperfValid] == 0)
                or (timeStep[self._idUnderperfValid] == np.nan)
                or (timeStep[self._idOrgNFeat] == 0)
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
        if (
            (timeStep[self._idUnderperfValid] == 0)
            or (timeStep[self._idUnderperfValid] == np.nan)
            or (timeStep[self._idOrgNFeat] == 0)
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
            if (
                (timeStep[self._idUnderperfProba] > underperformThreshold)
                or (timeStep[self._idUnderperfProba] == np.nan)
                or (timeStep[self._idOrgNFeat] == 0)
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
        _, dset, f = getStackedTurbineData(turbine, recompute=True, verbose=True)
        print(dset.shape)
        f.close()
