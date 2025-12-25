from os import PathLike
from pathlib import Path
from typing import Optional
import pandas as pd
from data_reader import parquet


from functools import cache


DATA_PATH = None

@cache
def listTurbines(dataPath: Optional[PathLike] = None) -> list[str]:
    if dataPath is None:
        raise ValueError("dataPath must be provided")
    
    # if dataPath is parquet file
    if str(dataPath).endswith(".parquet"):
        dfData = parquet.read(str(dataPath))
    elif str(dataPath).endswith(".csv"):
        dfData = pd.read_csv(dataPath)
    else:
        raise ValueError("Unsupported data file format. Only parquet and csv are supported.")
    return dfData["turbineid"].unique().tolist()


def readTurbine(
    name: Optional[str] = None,
    dataPath: Optional[PathLike] = None,
) -> pd.DataFrame:
    if dataPath is None:
        dataPath = DATA_PATH
    
    if str(dataPath).endswith(".parquet"):
        dfData = parquet.read(str(dataPath))
    elif str(dataPath).endswith(".csv"):
        dfData = pd.read_csv(dataPath)
    else:
        raise ValueError("Unsupported data file format. Only parquet and csv are supported.")
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


def getDataRange():
    return {
        "avgpower": [0, 2050],
        "avgrotorspeed": [0, 18],
        "avgwindspeed": [3, 15],
        "density": [1, 1.3],
        "ambienttemperature": [-5, 30],
        "avgwinddirection": [0, 360],
    }
