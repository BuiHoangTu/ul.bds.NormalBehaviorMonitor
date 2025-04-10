from typing import Optional
import pandas as pd
from data import parquet


from functools import cache


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