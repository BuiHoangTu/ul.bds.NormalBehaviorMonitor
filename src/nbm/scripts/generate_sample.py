#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
from datetime import datetime
import hashlib
from data_reader.data import readTurbine

# read the next args
parser = argparse.ArgumentParser(description="Generate sample data for turbines")
parser.add_argument(
    "--inputFile", type=str, required=True, help="Path to the input turbine data file"
)
parser.add_argument(
    "--outputFile",
    type=str,
    default="data/raw/sample.csv",
    help="Path to the output sample data file",
)
args = parser.parse_args()

fullData = readTurbine(dataPath=args.inputFile)


# In[2]:


fullData["turbineid"].unique()


# In[3]:


# for each turbine id, slice random 6 months of data
import numpy as np
import pandas as pd


df = fullData
df["datetime"] = pd.to_datetime(df["datetime"])

rng = np.random.default_rng(seed=42)

# --- 1. Create stable random ID mapping for turbineid ---
SALT = datetime.now().isoformat()


def hash_turbineid(tid):
    return hashlib.sha256((SALT + tid).encode()).hexdigest()[:12]


df["turbineid"] = df["turbineid"].apply(hash_turbineid)


# --- 2. 6 continuous months and shift time ---
# random new start time
random_start_time = pd.Timestamp("2010-01-01")


def process_turbine(group):
    group = group.sort_values("datetime")

    min_time = group["datetime"].min()
    max_time = group["datetime"].max()

    # latest possible start so that 6 months fit
    latest_start = max_time - pd.DateOffset(months=6)
    if latest_start <= min_time:
        return pd.DataFrame()  # not enough data

    # random slice start
    slice_start = min_time + (latest_start - min_time) * rng.random()
    slice_end = slice_start + pd.DateOffset(months=6)

    sliced = group[
        (group["datetime"] >= slice_start) & (group["datetime"] < slice_end)
    ].copy()

    if sliced.empty:
        return pd.DataFrame()

    # shift datetime
    sliced["datetime"] = (
        sliced["datetime"] - sliced["datetime"].min()
    ) + random_start_time

    return sliced


# --- 3. Apply per turbine ---
result = (
    df.groupby("turbineid", group_keys=False)
    .apply(process_turbine)
    .reset_index(drop=True)
)


# In[4]:


retainingCols = [
    "turbineid",
    "datetime",
    "underperformanceprobability",
    "underperformanceprobabilityvalid",
    "normalbehaviour",
    "avgwinddirection",
    "avgpower",
    "avgrotorspeed",
    "avgwindspeed",
    "density",
    "ambienttemperature",
]

result = result[retainingCols]


# In[5]:


result.to_csv(args.outputFile, index=False)
