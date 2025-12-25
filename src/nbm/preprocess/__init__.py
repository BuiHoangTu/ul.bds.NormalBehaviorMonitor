from data_reader.data import getDataRange, listTurbines
import numpy as np
import pandas as pd
from preprocess.cls_dataset import TurbineDataset
from preprocess.normalize import createFeatureTransformer
from preprocess.split_data import splitIndices
from preprocess.transform_to_3d import generateStackedTurbineData


def fullPrepare(
    abnormalTraining: bool,
    dataPath=None,
    testRatio=0.2,
    valRatio=0.2,
):
    """
    Filter the data for valid samples, stack into time series, seperate into normal and abnormal data.
    """

    N_STEPS_PER_SAMPLE = 128
    IMMUTE_FEATS = [
        "datetime",
        "underperformanceprobability",
        "normalbehaviour",
    ]
    ANGLE_FEATS = ["avgwinddirection"]

    targetFeatRange = getDataRange()
    targetFeats = list(targetFeatRange.keys())

    rangedFeatRanges = {
        k: targetFeatRange[k] for k in targetFeats if k not in ANGLE_FEATS
    }

    transformer = createFeatureTransformer(
        rangedFeatRanges=rangedFeatRanges,
        angleFeats=ANGLE_FEATS,
        immuteFeats=[],  # immuteFeats are handled below
    )

    ## Eval functions
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
        return normalness.sum() >= np.int32(127)

    def evalAbn_Valid(
        data2dDf: pd.DataFrame,
    ):
        if evalUnderperformValid(data2dDf) is False:
            return False

        normalness = data2dDf["normalbehaviour"].astype(bool)
        return ~(normalness.sum() >= np.int32(127))

    # prepare datasets for each turbine
    turbines = listTurbines(dataPath=dataPath)

    normTrainSets = []
    normValSets = []
    normTestSets = []

    abnTrainSets = []
    abnValSets = []
    abnTestSets = []

    lastIndexer = None

    for turbine in turbines:
        print(f"Processing turbine: {turbine}")
        indexer, (dataNorm, dataAbn) = generateStackedTurbineData(
            turbine,
            n_timesteps=N_STEPS_PER_SAMPLE,
            conditions=[evalNorm_Valid, evalAbn_Valid],
            dataPath=dataPath,
        )

        print(f"Normal valid data: {dataNorm.shape[0]}")
        print(f"Abnormal valid data: {dataAbn.shape[0]}")
        print("=" * 50)

        ## split train, val, test
        # for normal data
        trainIndices, valIndices, testIndices = splitIndices(
            list(range(dataNorm.shape[0])),
            testRatio=testRatio,
            valRatio=valRatio,
        )
        normTrainSets.append(
            TurbineDataset.from3dNumpy(
                dataNorm, indexer, targetFeats, transformer, IMMUTE_FEATS, trainIndices
            )
        )
        normValSets.append(
            TurbineDataset.from3dNumpy(
                dataNorm, indexer, targetFeats, transformer, IMMUTE_FEATS, valIndices
            )
        )
        normTestSets.append(
            TurbineDataset.from3dNumpy(
                dataNorm, indexer, targetFeats, transformer, IMMUTE_FEATS, testIndices
            )
        )

        # for abnormal data
        if abnormalTraining is True:  # split like normal data
            trainIndices, valIndices, testIndices = splitIndices(
                list(range(dataAbn.shape[0])),
                testRatio=testRatio,
                valRatio=valRatio,
            )
            abnTrainSets.append(
                TurbineDataset.from3dNumpy(
                    dataAbn,
                    indexer,
                    targetFeats,
                    transformer,
                    IMMUTE_FEATS,
                    trainIndices,
                )
            )
        else:  # split 1:1 for val and test
            valIndices, _, testIndices = splitIndices(
                list(range(dataAbn.shape[0])),
                testRatio=0.5,
                valRatio=0.0,
            )
            trainIndices = []
            pass
        abnValSets.append(
            TurbineDataset.from3dNumpy(
                dataAbn, indexer, targetFeats, transformer, IMMUTE_FEATS, valIndices
            )
        )
        abnTestSets.append(
            TurbineDataset.from3dNumpy(
                dataAbn, indexer, targetFeats, transformer, IMMUTE_FEATS, testIndices
            )
        )

        if lastIndexer is not None and lastIndexer != indexer:
            raise ValueError(
                f"At turbine {turbine}, indexer changed from {lastIndexer} to {indexer}. "
            )
        lastIndexer = indexer

    # merge all turbines' data
    normTrainSet = TurbineDataset.merge(normTrainSets)
    normValSet = TurbineDataset.merge(normValSets)
    normTestSet = TurbineDataset.merge(normTestSets)

    abnTrainSet = TurbineDataset.merge(abnTrainSets)
    abnValSet = TurbineDataset.merge(abnValSets)
    abnTestSet = TurbineDataset.merge(abnTestSets)

    return (
        TurbineDataset.merge([normTrainSet, abnTrainSet]),
        TurbineDataset.merge([normValSet, abnValSet]),
        TurbineDataset.merge([normTestSet, abnTestSet]),
    )


def loadConfigs(configPath: str) -> dict:
    import yaml

    with open(configPath, "r") as f:
        configs = yaml.safe_load(f)

    return configs["preprocessing"]


if __name__ == "__main__":
    # read the cmd args
    import argparse

    parser = argparse.ArgumentParser()
    # read the process type
    parser.add_argument(
        "--abnormal-training",
        action="store_true",
        default=True,
        help="Whether to include abnormal data in training set",
    )
    parser.add_argument(
        "--test-ratio", type=float, default=0.2, help="Ratio of test set"
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.2, help="Ratio of validation set"
    )
    parser.add_argument(
        "--random-seed", type=int, default=42, help="Random seed for shuffling data"
    )
    # read the directory
    parser.add_argument(
        "--input-path",
        type=str,
        default=None,
        help="Input file of raw data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory to save the datasets",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="train-config.yaml",
        help="Path to the preprocessing config file",
    )

    args = parser.parse_args()

    # load configs
    configs = loadConfigs(args.config_path)
    # override args with configs
    configs.update({k: v for k, v in vars(args).items() if v is not None})

    import random

    np.random.seed(configs["random_seed"])
    random.seed(configs["random_seed"])

    (trainSet, valSet, testSet) = fullPrepare(
        abnormalTraining=configs["abnormal_training"],
        dataPath=configs["input_path"],
        testRatio=configs["test_ratio"],
        valRatio=configs["val_ratio"],
    )

    print(f"Train set size: {len(trainSet)}")
    print(f"Val set size: {len(valSet)}")
    print(f"Test set size: {len(testSet)}")

    if configs["output_dir"] is not None:
        from pathlib import Path

        output_dir = Path(configs["output_dir"])
        trainSet.save(output_dir / "trainSet")
        valSet.save(output_dir / "valSet")
        testSet.save(output_dir / "testSet")
