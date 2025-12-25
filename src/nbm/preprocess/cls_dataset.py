from os import PathLike
from typing import Iterable, Optional, Union
from deprecated import deprecated
import numpy as np
from torch.utils.data import Dataset

from preprocess.cls_indexer import Indexer
from preprocess.transform_to_3d import TurbineData


IMMUTE_GROUP = "imm"
TARGET_GROUP = "inp"


class TurbineDataset(Dataset):
    """Contains dataset and the mask for the dataset with 1 addition dimension for channel

    Args:
        Dataset (_type_): _description_
    """

    def __init__(self, items, indexer: Indexer):
        self.items = items
        self.indexer = indexer

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

    def save(self, prefix: PathLike):
        np.save(str(prefix) + "_items.npy", self.items)
        self.indexer.save(str(prefix) + "_indexer.pkl")

    @staticmethod
    def load(prefix: PathLike):
        items = np.load(str(prefix) + "_items.npy")
        indexer = Indexer.load(str(prefix) + "_indexer.pkl")

        return TurbineDataset(items, indexer)

    _EMPTY_INSTANCE = None

    @classmethod
    def Empty(cls):
        if cls._EMPTY_INSTANCE is None:
            cls._EMPTY_INSTANCE = TurbineDataset([], Indexer())
        return cls._EMPTY_INSTANCE

    @staticmethod
    @deprecated
    def fromTurbineData(
        turbineData: TurbineData,
        rowIndices: list[int],
        featNames: list[str],
        transformer=None,
        immuteFeats: list[str] = [],
    ):
        """Create TurbineDataset from turbineData3d

        Args:
            turbineData (TurbineData): The whole data
            rowIndices (list[int]): Indices to use for the dataset
            featNames (list[str]): features to use
            transformer (TransformerMixin, optional): transformer to use for the data. Defaults to None.
            immuteFeats (list[str], optional): features that should not be transformed. Defaults to [].

        """

        return TurbineDataset.from3dNumpy(
            turbineData.data3d,  # type: ignore # dset is eqivalent to numpy
            Indexer(tags=list(turbineData.columns)),
            rowIndices=rowIndices,
            featNames=featNames,
            transformer=transformer,
            immuteFeats=immuteFeats,
        )

    @staticmethod
    def from3dNumpy(
        turbineNp: np.ndarray,
        indexer: Indexer,
        featNames: list[str],
        transformer=None,
        immuteFeats: list[str] = [],
        rowIndices: Optional[list[int]] = None,
    ):

        if rowIndices is None:
            rowIndices = list(range(turbineNp.shape[0]))

        turbineNp = turbineNp[rowIndices, ...]  # select only the rows we want

        # check if turbineNp is empty
        if turbineNp.shape[0] == 0:
            return TurbineDataset.Empty()

        turbineNpFeat = indexer.slice(turbineNp, featNames, dim=1)
        turbineNpImmute = indexer.slice(turbineNp, immuteFeats, dim=1)

        if transformer and turbineNpFeat.shape[0] > 0:
            # transform the features
            n_batch, n_feat, n_time = turbineNpFeat.shape

            npFeatAs2d = turbineNpFeat.transpose((0, 2, 1)).reshape(
                n_batch * n_time, n_feat
            )
            npFeatAs2d = transformer.transform(npFeatAs2d)

            # placehold with -1 because n_feat might change after transformation
            turbineNpFeat = npFeatAs2d.reshape(n_batch, n_time, -1).transpose((0, 2, 1))

            # check if the names can be transformed
            try:
                featNames = transformer.get_feature_names_out(featNames).tolist()
                # check if the number of features matched
                if turbineNpFeat.shape[1] != len(featNames):
                    raise ValueError(
                        "Transformer's out_names does not match the actual number of out_features."
                        + " Expected: "
                        + str(turbineNpFeat.shape[1])
                        + " features, but got: "
                        + str(len(featNames))
                        + " features' names."
                    )
            except AttributeError:
                # if the transformer does not have names_out, check if the number of features changed
                if turbineNpFeat.shape[1] != len(featNames):
                    raise ValueError(
                        "Transformer does not have get_feature_names_out method"
                        + " and the number of features changed."
                        + " From: "
                        + str(len(featNames))
                        + " features To: "
                        + str(turbineNpFeat.shape[1])
                        + " features. You must provide the new feature names."
                    )

        # concatenate the features and immute features
        turbineNp = np.concatenate((turbineNpFeat, turbineNpImmute), axis=1)

        args = {
            TARGET_GROUP: featNames,
            IMMUTE_GROUP: immuteFeats,
        }
        outIndexer = Indexer(defaultDim=0, **args)

        return TurbineDataset(turbineNp, outIndexer)

    @staticmethod
    def merge(datasets):
        """Merge multiple datasets into one dataset

        Args:
            datasets (list[TurbineDataset]): List of datasets to merge

        Returns:
            TurbineDataset: Merged dataset
        """
        return mergeTurbineDatasets(datasets)


def mergeTurbineDatasets(dataSets: list[TurbineDataset]) -> TurbineDataset:
    dataSets = [ds for ds in dataSets if len(ds) > 0]
    
    # check if any non-empty dataset
    if len(dataSets) == 0:
        return TurbineDataset.Empty()
    
    ### check indexer is the same
    indexers = [dataSet.indexer for dataSet in dataSets]
    ## check len
    lengths = set([len(indexer) for indexer in indexers])
    if len(lengths) != 1:
        raise ValueError("Indexers are not the same length. Found: " + str(lengths))
    ## check indexer is the same
    indexer = indexers[0]
    for i in range(1, len(indexers)):
        if indexer != indexers[i]:
            raise ValueError(
                "Indexers are not the same. Found: {"
                + str(indexer)
                + "} and {"
                + str(indexers[i])
                + "}"
            )

    ### merge items
    items = []
    for dataSet in dataSets:
        items.extend(dataSet.items)
    items = np.array(items)

    return TurbineDataset(items, indexer)


def toTurbineDatasets(
    turbineData: TurbineData,
    indiceses: Union[list[int], Iterable[list[int]]],
    featNames: list[str],
    transformer=None,
    immuteFeats: list[str] = [],
) -> tuple[TurbineDataset, ...]:
    """Quickly create multiple TurbineDataset from a list of indices

    Args:
        turbineData (TurbineData): The whole data
        indiceses (list[int] or list[list[int]]): Indices to use for each dataset
        featNames (list[str]): features to use
        transformer (TransformerMixin, optional): transformer to use for the data. Defaults to None.
        immuteFeats (list[str], optional): features that should not be transformed. Defaults to [].

        Returns:
            tuple[TurbineDataset]: _description_
    """

    # check if indiceses is a list of indices or a list of list of indices
    indicesList = list(indiceses)
    if isinstance(indicesList[0], int):
        indicesList = [indicesList]

    datasets = []
    for indices in indicesList:
        dataset = TurbineDataset.fromTurbineData(
            turbineData,
            indices,  # type: ignore # intellisense bug
            featNames,
            transformer,
            immuteFeats,
        )
        datasets.append(dataset)

    return tuple(datasets)
