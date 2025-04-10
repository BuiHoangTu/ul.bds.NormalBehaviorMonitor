from typing import Iterable, Union
import numpy as np
from torch.utils.data import Dataset

from preprocess.cls_indexer import Indexer
from preprocess.transform_data import TurbineData


class TurbineDataset(Dataset):
    """Contains dataset and the mask for the dataset with 1 addition dimension for channel

    Args:
        Dataset (_type_): _description_
    """

    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

    def save(self, path):
        np.save(path + "_items.npy", self.items)

    @staticmethod
    def load(path):
        items = np.load(path + "_items.npy")
        return TurbineDataset(items)

    @staticmethod
    def fromTurbineData(
        turbineData: TurbineData,
        rowIndices: list[int],
        featNames: list[str],
        transform=None,
        immuteFeats: list[str] = [],
    ):
        """Create TurbineDataset from turbineData3d

        Args:
            turbineData (TurbineData): The whole data
            rowIndices (list[int]): Indices to use for the dataset
            featNames (list[str]): features to use
            transform (Callable, optional): how to transform the data. Defaults to None.
            immuteFeats (list[str], optional): features that should not be transformed. Defaults to [].

        """

        featIndices = [turbineData.getIdOfColumn(featName) for featName in featNames]
        immuteIndices = [
            turbineData.getIdOfColumn(featName) for featName in immuteFeats
        ]
        data3d = turbineData.data3d

        items = []
        for rowIdx in rowIndices:
            itemFeat = data3d[rowIdx][:, featIndices]
            itemImmuteFeat = data3d[rowIdx][:, immuteIndices]

            if transform:
                itemFeat = transform(itemFeat)

            item = np.concatenate((itemFeat, itemImmuteFeat), axis=1)

            items.append(item)

        indexer = Indexer(featNames, immuteFeats)

        return indexer, TurbineDataset(items)


def toTurbineDatasets(
    turbineData: TurbineData,
    indiceses: Union[list[int], Iterable[list[int]]],
    featNames: list[str],
    transform=None,
    immuteFeats: list[str] = [],
) -> tuple[Indexer, tuple[TurbineDataset, ...]]:
    """Quickly create multiple TurbineDataset from a list of indices

    Args:
        turbineData (TurbineData): The whole data
        indiceses (list[int] or list[list[int]]): Indices to use for each dataset
        featNames (list[str]): features to use
        transform (Callable, optional): how to transform the data. Defaults to None.
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
        indexer, dataset = TurbineDataset.fromTurbineData(
            turbineData,
            indices,  # type: ignore # intellisense bug
            featNames,
            transform,
            immuteFeats,
        )
        datasets.append(dataset)

    return indexer, tuple(datasets)
