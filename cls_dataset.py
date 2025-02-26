import numpy as np
from torch.utils.data import Dataset


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
    def fromTurbine3d(
        turbineData3d,
        rowIndices,
        featIndices,
        transform=None,
    ):
        """Create TurbineDataset from turbineData3d

        Args:
            turbineData3d (3d numpy array): The whole data
            indices (list[int]): Indices to use for the dataset
            featIndices (list[int]): features to use
            transform (Callable, optional): how to transform the data. Defaults to None.
            validColId (int, optional): masking feature for invalid data. Defaults to None.

        """

        items = []
        for rowIdx in rowIndices:
            item = turbineData3d[rowIdx][:, featIndices]

            if transform:
                item = transform(item)

            # reshape to match the expected input shape of the model
            # add dimension of channel (1 channel)
            itemShape = (1,) + item.shape
            item = item.reshape(itemShape)

            items.append(item)

        return TurbineDataset(items)


def toTurbineDatasets(
    turbineData3d,
    indiceses,
    featIndices,
    transform=None,
) -> tuple[TurbineDataset, ...]:
    """Quickly create multiple TurbineDataset from a list of indices

    Args:
        turbineData3d (3d numpy array): The whole data
        indiceses (list[int] or list[list[int]]): Indices to use for each dataset
        featIndices (list[int]): features to use
        transform (Callable, optional): how to transform the data. Defaults to None.
        validColId (int, optional): masking feature for invalid data. Defaults to None.

    Returns:
        tuple[TurbineDataset]: _description_
    """

    # check if indiceses is a list of indices or a list of list of indices
    if isinstance(indiceses[0], int):
        indiceses = [indiceses]

    datasets = []
    for indices in indiceses:
        dataset = TurbineDataset.fromTurbine3d(
            turbineData3d,
            indices,
            featIndices,
            transform,
        )
        datasets.append(dataset)

    return tuple(datasets)
