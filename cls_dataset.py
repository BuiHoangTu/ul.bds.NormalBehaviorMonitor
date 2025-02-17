import numpy as np
from torch.utils.data import Dataset


class TurbineDataset(Dataset):
    """Contains dataset and the mask for the dataset with 1 addition dimension for channel

    Args:
        Dataset (_type_): _description_
    """
    
    def __init__(
        self,
        turbineData3d,
        rowIndices,
        featIndices,
        transform=None,
        validColId=None,
    ):
        self.turbineData3d = turbineData3d
        self.rowIndices = rowIndices
        self.featIndices = featIndices
        self.transform = transform
        self.validColId = validColId

    def __len__(self):
        return len(self.rowIndices)

    def __getitem__(self, idx):
        rowIdx = self.rowIndices[idx]
        item = self.turbineData3d[rowIdx][:, self.featIndices]

        if self.transform:
            item = self.transform(item)

        if self.validColId is not None:
            # 1 for valid, 0 for invalid
            mask = self.turbineData3d[rowIdx][:, self.validColId].float()
        else:
            mask = np.ones(item.shape[0])

        # reshape to match the expected input shape of the model
        # add dimension of channel (1 channel)
        itemShape = (1,) + item.shape
        item = item.reshape(itemShape)

        maskShape = (1,) + mask.shape
        mask = mask.reshape(maskShape)

        return item, mask


def trainTestTurbineDataset(
    turbineData3d,
    trainIndices,
    testIndices,
    featIndices,
    transform=None,
):
    trainDataset = TurbineDataset(turbineData3d, trainIndices, featIndices, transform)
    testDataset = TurbineDataset(turbineData3d, testIndices, featIndices, transform)

    return trainDataset, testDataset
