from torch.utils.data import Dataset


class TurbineDataset(Dataset):
    def __init__(self, turbineData3d, rowIndices, featIndices, transform=None):
        self.turbineData3d = turbineData3d
        self.rowIndices = rowIndices
        self.featIndices = featIndices
        self.transform = transform

    def __len__(self):
        return len(self.rowIndices)

    def __getitem__(self, idx):
        rowIdx = self.rowIndices[idx]
        item = self.turbineData3d[rowIdx][:, self.featIndices]

        if self.transform:
            item = self.transform(item)

        # reshape to match the expected input shape of the model
        # add dimension of channel (1 channel)
        shape = (1,) + item.shape

        return item.reshape(shape)


def trainTestTurbineDataset(turbineData3d, trainIndices, testIndices, featIndices, transform=None):
    trainDataset = TurbineDataset(turbineData3d, trainIndices, featIndices, transform)
    testDataset = TurbineDataset(turbineData3d, testIndices, featIndices, transform)

    return trainDataset, testDataset