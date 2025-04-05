import random
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader
from support_classes.cls_dataset import toTurbineDatasets
from model_options.model_simple_singular import SimpleDecoder, SimpleEncoder
from prepare_data import TurbineData, listTurbines
from torchsummary import summary


RANDOM_SEED = 17
TEST_RATIO = 0.2
VAL_RATIO = 0.2
COMPRESSION = 2


def main():
    turbines = listTurbines()
    print(f"Turbines: {turbines}")

    turbineData = TurbineData(turbines[1], verbose=False)
    print("Data shape and columns:")
    print(turbineData.data3d.shape)
    print(turbineData.columns)
    print("=" * 50)

    turbineData.verbose = True
    normalIndices = turbineData.getNormalIndices(
        maxConsecutiveInvalid=0,  # 1 hours of consecutive invalid data
        maxInvalidRate=0.5,
        underperformThreshold=1,  # ignore underperf threshold
    )

    ## split training and testing data
    np.random.shuffle(normalIndices)
    n_test = int(len(normalIndices) * TEST_RATIO)
    testIndices = normalIndices[:n_test]
    trainIndices = normalIndices[n_test:]
    n_val = int(len(trainIndices) * VAL_RATIO)
    valIndices = trainIndices[:n_val]
    trainIndices = trainIndices[n_val:]

    print(f"Train: {len(trainIndices)} Val: {len(valIndices)} Test: {len(testIndices)}")
    print("=" * 50)

    targetFeats = [
        "avgpower",
        "avgrotorspeed",
        "avgwindspeed",
        "density",
        "ambienttemperature",
        "avgwinddirection",
    ]
    immuteFeats = [
        "datetime",
        "underperformanceprobability",
    ]
    # train the transformer
    stdScaler = StandardScaler()

    # h5py requires sorted indices
    sortedTrainIndices = trainIndices.copy()
    sortedTrainIndices.sort()

    trainData2d = turbineData.data3d[sortedTrainIndices, 0, :]
    targetFeatIndices = [turbineData.getIdOfColumn(feat) for feat in targetFeats]
    transformerTrainData = trainData2d[:, targetFeatIndices]
    print(f"Train data for scaler and imputer {transformerTrainData.shape}")

    transformerTrainData = stdScaler.fit_transform(transformerTrainData)

    indexer, (trainSet, valSet, testSet) = toTurbineDatasets(
        turbineData,
        (trainIndices, valIndices, testIndices),  # type: ignore
        targetFeats,
        stdScaler.transform,
        immuteFeats,
    )

    trainLoader = DataLoader(trainSet, batch_size=64, shuffle=True, pin_memory=True)
    valLoader = DataLoader(valSet, batch_size=64, shuffle=False, pin_memory=True)
    testLoader = DataLoader(testSet, batch_size=64, shuffle=False, pin_memory=True)

    first_batch = next(iter(trainLoader)).permute(0, 2, 1)
    inputShape = first_batch[0].size()
    print(f"Data shape: {inputShape}")
    
    class Autoencoder(nn.Module):
        def __init__(self, sampleShape, latent_dim: int):
            super().__init__()
            
            n_feat, n_time = sampleShape
            
            self.encoder = SimpleEncoder(n_feat, latent_dim)
            self.decoder = SimpleDecoder(sampleShape)

        def forward(self, x):
            latent = self.encoder(x)
            reconstructed = self.decoder(latent)
            return reconstructed


    # test if the model is working
    testModel = Autoencoder(inputShape, inputShape[0] * inputShape[1] // COMPRESSION)

    summary(testModel, inputShape, device="cpu")
    print("=" * 50)
    
    

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    main()
