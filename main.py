import numpy as np
from sklearn.discriminant_analysis import StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader
from cls_dataset import trainTestTurbineDataset
from masking import MaskedConv2d, maskedMseLoss
from prepare_data import TurbineData, listTurbines


TEST_RATIO = 0.2
SEED = 17
np.random.seed(SEED)


def main():
    turbines = listTurbines()
    print(f"Turbines: {turbines}")

    turbineData = TurbineData(turbines[0], verbose=True)
    print(f"Data shape: {turbineData.data3d.shape}")

    normalIndices = turbineData.getNormalIndices()

    ## split training and testing data
    np.random.shuffle(normalIndices)
    n_test = int(len(normalIndices) * TEST_RATIO)
    testIndices = normalIndices[:n_test]
    trainIndices = normalIndices[n_test:]

    targetFeats = [
        "avgwindspeed",
        "avgpower",
        "ambienttemperature",
        "avghumidity",
        "density",
    ]
    targetFeatIndices = [turbineData.getIdOfColumn(feat) for feat in targetFeats]

    stdScaler = StandardScaler()

    sortedTrainIndices = trainIndices.copy()
    sortedTrainIndices.sort()  # h5py requires sorted indices

    scalerTrainData = turbineData.data3d[sortedTrainIndices, 0, :][:, targetFeatIndices]
    print(scalerTrainData.shape)

    stdScaler.fit(scalerTrainData)

    trainSet, testSet = trainTestTurbineDataset(
        turbineData.data3d,
        trainIndices,
        testIndices,
        targetFeatIndices,
        stdScaler.transform,
    )

    print(f"Train sample shape: {trainSet[0][0].shape}")

    ## model
    class Encoder(nn.Module):
        def __init__(self, latentDim):
            super().__init__()
            self.latentDim = latentDim

            # 1 learnable embedding for invalid values
            self.maskEmbed = nn.Parameter(torch.zeros(1, 1, 1, 1))

            self.conv1 = MaskedConv2d(1, 8, kernel_size=3, stride=2, padding=1)  # 5->3
            self.conv2 = MaskedConv2d(8, 16, kernel_size=3, stride=2, padding=1)  # 3->2
            self.conv3 = MaskedConv2d(16, 32, kernel_size=2)  # 2->1
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(5728, 2048)
            self.fc2 = nn.Linear(2048, latentDim)

        def forward(self, x, mask):
            if mask is None:
                raise ValueError("Mask is required")

            xShapeLen = len(x.shape)
            if mask.shape != x.shape[0 : xShapeLen - 1]:
                raise ValueError(
                    "Mask shape must cover til x's time steps: "
                    + str(mask.shape)
                    + " != "
                    + str(x.shape[0 : xShapeLen - 1])
                )

            mask = torch.tensor(mask, dtype=torch.float32)
            maskAdd1 = mask.unsqueeze(-1).expand_as(x)
            x = x * maskAdd1 + (1 - maskAdd1) * self.maskEmbed

            x, mask = self.conv1(x, mask)
            x = torch.relu(x)
            x, mask = self.conv2(x, mask)
            x = torch.relu(x)
            x, mask = self.conv3(x, mask)
            x = torch.relu(x)
            x = self.flatten(x)
            x = self.fc1(x)
            x = torch.relu(x)
            x = self.fc2(x)

            return x

    class Decoder(nn.Module):
        def __init__(self, latentDim):
            super().__init__()
            self.latentDim = latentDim

            self.fc1 = nn.Linear(latentDim, 2048)
            self.fc2 = nn.Linear(2048, 5728)
            self.conv1 = nn.ConvTranspose2d(32, 16, kernel_size=2)
            self.conv2 = nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1)
            self.conv3 = nn.ConvTranspose2d(8, 1, kernel_size=3, stride=2, padding=1)

        def forward(self, x):
            x = self.fc1(x)
            x = torch.relu(x)
            x = self.fc2(x)
            x = torch.relu(x)
            x = x.view(-1, 32, 179, 1)
            x = self.conv1(x)
            x = torch.relu(x)
            x = self.conv2(x)
            x = torch.relu(x)
            x = self.conv3(x)
            x = torch.sigmoid(x)

            return x

    class Autoencoder(nn.Module):

        def __init__(self, latent_dim: int):
            super().__init__()
            self.encoder = Encoder(latent_dim)
            self.decoder = Decoder(latent_dim)

        def forward(self, x, mask=None):
            latent = self.encoder(x, mask)
            reconstructed = self.decoder(latent)
            return reconstructed

    # test if the model is working
    testModel = Autoencoder(1024)

    # pass a random tensor to the model
    x = torch.randn(32, 1, 720, 5)

    output = testModel(x, mask=np.ones(x.shape[:3]))

    print(f"Expected output shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    del testModel
    del x
    del output

    # training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    trainLoader = DataLoader(trainSet, batch_size=32, shuffle=True, pin_memory=True)
    testLoader = DataLoader(testSet, batch_size=32, shuffle=False, pin_memory=True)
    
    model = Autoencoder(1024).to(device)
    criterion = maskedMseLoss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        model.train()
        epoLoss = 0

        for i, (x, mask) in enumerate(trainLoader):
            x = x.to(device)
            
            reconstructions = model(x, mask)
            loss = criterion(reconstructions, x, mask)
            epoLoss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if i % 10 == 0:
                print(f"Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}")
                
        print(f"Epoch: {epoch}, AvgLoss: {epoLoss / len(trainLoader)}")
        


if __name__ == "__main__":
    main()
