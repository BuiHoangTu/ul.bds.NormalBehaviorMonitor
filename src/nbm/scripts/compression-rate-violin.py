from notebook_wrapper import NotebookWrapper
from matplotlib import pyplot as plt

nb = NotebookWrapper(
    "./dev.ipynb",
    inputVariable=["COMPRESSION"],
    outputVariable=["auc"],
)

STEP = 8
compressionRates = [i for i in range(16, 96, STEP)]
stables = []
unstables = []
stableMAEs = []
unstableMAEs = []
plt.figure(figsize=(10, 6))
for i, compression in enumerate(compressionRates):
    print(f"Iteration {i + 1}: Compression Rate = {compression}")

    try:
        actual, outputs, indexer, targetFeats = nb.run(
            compression,
        )

        actualNp = actual.cpu().numpy()
        mask = indexer.slice(actualNp, ["underperformanceprobability"], dim=1) < 0.7

        actualNp = indexer.slice(actualNp, targetFeats, dim=1)
        actualStable = actualNp[mask.repeat(actualNp.shape[1], axis=1)]
        actualUnstable = actualNp[~mask.repeat(actualNp.shape[1], axis=1)]

        outputsNp = outputs.cpu().numpy()
        outputsNp = indexer.slice(outputsNp, targetFeats, dim=1)
        outputsStable = outputsNp[mask.repeat(outputsNp.shape[1], axis=1)]
        outputsUnstable = outputsNp[~mask.repeat(outputsNp.shape[1], axis=1)]

        stableDiff = actualStable - outputsStable
        unstableDiff = actualUnstable - outputsUnstable
        
        stableMAE = abs(stableDiff).mean()
        unstableMAE = abs(unstableDiff).mean()

        # shift the graph as the error rise
        stables.append(stableDiff + stableMAE)
        unstables.append(unstableDiff + unstableMAE)
        
        stableMAEs.append(stableMAE)
        unstableMAEs.append(unstableMAE)

    except Exception as e:
        print(f"Error during iteration {i + 1}: {e}")
        # print("Stop early")
        # continue

plt.violinplot(
    stables,
    positions=compressionRates,
    widths=STEP // 2,
    side="low",
    showmeans=True,
    showmedians=True,
    showextrema=False,
)
plt.violinplot(
    unstables,
    positions=compressionRates,
    widths=STEP // 2,
    side="high",
    showmeans=True,
    showmedians=True,
    showextrema=False,
)

plt.plot(
    compressionRates,
    [x.mean() for x in stableMAEs],
    label="Stable MAE",
    color="blue",
    marker=".",
)

plt.title("Reconstruction Error vs Compression Rate")
plt.xlabel("Compression Rate")
plt.ylabel("Reconstruction Error")
plt.grid()
plt.savefig("output/compression_rate-mae-extended-model.svg")
