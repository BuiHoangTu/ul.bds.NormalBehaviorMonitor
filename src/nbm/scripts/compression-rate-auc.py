from notebook_wrapper import NotebookWrapper
from matplotlib import pyplot as plt

nb = NotebookWrapper(
    "./notebooks/dev-classify.ipynb",
    inputVariable=["aeStruct"],
    outputVariable=["auc"],
)

inputs = []
for outDim in range(56, 28, -2):

    inputs.append(
        [
            (7, 14, 16),
            (14, 28, 32),
            (28, outDim, 64),
        ]
    )


outputs = []
plt.figure(figsize=(10, 6))
for i, input in enumerate(inputs):
    c_rate = 56 / input[-1][1]

    try:
        output = nb.run(
            input,
        )
        outputs.append(output)
        
        print(f"Iteration {i + 1}: Input = {input}, Compression Rate = {c_rate}, AUC = {output}")

    except Exception as e:
        print(f"Error during iteration {i + 1}: {e}")
        # print("Stop early")
        # continue

compressions = [56 / input[-1][1] for input in inputs]

plt.plot(compressions, outputs, marker="o", linestyle="-", color="b", label="AUC")
plt.xticks(compressions, rotation=45)
plt.xlabel("Compression Rate (times)")
plt.ylabel("AUC")
plt.title("AUC vs Compression Rate")
plt.legend()
plt.tight_layout()
plt.grid(True)
plt.savefig("output/compression_rate-dev_classify-mean_1-2.svg")
