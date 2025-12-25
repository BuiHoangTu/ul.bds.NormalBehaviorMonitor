from notebook_wrapper import NotebookWrapper

def sigmoidEval(underPerfProba):
    import torch
    return 1 / (1 + torch.exp(-10 * (underPerfProba - 0.7)))


def stepEval(underPerfProba):
    import torch
    return torch.where(underPerfProba < 0.7, torch.tensor(1.0), torch.tensor(0.0))


def sqrtEval(underPerfProba):
    import torch
    return torch.sqrt(1 - underPerfProba)


def squareEval(underPerfProba):
    return (1 - underPerfProba) ** 2


methods = {
    "sigmoid": sigmoidEval,
    "step": stepEval,
    "sqrt": sqrtEval,
    "square": squareEval,
}

for methodName, method in methods.items():
    try:
        nb = NotebookWrapper(
            "./dev.ipynb",
            inputVariable=["weightEval"],
            outputVariable=["testLoss"],
        )
        reconstErr=nb.export(
            f"./output/weight-cal-method-{methodName}.ipynb",
            method,
        )

        print(f"Method: {methodName}, Reconstruction Error: {reconstErr}")
    except Exception as e:
        print(f"Error during method {methodName}: {str(e)}")
        raise e
