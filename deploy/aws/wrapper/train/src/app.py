import subprocess
import sys
import argparse
from pathlib import Path
import logging

import torch
from torch.utils.data import DataLoader


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Data preprocessing for normal behavior monitoring"
    )
    parser.add_argument(
        "--source-code",
        type=str,
        help="Input directory containing source code",
    )
    parser.add_argument(
        "--preprocessed",
        type=str,
        help="Input directory containing preprocessed data",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output directory for model",
    )

    return parser.parse_args()


def main(args):
    mounted_code_path = Path(args.source_code)
    preprocessed_data_path = Path(args.preprocessed)
    output_dir = Path(args.output)

    logger.info(f"Mounted code path: {mounted_code_path}")
    logger.info(f"Preprocessed data path: {preprocessed_data_path}")
    logger.info(f"Output directory: {output_dir}")

    # import code from mounted_code_path
    try:
        sys.path.insert(0, str(mounted_code_path))
        logger.info(f"Added {mounted_code_path} to Python path")

        from main import trainModel
        from preprocess.cls_dataset import TurbineDataset
        from train.trainer import infer

        logger.info("Successfully imported trainModel module")
    except ImportError as e:
        logger.error("Failed to import trainModel module", exc_info=e)
        logger.info(f"Current Python path:{sys.path}")
        raise

    # read preprocessed data
    try:
        trainSet = TurbineDataset.load(
            str((preprocessed_data_path / "train").resolve())
        )
        valSet = TurbineDataset.load(str((preprocessed_data_path / "val").resolve()))
        testSet = TurbineDataset.load(str((preprocessed_data_path / "test").resolve()))

        logger.info("Loading preprocessed data completed successfully")

        logger.info(f"Number of training samples: {len(trainSet)}")
        logger.info(f"Number of validation samples: {len(valSet)}")
        logger.info(f"Number of test samples: {len(testSet)}")
    except Exception as e:
        logger.error("Loading preprocessed data failed", exc_info=e)
        raise

    trainLoader = DataLoader(trainSet, batch_size=64, shuffle=True, pin_memory=True)
    valLoader = DataLoader(valSet, batch_size=64, shuffle=False, pin_memory=True)
    testLoader = DataLoader(testSet, batch_size=64, shuffle=False, pin_memory=True)

    # train model
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        trainInjection, (model, trainLosses, valLosses) = trainModel(
            trainLoader, valLoader, compression=6, device=device,
        )

        logger.info("Train Losses: %s", trainLosses)
        logger.info("Validation Losses: %s", valLosses)

    except Exception as e:
        logger.error("Training model failed", exc_info=e)
        raise

    # evaluate model
    try: 
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        reconst, actual = infer(model, device, testLoader, trainInjection.inferBatch)
        
        y_pred = reconst.cpu().numpy()
        y_true = actual.cpu().numpy()
        
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        logger.info("Mean Squared Error: %s", mse)
        logger.info("Mean Absolute Error: %s", mae)
        logger.info("R2 Score: %s", r2)

    except Exception as e:
        logger.error("evaluate failed", exc_info=e)
        raise

    # save model
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        subprocess.check_call(["chmod", "-R", "777", output_dir])

        torch.onnx.export(
            model,
            next(iter(trainLoader))[0],
            str((output_dir / "model.onnx").resolve()),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        )
    except Exception as e:
        logger.error("Failed to save Model", exc_info=e)
        raise


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    args = parse_args()
    main(args)
