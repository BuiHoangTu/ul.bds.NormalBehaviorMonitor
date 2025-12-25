import hashlib
import os
import subprocess
import sys
import argparse
from pathlib import Path
import logging
import boto3
import json
from training_trigger import createTrainingJob


MOUNTED_CODE_PATH = Path("/opt/ml/processing/input/source-code")
OUTPUT_DIR = Path("/opt/ml/processing/output")

logger = logging.getLogger(__name__)


def callTrainingJob():
    createTrainingJob()


def hashDir(dirPath):
    h = hashlib.sha256(usedforsecurity=False)

    for root, _, files in sorted(os.walk(dirPath)):
        for file in sorted(files):
            filePath = os.path.join(root, file)
            with open(filePath, "rb") as f:
                while chunk := f.read(8192):
                    h.update(chunk)
    return h.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Data preprocessing for normal behavior monitoring"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=MOUNTED_CODE_PATH,
        help="Input directory containing source code",
    )
    parser.add_argument(
        "--in-hash-dir",
        type=str,
        help="Directory to store the hash of the input directory",
    )
    parser.add_argument(
        "--in-hash-out",
        type=str,
        help="Directory to store the new hash of the input directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=OUTPUT_DIR,
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--decrypt-keys",
        type=str,
        help="Secret name for decryption keys in AWS Secrets Manager",
    )
    parser.add_argument(
        "--region",
        type=str,
        help="AWS region for the secret",
    )
    return parser.parse_args()


def checkFolderChanged(hash_dir, hash_out_dir, mounted_code_path):
    # check the hash of data and preprocess it if it has changed
    if hash_dir.exists() is False:
        hash_dir.mkdir(parents=True, exist_ok=True)
    if hash_out_dir.exists() is False:
        hash_out_dir.mkdir(parents=True, exist_ok=True)

    hashChanged = False

    pDataHash = hash_dir / "data.hash"
    currDataHash = hashDir(mounted_code_path / "data")
    pDataHashOut = hash_out_dir / "data.hash"
    try:
        lastDataHash = pDataHash.read_text()
    except FileNotFoundError:
        lastDataHash = None
    if currDataHash != lastDataHash:
        hashChanged = True
        pDataHashOut.write_text(currDataHash)

    pPreprocessHash = hash_dir / "preprocess.hash"
    currPreprocessHash = hashDir(mounted_code_path / "preprocess")
    pPreprocessHashOut = hash_out_dir / "preprocess.hash"
    try:
        lastPreprocessHash = pPreprocessHash.read_text()
    except FileNotFoundError:
        lastPreprocessHash = None
    if currPreprocessHash != lastPreprocessHash:
        hashChanged = True
        pPreprocessHashOut.write_text(currPreprocessHash)

    return hashChanged


def main(args):
    mounted_code_path = Path(args.input)
    hash_dir = Path(args.in_hash_dir)
    hash_out_dir = Path(args.in_hash_out)
    output_dir = Path(args.output)
    decrypt_keys = args.decrypt_keys
    region = args.region

    logger.info(f"Mounted code path: {mounted_code_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Decrypt keys: {decrypt_keys}")
    logger.info(f"AWS region: {region}")

    hashChanged = checkFolderChanged(hash_dir, hash_out_dir, mounted_code_path)
    if hashChanged is False:
        logger.info("No changes detected in the data or preprocessing code. Exiting.")
        callTrainingJob()
        logger.info("Exiting without re-preprocessing.")
        return

    # Add the mounted code from S3
    sys.path.insert(0, str(mounted_code_path))
    logger.info(f"Added {mounted_code_path} to Python path")

    # Add secrets for data decryption
    secretManager = boto3.client("secretsmanager", region_name=args.region)
    keysResponse = secretManager.get_secret_value(SecretId=decrypt_keys)
    keys = json.loads(keysResponse["SecretString"])
    PK = keys["PK"]
    PK2 = keys["PK2"]

    os.environ["PK"] = PK
    os.environ["PK2"] = PK2

    try:
        from preprocess import fullPrepare

        logger.info("Successfully imported preprocessing module")
    except ImportError as e:
        logger.error("Failed to import preprocessing module", exc_info=e)
        logger.info(f"Current Python path:{sys.path}")
        raise

    try:
        trainData, valData, testData = fullPrepare()
        logger.info("Data preprocessing completed successfully")
        logger.info(f"Number of training samples: {len(trainData)}")
        logger.info(f"Number of validation samples: {len(valData)}")
        logger.info(f"Number of test samples: {len(testData)}")
    except Exception as e:
        logger.error("Data preprocessing failed", exc_info=e)
        raise

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        subprocess.check_call(["chmod", "-R", "777", output_dir])

        trainData.save(str((output_dir / "train").resolve()))
        valData.save(str((output_dir / "val").resolve()))
        testData.save(str((output_dir / "test").resolve()))
    except Exception as e:
        logger.error("Failed to save preprocessed data", exc_info=e)
        raise

    logger.info("Preprocessed data saved successfully")
    callTrainingJob()


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    args = parse_args()
    main(args)
