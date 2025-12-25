import os
import time
import boto3
from datetime import datetime

SQS_QUEUE_URL = os.environ["SQS_QUEUE_URL"]
MAX_BATCH_SIZE = 10000

sagemaker = boto3.client("sagemaker")
sqs = boto3.client("sqs")

INPUT_DIR = "/opt/ml/processing/input/source-code"
INPUT_HASH_DIR = "/opt/ml/processing/input/source-code-hash"
INPUT_HASH_OUT = "/opt/ml/processing/output/source-code-hash-out"
OUTPUT_DIR = "/opt/ml/processing/output/preprocessed"


def createProcessingJob():
    bucket = os.environ["OUT_BUCKET"]

    # Create a unique job name
    job_name = f"nbm-preprocessing-job-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"

    # Start the processing job
    appArgs = []
    appArgs.extend(["--input", INPUT_DIR])
    appArgs.extend(["--in-hash-dir", INPUT_HASH_DIR])
    appArgs.extend(["--in-hash-out", INPUT_HASH_OUT])
    appArgs.extend(["--output", OUTPUT_DIR])
    appArgs.extend(["--decrypt-keys", os.environ["KEY_NAME"]])
    appArgs.extend(["--region", os.environ["KEY_REGION"]])
    response = sagemaker.create_processing_job(
        ProcessingJobName=job_name,
        ProcessingResources={
            "ClusterConfig": {
                "InstanceCount": 1,
                "InstanceType": "ml.r5.xlarge",
                "VolumeSizeInGB": 30,
            }
        },
        AppSpecification={
            "ImageUri": os.environ["IMAGE_URI"],
            "ContainerArguments": appArgs,
            "Environment": {
                "TRAIN_ROLE_ARN": os.environ["TRAIN_ROLE_ARN"],
                "TRAIN_IMAGE_URI": os.environ["TRAIN_IMAGE_URI"],
                "SOURCE_CODE_BUCKET": os.environ["SOURCE_CODE_BUCKET"],
                "SOURCE_CODE_PREFIX": os.environ["SOURCE_CODE_PREFIX"],
                "PROCESSED_BUCKET": os.environ["PROCESSED_BUCKET"],
                "PROCESSED_PREFIX": os.environ["PROCESSED_PREFIX"],
                "TRAIN_OUT_BUCKET": os.environ["TRAIN_OUT_BUCKET"],
                "TRAIN_OUT_PREFIX": os.environ["TRAIN_OUT_PREFIX"],
                "REGION": os.environ["REGION"],
            },
        },
        RoleArn=os.environ["PROCESSING_ROLE_ARN"],
        ProcessingInputs=[
            {
                "InputName": "input",
                "S3Input": {
                    "S3Uri": f"s3://{bucket}/{os.environ['IN_PREFIX']}/",
                    "LocalPath": INPUT_DIR,
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                },
            },
            {
                "InputName": "input-hash",
                "S3Input": {
                    "S3Uri": f"s3://{bucket}/{os.environ['IN_HASH_PREFIX']}/",
                    "LocalPath": INPUT_HASH_DIR,
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                },
            },
        ],
        ProcessingOutputConfig={
            "Outputs": [
                {
                    "OutputName": "output-1",
                    "S3Output": {
                        "S3Uri": f"s3://{os.environ['OUT_BUCKET']}/{os.environ['OUT_PREFIX']}/",
                        "LocalPath": OUTPUT_DIR,
                        "S3UploadMode": "EndOfJob",
                    },
                },
                {
                    "OutputName": "input-hash-out",
                    "S3Output": {
                        "S3Uri": f"s3://{bucket}/{os.environ['IN_HASH_PREFIX']}/",
                        "LocalPath": INPUT_HASH_OUT,
                        "S3UploadMode": "EndOfJob",
                    },
                },
            ]
        },
    )
    return job_name, response


def lambda_handler(event, context):
    batchSize = len(event["Records"])

    if batchSize >= MAX_BATCH_SIZE:
        # wait then check if the queue is still empty
        time.sleep(30)

        attrs = sqs.get_queue_attributes(
            QueueUrl=SQS_QUEUE_URL, AttributeNames=["ApproximateNumberOfMessages"]
        )
        remaining = int(attrs["Attributes"]["ApproximateNumberOfMessages"])
        if remaining > 0:
            # if there are messages in the queue, return

            print(f"There are {remaining} messages in the queue.")
            print("Skipping processing job creation.")

            return {
                "statusCode": 200,
                "body": {
                    "msg": "Queue is not empty, skipping processing job creation.",
                    "message_count": remaining,
                },
            }

    job_name, attrs = createProcessingJob()

    return {
        "statusCode": 200,
        "body": {
            "msg": f"Started processing job {job_name}",
            "job_response": attrs,
        },
    }
