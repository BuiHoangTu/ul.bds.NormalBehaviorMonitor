from datetime import datetime
import os
import boto3


# The input channels are mounted here
IN_DIR = "/opt/ml/input/data"
# The output directory is mounted here
OUT_DIR = "/opt/ml/model"


sagemaker = boto3.client("sagemaker")


def createTrainingJob():
    jobName = f"nbm-training-job-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"

    sourceCodeChannel =  "source-code"
    sourceCodeMount = IN_DIR + "/" + sourceCodeChannel
    preprocessedChannel = "preprocessed"
    preprocessedMount = IN_DIR + "/" + preprocessedChannel

    appArgs = []
    appArgs.extend(["--source-code", sourceCodeMount])
    appArgs.extend(["--preprocessed", preprocessedMount])
    appArgs.extend(["--output", OUT_DIR])

    res = sagemaker.create_training_job(
        TrainingJobName=jobName,
        AlgorithmSpecification={
            "TrainingImage": os.environ["TRAIN_IMAGE_URI"],
            "TrainingInputMode": "File",
            "MetricDefinitions": [
                {"Name": "mse", "Regex": "Mean Squared Error: ([0-9\\.eE+-]+)"},
                {"Name": "mae", "Regex": "Mean Absolute Error: ([0-9\\.eE+-]+)"},
                {"Name": "r2", "Regex": "R2 Score: ([0-9\\.eE+-]+)"},
                {
                    "Name": "train_loss",
                    "Regex": "Train Losses: \\[?([0-9\\.,\\s]+)\\]?",
                },
                {
                    "Name": "val_loss",
                    "Regex": "Validation Losses: \\[?([0-9\\.,\\s]+)\\]?",
                },
            ],
            "ContainerArguments": appArgs,
        },
        RoleArn=os.environ["TRAIN_ROLE_ARN"],
        ResourceConfig={
            "InstanceType": "ml.g5.xlarge",
            "InstanceCount": 1,
            "VolumeSizeInGB": 30,
        },
        InputDataConfig=[
            {
                "ChannelName": sourceCodeChannel,
                "DataSource": {
                    "S3DataSource": {
                        "S3Uri": f"s3://{os.environ['SOURCE_CODE_BUCKET']}/{os.environ['SOURCE_CODE_PREFIX']}/",
                        "S3DataType": "S3Prefix",
                    }
                },
            },
            {
                "ChannelName": preprocessedChannel,
                "DataSource": {
                    "S3DataSource": {
                        "S3Uri": f"s3://{os.environ['PROCESSED_BUCKET']}/{os.environ['PROCESSED_PREFIX']}/",
                        "S3DataType": "S3Prefix",
                    }
                },
            },
        ],
        OutputDataConfig={
            "S3OutputPath": f"s3://{os.environ['TRAIN_OUT_BUCKET']}/{os.environ['TRAIN_OUT_PREFIX']}/",
        },
        StoppingCondition={"MaxRuntimeInSeconds": 3600},
    )

    return {
        "statusCode": 200,
        "body": f"Training job {jobName} started successfully.",
        "job_response": res,
    }
