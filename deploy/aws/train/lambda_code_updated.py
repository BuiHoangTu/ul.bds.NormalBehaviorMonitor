import boto3
import datetime
import os


def lambda_handler(event, context):
    sm = boto3.client("sagemaker")
    job_name = f"training-job-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"

    response = sm.create_training_job(
        TrainingJobName=job_name,
        AlgorithmSpecification={
            "TrainingImage": "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:2.0.1-cpu-py310-ubuntu20.04",
            "TrainingInputMode": "File",
        },
        RoleArn="arn:aws:iam::<YOUR_ACCOUNT_ID>:role/<SAGEMAKER_EXEC_ROLE>",
        InputDataConfig=[
            {
                "ChannelName": "training",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": "s3://your-bucket/input-data/",
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
                "ContentType": "text/csv",
                "InputMode": "File",
            },
        ],
        OutputDataConfig={"S3OutputPath": "s3://your-bucket/output/"},
        ResourceConfig={
            "InstanceType": "ml.m5.large",
            "InstanceCount": 1,
            "VolumeSizeInGB": 30,
        },
        StoppingCondition={"MaxRuntimeInSeconds": 3600},
        HyperParameters={"epochs": "10", "batch_size": "32", "lr": "0.001"},
    )

    return {"statusCode": 200, "body": f"Started training job: {job_name}"}
