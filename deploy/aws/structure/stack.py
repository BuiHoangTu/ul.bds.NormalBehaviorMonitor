from aws_cdk import (
    Stack,
    SymlinkFollowMode,
    aws_iam as iam,
    aws_s3 as s3,
    aws_ecr_assets as ecr_assets,
    aws_lambda as _lambda,
    aws_s3_notifications as s3n,
    Duration,
    aws_secretsmanager as secretsmanager,
    aws_sqs as sqs,
    aws_lambda_event_sources as lambda_event_sources,
    RemovalPolicy
)
from constructs import Construct


SOURCE_CODE_PREFIX = "source-code"
SOURCE_CODE_HASH_PREFIX = "source-code-hash"
DATA_PREFIX = "source-code/data"
PREPROCESS_PREFIX = "source-code/preprocess"
PREPROCESS_OUTPUT_PREFIX = "preprocessed"
TRAIN_OUTPUT_PREFIX = "trained"


class NBM_Stack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs):
        super().__init__(scope, construct_id, **kwargs)

        ## S3 Bucket
        self.bucket = s3.Bucket(
            self,
            "NormalBehaviorMonitoringBucket",
            bucket_name="normal-behaviour-monitoring",
            encryption=s3.BucketEncryption.S3_MANAGED,
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
        )

        ## GitHub
        self._addGithubAction()

        # Training img
        self.training_role, self.training_docker = self._addTrainingImage()
        # Preprocessing then init train
        self._addPreprocess()

    def _addGithubAction(self):
        # GitHub OIDC Provider
        oidc_provider = iam.OpenIdConnectProvider(
            self,
            "GitHubActionsOidcProvider",
            url="https://token.actions.githubusercontent.com",
            client_ids=["sts.amazonaws.com"],
            thumbprints=["6938fd4d98bab03faadb97b34396831e3780aea1"],
        )
        # GitHub Actions IAM Role
        github_role = iam.Role(
            self,
            "NBM_GitHubActionsRole",
            role_name="GitHubActionsRole",
            assumed_by=iam.FederatedPrincipal(
                federated=oidc_provider.open_id_connect_provider_arn,
                conditions={
                    "StringLike": {
                        "token.actions.githubusercontent.com:sub": "repo:BuiHoangTu/ul.bds.NormalBehaviorMonitor:*"
                    },
                    "StringEquals": {
                        "token.actions.githubusercontent.com:aud": "sts.amazonaws.com"
                    },
                },
                assume_role_action="sts:AssumeRoleWithWebIdentity",
            ),  # type: ignore
        )
        github_role.add_to_policy(  # Deny all AssumeRole
            iam.PolicyStatement(
                effect=iam.Effect.DENY, actions=["sts:AssumeRole"], resources=["*"]
            )
        )
        github_role.add_to_policy(  # Allow GitHub Actions to upload to S3
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=["s3:PutObject", "s3:PutObjectAcl", "s3:ListBucket"],
                resources=[
                    self.bucket.bucket_arn,
                    f"{self.bucket.bucket_arn}/{SOURCE_CODE_PREFIX}/*",
                ],
            )
        )

    def _addTrainingImage(self):
        # Training image
        training_docker = ecr_assets.DockerImageAsset(
            self,
            "NBM_TrainingImage",
            asset_name="training-nbm",
            directory="wrapper/train/src",
            follow_symlinks=SymlinkFollowMode.ALWAYS,
        )
        # Training role
        training_role = iam.Role(
            self,
            "NBM_TrainingRole",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),  # type: ignore
        )
        training_docker.repository.grant_pull(training_role)
        training_role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=["s3:ListBucket"],
                resources=[self.bucket.bucket_arn],
            )
        )
        training_role.add_to_policy(  # Allow S3 read access to preprocessed
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=["s3:GetObject"],
                resources=[
                    f"{self.bucket.bucket_arn}/{SOURCE_CODE_PREFIX}/*",
                    f"{self.bucket.bucket_arn}/{PREPROCESS_OUTPUT_PREFIX}/*",
                ],
            )
        )
        training_role.add_to_policy(  # Allow S3 write access to trained
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=["s3:PutObject"],
                resources=[f"{self.bucket.bucket_arn}/{TRAIN_OUTPUT_PREFIX}/*"],
            )
        )
        training_role.add_to_policy(  # Logs
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=[
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents",
                ],
                resources=["*"],
            )
        )

        return training_role, training_docker

    def _addPreprocess(self):
        # SQS queue for filtering S3 events
        event_queue = sqs.Queue(
            self,
            "S3SourceCodeUpdatedQueue",
            visibility_timeout=Duration.seconds(300),
            receive_message_wait_time=Duration.seconds(10),
        )
        self.bucket.add_event_notification(
            s3.EventType.OBJECT_CREATED,
            s3n.SqsDestination(event_queue),  # type: ignore
            s3.NotificationKeyFilter(prefix=SOURCE_CODE_PREFIX),
        )
        self.bucket.add_event_notification(
            s3.EventType.OBJECT_REMOVED,
            s3n.SqsDestination(event_queue),  # type: ignore
            s3.NotificationKeyFilter(prefix=SOURCE_CODE_PREFIX),
        )

        processing_role, processing_docker, keys = self.__addPreprocessImage()

        # Lambda function to trigger SageMaker processing job
        processing_trigger = _lambda.Function(
            self,
            "NBM_PreprocessingTrigger",
            runtime=_lambda.Runtime.PYTHON_3_12,
            handler="trigger.lambda_handler",
            code=_lambda.Code.from_asset("lambda/preprocessing_trigger"),
            environment={
                # env of preprocess
                "PROCESSING_ROLE_ARN": processing_role.role_arn,
                "IMAGE_URI": processing_docker.image_uri,
                "IN_BUCKET": self.bucket.bucket_name,
                "IN_PREFIX": SOURCE_CODE_PREFIX,
                "IN_HASH_BUCKET": self.bucket.bucket_name,
                "IN_HASH_PREFIX": SOURCE_CODE_HASH_PREFIX,
                "OUT_BUCKET": self.bucket.bucket_name,
                "OUT_PREFIX": PREPROCESS_OUTPUT_PREFIX,
                "KEY_REGION": self.region,
                "KEY_NAME": keys.secret_name,
                "SQS_QUEUE_URL": event_queue.queue_url,
                # env of training
                "TRAIN_ROLE_ARN": self.training_role.role_arn,
                "TRAIN_IMAGE_URI": self.training_docker.image_uri,
                "SOURCE_CODE_BUCKET": self.bucket.bucket_name,
                "SOURCE_CODE_PREFIX": SOURCE_CODE_PREFIX,
                "PROCESSED_BUCKET": self.bucket.bucket_name,
                "PROCESSED_PREFIX": PREPROCESS_OUTPUT_PREFIX,
                "TRAIN_OUT_BUCKET": self.bucket.bucket_name,
                "TRAIN_OUT_PREFIX": TRAIN_OUTPUT_PREFIX,
                "REGION": self.region,
            },
            reserved_concurrent_executions=1,
            timeout=Duration.minutes(5),
        )
        processing_trigger.add_to_role_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=["sagemaker:CreateProcessingJob"],
                resources=["*"],
            )
        )
        processing_trigger.add_to_role_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=["iam:PassRole"],
                resources=[processing_role.role_arn],
            )
        )
        # Add lambda to the SQS queue
        processing_trigger.add_event_source(
            lambda_event_sources.SqsEventSource(
                event_queue,
                batch_size=10000,  # as large as possible to avoid premature processing
                max_batching_window=Duration.seconds(30),
                max_concurrency=2,  # can't set to one, limit is set at lambda level
            )
        )
        # allow lambda to read number of messages left in the queue
        event_queue.grant(processing_trigger, "sqs:GetQueueAttributes")

    def __addPreprocessImage(self):
        # Preprocessing image
        processing_docker = ecr_assets.DockerImageAsset(
            self,
            "PreprocessingNBM",
            asset_name="preprocessing-nbm",
            directory="wrapper/processing/src",
            follow_symlinks=SymlinkFollowMode.ALWAYS,
        )
        # Preprocessing role
        processing_role = iam.Role(
            self,
            "NBM_PreprocessingRole",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),  # type: ignore
        )
        processing_docker.repository.grant_pull(processing_role)
        processing_role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=["s3:ListBucket"],
                resources=[self.bucket.bucket_arn],
            )
        )
        processing_role.add_to_policy(  # Allow S3 read access to source code and hashes
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=["s3:GetObject"],
                resources=[
                    f"{self.bucket.bucket_arn}/{SOURCE_CODE_PREFIX}/*",
                    f"{self.bucket.bucket_arn}/{SOURCE_CODE_HASH_PREFIX}/*",
                ],
            )
        )
        processing_role.add_to_policy(  # Allow S3 write access to preprocessed and hash
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=["s3:PutObject"],
                resources=[
                    f"{self.bucket.bucket_arn}/{PREPROCESS_OUTPUT_PREFIX}/*",
                    f"{self.bucket.bucket_arn}/{SOURCE_CODE_HASH_PREFIX}/data.hash",
                    f"{self.bucket.bucket_arn}/{SOURCE_CODE_HASH_PREFIX}/preprocess.hash",
                ],
            )
        )
        processing_role.add_to_policy(  # Allow CloudWatch access
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=[
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents",
                ],
                resources=["*"],
            )
        )
        processing_role.add_to_policy(  # Allow create training job
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=["sagemaker:CreateTrainingJob"],
                resources=["*"],
            )
        )

        # Data keys
        nbm_keys = secretsmanager.Secret(
            self,
            "NBM_DataKeys",
            secret_name="nbm-keys",
            description="Keys used for decrypting S3 parquet files in NBM",
        )
        nbm_keys.grant_read(processing_role)

        return processing_role, processing_docker, nbm_keys
