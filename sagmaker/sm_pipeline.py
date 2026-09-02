"""SageMaker Pipeline: Train -> Evaluate -> CreateModel -> Batch Transform.

WHERE THIS FILE RUNS: your laptop only.

It does NOT train or score data. It:
  1) builds a pipeline definition (JSON DAG)
  2) upserts that definition into the SageMaker Pipelines service
  3) starts an execution (optional)
  4) can poll status

SageMaker then runs each step on short-lived AWS compute and writes to S3.
Think: this file is like "kubectl apply" for an ML workflow — not the worker.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import boto3
import pandas as pd
from sagemaker.core import image_uris
from sagemaker.core.processing import ScriptProcessor
from sagemaker.core.shapes import (
    ProcessingInput,
    ProcessingOutput,
    ProcessingS3Input,
    ProcessingS3Output,
)
from sagemaker.core.transformer import Transformer
from sagemaker.core.workflow.parameters import ParameterString
from sagemaker.core.workflow.pipeline_context import PipelineSession
from sagemaker.mlops.workflow.model_step import ModelStep
from sagemaker.mlops.workflow.pipeline import Pipeline
from sagemaker.mlops.workflow.steps import ProcessingStep, TrainingStep, TransformStep
from sagemaker.serve.model_builder import ModelBuilder
from sagemaker.train import ModelTrainer
from sagemaker.train.configs import Compute, InputData, OutputDataConfig, SourceCode

# Columns the model uses as inputs (no label). Batch CSV must match this order.
FEATURE_COLUMNS = ["battery_power", "ram", "px_height", "px_width"]
# Fixed name in your AWS account; upsert updates the same pipeline.
PIPELINE_NAME = "mob-train-batch-transform"


def required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"Missing env var {name}")
    return value


def session_bundle():
    """Wire AWS identity for the SDK (profile + region + role + bucket)."""
    profile = required_env("AWS_PROFILE")
    region = required_env("AWS_REGION")
    bucket = required_env("S3_BUCKET")
    role = required_env("SAGEMAKER_ROLE")
    instance_type = os.environ.get("SM_INSTANCE", "ml.m5.large")

    # boto3 uses your local ~/.aws profile (sagemaker).
    boto_session = boto3.Session(profile_name=profile, region_name=region)

    # PipelineSession is special: when you call .train() / .transform() inside
    # build_pipeline(), it does NOT start jobs immediately. It records step
    # arguments into the pipeline definition instead.
    pipeline_session = PipelineSession(
        boto_session=boto_session,
        default_bucket=bucket,
    )
    return {
        "profile": profile,
        "region": region,
        "bucket": bucket,
        "role": role,
        "instance_type": instance_type,
        "boto_session": boto_session,
        "pipeline_session": pipeline_session,
    }


def prepare_batch_input(cfg: dict) -> str:
    """Build and upload the file that BatchTransform will score.

    Transform expects features only (no price_range label) and no header,
    because sm_src/inference.py reads CSV with header=None.
    """
    root = Path(__file__).resolve().parent
    test_csv = root / "test.csv"
    if not test_csv.is_file():
        raise SystemExit("test.csv missing")

    features = pd.read_csv(test_csv)[FEATURE_COLUMNS]
    local = root / "batch_input.csv"
    features.to_csv(local, header=False, index=False)

    key = "data/batch/input/input.csv"
    s3 = cfg["boto_session"].client("s3")
    s3.upload_file(str(local), cfg["bucket"], key)
    # Prefix URI (folder). TransformStep reads objects under this prefix.
    uri = f"s3://{cfg['bucket']}/data/batch/input"
    print(f"batch input -> {uri}/input.csv ({len(features)} rows)", flush=True)
    return uri


def _proc_input(name: str, s3_uri, local_path: str) -> ProcessingInput:
    """S3 -> container path mount for a Processing Job."""
    return ProcessingInput(
        input_name=name,
        s3_input=ProcessingS3Input(
            s3_uri=s3_uri,
            local_path=local_path,
            s3_data_type="S3Prefix",
            s3_input_mode="File",
        ),
    )


def _proc_output(name: str, local_path: str, s3_uri: str) -> ProcessingOutput:
    """Container path -> S3 upload when the Processing Job finishes."""
    return ProcessingOutput(
        output_name=name,
        s3_output=ProcessingS3Output(
            s3_uri=s3_uri,
            local_path=local_path,
            s3_upload_mode="EndOfJob",
        ),
    )


def build_pipeline(cfg: dict) -> Pipeline:
    """Define the DAG: Train -> Evaluate -> CreateModel -> BatchTransform.

    Still on your laptop. Nothing expensive runs here yet — we only describe
    what SageMaker should do later when an execution starts.
    """
    role = cfg["role"]
    bucket = cfg["bucket"]
    region = cfg["region"]
    instance_type = cfg["instance_type"]
    pipeline_session = cfg["pipeline_session"]
    # Worker scripts that will be uploaded and run INSIDE AWS containers.
    src_dir = str(Path(__file__).resolve().parent / "sm_src")
    evaluate_script = str(Path(src_dir) / "evaluate.py")

    # ---- Pipeline parameters (can override at start time) ------------------
    # Like workflow inputs: which S3 paths to use for this execution.
    train_data = ParameterString(
        name="TrainData",
        default_value=f"s3://{bucket}/data/train",
    )
    test_data = ParameterString(
        name="TestData",
        default_value=f"s3://{bucket}/data/test",
    )
    batch_data = ParameterString(
        name="BatchData",
        default_value=f"s3://{bucket}/data/batch/input",
    )

    # Official AWS SKLearn container image for this region/version.
    sklearn_image = image_uris.retrieve(
        framework="sklearn",
        region=region,
        version="1.2-1",
        py_version="py3",
        instance_type=instance_type,
        image_scope="training",
    )

    # ---- Step 1: Training --------------------------------------------------
    # ModelTrainer = "how to train" (image, script, instance, output path).
    # TrainingStep wraps it so the Pipeline can schedule it.
    trainer = ModelTrainer(
        training_image=sklearn_image,
        # Upload sm_src/ and run train.py inside the container.
        source_code=SourceCode(source_dir=src_dir, entry_script="train.py"),
        compute=Compute(instance_type=instance_type, instance_count=1),
        # Where SageMaker writes model.tar.gz after training.
        output_data_config=OutputDataConfig(
            s3_output_path=f"s3://{bucket}/output/pipeline/train"
        ),
        # Passed to train.py as CLI flags: --n-estimators 50 --max-depth 8
        hyperparameters={"n-estimators": 50, "max-depth": 8},
        role=role,
        sagemaker_session=pipeline_session,
        base_job_name="mob-pipeline-train",
    )
    # Because pipeline_session is a PipelineSession, this does NOT start a job.
    # It returns step arguments for the DAG.
    train_args = trainer.train(
        input_data_config=[
            # channel_name="train" becomes env SM_CHANNEL_TRAIN in the container.
            InputData(channel_name="train", data_source=train_data),
        ]
    )
    step_train = TrainingStep(name="Train", step_args=train_args)

    # ---- Step 2: Evaluate (Processing Job) ---------------------------------
    # Short-lived machine: mount model + test CSV, run evaluate.py, write
    # evaluation.json to S3. Exit non-zero if accuracy < threshold (= soft gate).
    # Same idea as ct/gate.py, but as a managed AWS job instead of a k8s Job.
    processor = ScriptProcessor(
        image_uri=sklearn_image,
        command=["python3"],
        role=role,
        instance_count=1,
        instance_type=instance_type,
        sagemaker_session=pipeline_session,
        base_job_name="mob-pipeline-eval",
    )
    eval_args = processor.run(
        code=evaluate_script,
        inputs=[
            _proc_input(
                "model",
                step_train.properties.ModelArtifacts.S3ModelArtifacts,
                "/opt/ml/processing/model",
            ),
            _proc_input("test", test_data, "/opt/ml/processing/test"),
        ],
        outputs=[
            _proc_output(
                "evaluation",
                "/opt/ml/processing/evaluation",
                f"s3://{bucket}/output/pipeline/evaluation",
            ),
        ],
        arguments=["--min-accuracy", "0.55"],
    )
    step_eval = ProcessingStep(
        name="Evaluate",
        step_args=eval_args,
        depends_on=[step_train],
    )

    # ---- Step 3: CreateModel -----------------------------------------------
    # Turns training output (model.tar.gz) into a SageMaker Model resource.
    # Batch Transform (and endpoints) need a Model name, not a raw S3 path.
    # `.properties.ModelArtifacts.S3ModelArtifacts` is a pipeline reference:
    # "use whatever S3 URI the Train step produced" — resolved at runtime.
    #
    # train.py already packs code/inference.py into model.tar.gz. We still
    # pass source_code here so ModelBuilder sets SAGEMAKER_PROGRAM=inference.py.
    # Wait for Evaluate so a failed quality gate stops packaging/serving.
    model_builder = ModelBuilder(
        image_uri=sklearn_image,
        s3_model_data_url=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        role_arn=role,
        sagemaker_session=pipeline_session,
        source_code=SourceCode(source_dir=src_dir, entry_script="inference.py"),
    )
    step_create_model = ModelStep(
        name="CreateModel",
        step_args=model_builder.build(),
        depends_on=[step_eval],
    )

    # ---- Step 4: Batch Transform -------------------------------------------
    # Score a whole S3 prefix. Machine starts, reads input, writes predictions,
    # then stops. No 24/7 endpoint.
    transformer = Transformer(
        # Again a runtime reference: model created by the previous step.
        model_name=step_create_model.properties.ModelName,
        instance_count=1,
        instance_type=instance_type,
        strategy="MultiRecord",  # pack multiple CSV lines per request
        assemble_with="Line",  # write one prediction line per input line
        accept="text/csv",
        output_path=f"s3://{bucket}/output/pipeline/transform",
        sagemaker_session=pipeline_session,
    )
    transform_args = transformer.transform(
        data=batch_data,
        content_type="text/csv",
        split_type="Line",  # split file by newline before inference
    )
    step_transform = TransformStep(
        name="BatchTransform",
        step_args=transform_args,
        # Explicit edge in the DAG: Transform waits for CreateModel.
        depends_on=[step_create_model],
    )

    # Order in `steps` plus depends_on / property links defines the DAG.
    return Pipeline(
        name=PIPELINE_NAME,
        parameters=[train_data, test_data, batch_data],
        steps=[step_train, step_eval, step_create_model, step_transform],
        sagemaker_session=pipeline_session,
    )


def cmd_upsert(cfg: dict) -> None:
    """Register/update the pipeline definition in AWS (no execution yet)."""
    prepare_batch_input(cfg)
    pipeline = build_pipeline(cfg)
    # Creates the pipeline if missing, updates definition if it exists.
    pipeline.upsert(role_arn=cfg["role"])
    print(f"upserted pipeline={PIPELINE_NAME}", flush=True)


def cmd_start(cfg: dict) -> None:
    """Upsert definition, then ask SageMaker to run one execution."""
    prepare_batch_input(cfg)
    pipeline = build_pipeline(cfg)
    pipeline.upsert(role_arn=cfg["role"])
    # After this returns, AWS owns the run. Laptop can disconnect.
    execution = pipeline.start(
        parameters={
            "TrainData": f"s3://{cfg['bucket']}/data/train",
            "TestData": f"s3://{cfg['bucket']}/data/test",
            "BatchData": f"s3://{cfg['bucket']}/data/batch/input",
        }
    )
    arn = getattr(execution, "arn", None) or getattr(execution, "pipeline_execution_arn", None)
    print(f"started execution={arn or execution}", flush=True)
    print("Watch: make sm-pipeline-status", flush=True)
    print(
        f"Console: SageMaker → Pipelines → {PIPELINE_NAME}",
        flush=True,
    )


def cmd_status(cfg: dict) -> None:
    """Poll recent executions and the latest run's step states (boto3)."""
    sm = cfg["boto_session"].client("sagemaker")
    executions = sm.list_pipeline_executions(
        PipelineName=PIPELINE_NAME,
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=5,
    ).get("PipelineExecutionSummaries", [])
    if not executions:
        print("no executions yet", flush=True)
        return
    for ex in executions:
        print(
            f"{ex['PipelineExecutionDisplayName']}  "
            f"{ex['PipelineExecutionStatus']}  "
            f"{ex['StartTime']}",
            flush=True,
        )
    latest = executions[0]["PipelineExecutionArn"]
    steps = sm.list_pipeline_execution_steps(PipelineExecutionArn=latest).get(
        "PipelineExecutionSteps", []
    )
    print("--- latest steps ---", flush=True)
    for step in steps:
        meta = step.get("Metadata", {})
        print(f"{step['StepName']}: {step['StepStatus']} {meta}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=["upsert", "start", "status", "prep"],
        help="prep=upload batch CSV | upsert=save DAG | start=run | status=poll",
    )
    args = parser.parse_args()
    cfg = session_bundle()

    if args.action == "prep":
        prepare_batch_input(cfg)
    elif args.action == "upsert":
        cmd_upsert(cfg)
    elif args.action == "start":
        cmd_start(cfg)
    elif args.action == "status":
        cmd_status(cfg)


if __name__ == "__main__":
    main()
