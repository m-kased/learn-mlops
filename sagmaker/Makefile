# ============================================================================
#  SageMaker lab (done) — make -C sm <target>
#  Train → Evaluate → CreateModel → Batch Transform pipeline
# ============================================================================

.DEFAULT_GOAL := help

AWS_PROFILE ?= sagemaker
AWS_REGION ?= us-east-1
S3_BUCKET ?= mobbucketsagemaker-jx-2025
SAGEMAKER_ROLE ?= arn:aws:iam::793799268730:role/AmazonSageMaker-ExecutionRole-20230120T164209-jx-2025
SM_INSTANCE ?= ml.m5.large

.PHONY: help
help: ## Show SageMaker targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# ---- S3 / identity ---------------------------------------------------------
.PHONY: whoami upload ls
whoami: ## Confirm AWS profile / account / region
	aws sts get-caller-identity --profile $(AWS_PROFILE) --region $(AWS_REGION)
	@echo "bucket=s3://$(S3_BUCKET) role=$(SAGEMAKER_ROLE)"

upload: ## Upload train/test CSV into S3 prefixes
	aws s3 cp train.csv s3://$(S3_BUCKET)/data/train/train.csv \
		--profile $(AWS_PROFILE) --region $(AWS_REGION)
	aws s3 cp test.csv s3://$(S3_BUCKET)/data/test/test.csv \
		--profile $(AWS_PROFILE) --region $(AWS_REGION)
	@printf '' | aws s3 cp - s3://$(S3_BUCKET)/models/.keep \
		--profile $(AWS_PROFILE) --region $(AWS_REGION)
	@printf '' | aws s3 cp - s3://$(S3_BUCKET)/output/.keep \
		--profile $(AWS_PROFILE) --region $(AWS_REGION)
	@$(MAKE) ls --no-print-directory

ls: ## List objects in the SageMaker bucket
	aws s3 ls s3://$(S3_BUCKET)/ --recursive --profile $(AWS_PROFILE) --region $(AWS_REGION)

# ---- Batch Transform (one-shot reference) ----------------------------------
.PHONY: deps batch batch-jobs batch-results cost-check
deps: ## Sync project deps with uv (pyproject.toml in this folder)
	@command -v uv >/dev/null || (echo "uv not found. Install: https://docs.astral.sh/uv/"; exit 1)
	uv sync

batch: ## Batch Transform: package model, score S3 CSV, write predictions
	AWS_PROFILE=$(AWS_PROFILE) AWS_REGION=$(AWS_REGION) \
		S3_BUCKET=$(S3_BUCKET) SAGEMAKER_ROLE=$(SAGEMAKER_ROLE) \
		SM_INSTANCE=$(SM_INSTANCE) \
		uv run python sm_batch.py

batch-jobs: ## List recent Batch Transform jobs
	aws sagemaker list-transform-jobs --max-results 10 --sort-by CreationTime \
		--sort-order Descending --profile $(AWS_PROFILE) --region $(AWS_REGION) \
		--query 'TransformJobSummaries[].{Name:TransformJobName,Status:TransformJobStatus,Created:CreationTime}' \
		--output table

batch-results: ## Show newest batch prediction objects under output/batch/
	aws s3 ls s3://$(S3_BUCKET)/output/batch/ --recursive \
		--profile $(AWS_PROFILE) --region $(AWS_REGION)
	@echo "--- preview newest .out (if any) ---"
	@key=$$(aws s3api list-objects-v2 --bucket $(S3_BUCKET) --prefix output/batch/ \
		--profile $(AWS_PROFILE) --region $(AWS_REGION) \
		--query 'reverse(sort_by(Contents,&LastModified))[0].Key' --output text); \
	if [ "$$key" != "None" ] && [ -n "$$key" ]; then \
		echo "s3://$(S3_BUCKET)/$$key"; \
		aws s3 cp "s3://$(S3_BUCKET)/$$key" - --profile $(AWS_PROFILE) --region $(AWS_REGION) | head -20; \
	else \
		echo "no batch outputs yet"; \
	fi

cost-check: ## Quick check for endpoints / notebooks / in-progress jobs
	@echo "=== endpoints ==="; aws sagemaker list-endpoints --profile $(AWS_PROFILE) --region $(AWS_REGION) --output table
	@echo "=== notebooks ==="; aws sagemaker list-notebook-instances --profile $(AWS_PROFILE) --region $(AWS_REGION) --output table
	@echo "=== transform InProgress ==="; aws sagemaker list-transform-jobs --status-equals InProgress --profile $(AWS_PROFILE) --region $(AWS_REGION) --output table
	@echo "=== training InProgress ==="; aws sagemaker list-training-jobs --status-equals InProgress --profile $(AWS_PROFILE) --region $(AWS_REGION) --output table
	@echo "=== pipeline executions ==="; aws sagemaker list-pipeline-executions --pipeline-name mob-train-batch-transform --max-results 5 --profile $(AWS_PROFILE) --region $(AWS_REGION) --output table 2>/dev/null || true

# ---- Pipeline: Train → Evaluate → CreateModel → Batch Transform ------------
.PHONY: pipeline-prep pipeline-upsert pipeline-start pipeline-status
pipeline-prep: ## Upload features-only batch CSV to S3
	AWS_PROFILE=$(AWS_PROFILE) AWS_REGION=$(AWS_REGION) \
		S3_BUCKET=$(S3_BUCKET) SAGEMAKER_ROLE=$(SAGEMAKER_ROLE) \
		uv run python sm_pipeline.py prep

pipeline-upsert: ## Create/update pipeline definition in SageMaker
	AWS_PROFILE=$(AWS_PROFILE) AWS_REGION=$(AWS_REGION) \
		S3_BUCKET=$(S3_BUCKET) SAGEMAKER_ROLE=$(SAGEMAKER_ROLE) \
		SM_INSTANCE=$(SM_INSTANCE) \
		uv run python sm_pipeline.py upsert

pipeline-start: ## Upsert + start pipeline execution
	AWS_PROFILE=$(AWS_PROFILE) AWS_REGION=$(AWS_REGION) \
		S3_BUCKET=$(S3_BUCKET) SAGEMAKER_ROLE=$(SAGEMAKER_ROLE) \
		SM_INSTANCE=$(SM_INSTANCE) \
		uv run python sm_pipeline.py start

pipeline-status: ## Show recent pipeline executions + latest step statuses
	AWS_PROFILE=$(AWS_PROFILE) AWS_REGION=$(AWS_REGION) \
		S3_BUCKET=$(S3_BUCKET) SAGEMAKER_ROLE=$(SAGEMAKER_ROLE) \
		uv run python sm_pipeline.py status
