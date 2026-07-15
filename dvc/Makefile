# ============================================================================
#  Data Versioning with DVC — make -C dvc <target>
#  DVC lives at repo root (.dvc/); this folder holds ops helpers.
# ============================================================================

.DEFAULT_GOAL := help

PROJECT     ?= gke-ml-500613
REGION      ?= us-central1
DVC_BUCKET  ?= learn-mlops-dvc-$(PROJECT)
DVC_REMOTE  ?= gcp-remote
DVC_URL     ?= gs://$(DVC_BUCKET)

.PHONY: help
help: ## Show DVC targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ---- GCS bucket lifecycle --------------------------------------------------
.PHONY: bucket-create buckets bucket-info bucket-files bucket-delete
bucket-create: ## Create the GCS bucket used by DVC
	gcloud storage buckets create gs://$(DVC_BUCKET) \
		--project=$(PROJECT) \
		--location=$(REGION) \
		--uniform-bucket-level-access

buckets: ## List GCS buckets in the project
	gcloud storage buckets list --project=$(PROJECT)

bucket-info: ## Describe the DVC GCS bucket
	gcloud storage buckets describe gs://$(DVC_BUCKET)

bucket-files: ## List all objects stored in the DVC bucket
	gcloud storage ls --recursive gs://$(DVC_BUCKET)

bucket-delete: ## Delete DVC bucket and all objects (CONFIRM=yes)
	@test "$(CONFIRM)" = "yes" || \
		(echo "Refusing deletion. Run: make -C dvc bucket-delete CONFIRM=yes"; exit 1)
	gcloud storage rm --recursive gs://$(DVC_BUCKET)

# ---- DVC remote + data operations (run from repo root) ---------------------
.PHONY: remote-add remotes status push pull
remote-add: ## Configure the GCS bucket as the default DVC remote
	cd .. && dvc remote add --force --default $(DVC_REMOTE) $(DVC_URL)

remotes: ## Show configured DVC remotes
	cd .. && dvc remote list

status: ## Show changed DVC-tracked data
	cd .. && dvc status

push: ## Upload tracked data to the GCS remote
	cd .. && dvc push

pull: ## Download tracked data from the GCS remote
	cd .. && dvc pull
