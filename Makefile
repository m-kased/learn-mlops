# ============================================================================
#  learn-mlops — active work: Phase 4 Drift / Monitoring
# ============================================================================

.DEFAULT_GOAL := help

CLUSTER_TARGETS := create-cluster destroy-cluster creds pool-gpu pools list nodes drivers \
	cron-run cron-install cron-uninstall cron-status schedule-help schedule-sa

.PHONY: help cluster-help labs-help triton-help kserve-help ct-help dvc-help sm-help \
	cluster-status $(CLUSTER_TARGETS)

help: ## Show module shortcuts
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "  Cluster:    make cluster-help"
	@echo "  Labs:       make labs-help     →  make -C labs <target>"
	@echo "  Triton:     make triton-help   →  make -C triton <target>"
	@echo "  KServe:     make kserve-help   →  make -C kserve <target>"
	@echo "  CT:         make ct-help       →  make -C ct <target>"
	@echo "  DVC:        make dvc-help      →  make -C dvc <target>"
	@echo "  SageMaker:  make sm-help       →  make -C sm <target>"

cluster-help: ## Show all cluster targets
	@$(MAKE) -C cluster help --no-print-directory

labs-help: ## Show all lab targets
	@$(MAKE) -C labs help --no-print-directory

triton-help: ## Show all Triton targets
	@$(MAKE) -C triton help --no-print-directory

kserve-help: ## Show all KServe targets
	@$(MAKE) -C kserve help --no-print-directory

ct-help: ## Show all CT (Continuous Training) targets
	@$(MAKE) -C ct help --no-print-directory

dvc-help: ## Show all DVC targets
	@$(MAKE) -C dvc help --no-print-directory

sm-help: ## Show SageMaker lab targets (done)
	@$(MAKE) -C sm help --no-print-directory

$(CLUSTER_TARGETS):
	$(MAKE) -C cluster $@

cluster-status: ## GKE cluster info
	$(MAKE) -C cluster status
