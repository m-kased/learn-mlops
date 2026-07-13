# ============================================================================
#  learn-mlops — GKE GPU / MLOps learning repo
#  Cluster · labs · triton  |  Next: KServe at repo root
# ============================================================================

.DEFAULT_GOAL := help

CLUSTER_TARGETS := create-cluster destroy-cluster creds pool-gpu pools list nodes drivers \
	cron-run cron-install cron-uninstall cron-status schedule-help schedule-sa

.PHONY: help cluster-help labs-help triton-help cluster-status $(CLUSTER_TARGETS)

help: ## Show top-level shortcuts
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "  Cluster:  make cluster-help"
	@echo "  Labs:     make labs-help     →  make -C labs <target>"
	@echo "  Triton:   make triton-help   →  make -C triton <target>"

cluster-help: ## Show all cluster targets
	@$(MAKE) -C cluster help --no-print-directory

labs-help: ## Show all lab targets
	@$(MAKE) -C labs help --no-print-directory

triton-help: ## Show all Triton targets
	@$(MAKE) -C triton help --no-print-directory

$(CLUSTER_TARGETS):
	$(MAKE) -C cluster $@

cluster-status: ## GKE cluster info
	$(MAKE) -C cluster status
