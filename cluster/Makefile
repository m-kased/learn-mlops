# GKE GPU cluster — lifecycle and inspection
# Usage: make -C cluster help   (or from root: make creds, make destroy-cluster, …)

# ---- Config (override on the CLI, e.g. `make create-cluster ZONE=us-central1-b`) ----
PROJECT         ?= gke-ml-500613
CLUSTER         ?= cluster-1
ZONE            ?= us-central1-a
RELEASE_CHANNEL ?= regular
DISK_TYPE       ?= pd-standard

# ---- System pool (cluster default pool) ------------------------------------
SYSTEM_MACHINE   ?= e2-standard-4
SYSTEM_DISK_SIZE ?= 50

# ---- GPU pool --------------------------------------------------------------
MACHINE_TYPE  ?= n1-standard-4
GPU_TYPE      ?= nvidia-tesla-t4
GPU_COUNT     ?= 1
GPU_DISK_SIZE ?= 100
GPU_POOL      ?= gpu-pool
GPU_MAX_NODES ?= 2
MAX_SHARED    ?= 4    # GPU time-sharing: virtual clients per physical GPU

.DEFAULT_GOAL := help

.PHONY: help
help: ## Show cluster targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ---- Lifecycle -------------------------------------------------------------
.PHONY: create-cluster
create-cluster: ## Create system pool + time-shared autoscaling T4 GPU pool
	gcloud container --project "$(PROJECT)" clusters create "$(CLUSTER)" \
		--zone "$(ZONE)" \
		--no-enable-basic-auth \
		--release-channel "$(RELEASE_CHANNEL)" \
		--machine-type "$(SYSTEM_MACHINE)" \
		--image-type "COS_CONTAINERD" \
		--disk-type "$(DISK_TYPE)" \
		--disk-size "$(SYSTEM_DISK_SIZE)" \
		--metadata disable-legacy-endpoints=true \
		--scopes "https://www.googleapis.com/auth/cloud-platform" \
		--num-nodes 1 \
		--node-locations "$(ZONE)" \
		--enable-ip-alias \
		--enable-autoupgrade \
		--enable-autorepair \
		--enable-shielded-nodes
	$(MAKE) pool-gpu
	@echo ">>> Ready: default-pool ($(SYSTEM_MACHINE) system) + $(GPU_POOL) (T4 spot, time-sharing x$(MAX_SHARED), autoscale 0..$(GPU_MAX_NODES))."
	@echo ">>> Next: make creds   then: make up"

.PHONY: destroy-cluster
destroy-cluster: ## DELETE the cluster (stops burning credit)
	gcloud container clusters delete "$(CLUSTER)" --zone "$(ZONE)" --quiet

.PHONY: creds
creds: ## Fetch kubeconfig credentials for kubectl
	gcloud container clusters get-credentials "$(CLUSTER)" --zone "$(ZONE)" --project "$(PROJECT)"

# ---- GPU node pool ---------------------------------------------------------
.PHONY: pool-gpu
pool-gpu: ## Add the T4 GPU pool: spot, time-sharing, autoscale 0..N
	gcloud container node-pools create "$(GPU_POOL)" \
		--cluster "$(CLUSTER)" --zone "$(ZONE)" --project "$(PROJECT)" \
		--machine-type "$(MACHINE_TYPE)" \
		--accelerator "type=$(GPU_TYPE),count=$(GPU_COUNT),gpu-driver-version=default,gpu-sharing-strategy=time-sharing,max-shared-clients-per-gpu=$(MAX_SHARED)" \
		--image-type "COS_CONTAINERD" \
		--disk-type "$(DISK_TYPE)" \
		--disk-size "$(GPU_DISK_SIZE)" \
		--spot \
		--num-nodes 0 \
		--enable-autoscaling --min-nodes 0 --max-nodes "$(GPU_MAX_NODES)" \
		--node-locations "$(ZONE)"

.PHONY: pools
pools: ## List node pools and their sizes/autoscaling
	gcloud container node-pools list --cluster "$(CLUSTER)" --zone "$(ZONE)" --project "$(PROJECT)"

# ---- Inspect ---------------------------------------------------------------
.PHONY: status
status: ## Show cluster info + node count
	gcloud container clusters describe "$(CLUSTER)" --zone "$(ZONE)" \
		--format="value(status,currentMasterVersion,currentNodeCount)"

.PHONY: list
list: ## List all clusters in the project
	gcloud container clusters list --project "$(PROJECT)"

.PHONY: nodes
nodes: ## Show nodes and GPU allocatable per node
	kubectl get nodes -o wide
	@echo "--- GPU allocatable ---"
	kubectl get nodes -o=jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.allocatable.nvidia\.com/gpu}{"\n"}{end}'

.PHONY: drivers
drivers: ## Check the NVIDIA GPU device plugin (GKE-managed)
	kubectl get pods -n kube-system -l k8s-app=nvidia-gpu-device-plugin -o wide
	kubectl get pods -n kube-system | grep -i nvidia || true

# ---- Cron safety (local machine — destroys cluster if still running) ---------
CRON_SCHEDULE ?= 0 23 * * *
CRON_SCRIPT   := $(abspath destroy-if-exists.sh)
CRON_LOG      := $(HOME)/.local/log/ml-gpu/destroy-cluster.log
CRON_MARKER   := destroy-if-exists.sh

.PHONY: cron-run cron-install cron-uninstall cron-status
cron-run: ## Run destroy-if-exists once now (dry run of what cron does)
	@chmod +x "$(CRON_SCRIPT)"
	"$(CRON_SCRIPT)"

cron-install: ## Install daily cron to delete cluster if it still exists
	@chmod +x "$(CRON_SCRIPT)"
	@mkdir -p "$(dir $(CRON_LOG))"
	@(crontab -l 2>/dev/null | grep -v "$(CRON_MARKER)" || true; \
	  echo "$(CRON_SCHEDULE) PROJECT=$(PROJECT) CLUSTER=$(CLUSTER) ZONE=$(ZONE) $(CRON_SCRIPT) >>$(CRON_LOG) 2>&1") \
	  | crontab -
	@echo "Cron installed: $(CRON_SCHEDULE)"
	@echo "  Script: $(CRON_SCRIPT)"
	@echo "  Log:    $(CRON_LOG)"
	@echo "Remove with: make -C cluster cron-uninstall"

cron-uninstall: ## Remove the cluster destroy cron job
	@crontab -l 2>/dev/null | grep -v "$(CRON_MARKER)" | crontab - || true
	@echo "Cron entry removed (if it was installed)."

cron-status: ## Show cron entry + recent log lines
	@echo "--- crontab ---"
	@crontab -l 2>/dev/null | grep "$(CRON_MARKER)" || echo "(not installed)"
	@echo "--- log (last 10 lines) ---"
	@tail -n 10 "$(CRON_LOG)" 2>/dev/null || echo "(no log yet — run: make cron-run)"

# ---- GitHub Actions schedule (works when laptop is asleep) -------------------
SCHEDULE_SA_SCRIPT := $(abspath setup-schedule-sa.sh)

.PHONY: schedule-help schedule-sa
schedule-help: ## Setup steps for GitHub Actions daily cluster destroy
	@echo "GitHub Actions destroys $(CLUSTER) if it still exists (see .github/workflows/destroy-cluster.yml)."
	@echo ""
	@echo "One-time setup:"
	@echo "  1. make schedule-sa          # creates SA + key.json"
	@echo "  2. GitHub → Settings → Secrets → Actions → GCP_SA_KEY = key.json contents"
	@echo "  3. rm key.json && git push     # workflow must be on default branch"
	@echo "  4. Actions → Destroy GKE cluster → Run workflow  (test)"
	@echo ""
	@echo "Schedule: daily 20:00 UTC (edit cron in workflow file)."
	@echo "Disable: delete or disable the workflow in GitHub UI."

schedule-sa: ## Create GCP SA + key.json for GitHub secret GCP_SA_KEY
	@chmod +x "$(SCHEDULE_SA_SCRIPT)"
	"$(SCHEDULE_SA_SCRIPT)"
