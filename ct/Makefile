# ============================================================================
#  Continuous Training (CT) — make -C ct <target>
#  train → evaluate → gate → triton export → kserve deploy → smoke test
# ============================================================================

.DEFAULT_GOAL := help

RUN_NAME ?= run-a
MIN_ACCURACY ?= 0.98
VERSION ?= 1
FORMAT ?= both
INSTANCES ?= 2

.PHONY: help
help: ## Show CT targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ---- Single clean training run (checkpoint on PVC) -------------------------
.PHONY: train
train: ## Train one MNIST run for the pipeline (RUN_NAME=run-a)
	kubectl apply -f ../labs/pvc.yaml
	kubectl create configmap mnist-code --from-file=train.py=../labs/train.py \
		--dry-run=client -o yaml | kubectl apply -f -
	kubectl delete job mnist-train --ignore-not-found
	kubectl apply -f train-job.yaml
	@echo "Waiting for training Job (GPU pull + train)..."
	kubectl wait --for=condition=complete job/mnist-train --timeout=900s
	kubectl logs job/mnist-train --tail=20

# ---- Evaluate trained checkpoint (accuracy → metrics.json on PVC) ----------
.PHONY: evaluate
evaluate: ## Evaluate checkpoint on MNIST test set (RUN_NAME=run-a)
	kubectl create configmap mnist-eval-code --from-file=evaluate.py \
		--dry-run=client -o yaml | kubectl apply -f -
	kubectl delete job mnist-eval --ignore-not-found
	kubectl apply -f eval-job.yaml
	@echo "Waiting for eval Job..."
	kubectl wait --for=condition=complete job/mnist-eval --timeout=300s
	kubectl logs job/mnist-eval

# ---- Quality gate ----------------------------------------------------------
.PHONY: gate
gate: ## Fail if accuracy < MIN_ACCURACY (default 0.98)
	kubectl create configmap mnist-gate-code --from-file=gate.py \
		--dry-run=client -o yaml | kubectl apply -f -
	kubectl delete job mnist-gate --ignore-not-found
	sed 's/MIN_ACCURACY_PLACEHOLDER/$(MIN_ACCURACY)/' gate-job.yaml | kubectl apply -f -
	@echo "Waiting for quality gate (min_accuracy=$(MIN_ACCURACY))..."
	kubectl wait --for=condition=complete job/mnist-gate --timeout=120s
	kubectl logs job/mnist-gate

# ---- Full pipeline ---------------------------------------------------------
.PHONY: pipeline
pipeline: ## train → evaluate → gate → export → deploy → smoke test
	@echo "=== CT pipeline: train/eval/gate/export/deploy/test ==="
	@echo "Freeing ml-data RWO PVC from serving pods if present..."
	-kubectl delete -f ../kserve/mnist-triton.yaml --ignore-not-found
	-kubectl wait --for=delete pod -l serving.kserve.io/inferenceservice=mnist-triton --timeout=180s 2>/dev/null || true
	-$(MAKE) -C ../triton clean --no-print-directory
	$(MAKE) train RUN_NAME=$(RUN_NAME)
	$(MAKE) evaluate RUN_NAME=$(RUN_NAME)
	$(MAKE) gate RUN_NAME=$(RUN_NAME) MIN_ACCURACY=$(MIN_ACCURACY)
	$(MAKE) -C ../triton export VERSION=$(VERSION) FORMAT=$(FORMAT) INSTANCES=$(INSTANCES) --no-print-directory
	$(MAKE) -C ../kserve deploy-mnist --no-print-directory
	$(MAKE) -C ../kserve test-mnist --no-print-directory
	@echo "=== CT pipeline passed ==="

.PHONY: pipeline-clean
pipeline-clean: ## Delete CT Jobs/ConfigMaps (keeps PVC, KServe, model files)
	kubectl delete -f train-job.yaml -f eval-job.yaml -f gate-job.yaml --ignore-not-found
	kubectl delete job mnist-train mnist-eval mnist-gate --ignore-not-found
	kubectl delete configmap mnist-code mnist-eval-code mnist-gate-code --ignore-not-found
