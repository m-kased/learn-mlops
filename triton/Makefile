# ============================================================================
#  NVIDIA Triton on GKE (learn-mlops) — make -C triton <target>
#  Needs: PVC ml-data + checkpoint from labs (make -C labs workloads)
# ============================================================================

.DEFAULT_GOAL := help

VERSION ?= 1
TEST_VERSION ?=
MODEL ?= mnist
FORMAT ?= both
INSTANCES ?= 2

.PHONY: help
help: ## Show Triton targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ---- Export (scales Triton down first — RWO PVC) ---------------------------
.PHONY: export
export: ## Export checkpoint to Triton repo (VERSION= / FORMAT= / INSTANCES=)
	@# ml-data is ReadWriteOnce: Triton and export Job cannot mount together.
	-kubectl scale deploy/triton --replicas=0 2>/dev/null || true
	-kubectl wait --for=delete pod -l app=triton --timeout=120s 2>/dev/null || true
	kubectl create configmap triton-export-code --from-file=export_model.py \
		--dry-run=client -o yaml | kubectl apply -f -
	kubectl delete job triton-export --ignore-not-found
	sed -e 's/MODEL_VERSION_PLACEHOLDER/$(VERSION)/' \
	    -e 's/FORMAT_PLACEHOLDER/$(FORMAT)/' \
	    -e 's/INSTANCES_PLACEHOLDER/$(INSTANCES)/' export-job.yaml | kubectl apply -f -
	@echo "Waiting for export Job (version=$(VERSION))..."
	kubectl wait --for=condition=complete job/triton-export --timeout=300s
	kubectl logs job/triton-export
	@echo "Export done. Bring Triton back: make deploy"

# ---- Deploy + test ---------------------------------------------------------
.PHONY: deploy
deploy: ## Deploy Triton server (GPU, reads model repo from PVC)
	kubectl apply -f triton.yaml
	kubectl rollout status deploy/triton --timeout=600s
	@echo "Ready. Test: make test   Status: make status"

.PHONY: up
up: export deploy ## Export model + deploy Triton

.PHONY: status
status: ## Triton Deployment, pods, Service
	kubectl get deploy,svc -l app=triton
	kubectl get pods -l app=triton -o wide
	@echo "Health (with port-forward): curl -s localhost:8000/v2/health/ready"

.PHONY: test
test: ## Infer (MODEL=mnist|mnist_onnx, TEST_VERSION=1|2)
	kubectl create configmap triton-test-code --from-file=test_infer.py \
		--dry-run=client -o yaml | kubectl apply -f -
	kubectl delete job triton-test --ignore-not-found
	sed -e 's/TEST_VERSION_PLACEHOLDER/$(TEST_VERSION)/' \
	    -e 's/TEST_MODEL_PLACEHOLDER/$(MODEL)/' test-job.yaml | kubectl apply -f -
	kubectl wait --for=condition=complete job/triton-test --timeout=300s
	kubectl logs job/triton-test

# ---- Model control API -----------------------------------------------------
.PHONY: models
models: ## List repo index (name stays after unload; look at state)
	@kubectl delete pod triton-api-curl --ignore-not-found >/dev/null 2>&1 || true
	@kubectl run triton-api-curl --rm -i --restart=Never --image=curlimages/curl:8.5.0 -- \
		curl -s -X POST http://triton:8000/v2/repository/index
	@echo ""

.PHONY: model-ready
model-ready: ## Is MODEL loaded? (HTTP 200 = yes, 4xx = unloaded)
	@kubectl delete pod triton-api-curl --ignore-not-found >/dev/null 2>&1 || true
	@kubectl run triton-api-curl --rm -i --restart=Never --image=curlimages/curl:8.5.0 -- \
		sh -c 'code=$$(curl -s -o /tmp/body -w "%{http_code}" http://triton:8000/v2/models/$(MODEL)/ready); echo "MODEL=$(MODEL) ready_HTTP=$$code"; cat /tmp/body; echo'

.PHONY: model-unload
model-unload: ## Unload MODEL (needs --model-control-mode=explicit)
	@kubectl delete pod triton-api-curl --ignore-not-found >/dev/null 2>&1 || true
	@kubectl run triton-api-curl --rm -i --restart=Never --image=curlimages/curl:8.5.0 -- \
		sh -c 'code=$$(curl -s -o /tmp/body -w "%{http_code}" -X POST http://triton:8000/v2/repository/models/$(MODEL)/unload); echo "unload_HTTP=$$code"; cat /tmp/body; echo'
	@echo "Next: make model-ready MODEL=$(MODEL)  (expect 4xx if unload worked)"

.PHONY: model-load
model-load: ## Load MODEL from /models disk
	@kubectl delete pod triton-api-curl --ignore-not-found >/dev/null 2>&1 || true
	@kubectl run triton-api-curl --rm -i --restart=Never --image=curlimages/curl:8.5.0 -- \
		sh -c 'code=$$(curl -s -o /tmp/body -w "%{http_code}" -X POST http://triton:8000/v2/repository/models/$(MODEL)/load); echo "load_HTTP=$$code"; cat /tmp/body; echo'
	@echo "Next: make model-ready MODEL=$(MODEL)  (expect 200)"

.PHONY: load
load: ## Concurrent load (128 req / 32 workers) — compare INSTANCES=1 vs 2
	kubectl create configmap triton-load-code --from-file=load_infer.py \
		--dry-run=client -o yaml | kubectl apply -f -
	kubectl delete job triton-load --ignore-not-found
	sed 's/LOAD_MODEL_PLACEHOLDER/$(MODEL)/' load-job.yaml | kubectl apply -f -
	kubectl wait --for=condition=complete job/triton-load --timeout=300s
	kubectl logs job/triton-load

.PHONY: instances-log
instances-log: ## Grep Triton logs for instance lines
	kubectl logs deploy/triton --tail=200 | grep -iE 'instance|mnist' || true

.PHONY: logs
logs: ## Stream Triton server logs
	kubectl logs -f deploy/triton

.PHONY: forward
forward: ## Port-forward Triton HTTP to localhost:8000 (blocking)
	@echo "Try: curl -s localhost:8000/v2/health/ready"
	kubectl port-forward svc/triton 8000:8000

.PHONY: clean
clean: ## Delete Triton Deployment, Jobs, ConfigMaps (keeps model on PVC)
	kubectl delete -f triton.yaml -f export-job.yaml -f test-job.yaml -f load-job.yaml --ignore-not-found
	kubectl delete job triton-export triton-test triton-load --ignore-not-found
	kubectl delete configmap triton-export-code triton-test-code triton-load-code --ignore-not-found
	kubectl delete pod triton-api-curl triton-metrics-curl --ignore-not-found
	@echo "Triton cleaned. PVC ml-data + model files kept. Prometheus: make -C ../labs prometheus-clean"
