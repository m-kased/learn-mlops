# ============================================================================
#  KServe for learn-mlops (done) — make -C kserve <target>
#  Mode: Standard (no Knative)
# ============================================================================

.DEFAULT_GOAL := help

KSERVE_VERSION       ?= v0.19.0
CERT_MANAGER_VERSION ?= v1.16.2

.PHONY: help
help: ## Show KServe targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# ---- Install ---------------------------------------------------------------
.PHONY: install-cert-manager
install-cert-manager: ## Install cert-manager via Helm
	helm repo add jetstack https://charts.jetstack.io --force-update
	helm upgrade --install cert-manager jetstack/cert-manager \
		--namespace cert-manager --create-namespace \
		--version $(CERT_MANAGER_VERSION) \
		--set crds.enabled=true \
		--wait --timeout 5m
	@echo "cert-manager ready."

.PHONY: install
install: install-cert-manager ## Install KServe Standard mode + serving runtimes
	helm upgrade --install kserve-crd oci://ghcr.io/kserve/charts/kserve-crd \
		--namespace kserve --create-namespace \
		--version $(KSERVE_VERSION) --wait --timeout 5m
	helm upgrade --install kserve-resources oci://ghcr.io/kserve/charts/kserve-resources \
		--version $(KSERVE_VERSION) \
		--namespace kserve --create-namespace \
		--set kserve.controller.deploymentMode=Standard \
		--wait --timeout 5m
	helm upgrade --install kserve-runtime-configs oci://ghcr.io/kserve/charts/kserve-runtime-configs \
		--version $(KSERVE_VERSION) \
		--namespace kserve --create-namespace \
		--set kserve.servingruntime.enabled=true \
		--wait --timeout 5m
	@$(MAKE) --no-print-directory status

.PHONY: status
status: ## Show KServe / cert-manager pods + runtimes
	@echo "=== cert-manager ==="
	kubectl get pods -n cert-manager
	@echo ""
	@echo "=== kserve ==="
	kubectl get pods -n kserve
	@echo ""
	@echo "=== CRDs ==="
	kubectl get crd | grep serving.kserve.io || true
	@echo ""
	@echo "=== serving runtimes ==="
	kubectl get clusterservingruntime 2>/dev/null | head -20 || true

.PHONY: inspect
inspect: ## Show Deployments/Services owned by InferenceServices
	kubectl get inferenceservice -o wide
	kubectl get deploy,svc,pods -l serving.kserve.io/inferenceservice

# ---- Iris hello-world ------------------------------------------------------
.PHONY: deploy
deploy: ## Deploy sklearn-iris InferenceService
	kubectl apply -f sklearn-iris.yaml
	kubectl wait --for=condition=Ready inferenceservice/sklearn-iris --timeout=300s
	kubectl get inferenceservice sklearn-iris
	@echo "Ready. Test: make test"

.PHONY: test
test: ## Iris inference smoke test
	kubectl create configmap kserve-iris-input --from-file=iris-input.json \
		--dry-run=client -o yaml | kubectl apply -f -
	kubectl delete job kserve-iris-test --ignore-not-found
	kubectl apply -f iris-test-job.yaml
	kubectl wait --for=condition=complete job/kserve-iris-test --timeout=120s
	kubectl logs job/kserve-iris-test

.PHONY: logs
logs: ## Stream sklearn-iris logs
	kubectl logs -l serving.kserve.io/inferenceservice=sklearn-iris -c kserve-container -f

# ---- Transformer -----------------------------------------------------------
.PHONY: deploy-xform
deploy-xform: ## Deploy iris with custom Transformer
	kubectl create configmap iris-transformer-code --from-file=iris_transformer.py \
		--dry-run=client -o yaml | kubectl apply -f -
	kubectl apply -f iris-xform.yaml
	kubectl wait --for=condition=Ready inferenceservice/iris-xform --timeout=600s
	kubectl get deploy,svc -l serving.kserve.io/inferenceservice=iris-xform
	@echo "Ready. Test: make test-xform"

.PHONY: test-xform
test-xform: ## Custom features JSON via transformer
	kubectl create configmap kserve-xform-input --from-file=iris-xform-input.json \
		--dry-run=client -o yaml | kubectl apply -f -
	kubectl delete job kserve-xform-test --ignore-not-found
	kubectl apply -f iris-xform-test-job.yaml
	kubectl wait --for=condition=complete job/kserve-xform-test --timeout=180s
	kubectl logs job/kserve-xform-test

# ---- InferenceGraph --------------------------------------------------------
.PHONY: deploy-graph
deploy-graph: ## Iris Splitter graph 80/20
	kubectl apply -f sklearn-iris.yaml -f sklearn-iris-v2.yaml
	kubectl wait --for=condition=Ready inferenceservice/sklearn-iris --timeout=300s
	kubectl wait --for=condition=Ready inferenceservice/sklearn-iris-v2 --timeout=300s
	kubectl apply -f iris-graph.yaml
	kubectl wait --for=condition=Ready inferencegraph/iris-split --timeout=300s || true
	kubectl get inferencegraph iris-split
	@echo "Ready. Test: make test-graph"

.PHONY: test-graph
test-graph: ## 20 requests through Splitter
	kubectl delete job kserve-graph-test --ignore-not-found
	kubectl apply -f iris-graph-test-job.yaml
	kubectl wait --for=condition=complete job/kserve-graph-test --timeout=180s
	kubectl logs job/kserve-graph-test

# ---- MNIST + Triton runtime (needs PVC from labs/triton export) ------------
.PHONY: deploy-mnist
deploy-mnist: ## MNIST via KServe Triton + PVC
	@echo "Need: make -C ../labs workloads && make -C ../triton export"
	@echo "And: make -C ../triton clean  (free RWO PVC)"
	kubectl apply -f mnist-triton.yaml
	kubectl wait --for=condition=Ready inferenceservice/mnist-triton --timeout=600s
	kubectl get inferenceservice mnist-triton
	@echo "Ready. Test: make test-mnist"

.PHONY: test-mnist
test-mnist: ## MNIST inference via KServe Triton
	kubectl delete job kserve-mnist-test --ignore-not-found
	kubectl apply -f mnist-test-job.yaml
	kubectl wait --for=condition=complete job/kserve-mnist-test --timeout=180s
	kubectl logs job/kserve-mnist-test

.PHONY: undeploy
undeploy: ## Delete InferenceServices / graphs / test Jobs
	kubectl delete -f iris-graph.yaml -f iris-graph-test-job.yaml --ignore-not-found
	kubectl delete -f iris-xform.yaml -f iris-xform-test-job.yaml --ignore-not-found
	kubectl delete -f sklearn-iris.yaml -f sklearn-iris-v2.yaml -f iris-test-job.yaml --ignore-not-found
	kubectl delete -f mnist-triton.yaml -f mnist-test-job.yaml --ignore-not-found
	kubectl delete job kserve-iris-test kserve-mnist-test kserve-graph-test kserve-xform-test --ignore-not-found
	kubectl delete configmap kserve-iris-input iris-transformer-code kserve-xform-input --ignore-not-found

.PHONY: clean
clean: undeploy ## Uninstall KServe + cert-manager
	-helm uninstall kserve-runtime-configs -n kserve 2>/dev/null || true
	-helm uninstall kserve-resources -n kserve 2>/dev/null || true
	-helm uninstall kserve-crd -n kserve 2>/dev/null || true
	-kubectl delete namespace kserve --ignore-not-found
	-helm uninstall cert-manager -n cert-manager 2>/dev/null || true
	-kubectl delete namespace cert-manager --ignore-not-found
	@echo "KServe stack removed."
