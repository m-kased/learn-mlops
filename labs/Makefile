# ============================================================================
#  GPU ML labs (run from repo root: make -C labs <target>)
# ============================================================================

QUEUE_NAME ?= inference:queue

.DEFAULT_GOAL := help

.PHONY: help
help: ## Show lab targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# ---- GPU smoke test --------------------------------------------------------
.PHONY: gpu-test
gpu-test: ## Run a pod that prints `nvidia-smi` then exits
	kubectl apply -f gpu-test.yaml
	@echo "Waiting for pod to finish... (Ctrl-C is fine, then run 'make gpu-logs')"
	kubectl wait --for=condition=Ready pod/gpu-test --timeout=300s || true
	$(MAKE) gpu-logs

.PHONY: gpu-logs
gpu-logs: ## Show logs from the gpu-test pod
	kubectl logs gpu-test

.PHONY: gpu-clean
gpu-clean: ## Delete the gpu-test pod
	kubectl delete -f gpu-test.yaml --ignore-not-found

# ---- PyTorch GPU check -----------------------------------------------------
.PHONY: pytorch-test
pytorch-test: ## Run a pod that verifies PyTorch can use the GPU (real matmul)
	kubectl apply -f pytorch-test.yaml
	@echo "Pulling the PyTorch image can take a few minutes the first time..."
	kubectl wait --for=condition=Ready pod/pytorch-test --timeout=600s || true
	$(MAKE) pytorch-logs

.PHONY: pytorch-logs
pytorch-logs: ## Show logs from the pytorch-test pod (use after pytorch-test)
	kubectl logs -f pytorch-test

.PHONY: pytorch-clean
pytorch-clean: ## Delete the pytorch-test pod
	kubectl delete -f pytorch-test.yaml --ignore-not-found

# ---- Persistent storage ----------------------------------------------------
.PHONY: data
data: ## Create the shared PersistentVolumeClaim (datasets + checkpoints)
	kubectl apply -f pvc.yaml
	kubectl get pvc ml-data

.PHONY: data-clean
data-clean: ## Delete the PVC (DESTROYS stored datasets/checkpoints)
	kubectl delete -f pvc.yaml --ignore-not-found

# ---- MNIST training Job ----------------------------------------------------
.PHONY: train
train: ## Train MNIST on the GPU as a Job (creates PVC + code ConfigMap first)
	kubectl apply -f pvc.yaml
	kubectl create configmap mnist-code --from-file=train.py \
		--dry-run=client -o yaml | kubectl apply -f -
	kubectl delete job mnist-train --ignore-not-found
	kubectl apply -f mnist-job.yaml
	@echo "Job submitted. Follow it with: make train-logs"

.PHONY: train-logs
train-logs: ## Stream logs from the training Job (waits for the pod to start)
	kubectl wait --for=condition=Ready pod -l job-name=mnist-train --timeout=600s || true
	kubectl logs -f job/mnist-train

.PHONY: train-clean
train-clean: ## Delete the training Job and its code ConfigMap (keeps the PVC)
	kubectl delete -f mnist-job.yaml --ignore-not-found
	kubectl delete -f mnist-job-b.yaml --ignore-not-found
	kubectl delete configmap mnist-code --ignore-not-found

# ---- Real workloads (two concurrent training Jobs) ---------------------------
.PHONY: workloads
workloads: ## Submit two MNIST training Jobs that share the GPU (like real batch work)
	kubectl apply -f pvc.yaml
	kubectl create configmap mnist-code --from-file=train.py \
		--dry-run=client -o yaml | kubectl apply -f -
	kubectl delete job mnist-train mnist-train-b --ignore-not-found
	kubectl apply -f mnist-job.yaml -f mnist-job-b.yaml
	@echo "Two Jobs submitted. Status: make workloads-status  Logs: make workloads-logs-a"

.PHONY: workloads-status
workloads-status: ## Show both training Jobs and their pods
	kubectl get jobs mnist-train mnist-train-b
	kubectl get pods -l 'job-name in (mnist-train, mnist-train-b)' -o wide

.PHONY: workloads-logs-a
workloads-logs-a: ## Stream logs from training Job A (run-a)
	kubectl logs -f job/mnist-train

.PHONY: workloads-logs-b
workloads-logs-b: ## Stream logs from training Job B (run-b)
	kubectl logs -f job/mnist-train-b

.PHONY: workloads-gpu
workloads-gpu: ## nvidia-smi on the GPU node via a running trainer pod
	kubectl exec -it $$(kubectl get pod -l job-name=mnist-train -o jsonpath='{.items[0].metadata.name}') -- nvidia-smi

.PHONY: workloads-clean
workloads-clean: ## Delete both training Jobs (keeps PVC + checkpoints)
	kubectl delete -f mnist-job.yaml -f mnist-job-b.yaml --ignore-not-found
	kubectl delete configmap mnist-code --ignore-not-found

# ---- Interactive dev pod ---------------------------------------------------
.PHONY: dev-up
dev-up: ## Start a long-lived GPU pod for interactive work (creates PVC first)
	kubectl apply -f pvc.yaml
	kubectl apply -f dev-pod.yaml
	kubectl wait --for=condition=Ready pod/dev --timeout=600s || true
	@echo "Ready. Open a shell with: make dev-shell"

.PHONY: dev-shell
dev-shell: ## Open an interactive shell inside the dev pod
	kubectl exec -it dev -- bash

.PHONY: gpu-watch
gpu-watch: ## Live GPU utilization via nvidia-smi inside the dev pod (Ctrl-C to stop)
	kubectl exec -it dev -- nvidia-smi -l 1

.PHONY: dev-clean
dev-clean: ## Delete the dev pod (frees the GPU; keeps the PVC)
	kubectl delete -f dev-pod.yaml --ignore-not-found

# ---- GPU time-sharing demo (C) ---------------------------------------------
.PHONY: share-demo
share-demo: ## Launch 3 pods that share ONE physical T4 via time-sharing
	kubectl apply -f share-demo.yaml
	@echo "Watch them all land on the same node with: make share-show"

.PHONY: share-show
share-show: ## Show the 3 sharing pods (same node) + allocatable GPUs
	kubectl get pods -l app=gpu-share -o wide
	@echo "--- GPU allocatable per node (physical x max-shared-clients) ---"
	kubectl get nodes -o=jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.allocatable.nvidia\.com/gpu}{"\n"}{end}'
	@echo "--- processes on the GPU (all 3 pods share it) ---"
	-kubectl exec deploy/gpu-share -- nvidia-smi

.PHONY: share-clean
share-clean: ## Delete the time-sharing demo
	kubectl delete -f share-demo.yaml --ignore-not-found

# ---- Inference serving (E) -------------------------------------------------
.PHONY: serve
serve: ## Deploy MNIST inference (Deployment + Service) on a shared GPU
	kubectl create configmap serve-code --from-file=serve.py \
		--dry-run=client -o yaml | kubectl apply -f -
	kubectl apply -f serve.yaml
	kubectl rollout status deploy/mnist-serve --timeout=600s

.PHONY: serve-forward
serve-forward: ## Port-forward inference to localhost:8080 (blocking)
	@echo "In another terminal:  curl -s localhost:8080/predict/sample | python3 -m json.tool"
	kubectl port-forward svc/mnist-serve 8080:80

.PHONY: serve-test
serve-test: ## One-off prediction from inside the cluster
	kubectl run curl-test --rm -it --restart=Never --image=curlimages/curl -- \
		curl -s http://mnist-serve/predict/sample

.PHONY: serve-clean
serve-clean: ## Delete inference Deployment, Service and code ConfigMap
	kubectl delete -f serve.yaml --ignore-not-found
	kubectl delete configmap serve-code --ignore-not-found

# ---- HPA on serving --------------------------------------------------------
.PHONY: serve-hpa
serve-hpa: ## Attach a HorizontalPodAutoscaler to mnist-serve (CPU-based)
	kubectl apply -f serve-hpa.yaml
	kubectl get hpa mnist-serve

.PHONY: serve-hpa-status
serve-hpa-status: ## Show HPA decision + mnist-serve pod count
	kubectl get hpa mnist-serve
	kubectl get deploy mnist-serve
	kubectl get pods -l app=mnist-serve -o wide

.PHONY: serve-load
serve-load: ## Start HTTP load generator (to trigger HPA scale-up)
	kubectl apply -f serve-load.yaml
	@echo "Load running. Watch scaling with: make serve-hpa-status"

.PHONY: serve-load-clean
serve-load-clean: ## Stop the load generator
	kubectl delete -f serve-load.yaml --ignore-not-found

.PHONY: serve-hpa-clean
serve-hpa-clean: ## Remove the HPA (Deployment stays at current replica count)
	kubectl delete -f serve-hpa.yaml --ignore-not-found

# ---- RPS scaling (Prometheus + KEDA) ---------------------------------------
.PHONY: prometheus prometheus-status prometheus-clean
prometheus: ## Deploy in-cluster Prometheus (scrapes mnist-serve /metrics)
	kubectl apply -f prometheus.yaml
	kubectl rollout status deploy/prometheus --timeout=300s
	@echo "Wait ~30s for first scrape, then: make prometheus-status"

prometheus-status: ## Show Prometheus + current RPS query
	kubectl get pods,svc -l app=prometheus
	@echo "RPS query (sum rate over 1m):"
	@kubectl exec deploy/prometheus -- wget -qO- \
		'http://localhost:9090/api/v1/query?query=sum(rate(http_requests_total{app="mnist-serve"}[1m]))' \
		2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); r=d.get('data',{}).get('result',[]); print(r[0]['value'][1] if r else 'no data yet')" || echo "no data yet"

prometheus-clean: ## Delete Prometheus
	kubectl delete -f prometheus.yaml --ignore-not-found

.PHONY: serve-rps serve-rps-status serve-rps-clean
serve-rps: ## KEDA scales mnist-serve on HTTP RPS (needs prometheus + serve)
	-kubectl delete -f serve-hpa.yaml --ignore-not-found
	kubectl apply -f serve-rps-scaledobject.yaml
	@echo "ScaledObject applied. Start load: make serve-load"
	@echo "Watch scaling: make serve-rps-status"

serve-rps-status: ## ScaledObject + mnist-serve replicas + RPS
	kubectl get scaledobject,hpa -l scaledobject.keda.sh/name=mnist-serve 2>/dev/null \
		|| kubectl get scaledobject mnist-serve,hpa 2>/dev/null || kubectl get scaledobject,hpa
	kubectl get deploy mnist-serve
	kubectl get pods -l app=mnist-serve -o wide
	@$(MAKE) --no-print-directory prometheus-status

serve-rps-clean: ## Remove KEDA RPS ScaledObject (stops RPS autoscaling)
	kubectl delete -f serve-rps-scaledobject.yaml --ignore-not-found

# ---- Queue depth (Redis + KEDA workers) ------------------------------------
.PHONY: queue-redis
queue-redis: ## Deploy Redis queue backend
	kubectl apply -f redis.yaml
	kubectl rollout status deploy/redis --timeout=120s
	@echo "Test with: make queue-redis-test"

.PHONY: queue-redis-status
queue-redis-status: ## Show Redis pod + current queue length
	kubectl get pods -l app=redis -o wide
	-kubectl exec deploy/redis -- redis-cli LLEN $(QUEUE_NAME)

.PHONY: queue-redis-test
queue-redis-test: ## Push 3 test jobs and print queue length
	kubectl exec deploy/redis -- redis-cli LPUSH $(QUEUE_NAME) job-1 job-2 job-3
	@echo -n "Queue length: "
	@kubectl exec deploy/redis -- redis-cli LLEN $(QUEUE_NAME)

.PHONY: queue-redis-drain
queue-redis-drain: ## Clear the queue
	kubectl exec deploy/redis -- redis-cli DEL $(QUEUE_NAME)

.PHONY: queue-redis-clean
queue-redis-clean: ## Delete Redis Deployment + Service
	kubectl delete -f redis.yaml --ignore-not-found

.PHONY: queue-ckpt queue-worker queue-worker-logs queue-worker-status queue-worker-clean
queue-ckpt: ## Export model weights from PVC into ConfigMap (for multi-worker KEDA)
	@POD=$$(kubectl get pod -l app=mnist-serve -o jsonpath='{.items[0].metadata.name}' 2>/dev/null); \
	if [ -z "$$POD" ]; then \
	  echo "Need mnist-serve (make serve) or a pod with /data/checkpoints/run-a/mnist.pt on ml-data"; \
	  exit 1; \
	fi; \
	echo "Exporting weights from $$POD (model only, fits ConfigMap 1MiB limit)..."; \
	kubectl exec "$$POD" -- python -c 'import torch; c=torch.load("/data/checkpoints/run-a/mnist.pt", map_location="cpu", weights_only=False); torch.save({"model": c["model"]}, "/tmp/mnist-ckpt.pt")'; \
	kubectl cp "default/$$POD:/tmp/mnist-ckpt.pt" /tmp/mnist-ckpt.pt; \
	SIZE=$$(stat -c%s /tmp/mnist-ckpt.pt); \
	if [ "$$SIZE" -gt 1048576 ]; then \
	  echo "Error: checkpoint is $$SIZE bytes (ConfigMap data limit is 1MiB)"; \
	  rm -f /tmp/mnist-ckpt.pt; exit 1; \
	fi; \
	kubectl delete configmap mnist-ckpt --ignore-not-found; \
	kubectl create configmap mnist-ckpt --from-file=mnist.pt=/tmp/mnist-ckpt.pt; \
	rm -f /tmp/mnist-ckpt.pt; \
	echo "ConfigMap mnist-ckpt ready ($$SIZE bytes)."

queue-worker: queue-ckpt ## Deploy GPU worker that pulls jobs from Redis
	kubectl create configmap queue-worker-code --from-file=queue_worker.py \
		--dry-run=client -o yaml | kubectl apply -f -
	kubectl apply -f queue-worker.yaml
	kubectl rollout status deploy/queue-worker --timeout=600s
	@echo "Push jobs: make queue-produce   Watch: make queue-worker-logs"

queue-worker-logs: ## Stream worker logs (see each job processed)
	kubectl logs -f deploy/queue-worker

queue-worker-status: ## Worker pod + queue length
	kubectl get pods -l app=queue-worker -o wide
	-kubectl exec deploy/redis -- redis-cli LLEN $(QUEUE_NAME)

queue-worker-clean: ## Delete the queue worker
	kubectl delete -f queue-worker.yaml --ignore-not-found
	kubectl delete configmap queue-worker-code mnist-ckpt --ignore-not-found

.PHONY: queue-produce
queue-produce: ## Push 10 MNIST inference jobs (image indices 0-9)
	kubectl exec deploy/redis -- redis-cli LPUSH $(QUEUE_NAME) \
		0 1 2 3 4 5 6 7 8 9
	@echo -n "Queue length: "
	@kubectl exec deploy/redis -- redis-cli LLEN $(QUEUE_NAME)

.PHONY: queue-produce-many
queue-produce-many: ## Push 81 jobs to trigger KEDA scale-up
	kubectl exec deploy/redis -- sh -c 'for i in $$(seq 0 80); do redis-cli LPUSH $(QUEUE_NAME) $$i; done'
	@echo -n "Queue length: "
	@kubectl exec deploy/redis -- redis-cli LLEN $(QUEUE_NAME)

.PHONY: queue-keda queue-keda-status queue-keda-clean
queue-keda: ## KEDA scales workers on Redis queue depth
	kubectl apply -f queue-scaledobject.yaml
	@echo "ScaledObject applied. Watch: make queue-keda-status"

queue-keda-status: ## ScaledObject + worker replicas + queue length
	kubectl get scaledobject,hpa -l scaledobject.keda.sh/name=queue-worker 2>/dev/null || kubectl get scaledobject,hpa
	kubectl get deploy queue-worker
	kubectl get pods -l app=queue-worker -o wide
	@echo -n "Queue length: "
	@kubectl exec deploy/redis -- redis-cli LLEN $(QUEUE_NAME)

queue-keda-clean: ## Remove KEDA ScaledObject (stops autoscaling workers)
	kubectl delete -f queue-scaledobject.yaml --ignore-not-found
