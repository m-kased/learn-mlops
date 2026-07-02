#!/usr/bin/env bash
# Destroy the GKE cluster if it still exists (for cron / cost safety).
# Config via env or defaults below (match cluster/Makefile).

set -euo pipefail

PROJECT="${PROJECT:-gke-ml-500613}"
CLUSTER="${CLUSTER:-cluster-1}"
ZONE="${ZONE:-us-central1-a}"
LOG_DIR="${LOG_DIR:-${HOME}/.local/log/ml-gpu}"

mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/destroy-cluster.log"

# Cron has a minimal PATH — include common gcloud locations.
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${HOME}/google-cloud-sdk/bin:${PATH}"

log() {
  echo "$(date -Is) $*" | tee -a "$LOG_FILE"
}

if ! command -v gcloud >/dev/null 2>&1; then
  log "ERROR: gcloud not found in PATH"
  exit 1
fi

if gcloud container clusters describe "$CLUSTER" \
    --zone "$ZONE" --project "$PROJECT" >/dev/null 2>&1; then
  log "Cluster ${CLUSTER} exists in ${PROJECT}/${ZONE} — deleting..."
  if gcloud container clusters delete "$CLUSTER" \
      --zone "$ZONE" --project "$PROJECT" --quiet >>"$LOG_FILE" 2>&1; then
    log "Cluster ${CLUSTER} deleted."
  else
    log "ERROR: delete failed (see log)"
    exit 1
  fi
else
  log "Cluster ${CLUSTER} not found — nothing to do."
fi
