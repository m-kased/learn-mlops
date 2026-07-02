#!/usr/bin/env bash
# Create a GCP service account for GitHub Actions cluster destroy (minimal scope).
# Outputs key.json — add its contents to GitHub secret GCP_SA_KEY, then delete the file.

set -euo pipefail

PROJECT="${PROJECT:-gke-ml-500613}"
SA_NAME="${SA_NAME:-github-gke-destroy}"
SA_EMAIL="${SA_NAME}@${PROJECT}.iam.gserviceaccount.com"
KEY_FILE="${KEY_FILE:-key.json}"

export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${HOME}/google-cloud-sdk/bin:${PATH}"

if ! command -v gcloud >/dev/null 2>&1; then
  echo "ERROR: gcloud not found" >&2
  exit 1
fi

echo "Project: $PROJECT"
echo "Service account: $SA_EMAIL"

if ! gcloud iam service-accounts describe "$SA_EMAIL" --project="$PROJECT" >/dev/null 2>&1; then
  gcloud iam service-accounts create "$SA_NAME" \
    --project="$PROJECT" \
    --display-name="GitHub Actions GKE cluster destroy"
  echo "Created service account."
else
  echo "Service account already exists."
fi

gcloud projects add-iam-policy-binding "$PROJECT" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/container.clusterAdmin" \
  --condition=None >/dev/null

echo "Granted roles/container.clusterAdmin (list/get/delete clusters)."

if [[ -f "$KEY_FILE" ]]; then
  echo "ERROR: $KEY_FILE already exists — move it aside first." >&2
  exit 1
fi

gcloud iam service-accounts keys create "$KEY_FILE" \
  --iam-account="$SA_EMAIL" \
  --project="$PROJECT"

echo ""
echo "Created $KEY_FILE"
echo ""
echo "Next steps:"
echo "  1. GitHub repo → Settings → Secrets and variables → Actions"
echo "  2. New repository secret:  GCP_SA_KEY  =  entire contents of $KEY_FILE"
echo "  3. rm $KEY_FILE   (never commit this file)"
echo "  4. Push .github/workflows/destroy-cluster.yml and check Actions tab"
echo "  5. Test: Actions → Destroy GKE cluster → Run workflow"
