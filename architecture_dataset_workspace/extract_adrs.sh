#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_DIR="./architecture_dataset_workspace"
DATASET_DIR="$WORKSPACE_DIR/compiled_adr_dataset"

mkdir -p "$DATASET_DIR"
cd "$WORKSPACE_DIR" || exit

echo "Cloning target repositories..."

# Use --depth 1 for shallow clones to save time/space
git clone --depth 1 https://github.com/openedx/edx-platform.git
git clone --depth 1 https://github.com/backstage/backstage.git
git clone --depth 1 https://github.com/adr/madr.git
git clone --depth 1 https://github.com/alphagov/govuk-docker.git
git clone --depth 1 https://github.com/petabridge/memorizer.git

echo "Extracting Open edX ADRs (RST format)..."
find edx-platform -type f -name "*.rst" -path "*/docs/decisions/*" | while read -r file; do
    basename=$(basename "$file")
    cp "$file" "$DATASET_DIR/edx_${basename}"
done

echo "Extracting Backstage ADRs (Markdown)..."
find backstage -type f -name "*.md" -path "*/docs/architecture-decisions/*" | while read -r file; do
    basename=$(basename "$file")
    cp "$file" "$DATASET_DIR/backstage_${basename}"
done

echo "Extracting MADR standard ADRs (Markdown)..."
find madr -type f -name "*.md" -path "*/docs/decisions/*" | while read -r file; do
    basename=$(basename "$file")
    cp "$file" "$DATASET_DIR/madr_${basename}"
done

echo "Extracting GOV.UK ADRs (Markdown)..."
find govuk-docker -type f -name "*.md" -path "*/docs/adr/*" | while read -r file; do
    basename=$(basename "$file")
    cp "$file" "$DATASET_DIR/govuk_${basename}"
done

echo "Extracting Memorizer ADRs (Markdown)..."
find memorizer -type f -name "*.md" -path "*/docs/adr/*" | while read -r file; do
    basename=$(basename "$file")
    cp "$file" "$DATASET_DIR/memorizer_${basename}"
done

echo "Extraction complete. Total ADRs collected:"
ls -1 "$DATASET_DIR" | wc -l
