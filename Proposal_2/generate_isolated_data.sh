#!/usr/bin/env bash
# Generate all 4 isolated degradation datasets for MoE expert pre-training.
# Run from the Proposal_2/ directory.

set -euo pipefail
cd "$(dirname "$0")"

GT_DIR="../train/GT"
OUTPUT_ROOT="data"
SEED=42

echo "=== Generating isolated datasets for MoE experts ==="
echo "GT directory: $GT_DIR"
echo "Output root:  $OUTPUT_ROOT"
echo ""

conda run -n ml_env python generate_denoise_pairs.py \
    --gt-dir "$GT_DIR" \
    --output-root "$OUTPUT_ROOT" \
    --seed "$SEED" \
    --type all

echo ""
echo "=== All datasets generated ==="
echo "Datasets:"
echo "  $OUTPUT_ROOT/blur_only/     (blur expert)"
echo "  $OUTPUT_ROOT/gaussian_only/ (gaussian denoise expert)"
echo "  $OUTPUT_ROOT/speckle_only/  (speckle denoise expert)"
echo "  $OUTPUT_ROOT/upsample_only/ (upsample expert)"
