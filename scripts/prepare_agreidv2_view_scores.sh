#!/usr/bin/env bash
# Build the AG-ReID.v2 aerial Far/Near view scores used by three-domain Stage 2.
#
# Usage (from anywhere):
#   bash scripts/prepare_agreidv2_view_scores.sh <V2_OUTER_DIR> [MASK_DIR] [GPU]
#
#   V2_OUTER_DIR  outer AG-ReID.v2 directory (contains exp*.txt and AG-ReID.v2/)
#   MASK_DIR      SCHP masks mirroring train_all; default <V2_OUTER_DIR>_SCHP/train_all
#   GPU           CUDA device index; default 0
#
# Folds follow AG-ReID.v1: the A field defines the fold (aerial c00/c10/c20,
# ground c02,c03 / c12,c13 / c22,c23).  Identities include A, so folds are
# identity-disjoint and the strict camera_identity split removes nothing.
# Every aerial score is out-of-fold (scored by the model that never saw it).
set -euo pipefail

DATA_DIR=${1:?usage: $0 <V2_OUTER_DIR> [MASK_DIR] [GPU]}
DATA_DIR=${DATA_DIR%/}
MASK_DIR=${2:-${DATA_DIR}_SCHP/train_all}
GPU=${3:-0}
TRAIN_DIR="$DATA_DIR/AG-ReID.v2/train_all"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$REPO_ROOT/view_domain/outputs/agreidv2"
SCORES="$OUT/agreidv2_aerial_view_scores_foreground.csv"
PY=${PYTHON:-python}

cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1
mkdir -p "$OUT"

echo "[v2-view] $(date) 1/5 SCHP mask preflight"
"$PY" tools/preflight_agreidv2.py --data-dir "$DATA_DIR" --mask-dir "$MASK_DIR"

echo "[v2-view] $(date) 2/5 manifest"
"$PY" view_domain/build_view_manifest.py \
  --aerial-dir "$TRAIN_DIR" --ground-dir "$TRAIN_DIR" \
  --aerial-cameras c00,c10,c20 \
  --ground-folds 'c02,c03;c12,c13;c22,c23' \
  --output "$OUT/view_manifest.csv"

echo "[v2-view] $(date) 3/5 foreground classifier (3 folds)"
"$PY" view_domain/train_view_classifier.py \
  --manifest "$OUT/view_manifest.csv" \
  --all-folds --mode foreground --split-mode camera_identity \
  --checkpoint-root "$OUT/checkpoints" \
  --aerial-source-root "$TRAIN_DIR" --ground-source-root "$TRAIN_DIR" \
  --aerial-mask-root "$MASK_DIR" --ground-mask-root "$MASK_DIR" \
  --batch-size 64 --epochs 30 --workers 8 --seed 1 --amp

echo "[v2-view] $(date) 4/5 out-of-fold aerial scores"
"$PY" view_domain/score_aerial_images.py \
  --manifest "$OUT/view_manifest.csv" \
  --foreground-checkpoint-dir "$OUT/checkpoints/foreground" \
  --aerial-source-root "$TRAIN_DIR" --aerial-mask-root "$MASK_DIR" \
  --near-ratio 0.50 --grouping per-camera --workers 8 \
  --output "$SCORES"

echo "[v2-view] $(date) 5/5 analysis + score preflight"
"$PY" view_domain/analyze_view_scores.py \
  --scores "$SCORES" --output-dir "$OUT/analysis" --near-ratio 0.50
"$PY" tools/preflight_agreidv2.py --data-dir "$DATA_DIR" \
  --mask-dir "$MASK_DIR" --view-score-file "$SCORES" --view-near-ratio 0.50

echo "[v2-view] $(date) done: $SCORES"
echo "[v2-view] inspect $OUT/analysis before running three-domain Stage 2"
