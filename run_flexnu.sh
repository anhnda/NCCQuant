#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_flexnu.sh -- FlexNu: FlexRound-style learnable rounding on a jointly
# learned, sorted, non-uniform per-block codebook.
#
# Runs the ablation ladder so the two mechanisms can be attributed separately:
#
#   A  freeze both          -> must reproduce codebookK exactly (sanity)
#   B  codebook only        -> FITTING W. Assignment stays nearest-neighbour.
#   C  divisor only         -> ESCAPING via the loss. Non-nearest assignments,
#                              reachable only through G's off-diagonals.
#   D  joint                -> both, with lr_cb << lr_scale
#   E  codebookK baseline   -> reference
#
# B and C are different in kind, not in degree. On synthetic anisotropic Grams
# C beat B roughly 7x (-80% vs -11%) while moving only ~6.7% of weights off
# their nearest codeword. On isotropic G both are exactly 0, as theory requires.
# Expect C (or D) to win on real activations; if B wins, the Gram is closer to
# diagonal than expected and that itself is the finding.
#
# Read the printed "FlexNu reconstruction energy Tr(R G R^T)" line for each.
#
# GPU + torch required. Nothing is installed by this script. Run manually.
# ---------------------------------------------------------------------------
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-/path/to/Mistral-7B-v0.3}
OUT_ROOT=${OUT_ROOT:-./quantized_models/flexnu}
LOG_DIR=./logs
mkdir -p "$LOG_DIR" "$OUT_ROOT"
LOG_FILE="$LOG_DIR/flexnu_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== run started $(date) ==="

BITS=3
CB_BLOCK=64
N_CALIB=128
MAX_LEN=512

# FlexNu hyper-params.
ITERS=400
# lr_cb deliberately tiny: on anisotropic G every increase in it monotonically
# hurt (81% -> 37% as lr_cb went 0 -> 3e-3). The codebook chases W and drags the
# thresholds out from under the divisor.
LR_CB=1e-5
LR_SCALE=3e-3
TAU_FRAC=0.5
# 0.0 = joint. Do NOT stage: it starts the divisor at the nearest-neighbour
# point it exists to escape (joint -80%, staged -0.3%).
STAGE_FRAC=0.0

# Memory knobs. row-block bounds the [row_block, in, K-1] threshold tensor;
# gram-group-size bounds how many [in,in] Grams are held at once (each group
# costs one extra calibration pass).
ROW_BLOCK=128
GRAM_GROUP=8
GRAM_DEV=cpu

COMMON="--model-path $MODEL_PATH --bits-unused \
  --cb-block-size $CB_BLOCK --n-calib $N_CALIB --max-length $MAX_LEN \
  --no-ncc --flexnu-iters $ITERS --flexnu-lr-cb $LR_CB \
  --flexnu-lr-scale $LR_SCALE --flexnu-tau-frac $TAU_FRAC \
  --flexnu-stage-frac $STAGE_FRAC --flexnu-row-block $ROW_BLOCK \
  --gram-group-size $GRAM_GROUP --gram-store-device $GRAM_DEV \
  --flexnu-verbose"
# NOTE: --bits-unused above is a placeholder reminder that bit width is chosen
# by the quantizer NAME (flexnu3 / flexnu4), not a flag. Remove it.
COMMON=${COMMON/--bits-unused/}

run () {   # run <tag> <extra-args...>
  local tag=$1; shift
  echo
  echo "############################################################"
  echo "# $tag   ($(date))"
  echo "############################################################"
  python quantize.py $COMMON \
    --quantizer "flexnu${BITS}" \
    --output-dir "$OUT_ROOT/$tag" \
    "$@"
}

# E -- baseline reference (plain learned codebook, no Gram, no FlexRound).
echo "############ E  codebook${BITS} baseline ############"
python quantize.py \
  --model-path "$MODEL_PATH" --quantizer "codebook${BITS}" \
  --cb-block-size $CB_BLOCK --n-calib $N_CALIB --max-length $MAX_LEN \
  --no-ncc --output-dir "$OUT_ROOT/E_codebook${BITS}"

run "A_freeze_both"   --flexnu-freeze-codebook --flexnu-freeze-scale
run "B_codebook_only" --flexnu-freeze-scale
run "C_divisor_only"  --flexnu-freeze-codebook
run "D_joint"

echo
echo "=== done $(date) ==="
echo "log: $LOG_FILE"
