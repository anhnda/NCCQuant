#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_flexnu.sh -- FlexNu: FlexRound-style learnable rounding on a sorted,
# non-uniform per-block codebook.
#
# Runs the ablation ladder so the two mechanisms can be attributed separately:
#
#   E  codebookK baseline   -> reference (plain learned codebook, no Gram)
#   A  freeze both          -> must reproduce codebookK exactly (sanity)
#   B  codebook only        -> FITTING W. Assignment stays nearest-neighbour.
#   C  divisor only         -> ESCAPING via the loss. Non-nearest assignments,
#                              reachable only through G's off-diagonals.
#   D  joint                -> both, with lr_cb << lr_scale
#
# B and C are different in kind, not in degree. On synthetic anisotropic Grams
# C beat B roughly 7x (-80% vs -11%) while moving only ~6.7% of weights off
# their nearest codeword. On isotropic G both are exactly 0, as theory requires.
# Expect C (or D) to win on real activations; if B wins, the Gram is closer to
# diagonal than expected and that itself is the finding.
#
# DISK. Each cell writes a full checkpoint (~14 GB for a 7B model); the whole
# ladder would be ~70 GB. So each cell is quantize -> eval -> preserve ppl.json
# -> rm -rf the weights, and only the json + logs survive. Peak disk is
# therefore ONE checkpoint, not five. Set KEEP_CKPT=1 to keep them all, or
# KEEP_CKPT=C_divisor_only to keep just the winner for downstream work.
#
# Outputs that survive a run:
#   $LOG_DIR/flexnu_<ts>.log            full stdout of every cell
#   $LOG_DIR/flexnu_<ts>/<cell>.log     per-cell log, split out for diffing
#   $OUT_ROOT/ppl/<cell>.json           perplexity per cell
#   $OUT_ROOT/summary_<ts>.tsv          one row per cell, for pasting into a table
#
# GPU + torch required. Nothing is installed by this script. Run manually.
# ---------------------------------------------------------------------------
set -euo pipefail
# Llama 3.1. /home/DATA/prometheus/anh/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b
# Mistral 7B /home/DATA/prometheus/anh/.cache/huggingface/hub/models--mistralai--Mistral-7B-v0.3/snapshots/caa1feb0e54d415e2df31207e5f4e273e33509b1 
# Qwen2.5 /home/DATA/prometheus/anh/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B/snapshots/d149729398750b98c0af14eb82c78cfe92750796 

MODEL_PATH=${MODEL_PATH:-/home/DATA/prometheus/anh/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b}
OUT_ROOT=${OUT_ROOT:-./quantized_models/flexnu}
LOG_DIR=${LOG_DIR:-./logs}

# Which checkpoints to keep. "" = keep none (default, disk-safe).
#   KEEP_CKPT=1                 keep all
#   KEEP_CKPT=C_divisor_only    keep only that cell (space-separated list ok)
KEEP_CKPT=${KEEP_CKPT:-}

TS=$(date +%Y%m%d_%H%M%S)
CELL_LOG_DIR="$LOG_DIR/flexnu_$TS"
PPL_DIR="$OUT_ROOT/ppl"
SUMMARY="$OUT_ROOT/summary_$TS.tsv"
mkdir -p "$LOG_DIR" "$CELL_LOG_DIR" "$OUT_ROOT" "$PPL_DIR"

LOG_FILE="$LOG_DIR/flexnu_$TS.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== run started $(date) ==="
echo "model:    $MODEL_PATH"
echo "log:      $LOG_FILE"
echo "per-cell: $CELL_LOG_DIR"
echo "ppl:      $PPL_DIR"
echo "keep:     ${KEEP_CKPT:-<none, checkpoints deleted after eval>}"

BITS=3
CB_BLOCK=64
N_CALIB=128
MAX_LEN=512

# ---- eval config ----------------------------------------------------------
EVAL_DATASETS=${EVAL_DATASETS:-"wikitext2 c4"}   # e.g. EVAL_DATASETS="wikitext2 c4"
EVAL_WINDOW=2048
EVAL_STRIDE=512

# ---- FlexNu hyper-params --------------------------------------------------
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

FLEX_COMMON="--cb-block-size $CB_BLOCK --n-calib $N_CALIB --max-length $MAX_LEN \
  --no-ncc --flexnu-iters $ITERS --flexnu-lr-cb $LR_CB \
  --flexnu-lr-scale $LR_SCALE --flexnu-tau-frac $TAU_FRAC \
  --flexnu-stage-frac $STAGE_FRAC --flexnu-row-block $ROW_BLOCK \
  --gram-group-size $GRAM_GROUP --gram-store-device $GRAM_DEV \
  --flexnu-verbose"

printf 'cell\tquantizer\tppl_wikitext2\tppl_c4\tenergy_drop_pct\n' > "$SUMMARY"

# --------------------------------------------------------------------------- #
# keep_ckpt <tag> -> 0 if this cell's weights should be preserved
# --------------------------------------------------------------------------- #
keep_ckpt () {
  local tag=$1
  [ -z "$KEEP_CKPT" ] && return 1
  [ "$KEEP_CKPT" = "1" ] && return 0
  for k in $KEEP_CKPT; do [ "$k" = "$tag" ] && return 0; done
  return 1
}

# --------------------------------------------------------------------------- #
# cell <tag> <quantizer> [extra quantize.py args...]
#   quantize -> eval -> preserve ppl.json + energy -> delete weights
# --------------------------------------------------------------------------- #
cell () {
  local tag=$1; shift
  local quant=$1; shift
  local dir="$OUT_ROOT/$tag"
  local clog="$CELL_LOG_DIR/$tag.log"

  echo
  echo "############################################################"
  echo "# $tag   [$quant]   ($(date))"
  echo "############################################################"

  # ---- 1. quantize --------------------------------------------------------
  # Never let one cell abort the ladder; record the failure and move on.
  set +e
  if [ "$quant" = "codebook${BITS}" ]; then
    python quantize.py \
      --model-path "$MODEL_PATH" --quantizer "$quant" \
      --cb-block-size $CB_BLOCK --n-calib $N_CALIB --max-length $MAX_LEN \
      --no-ncc --output-dir "$dir" "$@" 2>&1 | tee "$clog"
  else
    python quantize.py \
      --model-path "$MODEL_PATH" --quantizer "$quant" \
      $FLEX_COMMON --output-dir "$dir" "$@" 2>&1 | tee "$clog"
  fi
  local q_rc=${PIPESTATUS[0]}
  set -e

  if [ "$q_rc" -ne 0 ] || [ ! -d "$dir" ]; then
    echo "!!! $tag: quantize exited $q_rc / checkpoint missing -- skipping eval"
    printf '%s\t%s\tFAIL\tFAIL\tNA\n' "$tag" "$quant" >> "$SUMMARY"
    rm -rf "$dir"
    return 0
  fi

  # ---- 2. eval ------------------------------------------------------------
  echo ">>> [$tag] eval_ppl"
  set +e
  python eval_ppl.py \
    --model-path "$dir" \
    --datasets $EVAL_DATASETS \
    --max-length $EVAL_WINDOW --stride $EVAL_STRIDE \
    --tag "$tag" 2>&1 | tee -a "$clog"
  local e_rc=${PIPESTATUS[0]}
  set -e
  [ "$e_rc" -ne 0 ] && echo "!!! $tag: eval exited $e_rc"

  # ---- 3. preserve the small artifacts ------------------------------------
  [ -f "$dir/ppl.json" ] && cp "$dir/ppl.json" "$PPL_DIR/$tag.json"

  # One summary row: perplexities from the json, energy drop from the log.
  python - "$tag" "$quant" "$PPL_DIR/$tag.json" "$clog" "$SUMMARY" <<'PY'
import json, os, re, sys
tag, quant, pj, clog, summ = sys.argv[1:6]
def ppl(name):
    if not os.path.exists(pj):
        return "NA"
    d = json.load(open(pj))
    return f"{d[name]['perplexity']:.4f}" if name in d else "NA"
drop = "NA"
if os.path.exists(clog):
    m = re.findall(r"\(([-+]?\d+\.\d+)% drop\)", open(clog, errors="ignore").read())
    if m:
        drop = m[-1]
with open(summ, "a") as f:
    f.write(f"{tag}\t{quant}\t{ppl('wikitext2')}\t{ppl('c4')}\t{drop}\n")
PY

  # ---- 4. delete the weights ----------------------------------------------
  if keep_ckpt "$tag"; then
    echo ">>> [$tag] KEEPING checkpoint at $dir"
  else
    local sz
    sz=$(du -sh "$dir" 2>/dev/null | cut -f1)
    rm -rf "$dir"
    echo ">>> [$tag] deleted checkpoint (${sz:-?} reclaimed)"
  fi
  echo "<<< done $tag"
}

# --------------------------------------------------------------------------- #
# The ladder. E first so the baseline is on record before anything else runs.
# --------------------------------------------------------------------------- #
cell "E_codebook${BITS}"  "codebook${BITS}"
cell "A_freeze_both"      "flexnu${BITS}"  --flexnu-freeze-codebook --flexnu-freeze-scale
cell "B_codebook_only"    "flexnu${BITS}"  --flexnu-freeze-scale
cell "C_divisor_only"     "flexnu${BITS}"  --flexnu-freeze-codebook
cell "D_joint"            "flexnu${BITS}"

echo
echo "############################################################"
echo "# summary"
echo "############################################################"
# `column -t -s $'\t'` collapses runs of tabs and drops empty cells on some
# util-linux builds, so format it directly instead.
awk -F'\t' '{printf "%-18s %-12s %-14s %-10s %s\n", $1, $2, $3, $4, $5}' "$SUMMARY"

echo
echo "=== done $(date) ==="
echo "log:      $LOG_FILE"
echo "per-cell: $CELL_LOG_DIR"
echo "ppl:      $PPL_DIR"
echo "summary:  $SUMMARY"
du -sh "$OUT_ROOT" 2>/dev/null || true
