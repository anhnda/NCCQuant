#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_gq.sh -- GuidedQuant ladder. Standalone: does not use quantize.py or
# run_flexnu.sh, and shares nothing with them but eval_ppl.py.
#
# THE LADDER IS THE EXPERIMENT. Three cells, each isolating one claim:
#
#   E  codebookK   SqueezeLLM-style weighted k-means only, no coordinate
#                  descent. This is GuidedQuant's own initialisation, so it is
#                  the floor everything else must beat.
#   N  nosal       LNQ against the UNWEIGHTED Hessian E[x x^T].
#                  (N - E) is what off-diagonal activation structure buys.
#   G  guidedquant LNQ against the SALIENCY-WEIGHTED Hessian, plus a k-means
#                  init weighted by squared end-loss weight gradients.
#                  (G - N) is what END-LOSS saliency buys on top.
#
# At the default NUM_GROUPS=1 (full row, one Hessian per layer) N and G differ
# ONLY in the signal: unweighted E[x x^T] + diag init, versus saliency-weighted
# sum_t s_t x_t x_t^T + per-weight gradient init. The coordinate descent is
# identical. That makes (G - N) a clean read on whether the backward pass over
# the end loss is worth its cost. Raising NUM_GROUPS adds a second axis on top.
#
# Reading the result: if N ~ E, the off-diagonals are not helping and the extra
# machinery is wasted. If G ~ N, the saliency signal is not helping and the
# backward pass is wasted. Both are legitimate findings; the point of running
# all three is that a single number cannot tell you which.
#
# COST. Cell G runs a full backward pass over calibration (once, cached to
# --saliency-dir), then per block: one forward replay to build [G, in, in]
# Hessians, then coordinate descent per output group. It is much slower than
# cell E. Start with a small model and N_CALIB=32 to check it runs at all.
#
# MEMORY. Phase 2 holds NUM_GROUPS x [in, in] fp32 per Linear in the current
# block. At the default NUM_GROUPS=1 that is ~870 MB above the model for a 7B
# block. Raising it multiplies that directly.
# ---------------------------------------------------------------------------
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-/path/to/Llama-2-7b-hf}
OUT_ROOT=${OUT_ROOT:-./quantized_models/gq}
LOG_DIR=${LOG_DIR:-./logs}
KEEP_CKPT=${KEEP_CKPT:-}          # ""=none, 1=all, or space-separated cells

BITS=${BITS:-3}
NUM_GROUPS=${NUM_GROUPS:-1}
N_CALIB=${N_CALIB:-128}
CALIB_LEN=${CALIB_LEN:-2048}
CALIB_DATASET=${CALIB_DATASET:-wikitext2}

# LNQ hyper-params. Reference defaults are iters=3, cd_cycles=4.
ITERS=${ITERS:-3}
CD_CYCLES=${CD_CYCLES:-4}
RIDGE=${RIDGE:-1e-7}
ROW_BLOCK=${ROW_BLOCK:-64}
CD_BLOCK=${CD_BLOCK:-128}
KMEANS_INIT=${KMEANS_INIT:-kmeans++}
INIT_ITERS=${INIT_ITERS:-50}
ROW_CHUNK=${ROW_CHUNK:-1024}

# Phase-1 artefacts are expensive and model-specific, not cell-specific:
# compute once, reuse across cells.
SALIENCY_DIR=${SALIENCY_DIR:-./cache/saliency_$(basename "$MODEL_PATH")_g${NUM_GROUPS}_s${N_CALIB}_l${CALIB_LEN}}

SEQLEN=${SEQLEN:-2048}
EVAL_DATASETS=${EVAL_DATASETS:-"wikitext2 c4"}
EVAL_METHOD=${EVAL_METHOD:-block}
EVAL_DTYPE=${EVAL_DTYPE:-fp16}
DATASET_CACHE=${DATASET_CACHE:-./dataset_cache}

TS=$(date +%Y%m%d_%H%M%S)
CELL_LOG_DIR="$LOG_DIR/gq_$TS"
PPL_DIR="$OUT_ROOT/ppl"
SUMMARY="$OUT_ROOT/summary_$TS.tsv"
mkdir -p "$LOG_DIR" "$CELL_LOG_DIR" "$OUT_ROOT" "$PPL_DIR" "$(dirname "$SALIENCY_DIR")"

LOG_FILE="$LOG_DIR/gq_$TS.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== run started $(date) ==="
echo "model:     $MODEL_PATH"
echo "bits:      $BITS   groups: $NUM_GROUPS   full row"
echo "calib:     $N_CALIB x $CALIB_LEN ($CALIB_DATASET)"
echo "lnq:       iters=$ITERS cd_cycles=$CD_CYCLES ridge=$RIDGE"
echo "saliency:  $SALIENCY_DIR"
echo "eval:      method=$EVAL_METHOD seqlen=$SEQLEN dtype=$EVAL_DTYPE"
echo "log:       $LOG_FILE"
echo "keep:      ${KEEP_CKPT:-<none, checkpoints deleted after eval>}"

EVAL_COMMON="--datasets $EVAL_DATASETS --seqlen $SEQLEN --method $EVAL_METHOD \
  --dtype $EVAL_DTYPE --cache-dir $DATASET_CACHE"

{
  echo "# model=$MODEL_PATH bits=$BITS block=full_row groups=$NUM_GROUPS"
  echo "# calib: $N_CALIB x $CALIB_LEN ($CALIB_DATASET)"
  echo "# lnq: iters=$ITERS cd_cycles=$CD_CYCLES ridge=$RIDGE"
  echo "# eval: method=$EVAL_METHOD seqlen=$SEQLEN dtype=$EVAL_DTYPE"
  printf 'cell\tmethod\twikitext2\tc4\tptb_new\n'
} > "$SUMMARY"

keep_ckpt () {
  local tag=$1
  [ -z "$KEEP_CKPT" ] && return 1
  [ "$KEEP_CKPT" = "1" ] && return 0
  for k in $KEEP_CKPT; do [ "$k" = "$tag" ] && return 0; done
  return 1
}

summarize () {
  local tag=$1 method=$2 dir=$3
  [ -f "$dir/ppl.json" ] && cp "$dir/ppl.json" "$PPL_DIR/$tag.json"
  python - "$tag" "$method" "$PPL_DIR/$tag.json" "$SUMMARY" <<'PY'
import json, os, sys
tag, method, pj, summ = sys.argv[1:5]
ppl = {}
if os.path.exists(pj):
    try:
        ppl = json.load(open(pj)).get("ppl", {}) or {}
    except Exception as exc:
        print(f"!!! could not parse {pj}: {exc}")
def g(n):
    v = ppl.get(n)
    return f"{v:.4f}" if isinstance(v, (int, float)) else "NA"
with open(summ, "a") as f:
    f.write(f"{tag}\t{method}\t{g('wikitext2')}\t{g('c4')}\t{g('ptb-new')}\n")
PY
}

finish_cell () {
  local tag=$1 dir=$2 clog=$3
  echo ">>> [$tag] eval_ppl ($EVAL_METHOD, seqlen=$SEQLEN)"
  set +e
  python eval_ppl.py --model-path "$dir" $EVAL_COMMON --tag "$tag" 2>&1 | tee -a "$clog"
  set -e
  summarize "$tag" "$4" "$dir"
  if keep_ckpt "$tag"; then
    echo ">>> [$tag] KEEPING checkpoint at $dir"
  else
    local sz; sz=$(du -sh "$dir" 2>/dev/null | cut -f1)
    rm -rf "$dir"
    echo ">>> [$tag] deleted checkpoint (${sz:-?} reclaimed)"
  fi
  echo "<<< done $tag"
}

# --------------------------------------------------------------------------- #
# E: init only. Uses the existing quantize.py path, since codebookK is the
# SqueezeLLM-style k-means that lives there.
# --------------------------------------------------------------------------- #
cell_E () {
  local tag="E_codebook${BITS}" dir="$OUT_ROOT/E_codebook${BITS}"
  local clog="$CELL_LOG_DIR/$tag.log"
  echo; echo "############################################################"
  echo "# $tag   [k-means init only]   ($(date))"
  echo "############################################################"
  set +e
  python quantize.py --model-path "$MODEL_PATH" --quantizer "codebook${BITS}" \
    --cb-block-size -1 --n-calib $N_CALIB --max-length $CALIB_LEN \
    --no-ncc --output-dir "$dir" 2>&1 | tee "$clog"
  local rc=${PIPESTATUS[0]}
  set -e
  if [ "$rc" -ne 0 ] || [ ! -d "$dir" ]; then
    echo "!!! $tag: quantize exited $rc -- skipping eval"
    printf '%s\tkmeans\tFAIL\tFAIL\tFAIL\n' "$tag" >> "$SUMMARY"
    rm -rf "$dir"; return 0
  fi
  finish_cell "$tag" "$dir" "$clog" "kmeans"
}

# --------------------------------------------------------------------------- #
# N / G: the standalone driver. --nosal switches the Hessian between
# unweighted-single-group and saliency-weighted-per-group.
# --------------------------------------------------------------------------- #
cell_gq () {
  local tag=$1 method=$2; shift 2
  local dir="$OUT_ROOT/$tag" clog="$CELL_LOG_DIR/$tag.log"
  echo; echo "############################################################"
  echo "# $tag   [$method]   ($(date))"
  echo "############################################################"
  set +e
  python quantize_guidedquant.py \
    --model-path "$MODEL_PATH" --bits $BITS --num-groups $NUM_GROUPS \
    --n-calib $N_CALIB --seq-len $CALIB_LEN --calib-dataset $CALIB_DATASET \
    --iters $ITERS --cd-cycles $CD_CYCLES --ridge $RIDGE \
    --row-block $ROW_BLOCK --cd-block $CD_BLOCK \
    --kmeans-init $KMEANS_INIT --init-iters $INIT_ITERS \
    --row-chunk $ROW_CHUNK --saliency-dir "$SALIENCY_DIR" \
    --output-dir "$dir" --verbose "$@" 2>&1 | tee "$clog"
  local rc=${PIPESTATUS[0]}
  set -e
  if [ "$rc" -ne 0 ] || [ ! -d "$dir" ]; then
    echo "!!! $tag: quantize exited $rc -- skipping eval"
    printf '%s\t%s\tFAIL\tFAIL\tFAIL\n' "$tag" "$method" >> "$SUMMARY"
    rm -rf "$dir"; return 0
  fi
  finish_cell "$tag" "$dir" "$clog" "$method"
}

cell_E
# --nosal needs no saliency at all, so it must not read (or write) the shared
# cache -- the cached tensors are shaped by --num-groups and are irrelevant here.
SALIENCY_DIR_SAVED="$SALIENCY_DIR"
SALIENCY_DIR=""
cell_gq "N_nosal${BITS}"      "lnq-nosal"    --nosal
SALIENCY_DIR="$SALIENCY_DIR_SAVED"
cell_gq "G_guidedquant${BITS}" "guidedquant"

echo
echo "############################################################"
echo "# summary"
echo "############################################################"
grep '^#' "$SUMMARY"
grep -v '^#' "$SUMMARY" |
  awk -F'\t' '{printf "%-22s %-14s %-11s %-11s %s\n", $1,$2,$3,$4,$5}'

python - "$SUMMARY" "$BITS" <<'PY'
import sys
rows = {}
for line in open(sys.argv[1]):
    if line.startswith("#") or line.startswith("cell\t"):
        continue
    f = line.rstrip("\n").split("\t")
    if len(f) >= 3:
        rows[f[0]] = f[2]
b = sys.argv[2]
e, n, g = (rows.get(f"E_codebook{b}"), rows.get(f"N_nosal{b}"),
           rows.get(f"G_guidedquant{b}"))
def fl(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None
e, n, g = fl(e), fl(n), fl(g)
print()
if e is not None and n is not None:
    print(f"off-diagonal gain (E -> N): {e - n:+.4f}"
          + ("" if e - n > 0 else "   <- CD did not beat its own init"))
if n is not None and g is not None:
    print(f"saliency gain     (N -> G): {n - g:+.4f}"
          + ("" if n - g > 0 else "   <- end-loss weighting did not help"))
if e is not None and g is not None:
    print(f"total             (E -> G): {e - g:+.4f}")
PY

echo
echo "=== done $(date) ==="
echo "log:      $LOG_FILE"
echo "summary:  $SUMMARY"
