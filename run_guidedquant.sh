#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_guidedquant.sh -- LNQ ladder, full row, same protocol as run_flexnu.sh.
#
# WHAT THIS IS, PRECISELY. The method implemented is LNQ: alternating
# coordinate descent on the layer objective  sum_i dw_i^T H dw_i, with H the
# activation Gram G = E[x x^T] collected from calibration.
#
#   update_P  reassigns each input column against a Hessian-CORRECTED target,
#             so an assignment may be non-nearest-neighbour in raw weight
#             space. The off-diagonals of G are what make that possible.
#   update_C  re-solves the codebook by least squares in the whitened space
#             from the Cholesky factor of H.
#
# It is NOT GuidedQuant proper. GuidedQuant additionally weights H by end-loss
# saliency, which requires per-group output gradients from a backward pass over
# the calibration set. This repo collects G only, which is what GuidedQuant own
# code calls the ordinary LNQ Hessian "without GuidedQuant saliency". The
# script keeps the name for continuity with the paper being compared to, but
# the cells are labelled lnq* so no result gets written up as GuidedQuant.
#
# CELLS
#   E  codebookK        baseline: SqueezeLLM-style weighted k-means, no CD
#   L  lnqK             init = E, then LNQ alternation against the full Gram
#
# The pair is the experiment. E is exactly the LNQ initialisation, so
# (L - E) isolates what coordinate descent against the off-diagonals buys.
# If L does not beat E, the off-diagonal information is not helping and no
# amount of tuning the CD schedule will change that.
#
# WATCH: --lnq-verbose prints the objective every alternation. It must decrease
# monotonically. If it rises, the damping is too low for this Gram (raise
# --lnq-damp) -- that is a conditioning problem, not a quantizer bug.
# ---------------------------------------------------------------------------
# BEFORE THE LADDER: run the fp16 sanity check.
#
#   SANITY_ONLY=1 MODEL_PATH=meta-llama/Llama-2-7b-hf bash run_guidedquant.sh
#
# It must print wikitext2 PPL 5.47 at seqlen 2048. Off by >0.02 means a
# tokenization or model-loading bug, NOT a quantizer bug.
# ---------------------------------------------------------------------------
# DISK: each cell writes a full checkpoint (~14 GB for 7B). Each cell is
# quantize -> eval -> keep ppl.json -> rm -rf weights, so peak disk is ONE
# checkpoint. KEEP_CKPT=1 keeps all; KEEP_CKPT=L_lnq3 keeps just that cell.
#
# SURVIVES A RUN:
#   $LOG_DIR/guidedquant_<ts>.log          full stdout
#   $LOG_DIR/guidedquant_<ts>/<cell>.log   per-cell, for diffing
#   $OUT_ROOT/ppl/<cell>.json              full eval json
#   $OUT_ROOT/summary_<ts>.tsv             one row per cell
#
# GPU + torch required. Nothing is installed here. Run manually.
# ---------------------------------------------------------------------------
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-/path/to/Mistral-7B-v0.3}
OUT_ROOT=${OUT_ROOT:-./quantized_models/guidedquant}
LOG_DIR=${LOG_DIR:-./logs}
KEEP_CKPT=${KEEP_CKPT:-L_lnq3}          # ""=none, 1=all, or space-separated cell names

# ---- eval protocol --------------------------------------------------------
# seqlen 2048 for Llama-1/2 and Mistral; 8192 for Llama-3 / Qwen3. Numbers at
# different seqlen are NOT comparable, so it is recorded in every json and in
# the summary header.
SEQLEN=${SEQLEN:-2048}
EVAL_DATASETS=${EVAL_DATASETS:-"wikitext2 c4"}   # e.g. "wikitext2 c4 ptb-new"
EVAL_METHOD=${EVAL_METHOD:-block}             # block = paper standard
EVAL_STRIDE=${EVAL_STRIDE:-512}               # only used when method=sliding
EVAL_DTYPE=${EVAL_DTYPE:-fp16}
DATASET_CACHE=${DATASET_CACHE:-./dataset_cache}

# ---- sanity check ---------------------------------------------------------
# Reference fp16 wikitext2 PPL at seqlen 2048, from eval_ppl.py's docstring.
SANITY=${SANITY:-0}                # 1 = run it before the ladder
SANITY_ONLY=${SANITY_ONLY:-0}      # 1 = run it and exit
SANITY_STRICT=${SANITY_STRICT:-1}  # 1 = abort the ladder if it fails
SANITY_MODEL=${SANITY_MODEL:-}     # default: MODEL_PATH (the unquantized model)
SANITY_EXPECT=${SANITY_EXPECT:-}   # e.g. 5.47; empty = report only, no gate
SANITY_TOL=${SANITY_TOL:-0.02}

TS=$(date +%Y%m%d_%H%M%S)
CELL_LOG_DIR="$LOG_DIR/guidedquant_$TS"
PPL_DIR="$OUT_ROOT/ppl"
SUMMARY="$OUT_ROOT/summary_$TS.tsv"
mkdir -p "$LOG_DIR" "$CELL_LOG_DIR" "$OUT_ROOT" "$PPL_DIR"

LOG_FILE="$LOG_DIR/guidedquant_$TS.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== run started $(date) ==="
echo "model:    $MODEL_PATH"
echo "eval:     method=$EVAL_METHOD seqlen=$SEQLEN dtype=$EVAL_DTYPE "\
"datasets='$EVAL_DATASETS'"
[ "$EVAL_METHOD" = "sliding" ] && echo "          stride=$EVAL_STRIDE  "\
"(NOTE: sliding numbers are NOT comparable to paper tables)"
echo "log:      $LOG_FILE"
echo "per-cell: $CELL_LOG_DIR"
echo "ppl:      $PPL_DIR"
echo "keep:     ${KEEP_CKPT:-<none, checkpoints deleted after eval>}"

BITS=${BITS:-3}
# -1 = FULL ROW (default): one learned codebook per output row, spanning all
# input channels. Set CB_BLOCK=64 (or any positive int) for block-wise.
CB_BLOCK=${CB_BLOCK:--1}
N_CALIB=${N_CALIB:-128}
CALIB_LEN=${CALIB_LEN:-512}     # calibration seqlen; unrelated to eval SEQLEN

# ---- LNQ hyper-params -----------------------------------------------------
# Outer alternations. The objective typically flattens well before 15; the
# verbose trace tells you where, and there is no benefit to running past it.
LNQ_ITERS=${LNQ_ITERS:-15}
# CD sweeps over input columns inside each update_P. 2 matches the reference.
LNQ_CD_CYCLES=${LNQ_CD_CYCLES:-2}
# Hessian damping as a fraction of mean(diag(G)). Calibration with fewer
# samples than in_features gives a rank-deficient Gram, and the Cholesky in
# update_C needs positive definiteness. Raise this if the objective rises.
LNQ_DAMP=${LNQ_DAMP:-1e-2}
LNQ_RIDGE=${LNQ_RIDGE:-1e-7}

# Memory knobs. row-block bounds the [row_block, in, K] one-hot in update_C;
# cd-block bounds the residual update stride in update_P.
LNQ_ROW_BLOCK=${LNQ_ROW_BLOCK:-64}
LNQ_CD_BLOCK=${LNQ_CD_BLOCK:-128}
GRAM_GROUP=${GRAM_GROUP:-8}
GRAM_DEV=${GRAM_DEV:-cpu}

LNQ_COMMON="--cb-block-size $CB_BLOCK --n-calib $N_CALIB --max-length $CALIB_LEN \
  --no-ncc --lnq-iters $LNQ_ITERS --lnq-cd-cycles $LNQ_CD_CYCLES \
  --lnq-damp $LNQ_DAMP --lnq-ridge $LNQ_RIDGE \
  --lnq-row-block $LNQ_ROW_BLOCK --lnq-cd-block $LNQ_CD_BLOCK \
  --gram-group-size $GRAM_GROUP --gram-store-device $GRAM_DEV \
  --lnq-verbose"

EVAL_COMMON="--datasets $EVAL_DATASETS --seqlen $SEQLEN --method $EVAL_METHOD \
  --dtype $EVAL_DTYPE --cache-dir $DATASET_CACHE"
[ "$EVAL_METHOD" = "sliding" ] && EVAL_COMMON="$EVAL_COMMON --stride $EVAL_STRIDE"

# --------------------------------------------------------------------------- #
# fp16 sanity check. Tokenization bugs are silent and poison every cell, so
# this runs first and gates the ladder.
# --------------------------------------------------------------------------- #
run_sanity () {
  local sm="${SANITY_MODEL:-$MODEL_PATH}"
  local sj="$PPL_DIR/_sanity_fp16.json"
  local sl="$CELL_LOG_DIR/_sanity_fp16.log"

  echo
  echo "############################################################"
  echo "# SANITY  fp16 baseline: $sm"
  echo "############################################################"
  echo "# Reference: Llama-2-7B wikitext2 seqlen2048 -> 5.47, 13B -> 4.88."
  echo "# A miss here is tokenization or model loading, NOT the quantizer."

  set +e
  python eval_ppl.py --model-path "$sm" $EVAL_COMMON \
    --tag "_sanity_fp16" --out-json "$sj" 2>&1 | tee "$sl"
  local rc=${PIPESTATUS[0]}
  set -e
  if [ "$rc" -ne 0 ]; then
    echo "!!! sanity eval exited $rc"
    [ "$SANITY_STRICT" = "1" ] && { echo "aborting (SANITY_STRICT=1)"; exit 1; }
    return 0
  fi

  if [ -n "$SANITY_EXPECT" ]; then
    python - "$sj" "$SANITY_EXPECT" "$SANITY_TOL" "$SANITY_STRICT" <<'PY'
import json, sys
js, expect, tol, strict = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
d = json.load(open(js))
got = d.get("ppl", {}).get("wikitext2")
if got is None:
    print("!!! sanity: no wikitext2 PPL in json")
    sys.exit(1 if strict == "1" else 0)
delta = abs(got - expect)
print(f"sanity: wikitext2 PPL {got:.4f}  expected {expect:.4f}  |delta|={delta:.4f}")
if delta > tol:
    print(f"!!! SANITY FAILED: off by more than {tol}. Fix tokenization/loading "
          f"before trusting any ladder number.")
    sys.exit(1 if strict == "1" else 0)
print("sanity: OK")
PY
  else
    echo "sanity: no SANITY_EXPECT set -- reported only, not gated."
    echo "        (set SANITY_EXPECT=5.47 for Llama-2-7B to enforce it)"
  fi
}

if [ "$SANITY_ONLY" = "1" ]; then
  run_sanity
  echo "=== sanity-only run complete $(date) ==="
  exit 0
fi
[ "$SANITY" = "1" ] && run_sanity

# Header records the protocol, since PPL at different seqlen/method is not
# comparable and a bare table of numbers loses that.
{
  if [ "$CB_BLOCK" -le 0 ]; then CB_BLOCK_STR="full_row"; else CB_BLOCK_STR="$CB_BLOCK"; fi
  echo "# model=$MODEL_PATH bits=$BITS block=$CB_BLOCK_STR"
  if [ "$EVAL_METHOD" = "sliding" ]; then
    echo "# eval: method=sliding seqlen=$SEQLEN stride=$EVAL_STRIDE dtype=$EVAL_DTYPE"
  else
    echo "# eval: method=block seqlen=$SEQLEN dtype=$EVAL_DTYPE"
  fi
  echo "# lnq: iters=$LNQ_ITERS cd_cycles=$LNQ_CD_CYCLES damp=$LNQ_DAMP ridge=$LNQ_RIDGE"
  printf 'cell\tquantizer\twikitext2\tc4\tptb_new\tenergy_drop_pct\n'
} > "$SUMMARY"

keep_ckpt () {
  local tag=$1
  [ -z "$KEEP_CKPT" ] && return 1
  [ "$KEEP_CKPT" = "1" ] && return 0
  for k in $KEEP_CKPT; do [ "$k" = "$tag" ] && return 0; done
  return 1
}

# --------------------------------------------------------------------------- #
# cell <tag> <quantizer> [extra quantize.py args...]
#   quantize -> eval -> keep json -> delete weights
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
  # Never let one cell abort the ladder; record it and continue.
  set +e
  if [ "$quant" = "codebook${BITS}" ]; then
    python quantize.py \
      --model-path "$MODEL_PATH" --quantizer "$quant" \
      --cb-block-size $CB_BLOCK --n-calib $N_CALIB --max-length $CALIB_LEN \
      --no-ncc --output-dir "$dir" "$@" 2>&1 | tee "$clog"
  else
    python quantize.py \
      --model-path "$MODEL_PATH" --quantizer "$quant" \
      $LNQ_COMMON --output-dir "$dir" "$@" 2>&1 | tee "$clog"
  fi
  local q_rc=${PIPESTATUS[0]}
  set -e

  if [ "$q_rc" -ne 0 ] || [ ! -d "$dir" ]; then
    echo "!!! $tag: quantize exited $q_rc / checkpoint missing -- skipping eval"
    printf '%s\t%s\tFAIL\tFAIL\tFAIL\tNA\n' "$tag" "$quant" >> "$SUMMARY"
    rm -rf "$dir"
    return 0
  fi

  # ---- 2. eval ------------------------------------------------------------
  echo ">>> [$tag] eval_ppl ($EVAL_METHOD, seqlen=$SEQLEN)"
  set +e
  python eval_ppl.py --model-path "$dir" $EVAL_COMMON --tag "$tag" \
    2>&1 | tee -a "$clog"
  local e_rc=${PIPESTATUS[0]}
  set -e
  [ "$e_rc" -ne 0 ] && echo "!!! $tag: eval exited $e_rc"

  # ---- 3. keep the small artifacts ---------------------------------------
  [ -f "$dir/ppl.json" ] && cp "$dir/ppl.json" "$PPL_DIR/$tag.json"

  # Summary row. PPLs come from the json's nested "ppl" dict; the energy drop
  # is scraped from the quantize log.
  python - "$tag" "$quant" "$PPL_DIR/$tag.json" "$clog" "$SUMMARY" <<'PY'
import json, os, re, sys
tag, quant, pj, clog, summ = sys.argv[1:6]

ppl = {}
if os.path.exists(pj):
    try:
        ppl = json.load(open(pj)).get("ppl", {}) or {}
    except Exception as exc:
        print(f"!!! could not parse {pj}: {exc}")

def g(name):
    v = ppl.get(name)
    return f"{v:.4f}" if isinstance(v, (int, float)) else "NA"

drop = "NA"
if os.path.exists(clog):
    m = re.findall(r"\(([-+]?\d+\.\d+)% drop\)", open(clog, errors="ignore").read())
    if m:
        drop = m[-1]

with open(summ, "a") as f:
    f.write(f"{tag}\t{quant}\t{g('wikitext2')}\t{g('c4')}\t{g('ptb-new')}\t{drop}\n")
PY

  # ---- 4. delete the weights ---------------------------------------------
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
# The ladder. E first: it is the LNQ initialisation, so it must be on record
# before the CD result means anything.
# --------------------------------------------------------------------------- #
cell "E_codebook${BITS}"  "codebook${BITS}"
cell "L_lnq${BITS}"       "lnq${BITS}"

echo
echo "############################################################"
echo "# summary"
echo "############################################################"
grep '^#' "$SUMMARY"
grep -v '^#' "$SUMMARY" |
  awk -F'\t' '{printf "%-18s %-12s %-11s %-11s %-11s %s\n", $1,$2,$3,$4,$5,$6}'

# L must beat E. E is the LNQ initialisation, so if coordinate descent against
# the full Gram does not improve on it, the off-diagonals are not buying
# anything here -- flag that rather than bury it.
python - "$SUMMARY" "$BITS" <<'PY'
import sys
rows = {}
for line in open(sys.argv[1]):
    if line.startswith("#") or line.startswith("cell\t"):
        continue
    f = line.rstrip("\n").split("\t")
    if len(f) >= 3:
        rows[f[0]] = f[2]
bits = sys.argv[2]
e, l = rows.get(f"E_codebook{bits}"), rows.get(f"L_lnq{bits}")
try:
    if e and l and e != "NA" and l != "NA":
        e, l = float(e), float(l)
        d = e - l
        print(f"\ncheck: E={e:.4f}  L={l:.4f}  improvement={d:+.4f}"
              + ("  OK" if d > 0 else
                 "  !! LNQ did not beat its own init -- off-diagonals not "
                 "helping, or damping too high"))
except ValueError:
    pass
PY

echo
echo "=== done $(date) ==="
echo "log:      $LOG_FILE"
echo "per-cell: $CELL_LOG_DIR"
echo "ppl:      $PPL_DIR"
echo "summary:  $SUMMARY"
du -sh "$OUT_ROOT" 2>/dev/null || true