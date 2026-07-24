#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# sweep_flexnu.sh -- grid search over FlexNu hyper-params.
#
# Wraps run_flexnu.sh rather than reimplementing it: each grid point is one
# invocation with different env vars, so quantize/eval/summary logic stays in
# one place. Runs cell D_joint only by default -- cell E ignores every
# --flexnu-* flag, so it would produce an identical number for every point.
#
# Usage:
#   bash sweep_flexnu.sh                 # run the grid
#   DRY_RUN=1 bash sweep_flexnu.sh       # list points, run nothing
#   RESUME=1  bash sweep_flexnu.sh       # skip points already done
#
# Grid axes (space-separated, override from the environment):
#   LR_CB_GRID LR_SCALE_GRID TAU_GRID STAGE_GRID ITERS_GRID
#
# Example -- narrow sweep on the two axes that matter most:
#   LR_CB_GRID="1e-5 3e-5" LR_SCALE_GRID="1e-3 3e-3 1e-2" \
#   TAU_GRID="0.5" STAGE_GRID="0.0" bash sweep_flexnu.sh
# ---------------------------------------------------------------------------
set -euo pipefail

cd "$(dirname "$0")"

# ---- fixed config, applied to every point ---------------------------------
# These are held constant so the grid isolates the hyper-params. Anything set
# in the caller's environment wins.
export MODEL_PATH=${MODEL_PATH:-Llama-3.2-1B}
export BITS=${BITS:-4}
export CALIB_DATASET=${CALIB_DATASET:-c4}
export CALIB_LEN=${CALIB_LEN:-2048}
export N_CALIB=${N_CALIB:-128}
export GRAM_DEV=${GRAM_DEV:-cuda}
export ROW_BLOCK=${ROW_BLOCK:-2048}
export EVAL_EVERY=${EVAL_EVERY:-10}
export CELLS=${CELLS:-"D"}
# Checkpoints are ~1GB each and a grid produces dozens. Off by default; set
# KEEP_CKPT=D_joint to keep them (and watch the disk).
export KEEP_CKPT=${KEEP_CKPT:-}
# Sanity eval is a fixed fp16 number, identical for every point. Run it once
# up front instead of N times.
export SANITY=${SANITY:-0}

# ---- the grid -------------------------------------------------------------
# Defaults encode what the run_flexnu.sh comments already establish:
#   lr_cb  -- degrades monotonically past ~3e-4, so the grid stays below it
#   stage  -- 1.0 collapses the effect to 13%; only partial staging is probed
LR_CB_GRID=${LR_CB_GRID:-"3e-6 1e-5 3e-5 1e-4"}
LR_SCALE_GRID=${LR_SCALE_GRID:-"1e-3 3e-3 1e-2"}
TAU_GRID=${TAU_GRID:-"0.3 0.5 0.7"}
STAGE_GRID=${STAGE_GRID:-"0.0 0.25"}
ITERS_GRID=${ITERS_GRID:-"400"}

SWEEP_ROOT=${SWEEP_ROOT:-./sweeps}
# SWEEP_NAME makes the output dir stable, which is what RESUME needs -- a
# timestamped dir is fresh every invocation, so resume could never find prior
# work. Anonymous runs still get a timestamp so they do not collide.
if [ -n "${SWEEP_NAME:-}" ]; then
  SWEEP_DIR="$SWEEP_ROOT/$SWEEP_NAME"
elif [ "${RESUME:-0}" = "1" ]; then
  SWEEP_DIR="$SWEEP_ROOT/flexnu_resume"
  echo "note: RESUME=1 without SWEEP_NAME -- using $SWEEP_DIR"
else
  SWEEP_DIR="$SWEEP_ROOT/flexnu_$(date +%Y%m%d_%H%M%S)"
fi
SWEEP_TSV="$SWEEP_DIR/grid.tsv"
mkdir -p "$SWEEP_DIR"

# ---- enumerate ------------------------------------------------------------
POINTS=()
for it in $ITERS_GRID; do
for lc in $LR_CB_GRID; do
for ls in $LR_SCALE_GRID; do
for tf in $TAU_GRID; do
for sf in $STAGE_GRID; do
  POINTS+=("$it|$lc|$ls|$tf|$sf")
done; done; done; done; done

N=${#POINTS[@]}

echo "############################################################"
echo "# FlexNu sweep -- $N grid points"
echo "############################################################"
echo "model:    $MODEL_PATH   bits=$BITS"
echo "calib:    $N_CALIB x $CALIB_LEN ($CALIB_DATASET)"
echo "cells:    $CELLS"
echo "perf:     row_block=$ROW_BLOCK eval_every=$EVAL_EVERY gram_dev=$GRAM_DEV"
echo "out:      $SWEEP_DIR"
echo
echo "axes:"
echo "  iters     : $ITERS_GRID"
echo "  lr_cb     : $LR_CB_GRID"
echo "  lr_scale  : $LR_SCALE_GRID"
echo "  tau_frac  : $TAU_GRID"
echo "  stage_frac: $STAGE_GRID"
echo

if [ "${DRY_RUN:-0}" = "1" ]; then
  printf '%s\n' "${POINTS[@]}" | tr '|' '\t' | nl
  echo
  echo "DRY_RUN=1 -- nothing executed."
  exit 0
fi

# One fp16 sanity number for the whole sweep, not one per point.
if [ "${SWEEP_SANITY:-1}" = "1" ]; then
  echo ">>> fp16 sanity (once for the sweep)"
  SANITY_ONLY=1 SANITY=1 bash run_flexnu.sh 2>&1 \
    | tee "$SWEEP_DIR/_sanity.log" || echo "!!! sanity failed; continuing"
fi

if [ ! -f "$SWEEP_TSV" ]; then
  printf 'idx\titers\tlr_cb\tlr_scale\ttau\tstage\tcell\twikitext2\tc4\tptb_new\tenergy_drop_pct\trun_dir\n' \
    > "$SWEEP_TSV"
fi

i=0
START_ALL=$(date +%s)
for pt in "${POINTS[@]}"; do
  i=$((i+1))
  IFS='|' read -r it lc ls tf sf <<< "$pt"

  tag=$(printf 'i%s_cb%s_sc%s_t%s_st%s' "$it" "$lc" "$ls" "$tf" "$sf")
  run_dir="$SWEEP_DIR/$tag"

  # RESUME: a point is done if it produced a summary with a data row.
  if [ "${RESUME:-0}" = "1" ] && [ -d "$run_dir" ] && \
     ls "$run_dir"/summary_*.tsv >/dev/null 2>&1 && \
     grep -qv '^#' "$run_dir"/summary_*.tsv 2>/dev/null; then
    echo ">>> [$i/$N] $tag -- already done, skipping (RESUME=1)"
    continue
  fi

  echo
  echo "############################################################"
  echo "# [$i/$N] $tag   ($(date))"
  echo "#   iters=$it lr_cb=$lc lr_scale=$ls tau=$tf stage=$sf"
  echo "############################################################"
  # Remove any previous summary for this point. Without this a failed run
  # would be scraped against stale numbers from an earlier attempt and
  # silently reported as a success.
  rm -rf "$run_dir"
  mkdir -p "$run_dir"

  START=$(date +%s)
  # Each point gets its own OUT_ROOT/LOG_DIR so summaries never collide.
  set +e
  ITERS="$it" LR_CB="$lc" LR_SCALE="$ls" TAU_FRAC="$tf" STAGE_FRAC="$sf" \
  OUT_ROOT="$run_dir" LOG_DIR="$run_dir/logs" SANITY=0 \
    bash run_flexnu.sh 2>&1 | tee "$run_dir/run.log"
  rc=${PIPESTATUS[0]}
  set -e
  ELAPSED=$(( $(date +%s) - START ))

  if [ "$rc" -ne 0 ]; then
    echo "!!! [$i/$N] $tag exited $rc"
  fi
  echo ">>> [$i/$N] $tag done in ${ELAPSED}s (rc=$rc)"

  # Scrape this point's summary rows into the grid table.
  python - "$i" "$it" "$lc" "$ls" "$tf" "$sf" "$run_dir" "$SWEEP_TSV" "$rc" <<'PY'
import glob, os, sys

idx, it, lc, ls, tf, sf, run_dir, out_tsv, rc = sys.argv[1:10]

rows = []
if rc == "0":
    for path in sorted(glob.glob(os.path.join(run_dir, "summary_*.tsv"))):
        with open(path) as fh:
            for line in fh:
                line = line.rstrip("\n")
                if not line or line.startswith("#") or line.startswith("cell\t"):
                    continue
                rows.append(line.split("\t"))

if not rows:
    rows = [[f"FAIL_rc{rc}", "", "FAIL", "FAIL", "FAIL", "NA"]]

with open(out_tsv, "a") as fh:
    for r in rows:
        # summary row is: cell quantizer wikitext2 c4 ptb_new energy_drop
        cell = r[0] if len(r) > 0 else "?"
        wt   = r[2] if len(r) > 2 else "NA"
        c4   = r[3] if len(r) > 3 else "NA"
        ptb  = r[4] if len(r) > 4 else "NA"
        en   = r[5] if len(r) > 5 else "NA"
        fh.write("\t".join([idx, it, lc, ls, tf, sf, cell,
                            wt, c4, ptb, en, run_dir]) + "\n")
PY
done

TOTAL=$(( $(date +%s) - START_ALL ))
echo
echo "############################################################"
echo "# sweep complete -- ${TOTAL}s for $N points"
echo "############################################################"

# ---- ranked report --------------------------------------------------------
python - "$SWEEP_TSV" <<'PY'
import sys

path = sys.argv[1]
rows = []
with open(path) as fh:
    header = fh.readline().rstrip("\n").split("\t")
    for line in fh:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 11:
            continue
        rows.append(parts)

def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None

ok = [r for r in rows if fnum(r[7]) is not None]
bad = [r for r in rows if fnum(r[7]) is None]

if not ok:
    print("No successful points to rank.")
    sys.exit(0)

ok.sort(key=lambda r: fnum(r[7]))

print(f"\nRanked by wikitext2 PPL ({len(ok)} ok, {len(bad)} failed):\n")
hdr = f"{'#':>3}  {'iters':>5} {'lr_cb':>7} {'lr_scale':>8} {'tau':>4} {'stage':>5}  {'wt2':>8} {'c4':>8} {'ptb':>8} {'drop%':>6}"
print(hdr)
print("-" * len(hdr))
for n, r in enumerate(ok[:25], 1):
    print(f"{n:>3}  {r[1]:>5} {r[2]:>7} {r[3]:>8} {r[4]:>4} {r[5]:>5}  "
          f"{r[7]:>8} {r[8]:>8} {r[9]:>8} {r[10]:>6}")

best = ok[0]
print(f"\nBest: lr_cb={best[2]} lr_scale={best[3]} tau={best[4]} "
      f"stage={best[5]} iters={best[1]}  ->  wikitext2 {best[7]}")

# Per-axis marginals: median PPL holding one axis fixed. With a full grid this
# is a fair first cut at which axis actually moves the metric.
import statistics
print("\nPer-axis median wikitext2 (lower is better):")
for col, name in ((1, "iters"), (2, "lr_cb"), (3, "lr_scale"),
                  (4, "tau"), (5, "stage")):
    vals = {}
    for r in ok:
        vals.setdefault(r[col], []).append(fnum(r[7]))
    if len(vals) < 2:
        continue
    print(f"  {name}:")
    for k in sorted(vals, key=lambda k: statistics.median(vals[k])):
        print(f"    {k:>8} -> {statistics.median(vals[k]):.4f}  (n={len(vals[k])})")

if bad:
    print(f"\n{len(bad)} failed points:")
    for r in bad[:10]:
        print(f"  iters={r[1]} lr_cb={r[2]} lr_scale={r[3]} tau={r[4]} stage={r[5]}")
PY

echo
echo "grid tsv: $SWEEP_TSV"
