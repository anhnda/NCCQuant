#!/usr/bin/env python3
"""
Synthetic validation for FlexNu -- reproduces Appendix A of the paper.

Runs three experiments:

  main    Table 5. Ablation ladder (B / C / D) across three Gram regimes.
          The headline result: on anisotropic G the divisor (C) beats the
          codebook (B) by ~6x, and on isotropic G both are exactly zero.

  lrcb    Table 6. Sensitivity to the codebook learning rate at fixed
          lr_scale. Past ~3e-4 the codebook chases W and drags the
          thresholds out from under the divisor.

  stage   Table 7. Sensitivity to stage_frac. Partial staging is benign;
          full staging (stage_frac=1.0, i.e. codebook only, no divisor
          phase) collapses the effect.

Why the three Gram regimes matter
---------------------------------
The whole mechanism is governed by the OFF-DIAGONAL structure of
G = E[x x^T]. Proposition 1 of the paper: if G is diagonal, the loss
tr(R G R^T) separates across input coordinates and nearest-codeword
assignment is provably optimal -- so there is nothing for the divisor to
find. The three regimes span that axis deliberately:

    iid       G ~ I,        kappa ~ 4e1   -> expect EXACTLY zero (control)
    corr      random SPD,   kappa ~ 6e2   -> expect small
    lowrank   rank-16+noise, kappa ~ 7e4  -> expect large

The `iid` row is a positive control for the theory, not a weak baseline. If
it is ever nonzero, something is wrong with the implementation. Equally, it
is a warning: benchmarking this method on white-noise activations measures
nothing.

CPU-only and self-contained (a few minutes for the default grid). Nothing
here touches a real model.

Usage
-----
    python run_synthetic.py                      # all three, paper defaults
    python run_synthetic.py --exp main           # just the ladder
    python run_synthetic.py --seeds 5            # tighter error bars
    python run_synthetic.py --latex              # emit LaTeX tables
    python run_synthetic.py --json out.json      # machine-readable dump
    python run_synthetic.py --device cuda        # if you want it on GPU

Numbers in the paper were produced with the defaults below:
    --out-features 64 --in-features 256 --block-size 64 --bits 3
    --iters 600 --seeds 3
"""

from __future__ import annotations

import argparse
import json
import statistics as stat
import sys
from typing import Callable, Dict, List, Tuple

import torch

sys.path.insert(0, ".")
from quantizers import get_quantizer                      # noqa: E402
from quantizers.flexnu import FlexNuQuantizer             # noqa: E402


# --------------------------------------------------------------------------- #
# Problem generation
# --------------------------------------------------------------------------- #
def make_weights(out_f: int, in_f: int, seed: int, std: float,
                 device: str) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    return (torch.randn(out_f, in_f, generator=g) * std).to(device)


def make_gram(kind: str, in_f: int, seed: int, n_tokens: int,
              rank: int, mean_shift: float, device: str) -> torch.Tensor:
    """Second-moment Gram G = E[x x^T] for one of three anisotropy regimes.

    A nonzero mean_shift is deliberate: real activations are not centred, and
    G = Sigma + mu mu^T picks up a rank-one term from the mean. Setting it to
    zero makes all three regimes noticeably more benign.
    """
    g = torch.Generator(device="cpu").manual_seed(seed + 999)

    if kind == "iid":
        X = torch.randn(n_tokens, in_f, generator=g)

    elif kind == "corr":
        # Random SPD covariance via a Wishart-like construction; the ridge
        # keeps it invertible so kappa stays moderate.
        A = torch.randn(in_f, in_f, generator=g) / in_f ** 0.5
        C = A @ A.t() + 0.05 * torch.eye(in_f)
        X = torch.randn(n_tokens, in_f, generator=g) @ torch.linalg.cholesky(C).t()

    elif kind == "lowrank":
        # Rank-`rank` signal plus small isotropic noise. This is the regime
        # meant to resemble real activation statistics, which are dominated
        # by a few directions.
        U = torch.randn(in_f, rank, generator=g)
        X = (torch.randn(n_tokens, rank, generator=g) @ U.t()
             + 0.1 * torch.randn(n_tokens, in_f, generator=g))

    else:
        raise ValueError(f"unknown Gram regime {kind!r}")

    X = X + mean_shift
    return ((X.t() @ X) / n_tokens).to(device)


def cond_number(G: torch.Tensor) -> float:
    ev = torch.linalg.eigvalsh(G.double())
    return float(ev.max() / ev.clamp(min=1e-30).min())


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def energy(W_hat: torch.Tensor, W: torch.Tensor, G: torch.Tensor) -> float:
    """tr(R G R^T), the layer reconstruction objective."""
    R = (W_hat.float() - W.float())
    return float((R @ G * R).sum())


def moved_fraction(idx_a: torch.Tensor, idx_b: torch.Tensor) -> float:
    """Fraction of weights assigned to a different codeword. THE diagnostic:
    it directly measures whether the escape from nearest-neighbour is
    happening at all."""
    return 100.0 * float((idx_a != idx_b).float().mean())


# --------------------------------------------------------------------------- #
# One trial
# --------------------------------------------------------------------------- #
def run_cell(W: torch.Tensor, G: torch.Tensor, args, *,
             freeze_codebook: bool, freeze_scale: bool,
             lr_cb: float = None, lr_scale: float = None,
             stage_frac: float = None, seed: int = 0):
    q = FlexNuQuantizer(
        bits=args.bits,
        block_size=args.block_size,
        iters=args.iters,
        lr_cb=args.lr_cb if lr_cb is None else lr_cb,
        lr_scale=args.lr_scale if lr_scale is None else lr_scale,
        tau_frac=args.tau_frac,
        freeze_codebook=freeze_codebook,
        freeze_scale=freeze_scale,
        stage_frac=args.stage_frac if stage_frac is None else stage_frac,
        row_block=args.row_block,
        seed=seed,
    )
    q.set_gram(G)
    return q.quantize(W), q.last_diag


def baseline(W: torch.Tensor, args):
    """Per-block 1-D k-means codebook with nearest-codeword assignment: the
    existing method FlexNu is measured against."""
    name = f"codebook{args.bits}" if args.bits in (3, 4) else None
    if name is None:
        # codebookK only registers 3/4 bits; fall back to a frozen FlexNu,
        # which is the same thing (k-means init, nothing learned).
        q = FlexNuQuantizer(bits=args.bits, block_size=args.block_size, iters=0,
                            freeze_codebook=True, freeze_scale=True,
                            row_block=args.row_block)
        q.set_gram(None)
        return q.quantize(W)
    return get_quantizer(name, cb_block_size=args.block_size).quantize(W)


def agg(vals: List[float]) -> Tuple[float, float]:
    return stat.mean(vals), (stat.pstdev(vals) if len(vals) > 1 else 0.0)


def fmt(m: float, s: float, sign: bool = True) -> str:
    return f"{m:+.2f} ± {s:.2f}" if sign else f"{m:.2f} ± {s:.2f}"


# --------------------------------------------------------------------------- #
# Experiment 1: the ablation ladder (paper Table 5)
# --------------------------------------------------------------------------- #
def exp_main(args) -> Dict:
    print("\n" + "=" * 78)
    print("EXPERIMENT: ablation ladder across Gram regimes  (paper Table 5)")
    print("=" * 78)
    print("  B = codebook only (assignment stays nearest)  -> FITTING W")
    print("  C = divisor only  (assignment escapes)        -> ESCAPING via loss")
    print("  D = joint")
    print("  A = both frozen; sanity check, must equal baseline exactly")
    print("  values are % change in tr(R G R^T) vs k-means baseline "
          "(more negative = better)\n")

    out = {}
    for kind in args.regimes:
        acc = {k: [] for k in ("A", "B", "C", "D", "cond", "moved", "e_ref")}
        for seed in range(args.seeds):
            W = make_weights(args.out_features, args.in_features, seed,
                             args.weight_std, args.device)
            G = make_gram(kind, args.in_features, seed, args.n_tokens,
                          args.rank, args.mean_shift, args.device)
            acc["cond"].append(cond_number(G))

            ref = baseline(W, args)
            e_ref = energy(ref.W_dequant, W, G)
            acc["e_ref"].append(e_ref)

            for tag, fc, fs in (("A", True, True), ("B", False, True),
                                ("C", True, False), ("D", False, False)):
                r, _ = run_cell(W, G, args, freeze_codebook=fc,
                                freeze_scale=fs, seed=seed)
                acc[tag].append(100.0 * (energy(r.W_dequant, W, G) - e_ref) / e_ref)
                if tag == "C":
                    acc["moved"].append(moved_fraction(r.indices, ref.indices))

            # Structural invariant: the committed codebook must stay strictly
            # increasing, or the telescoping reconstruction is invalid and any
            # downstream neighbour query breaks.
            cb = r.block_codebooks
            assert bool((cb[..., 1:] > cb[..., :-1]).all()), \
                "codebook not strictly increasing"

        res = {k: agg(v) for k, v in acc.items()}
        out[kind] = res
        print(f"  {kind:8s} kappa={res['cond'][0]:.2e}  "
              f"A={fmt(*res['A'])}  B={fmt(*res['B'])}  "
              f"C={fmt(*res['C'])}  D={fmt(*res['D'])}  "
              f"moved={res['moved'][0]:.1f}%")

        if kind == "iid":
            # Positive control for Proposition 1: under (near-)diagonal G the
            # loss separates across coordinates and nearest-codeword assignment
            # is optimal, so EVERY learning cell must come out at ~0. A nonzero
            # value here means the implementation is finding "gains" that the
            # theory forbids, i.e. something is wrong -- not that the method is
            # unexpectedly good.
            for tag in ("B", "C", "D"):
                if abs(res[tag][0]) > 0.05:
                    print(f"    !! WARNING: cell {tag} = {res[tag][0]:+.2f}% "
                          f"under isotropic G. Proposition 1 requires ~0; "
                          f"check the implementation.")
        if abs(res["A"][0]) > 1e-3:
            print(f"    !! WARNING: cell A deviates from baseline by "
                  f"{res['A'][0]:+.4f}% -- should be exactly 0.")
    return out


# --------------------------------------------------------------------------- #
# Experiment 2: codebook learning rate (paper Table 6)
# --------------------------------------------------------------------------- #
def exp_lrcb(args) -> Dict:
    print("\n" + "=" * 78)
    print(f"EXPERIMENT: codebook LR sweep on '{args.sweep_regime}' G  "
          f"(paper Table 6)")
    print("=" * 78)
    print(f"  lr_scale fixed at {args.lr_scale:g}; energy reduction %, "
          "higher is better\n")

    out = {}
    for lr_cb in args.lrcb_grid:
        vals = []
        for seed in range(args.seeds):
            W = make_weights(args.out_features, args.in_features, seed,
                             args.weight_std, args.device)
            G = make_gram(args.sweep_regime, args.in_features, seed,
                          args.n_tokens, args.rank, args.mean_shift, args.device)
            # lr_cb == 0 is expressed by freezing, not by a zero LR: Adam with
            # lr=0 still advances its moment estimates, which is not the same
            # thing as holding the codebook fixed.
            _, d = run_cell(W, G, args, freeze_codebook=(lr_cb == 0.0),
                            freeze_scale=False, lr_cb=lr_cb, stage_frac=0.0,
                            seed=seed)
            vals.append(100.0 * d["energy_rel_drop"])
        out[lr_cb] = agg(vals)
        print(f"  lr_cb={lr_cb:<9g} drop = {fmt(*out[lr_cb], sign=False)}")
    return out


# --------------------------------------------------------------------------- #
# Experiment 3: staging (paper Table 7)
# --------------------------------------------------------------------------- #
def exp_stage(args) -> Dict:
    print("\n" + "=" * 78)
    print(f"EXPERIMENT: stage_frac sweep on '{args.sweep_regime}' G  "
          f"(paper Table 7)")
    print("=" * 78)
    print("  stage_frac = fraction of steps on the codebook before switching")
    print("  to the divisor. 1.0 = codebook only, no divisor phase at all.\n")

    out = {}
    for sf in args.stage_grid:
        vals = []
        for seed in range(args.seeds):
            W = make_weights(args.out_features, args.in_features, seed,
                             args.weight_std, args.device)
            G = make_gram(args.sweep_regime, args.in_features, seed,
                          args.n_tokens, args.rank, args.mean_shift, args.device)
            _, d = run_cell(W, G, args, freeze_codebook=False,
                            freeze_scale=False, stage_frac=sf, seed=seed)
            vals.append(100.0 * d["energy_rel_drop"])
        out[sf] = agg(vals)
        print(f"  stage_frac={sf:<6g} drop = {fmt(*out[sf], sign=False)}")
    return out


# --------------------------------------------------------------------------- #
# LaTeX emission (drops straight into the paper appendix)
# --------------------------------------------------------------------------- #
def emit_latex(main, lrcb, stage, args) -> str:
    L = []
    if main:
        L += [r"\begin{table}[h]", r"\centering",
              r"\caption{Reconstruction energy change vs.\ $k$-means baseline.}",
              r"\label{tab:main}", r"\begin{tabular}{lrrrrr}", r"\toprule",
              r"$G$ regime & $\kappa(G)$ & B (codebook) & C (divisor) "
              r"& D (joint) & moved \\", r"\midrule"]
        for k, v in main.items():
            c = v["cond"][0]
            mant, ex = f"{c:.1e}".split("e")
            L.append(f"{k:8s} & ${mant}\\times10^{{{int(ex)}}}$ & "
                     f"${v['B'][0]:.2f} \\pm {v['B'][1]:.2f}$ & "
                     f"${v['C'][0]:.2f} \\pm {v['C'][1]:.2f}$ & "
                     f"${v['D'][0]:.2f} \\pm {v['D'][1]:.2f}$ & "
                     f"${v['moved'][0]:.1f}\\%$ \\\\")
        L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    if lrcb:
        ks = list(lrcb)
        L += [r"\begin{table}[h]", r"\centering",
              rf"\caption{{Effect of $\eta_{{\text{{cb}}}}$ at "
              rf"$\eta_{{\text{{scale}}}}={args.lr_scale:g}$.}}",
              r"\label{tab:lrcb}",
              r"\begin{tabular}{l" + "r" * len(ks) + "}", r"\toprule",
              r"$\eta_{\text{cb}}$ & " +
              " & ".join(f"${k:g}$" for k in ks) + r" \\", r"\midrule",
              r"drop (\%) & " +
              " & ".join(f"${lrcb[k][0]:.1f} \\pm {lrcb[k][1]:.1f}$" for k in ks)
              + r" \\",
              r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    if stage:
        ks = list(stage)
        L += [r"\begin{table}[h]", r"\centering",
              r"\caption{Effect of \texttt{stage\_frac}.}", r"\label{tab:stage}",
              r"\begin{tabular}{l" + "r" * len(ks) + "}", r"\toprule",
              r"\texttt{stage\_frac} & " +
              " & ".join(f"${k:g}$" for k in ks) + r" \\", r"\midrule",
              r"drop (\%) & " +
              " & ".join(f"${stage[k][0]:.1f} \\pm {stage[k][1]:.1f}$" for k in ks)
              + r" \\",
              r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(L)


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(
        description="Synthetic validation for FlexNu (paper Appendix A)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--exp", nargs="+", default=["main", "lrcb", "stage"],
                   choices=["main", "lrcb", "stage"])
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--device", default="cpu")

    g = p.add_argument_group("problem")
    g.add_argument("--out-features", type=int, default=64)
    g.add_argument("--in-features", type=int, default=256)
    g.add_argument("--block-size", type=int, default=64)
    g.add_argument("--bits", type=int, default=3, choices=[2, 3, 4])
    g.add_argument("--weight-std", type=float, default=0.05)
    g.add_argument("--n-tokens", type=int, default=4096)
    g.add_argument("--rank", type=int, default=16,
                   help="signal rank for the 'lowrank' regime")
    g.add_argument("--mean-shift", type=float, default=0.3,
                   help="activation mean; G picks up a rank-one mu mu^T term")
    g.add_argument("--regimes", nargs="+",
                   default=["iid", "corr", "lowrank"],
                   choices=["iid", "corr", "lowrank"])

    g = p.add_argument_group("FlexNu")
    g.add_argument("--iters", type=int, default=600)
    g.add_argument("--lr-cb", type=float, default=1e-5)
    g.add_argument("--lr-scale", type=float, default=3e-3)
    g.add_argument("--tau-frac", type=float, default=0.5)
    g.add_argument("--stage-frac", type=float, default=0.0)
    g.add_argument("--row-block", type=int, default=32)

    g = p.add_argument_group("sweeps")
    g.add_argument("--sweep-regime", default="lowrank",
                   choices=["iid", "corr", "lowrank"])
    g.add_argument("--lrcb-grid", type=float, nargs="+",
                   default=[0.0, 1e-5, 1e-4, 3e-4, 1e-3, 3e-3])
    g.add_argument("--stage-grid", type=float, nargs="+",
                   default=[0.0, 0.25, 0.5, 0.75, 0.9, 1.0])

    g = p.add_argument_group("output")
    g.add_argument("--latex", action="store_true", help="emit LaTeX tables")
    g.add_argument("--json", type=str, default=None, help="dump results to JSON")
    args = p.parse_args()

    torch.set_grad_enabled(True)
    print(f"FlexNu synthetic validation | device={args.device} "
          f"seeds={args.seeds} bits={args.bits} "
          f"W=[{args.out_features},{args.in_features}] "
          f"block={args.block_size} iters={args.iters}")

    res = {"config": vars(args)}
    if "main" in args.exp:
        res["main"] = exp_main(args)
    if "lrcb" in args.exp:
        res["lrcb"] = exp_lrcb(args)
    if "stage" in args.exp:
        res["stage"] = exp_stage(args)

    if args.latex:
        print("\n" + "=" * 78 + "\nLATEX\n" + "=" * 78 + "\n")
        print(emit_latex(res.get("main"), res.get("lrcb"),
                         res.get("stage"), args))

    if args.json:
        ser = {"config": {k: v for k, v in vars(args).items()}}
        for key in ("main", "lrcb", "stage"):
            if key in res:
                ser[key] = {str(k): (list(v) if isinstance(v, tuple)
                                     else {kk: list(vv) for kk, vv in v.items()})
                            for k, v in res[key].items()}
        with open(args.json, "w") as f:
            json.dump(ser, f, indent=2)
        print(f"\nWrote {args.json}")

    print("\nReminder: these are synthetic weights and a synthetic Gram. The "
          "transferable claim is the ORDERING (C > B under anisotropy, "
          "C = B = 0 under isotropy), not the magnitudes.")


if __name__ == "__main__":
    main()
