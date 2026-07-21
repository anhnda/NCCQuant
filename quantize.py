"""
Quantization driver.

Loads a HuggingFace causal LM, builds a non-uniform codebook quantizer by name
(nf3 / nf4 / nvfp4 / codebook3 / codebook4), and quantizes every nn.Linear layer.

Two INDEPENDENT first-moment correctors are available (use either, neither, or
compare):
  --use-ncc  : NCC flips codewords under a budget so the per-channel first-moment
               error mu . e shrinks (changes the dequantized WEIGHTS).
  --bc       : naive bias correction folds the full mean output error
               mu @ (W_fp - W_q)^T into the layer's BIAS, leaving the quantized
               weights untouched. Standalone — it reads the plain quantized
               weights, NOT any NCC-corrected version.

Both read mu = E[x] (and, for NCC-Cov / James-Stein, the variance sigma_ii)
collected over a calibration set via forward hooks.

Global ASYM flag (quantizers.base_quantizer.ASYM) selects symmetric (absmax) vs
asymmetric (affine min/max) per-block mapping; exposed here via --asym/--no-asym.

NCC selection rule is chosen with --ncc-score:
    lite : eta = |mu_i| / g                       (covariance-free)
    cov  : eta = |mu_i| / ((sigma_ii + eps) * g)  (diagonal bias-variance)

Key option: --skip-lmhead (default TRUE). When set, the lm_head Linear is left in
full precision (it is large and quantizing it hurts perplexity most).

NOTE: per user preference, this script does not run torch on its own here; invoke
it explicitly with `python quantize.py ...`.
"""

from __future__ import annotations

import argparse
import gc
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import quantizers.base_quantizer as base_q
from quantizers import get_quantizer, apply_ncc, NEEDS_GRAM
from quantizers.gram_collect import GramCollector
from quantizers import apply_bias_correction


# --------------------------------------------------------------------------- #
# Calibration data
# --------------------------------------------------------------------------- #
def load_wikitext2_simple(n_samples: int = 128) -> List[str]:
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [it["text"] for it in ds if len(it["text"].strip()) > 0]
    return texts[:n_samples]


# --------------------------------------------------------------------------- #
# Activation-mean collection (mu = E[x] per Linear's input dimension)
# --------------------------------------------------------------------------- #
class ActMeanCollector:
    """Accumulates the running mean of the *input* activation for each Linear.

    For a Linear with weight [out, in], the input x has last-dim = in, which is
    exactly the dimension NCC's mu and BC's mu live on. We accumulate sum and
    count to form mu = E[x] in a streaming, memory-light way (no stored
    activations). Optionally also accumulates E[x^2] to give a per-input-channel
    variance estimate sigma_ii = E[x^2] - E[x]^2 (used by NCC score='cov' and,
    divided by the token count m, by the James-Stein stabiliser).
    """

    def __init__(self, want_var: bool = True):
        self.sum: Dict[str, torch.Tensor] = {}
        self.sumsq: Dict[str, torch.Tensor] = {}
        self.count: Dict[str, int] = {}
        self.want_var = want_var
        self.hooks = []

    def _hook(self, name: str):
        def hook(_m, inp, _out):
            x = inp[0] if isinstance(inp, tuple) else inp
            x = x.reshape(-1, x.shape[-1]).detach().float()  # [N, in]
            s = x.sum(dim=0).cpu()
            n = x.shape[0]
            if name not in self.sum:
                self.sum[name] = s
                self.count[name] = n
                if self.want_var:
                    self.sumsq[name] = (x * x).sum(dim=0).cpu()
            else:
                self.sum[name] += s
                self.count[name] += n
                if self.want_var:
                    self.sumsq[name] += (x * x).sum(dim=0).cpu()
        return hook

    def register(self, layers: List[Tuple[str, nn.Module]]):
        for name, module in layers:
            self.hooks.append(module.register_forward_hook(self._hook(name)))

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def mean(self, name: str) -> torch.Tensor:
        return self.sum[name] / max(1, self.count[name])

    def var(self, name: str):
        if not self.want_var or name not in self.sumsq:
            return None
        m = self.mean(name)
        ex2 = self.sumsq[name] / max(1, self.count[name])
        return (ex2 - m * m).clamp(min=0.0)


def is_lmhead(name: str) -> bool:
    return "lm_head" in name.lower() or name.endswith("lm_head")


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
@torch.no_grad()
def quantize_model(
    model,
    tokenizer,
    quantizer,
    calib_texts: List[str],
    device: str,
    use_ncc: bool = True,
    budget_p: float = 0.02,
    skip_lmhead: bool = True,
    n_calib: int = 128,
    max_length: int = 512,
    row_chunk: int = 1024,
    ncc_score: str = "lite",
    cov_eps: float = 1e-6,
    gap_floor: float = 1e-8,
    gap_floor_rel: float = 0.0,
    use_james_stein: bool = False,
    use_bc: bool = False,
    gram_group_size: int = 8,
    gram_store_device: str = "cpu",
    gram_max_tokens: int = None,
):
    # Gather target Linear layers.
    linears = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    if skip_lmhead:
        linears = [(n, m) for (n, m) in linears if not is_lmhead(n)]
    print(f"Quantizing {len(linears)} Linear layers "
          f"({'skipping' if skip_lmhead else 'including'} lm_head)")

    # Collect activation means (and variances) if NCC or BC is on.
    means: Dict[str, torch.Tensor] = {}
    varis: Dict[str, torch.Tensor] = {}
    collector_counts: Dict[str, int] = {}
    if use_ncc or use_bc:
        # cov rule and James-Stein need second moments; plain BC needs only mu.
        want_var = (use_ncc and ncc_score == "cov") or use_james_stein
        print(f"Collecting activation means "
              f"(want_var={want_var}, ncc={use_ncc}, score={ncc_score}, bc={use_bc}) ...")
        collector = ActMeanCollector(want_var=want_var)
        collector.register(linears)
        for i, text in enumerate(calib_texts[:n_calib]):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            model(**inputs, use_cache=False)
            if (i + 1) % 16 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
        collector.remove()
        for n, _ in linears:
            if n in collector.sum:
                means[n] = collector.mean(n)
                v = collector.var(n)
                if v is not None:
                    varis[n] = v
                collector_counts[n] = collector.count[n]   # token count m, for JS
        del collector
        gc.collect()

    # ------------------------------------------------------------------ #
    # Gram collection for Gram-objective quantizers (FlexNu).
    #
    # G = E[x x^T] is [in, in] per Linear -- 64 MB at in=4096, 822 MB at
    # in=14336. Holding one per layer for a whole model is not viable, so we
    # sweep in small GROUPS: register `gram_group_size` Linears, run the
    # calibration set once, quantize those layers, free their Grams, move on.
    # Cost is (n_layers / group_size) calibration passes; memory is bounded by
    # the group.
    # ------------------------------------------------------------------ #
    needs_gram = getattr(quantizer, "name", "") in NEEDS_GRAM
    if needs_gram:
        print(f"Quantizer {quantizer.name!r} uses the full-Gram objective "
              f"Tr(R G R^T); sweeping in groups of {gram_group_size} "
              f"(store_device={gram_store_device}).")

    def _gram_groups():
        """Yield (group_of_linears, gram_collector_or_None)."""
        if not needs_gram:
            yield linears, None
            return
        for gi in range(0, len(linears), gram_group_size):
            group = linears[gi:gi + gram_group_size]
            col = GramCollector(store_device=gram_store_device,
                                max_tokens=gram_max_tokens)
            col.register(group)
            with torch.no_grad():
                for i, text in enumerate(calib_texts[:n_calib]):
                    inputs = tokenizer(text, return_tensors="pt",
                                       truncation=True, max_length=max_length)
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    model(**inputs, use_cache=False)
                    if (i + 1) % 16 == 0:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()
            col.remove()
            yield group, col
            col.clear()
            del col
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Quantize layer by layer.
    total_flips = 0
    bias_before_sum = 0.0
    bias_after_sum = 0.0
    bc_layers = 0
    bc_created = 0
    bc_err_before_sum = 0.0
    flexnu_e_init = 0.0
    flexnu_e_final = 0.0
    pbar = tqdm(total=len(linears), desc="Quantizing layers")
    for group, gram_col in _gram_groups():
      for n, module in group:
          pbar.update(1)
          W = module.weight.data

          # Attach this layer's Gram before quantizing (FlexNu only). Passing
          # None degrades the objective to plain MSE rather than failing.
          G = None
          if needs_gram:
              G = (gram_col.get(n, device=W.device, dtype=torch.float32)
                   if gram_col is not None else None)
              quantizer.set_gram(G)

          res = quantizer.quantize(W, row_chunk=row_chunk)

          if needs_gram:
              quantizer.set_gram(None)
              d = getattr(quantizer, "last_diag", {}) or {}
              flexnu_e_init += d.get("energy_init", 0.0)
              flexnu_e_final += d.get("energy_final", 0.0)
              if gram_col is not None:
                  gram_col.free(n)
              del G

          W_out = res.W_dequant

          # ---- NCC path: flip codewords (changes the dequantized weights). ----
          if use_ncc and n in means:
              mu = means[n].to(W.device)
              # Raw per-input-channel activation variance == sigma_ii (for cov rule).
              sigma_ii = varis.get(n)
              if sigma_ii is not None:
                  sigma_ii = sigma_ii.to(W.device)
              # James-Stein needs variance OF THE MEAN = sigma_ii / m, not raw sigma_ii
              # (else the shrink factor is inflated by ~m and collapses mu).
              mu_var = None
              if use_james_stein and sigma_ii is not None:
                  m = max(1, collector_counts.get(n, 1))
                  mu_var = sigma_ii / m
              W_out, stats = apply_ncc(
                  W_fp=W, qres=res, mu=mu,
                  budget_p=budget_p,
                  use_james_stein=use_james_stein, mu_var=mu_var,
                  row_chunk=row_chunk,
                  score=ncc_score,
                  sigma_ii=sigma_ii,
                  cov_eps=cov_eps,
                  gap_floor=gap_floor,
                  gap_floor_rel=gap_floor_rel,
              )
              total_flips += stats.flips
              bias_before_sum += stats.bias_before
              bias_after_sum += stats.bias_after

          # Write the quantized weights (NCC-corrected if NCC ran, else plain).
          module.weight.data = W_out.to(W.dtype)

          # ---- BC path: STANDALONE naive bias correction. -------------------
          # Reads the PLAIN quantized weights (res.W_dequant), NOT W_out, so it is
          # independent of NCC: it folds the full mean output error of the plain
          # quantized layer into the bias. (Run BC without NCC for pure bias
          # correction; running both is a deliberate apples-to-oranges combo and
          # generally not intended.)
          if use_bc and n in means:
              mu = means[n].to(W.device)
              bc_stats = apply_bias_correction(
                  module=module, W_fp=W, W_q=res.W_dequant, mu=mu,
                  row_chunk=max(row_chunk, 4096),
              )
              bc_layers += 1
              bc_created += int(bc_stats.bias_created)
              bc_err_before_sum += bc_stats.out_err_before

          if torch.cuda.is_available():
              torch.cuda.empty_cache()

    pbar.close()

    print("Quantization complete.")
    if needs_gram and flexnu_e_init > 0:
        drop = 100.0 * (flexnu_e_init - flexnu_e_final) / max(flexnu_e_init, 1e-12)
        print(f"  FlexNu reconstruction energy Tr(R G R^T): "
              f"{flexnu_e_init:.6e} -> {flexnu_e_final:.6e}  ({drop:.2f}% drop)")
    if use_ncc:
        print(f"  NCC selection rule: {ncc_score}")
        print(f"  NCC total flips: {total_flips}")
        print(f"  Calibration first-moment error: "
              f"{bias_before_sum:.6e} -> {bias_after_sum:.6e}")
    if use_bc:
        print(f"  BC applied to {bc_layers} layers "
              f"({bc_created} new bias terms created)")
        print(f"  BC absorbed output first-moment energy: "
              f"{bc_err_before_sum:.6e} -> ~0")


def main():
    p = argparse.ArgumentParser(description="Non-uniform codebook quantization with NCC / BC")
    p.add_argument("--model-path", type=str, required=True, help="HF model name or local path")
    p.add_argument("--quantizer", type=str, default="nf4",
                   choices=["nf3", "nf4", "nvfp4", "codebook3", "codebook4",
                            "flexnu2", "flexnu3", "flexnu4"],
                   help="Non-uniform codebook to use. flexnu* = FlexRound-style "
                        "learnable rounding with a jointly learned sorted codebook "
                        "(needs the full activation Gram).")
    p.add_argument("--output-dir", type=str, default="./quantized_model")
    p.add_argument("--skip-lmhead", dest="skip_lmhead", action="store_true", default=True,
                   help="Skip quantizing lm_head (default: True)")
    p.add_argument("--no-skip-lmhead", dest="skip_lmhead", action="store_false",
                   help="Also quantize lm_head")
    p.add_argument("--use-ncc", dest="use_ncc", action="store_true", default=True,
                   help="Apply NCC first-moment correction (default: True)")
    p.add_argument("--no-ncc", dest="use_ncc", action="store_false")
    p.add_argument("--budget-p", type=float, default=0.02, help="NCC budget fraction p")
    p.add_argument("--n-calib", type=int, default=128)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--calib-dataset", type=str, default="wikitext2-simple",
                   choices=["wikitext2-simple"])
    p.add_argument("--seed", type=int, default=42)

    # Symmetric vs asymmetric per-block mapping (global ASYM flag).
    p.add_argument("--asym", dest="asym", action="store_true", default=True,
                   help="Asymmetric (affine min/max) quantization (default: True)")
    p.add_argument("--no-asym", dest="asym", action="store_false",
                   help="Symmetric (absmax) quantization")

    # NCC selection rule and its knobs.
    p.add_argument("--ncc-score", type=str, default="lite",
                   choices=["lite", "cov"],
                   help="NCC selection rule: 'lite' (|mu|/g, covariance-free) or "
                        "'cov' (|mu|/((sigma_ii+eps)*g), diagonal bias-variance).")
    p.add_argument("--cov-eps", type=float, default=1e-6,
                   help="Stabiliser added to sigma_ii in the 'cov' rule denominator "
                        "(ignored for 'lite').")
    p.add_argument("--gap-floor", type=float, default=1e-8,
                   help="Absolute lower bound on a feasible complementary gap "
                        "(Assumption 2, strict g>0; degenerate-codeword cleanup).")
    p.add_argument("--gap-floor-rel", type=float, default=0.0,
                   help="OPTIONAL relative gap floor as a fraction of the per-row "
                        "median gap (ablation only; contradicts the theory if >0).")
    p.add_argument("--use-james-stein", dest="use_james_stein",
                   action="store_true", default=False,
                   help="Apply James-Stein shrinkage to mu (ablation row; off by "
                        "default). Uses variance-of-the-mean = sigma_ii / m.")

    # Standalone bias correction.
    p.add_argument("--bc", dest="use_bc", action="store_true", default=False,
                   help="Apply STANDALONE naive bias correction: fold the mean "
                        "output error of the plain quantized weights into each "
                        "Linear's bias (default: False). Independent of NCC.")

    # quantizer-specific knobs (block = standard non-uniform scaling granularity)
    p.add_argument("--nf-block-size", type=int, default=64, help="NF3/NF4 block size (bnb default 64)")
    p.add_argument("--nvfp4-block-size", type=int, default=16, help="NVFP4 micro-block size")
    p.add_argument("--cb-block-size", type=int, default=64, help="learned-codebook block size")
    p.add_argument("--kmeans-iters", type=int, default=20, help="learned-codebook k-means iters")
    p.add_argument("--row-chunk", type=int, default=1024,
                   help="output rows processed at once (memory bound; no effect on result)")

    # ---- FlexNu -----------------------------------------------------------
    g = p.add_argument_group("FlexNu (flexnu2/3/4)")
    g.add_argument("--flexnu-iters", type=int, default=300,
                   help="Adam steps per row-block.")
    g.add_argument("--flexnu-lr-scale", type=float, default=3e-3,
                   help="LR for the FlexRound log-divisors delta2 / delta3.")
    g.add_argument("--flexnu-lr-cb", type=float, default=1e-5,
                   help="LR for the codebook params (anchor + softplus gaps).")
    g.add_argument("--flexnu-tau-frac", type=float, default=0.5,
                   help="Backward STE bump width as a fraction of the mean "
                        "codebook gap. Backward-only; no annealing schedule.")
    g.add_argument("--flexnu-no-s3", action="store_true",
                   help="Drop the per-output-channel divisor s3 (delta3).")
    g.add_argument("--flexnu-freeze-codebook", action="store_true",
                   help="ABLATION: fix the codebook at its k-means init; learn "
                        "only the FlexRound divisors.")
    g.add_argument("--flexnu-freeze-scale", action="store_true",
                   help="ABLATION: fix the divisors at 1; learn only the "
                        "codebook. Both freeze flags => reproduces codebookK.")
    g.add_argument("--flexnu-init", type=str, default="lloyd",
                   choices=["lloyd", "levels"],
                   help="Codebook init: 1-D k-means, or an affine map of a "
                        "canonical level shape onto each block's [min,max].")
    g.add_argument("--flexnu-row-block", type=int, default=128,
                   help="Rows optimised per sub-problem. THE memory knob: the "
                        "threshold sum is [row_block, in, K-1]. No effect on "
                        "the result.")
    g.add_argument("--flexnu-stage-frac", type=float, default=0.0,
                   help="Fraction of iters spent learning the codebook before "
                        "switching to the divisors. Default 0.0 = fully joint. "
                        "Staging defeats the mechanism: it starts the divisor "
                        "at the nearest-neighbour point it exists to escape.")
    g.add_argument("--flexnu-lambda-s2", type=float, default=0.0,
                   help="L2 on the log-divisors (safety rail; 0 = off).")
    g.add_argument("--flexnu-eval-every", type=int, default=1,
                   help="Steps between true-hard-energy evaluations for "
                        "best-iterate tracking. 1 recommended.")
    g.add_argument("--flexnu-verbose", action="store_true",
                   help="Print per-layer reconstruction energy drop.")
    g.add_argument("--gram-group-size", type=int, default=8,
                   help="Linears whose Gram is held at once. Each group costs "
                        "one extra calibration pass; memory is bounded by it.")
    g.add_argument("--gram-store-device", type=str, default="cpu",
                   choices=["cpu", "cuda"],
                   help="Where [in,in] Gram accumulators live.")
    g.add_argument("--gram-max-tokens", type=int, default=None,
                   help="Cap tokens accumulated per layer's Gram (None = all).")

    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Set the global asymmetric flag before any quantizer is built; quantize()
    # reads quantizers.base_quantizer.ASYM at call time.
    base_q.ASYM = args.asym
    print(f"ASYM mode: {base_q.ASYM}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | Quantizer: {args.quantizer} | "
          f"skip_lmhead={args.skip_lmhead} | ncc={args.use_ncc} "
          f"(score={args.ncc_score}) | bc={args.use_bc}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    model.eval()

    quantizer = get_quantizer(
        args.quantizer,
        nf_block_size=args.nf_block_size,
        nvfp4_block_size=args.nvfp4_block_size,
        cb_block_size=args.cb_block_size,
        n_iters=args.kmeans_iters,
        seed=args.seed,
        flexnu_iters=args.flexnu_iters,
        flexnu_lr_scale=args.flexnu_lr_scale,
        flexnu_lr_cb=args.flexnu_lr_cb,
        flexnu_tau_frac=args.flexnu_tau_frac,
        flexnu_no_s3=args.flexnu_no_s3,
        flexnu_freeze_codebook=args.flexnu_freeze_codebook,
        flexnu_freeze_scale=args.flexnu_freeze_scale,
        flexnu_stage_frac=args.flexnu_stage_frac,
        flexnu_eval_every=args.flexnu_eval_every,
        flexnu_init=args.flexnu_init,
        flexnu_row_block=args.flexnu_row_block,
        flexnu_lambda_s2=args.flexnu_lambda_s2,
        flexnu_verbose=args.flexnu_verbose,
    )
    print(f"Loaded quantizer: {quantizer}")

    calib_texts = load_wikitext2_simple(n_samples=args.n_calib)

    quantize_model(
        model, tokenizer, quantizer, calib_texts, device,
        use_ncc=args.use_ncc, budget_p=args.budget_p,
        skip_lmhead=args.skip_lmhead, n_calib=args.n_calib,
        max_length=args.max_length, row_chunk=args.row_chunk,
        ncc_score=args.ncc_score,
        cov_eps=args.cov_eps,
        gap_floor=args.gap_floor,
        gap_floor_rel=args.gap_floor_rel,
        use_james_stein=args.use_james_stein,
        use_bc=args.use_bc,
        gram_group_size=args.gram_group_size,
        gram_store_device=args.gram_store_device,
        gram_max_tokens=args.gram_max_tokens,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving to {args.output_dir} ...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()