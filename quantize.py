"""
Quantization driver.

Loads a HuggingFace causal LM, builds a non-uniform codebook quantizer by name
(nf3 / nf4 / nvfp4 / codebook3 / codebook4), and quantizes every nn.Linear layer.
Optionally applies NCC first-moment correction, for which it collects the
per-layer activation mean mu = E[x] over a calibration set via forward hooks.

Design mirrors the uploaded AWQ XL script: bfloat16 load, device_map="auto",
batched calibration hooks, aggressive cleanup. The base group-wise uniform
quantizer is replaced by the per-channel non-uniform codebook quantizer; the
salience grid-search is replaced by NCC's first-moment correction.

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

from quantizers import get_quantizer, apply_ncc


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
    exactly the dimension NCC's mu lives on. We accumulate sum and count to form
    mu = E[x] in a streaming, memory-light way (no stored activations).
    Optionally also accumulates E[x^2] to give NCC a variance estimate for the
    James-Stein stabiliser.
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
):
    # Gather target Linear layers.
    linears = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    if skip_lmhead:
        linears = [(n, m) for (n, m) in linears if not is_lmhead(n)]
    print(f"Quantizing {len(linears)} Linear layers "
          f"({'skipping' if skip_lmhead else 'including'} lm_head)")

    # Collect activation means only if NCC is on.
    means: Dict[str, torch.Tensor] = {}
    varis: Dict[str, torch.Tensor] = {}
    if use_ncc:
        print("Collecting activation means for NCC ...")
        collector = ActMeanCollector(want_var=True)
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
        del collector
        gc.collect()

    # Quantize layer by layer.
    total_flips = 0
    bias_before_sum = 0.0
    bias_after_sum = 0.0
    for n, module in linears:
        W = module.weight.data
        res = quantizer.quantize(W)
        W_out = res.W_dequant

        if use_ncc and n in means:
            mu = means[n].to(W.device)
            mu_var = varis.get(n)
            if mu_var is not None:
                mu_var = mu_var.to(W.device)
            W_out, stats = apply_ncc(
                W_fp=W, qres=res, mu=mu,
                budget_p=budget_p, use_james_stein=True, mu_var=mu_var,
            )
            total_flips += stats.flips
            bias_before_sum += stats.bias_before
            bias_after_sum += stats.bias_after

        module.weight.data = W_out.to(W.dtype)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("Quantization complete.")
    if use_ncc:
        print(f"  NCC total flips: {total_flips}")
        print(f"  Calibration first-moment error: "
              f"{bias_before_sum:.6e} -> {bias_after_sum:.6e}")


def main():
    p = argparse.ArgumentParser(description="Non-uniform codebook quantization with NCC")
    p.add_argument("--model-path", type=str, required=True, help="HF model name or local path")
    p.add_argument("--quantizer", type=str, default="nf4",
                   choices=["nf3", "nf4", "nvfp4", "codebook3", "codebook4"],
                   help="Non-uniform codebook to use")
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
    # quantizer-specific knobs
    p.add_argument("--block-size", type=int, default=16, help="NVFP4 micro-block size")
    p.add_argument("--kmeans-iters", type=int, default=20, help="learned-codebook k-means iters")
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | Quantizer: {args.quantizer} | skip_lmhead={args.skip_lmhead}")

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
        block_size=args.block_size,
        n_iters=args.kmeans_iters,
        seed=args.seed,
    )
    print(f"Loaded quantizer: {quantizer}")

    calib_texts = load_wikitext2_simple(n_samples=args.n_calib)

    quantize_model(
        model, tokenizer, quantizer, calib_texts, device,
        use_ncc=args.use_ncc, budget_p=args.budget_p,
        skip_lmhead=args.skip_lmhead, n_calib=args.n_calib,
        max_length=args.max_length,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving to {args.output_dir} ...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
