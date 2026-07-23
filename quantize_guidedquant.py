#!/usr/bin/env python
"""
Standalone GuidedQuant driver. Independent of quantize.py.

Three phases, in this order, because each depends on the last:

  1. SALIENCY + WEIGHT GRADIENTS
     One backward pass of the LM cross-entropy over calibration sequences.
     Produces, per (layer, module): saliency [N, seq, G] (squared gradient
     w.r.t. output activations, grouped over output channels) and wgrad
     [out, in] (squared gradient w.r.t. the weights).

  2. SALIENCY-WEIGHTED HESSIANS, one transformer block at a time
     Replays the SAME sequences in the SAME order and accumulates
     H[g] = sum_t s_tg x_t x_t^T for each Linear in that block.

  3. QUANTIZE that block, free its Hessians, move on.

WHY THE ORDER MATTERS. The Hessian collector consumes saliency by index: batch
i of the replay is paired with rows [i, i+bsz) of the stored saliency. If the
replay order differs from the backward-pass order, every token gets someone
else's saliency and the result is quietly wrong rather than obviously broken.
Both phases therefore iterate the identical `seqs` list.

MEMORY. Phase 2 holds [G, in, in] fp32 per Linear in the block. At the default
G=1 that is one Hessian per Linear: 64MB at in=4096, 484MB at in=11008, so a
Llama-2-7B block (7 Linears) costs roughly 870 MB on top of the model. Raising
--num-groups multiplies this directly.

USAGE
  python quantize_guidedquant.py --model-path <hf-path> --bits 3 \
      --n-calib 128 --seq-len 2048 --output-dir out/
"""

from __future__ import annotations

import argparse
import gc
import os
from typing import Dict, List

import torch
import torch.nn as nn

from quantizers.guidedquant import GuidedQuantQuantizer
from quantizers.guidedquant_collect import (
    collect_saliency_and_wgrad,
    SaliencyHessianCollector,
    _get_layers,
    _module_map,
)


# --------------------------------------------------------------------------- #
def load_calibration(tokenizer, n_samples: int, seq_len: int,
                     dataset: str = "wikitext2") -> List[torch.Tensor]:
    """Fixed-length token sequences. Deterministic order -- phases 1 and 2
    both iterate this list, and they must agree."""
    from datasets import load_dataset

    if dataset == "wikitext2":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        text = "\n\n".join(it["text"] for it in ds if it["text"].strip())
    elif dataset == "c4":
        ds = load_dataset("allenai/c4", "en",
                          data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
                          split="train", streaming=True)
        parts, n = [], 0
        for it in ds:
            parts.append(it["text"])
            n += len(it["text"])
            if n > n_samples * seq_len * 8:
                break
        text = "\n\n".join(parts)
    else:
        raise ValueError(f"unknown calibration dataset {dataset!r}")

    enc = tokenizer(text, return_tensors="pt").input_ids[0]
    total = enc.numel() // seq_len
    if total < n_samples:
        print(f"  [calib] only {total} full sequences available, "
              f"asked for {n_samples}")
        n_samples = total
    return [enc[i * seq_len:(i + 1) * seq_len].clone() for i in range(n_samples)]


# --------------------------------------------------------------------------- #
@torch.no_grad()
def _block_inputs(model, layers, seqs, device, verbose=True):
    """Capture the hidden states entering block 0, plus the kwargs each block
    needs. Lets phase 2 replay one block at a time without re-running the
    whole model for every block."""
    cache = {"kwargs": None}
    inps = []

    class Catcher(nn.Module):
        def __init__(self, mod):
            super().__init__()
            self.mod = mod

        def forward(self, hidden_states, **kw):
            inps.append(hidden_states.detach())
            cache["kwargs"] = kw
            raise RuntimeError("_stop_")

    layers[0] = Catcher(layers[0])
    for t in seqs:
        try:
            model(input_ids=t.to(device).unsqueeze(0))
        except RuntimeError as e:
            if "_stop_" not in str(e):
                raise
    layers[0] = layers[0].mod
    if verbose:
        print(f"  [replay] captured {len(inps)} block-0 inputs", flush=True)
    return inps, cache["kwargs"]


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(
        description="GuidedQuant: saliency-weighted LNQ, standalone pipeline")
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument("--output-dir", type=str, default="./quantized_guidedquant")
    p.add_argument("--bits", type=int, default=3, choices=[2, 3, 4])
    p.add_argument("--num-groups", type=int, default=1,
                   help="output-channel groups; each gets its own Hessian")
    p.add_argument("--n-calib", type=int, default=128)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--calib-dataset", type=str, default="wikitext2",
                   choices=["wikitext2", "c4"])
    p.add_argument("--iters", type=int, default=3,
                   help="LNQ outer alternations (reference: 3)")
    p.add_argument("--cd-cycles", type=int, default=4,
                   help="coordinate-descent sweeps per update_P (reference: 4)")
    p.add_argument("--ridge", type=float, default=1e-7)
    p.add_argument("--row-block", type=int, default=64)
    p.add_argument("--cd-block", type=int, default=128)
    p.add_argument("--kmeans-init", type=str, default="kmeans++",
                   choices=["kmeans++", "quantile"])
    p.add_argument("--init-iters", type=int, default=50)
    p.add_argument("--row-chunk", type=int, default=1024)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--saliency-dir", type=str, default=None,
                   help="cache dir for phase-1 artefacts; reused if present. "
                        "Empty string disables caching (used by --nosal, which "
                        "needs no saliency).")
    p.add_argument("--nosal", action="store_true",
                   help="ablation: unweighted Hessian (s == 1), single group. "
                        "Reproduces the paper's `nosal` row.")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"GuidedQuant | bits={args.bits} groups={args.num_groups} "
          f"nosal={args.nosal} | device={device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, trust_remote_code=True,
    )
    model.eval()
    model.config.use_cache = False

    seqs = load_calibration(tokenizer, args.n_calib, args.seq_len,
                            args.calib_dataset)
    print(f"  [calib] {len(seqs)} sequences x {args.seq_len} tokens")

    # ---------------- phase 1: saliency + weight gradients ------------------ #
    n_groups = 1 if args.nosal else args.num_groups
    sal = wgrad = None
    cache_ok = False
    if args.saliency_dir and os.path.isdir(args.saliency_dir):
        try:
            layers_probe = _get_layers(model)
            sal, wgrad = [], []
            for li in range(len(layers_probe)):
                d = torch.load(os.path.join(args.saliency_dir, f"l{li}.pt"),
                               map_location="cpu")
                sal.append(d["saliency"])
                wgrad.append(d["wgrad"])
            cache_ok = True
            print(f"  [saliency] loaded from {args.saliency_dir}")
        except Exception as exc:
            print(f"  [saliency] cache unusable ({exc}); recomputing")
            sal = wgrad = None

    if not cache_ok:
        if args.nosal:
            # No backward pass needed: s == 1 and no wgrad weighting.
            print("  [saliency] skipped (--nosal)")
            layers_probe = _get_layers(model)
            sal = [{n: None for n in _module_map(l)} for l in layers_probe]
            wgrad = [{n: None for n in _module_map(l)} for l in layers_probe]
        else:
            print("  [saliency] backward pass over calibration (LM loss)")
            sal, wgrad = collect_saliency_and_wgrad(
                model, seqs, num_groups=n_groups,
                save_dir=args.saliency_dir, device=device,
                verbose=args.verbose,
            )

    # ---------------- phases 2+3: per block --------------------------------- #
    model = model.to(device)
    layers = _get_layers(model)
    inps, kwargs = _block_inputs(model, layers, seqs, device,
                                 verbose=args.verbose)

    quantizer = GuidedQuantQuantizer(
        bits=args.bits, block_size=-1,
        cd_cycles=args.cd_cycles, iters=args.iters, ridge=args.ridge,
        cd_block_size=args.cd_block, row_block=args.row_block,
        init_iters=args.init_iters, kmeans_init=args.kmeans_init,
        seed=args.seed, verbose=args.verbose,
    )
    print(f"Quantizer: {quantizer}")

    n_layers = len(layers)
    for li in range(n_layers):
        block = layers[li].to(device)
        mods: Dict[str, nn.Linear] = _module_map(block)

        # -- phase 2: saliency-weighted Hessians for this block --------------
        col = SaliencyHessianCollector(store_device=device)
        for name, m in mods.items():
            s = sal[li].get(name) if sal[li] else None
            if s is None:
                # nosal, or a module the hooks never saw. expand() gives a
                # [N, seq, 1] view over a single 1.0 with no real allocation --
                # the collector only ever slices and reshapes it.
                s = torch.ones(1, 1, 1).expand(len(seqs), args.seq_len, 1)
            col.register(name, m, s)

        with torch.no_grad():
            outs = []
            for x in inps:
                o = block(x.to(device), **kwargs)
                outs.append((o[0] if isinstance(o, tuple) else o).detach())
        col.remove()

        # -- phase 3: quantize every Linear in the block ---------------------
        for name, m in mods.items():
            W = m.weight.data
            quantizer.set_hessian(col.get(name, device=device))
            quantizer.set_wgrad(wgrad[li].get(name) if wgrad[li] else None)
            res = quantizer.quantize(W.to(device), row_chunk=args.row_chunk)
            m.weight.data = res.W_dequant.to(W.dtype).to(W.device)
            quantizer.set_hessian(None)
            quantizer.set_wgrad(None)
            del res

        col.clear()
        del col
        gc.collect()
        torch.cuda.empty_cache()

        # This block's outputs are the next block's inputs.
        inps = outs
        layers[li] = block.cpu()
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  [{li + 1}/{n_layers}] block quantized", flush=True)

    # ---------------- save --------------------------------------------------
    print(f"Saving to {args.output_dir} ...")
    os.makedirs(args.output_dir, exist_ok=True)
    model.config.use_cache = True
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
