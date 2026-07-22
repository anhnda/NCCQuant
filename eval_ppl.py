#!/usr/bin/env python3
"""
Standalone single-model perplexity evaluation.

compare_slicing.py evaluates a PAIR of checkpoints against each other, which is
the wrong shape for the ablation ladder: there we produce one checkpoint per
cell, need its PPL, and then want the weights deleted before the next cell runs
(a 7B fp16 checkpoint is ~14 GB, and the ladder has five of them).

This script evaluates ONE model and writes ppl.json next to it, so run_flexnu.sh
can do quantize -> eval -> preserve json -> rm -rf checkpoint.

Sliding-window PPL, matching compare_slicing.py's convention: stride `--stride`,
window `--max-length`, context tokens masked with -100 so each token is scored
exactly once.

    python eval_ppl.py --model-path ./quantized_models/flexnu/C_divisor_only \
        --datasets wikitext2 c4 --max-length 2048 --stride 512
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from typing import List, Optional

import torch
from tqdm import tqdm


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #
def load_wikitext2_test(n_samples: Optional[int] = None) -> List[str]:
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [t for t in ds["text"] if t.strip()]
    if n_samples:
        texts = texts[:n_samples]
    # WikiText-2 PPL is conventionally computed over the concatenated test set.
    return ["\n\n".join(texts)]


def load_c4_validation(n_samples: int = 500) -> List[str]:
    from datasets import load_dataset
    ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    out = []
    for i, ex in enumerate(ds):
        if i >= n_samples:
            break
        if ex.get("text", "").strip():
            out.append(ex["text"])
    return out


# --------------------------------------------------------------------------- #
# Sliding-window perplexity
# --------------------------------------------------------------------------- #
@torch.no_grad()
def sliding_window_ppl(model, tokenizer, texts: List[str], device: str,
                       max_length: int = 2048, stride: int = 512,
                       token_cap_windows: int = 200) -> Optional[dict]:
    nlls = []
    total_tokens = 0

    for text in texts:
        enc = tokenizer(text, return_tensors="pt")
        input_ids = enc.input_ids

        # Some tokenizers omit BOS; add it so the first token is scoreable.
        if tokenizer.bos_token_id is not None and (
                input_ids.numel() == 0 or input_ids[0, 0] != tokenizer.bos_token_id):
            bos = torch.tensor([[tokenizer.bos_token_id]], dtype=input_ids.dtype)
            input_ids = torch.cat([bos, input_ids], dim=1)

        # Safety cap, same as compare_slicing.py.
        if input_ids.size(1) > max_length * token_cap_windows:
            input_ids = input_ids[:, : max_length * token_cap_windows]

        input_ids = input_ids.to(device)
        seq_len = input_ids.size(1)
        if seq_len < 2:
            continue

        prev_end_loc = 0
        pbar = tqdm(range(0, seq_len, stride), desc="  windows",
                    unit="win", leave=False)
        for begin_loc in pbar:
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc

            chunk = input_ids[:, begin_loc:end_loc]
            target = chunk.clone()
            if begin_loc > 0:
                target[:, :-trg_len] = -100      # score only the new stride
            if target.size(1) == 0:
                break

            out = model(chunk, labels=target)
            nlls.append(out.loss.float() * trg_len)
            prev_end_loc = end_loc

            if nlls:
                cur = torch.stack(nlls).sum()
                pbar.set_postfix({
                    "PPL": f"{torch.exp(cur / (total_tokens + prev_end_loc)).item():.4f}"})
            if end_loc == seq_len:
                break

        total_tokens += seq_len

    if not nlls:
        return None
    ppl = torch.exp(torch.stack(nlls).sum() / total_tokens).item()
    return {"perplexity": ppl, "total_tokens": total_tokens}


def main():
    p = argparse.ArgumentParser(description="Single-model sliding-window perplexity")
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument("--datasets", type=str, nargs="+", default=["wikitext2"],
                   choices=["wikitext2", "c4"])
    p.add_argument("--max-length", type=int, default=2048, help="window size")
    p.add_argument("--stride", type=int, default=512)
    p.add_argument("--n-samples", type=int, default=2000,
                   help="documents to use (c4); wikitext2 uses the full test set")
    p.add_argument("--out-json", type=str, default=None,
                   help="default: <model-path>/ppl.json")
    p.add_argument("--tag", type=str, default=None,
                   help="label recorded in the json (e.g. the ablation cell)")
    args = p.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Evaluating {args.model_path} on {args.datasets} "
          f"(window={args.max_length}, stride={args.stride}) ...")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    model.eval()

    results = {"model_path": args.model_path,
               "tag": args.tag,
               "max_length": args.max_length,
               "stride": args.stride}

    for name in args.datasets:
        print(f"\n--- {name} ---")
        texts = (load_wikitext2_test() if name == "wikitext2"
                 else load_c4_validation(args.n_samples))
        r = sliding_window_ppl(model, tokenizer, texts, device,
                               max_length=args.max_length, stride=args.stride)
        if r is None:
            print(f"  {name}: no scoreable tokens")
            continue
        results[name] = r
        print(f"  {name} PPL = {r['perplexity']:.4f} "
              f"over {r['total_tokens']:,} tokens")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out = args.out_json or os.path.join(args.model_path, "ppl.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
