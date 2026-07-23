#!/usr/bin/env python3
"""
Standalone perplexity evaluation.

PRIMARY METRIC: non-overlapping block PPL, khop protocol GPTQ / AWQ /
OmniQuant / SpinQuant. Corpus duoc noi lai, tokenize mot lan, cat thanh cac
block [0:seqlen], [seqlen:2*seqlen], ... Moi block chay doc lap, khong mang
context tu block truoc, khong mask.

    Sanity check: Llama-2-7B fp16, wikitext2, seqlen=2048 -> 5.47
                  Llama-2-13B fp16, wikitext2, seqlen=2048 -> 4.88

Lech qua 0.02 = con bug o tokenize hoac load model. Dung chay tiep ladder.

SECONDARY METRIC (--method sliding): overlapping sliding window. Uoc luong PPL
sat dinh nghia hon (HF goi non-overlapping la xap xi "suboptimal" vi token dau
moi block bi score voi context rong), NHUNG cho so THAP hon dang ke. Khong bao
gio dat canh so trong bang paper. Luon ghi ro ctx_len va stride khi bao cao.

Usage:
    python eval_ppl.py --model-path ./quantized_models/flexnu/C_divisor_only \
        --datasets wikitext2 c4 --seqlen 2048

    python eval_ppl.py --model-path meta-llama/Llama-2-7b-hf \
        --datasets wikitext2 --seqlen 2048 --dtype fp16    # -> phai ra 5.47
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import torch
from tqdm import tqdm


# --------------------------------------------------------------------------- #
# Corpus loading
#
# CRITICAL - ba dieu duoi day quyet dinh con so co khop bang paper hay khong:
#   1. wikitext2 dung "\n\n".join va KHONG filter dong rong.
#   2. Tokenize MOT LAN tren toan corpus, KHONG chen BOS thu cong.
#   3. c4 dung get_c4_new (noi 1100 doc dau, cat 256*seqlen token).
# --------------------------------------------------------------------------- #
def _load_corpus_ids(name: str, tokenizer, seqlen: int,
                     cache_dir: Optional[Path] = None) -> torch.Tensor:
    """Tra ve input_ids [1, N] cua toan bo corpus test."""
    cache_file = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        tok_id = getattr(tokenizer, "name_or_path", "tok").replace("/", "_")
        # Chi c4 phu thuoc seqlen (cat 256*seqlen); wikitext2/ptb thi khong.
        suffix = f"_{seqlen}" if name == "c4" else ""
        cache_file = cache_dir / f"{name}_{tok_id}{suffix}.pkl"
        if cache_file.exists():
            with open(cache_file, "rb") as fh:
                return pickle.load(fh)

    from datasets import load_dataset

    if name == "wikitext2":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        # KHONG filter: giu nguyen ca dong rong, dung nhu GPTQ datautils.py
        enc = tokenizer("\n\n".join(ds["text"]), return_tensors="pt").input_ids

    elif name == "c4":
        # get_c4_new: pin revision de moi lan chay ra dung cung mot tap text
        ds = load_dataset(
            "allenai/c4",
            "default",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            split="validation",
            revision="607bd4c8450a42878aa9ddc051a65a055450ef87",
        )
        enc = tokenizer(" ".join(ds[:1100]["text"]), return_tensors="pt").input_ids
        enc = enc[:, : 256 * seqlen]

    elif name == "ptb":
        ds = load_dataset("ptb_text_only", "penn_treebank", split="test")
        enc = tokenizer(" ".join(ds["sentence"]), return_tensors="pt").input_ids

    else:
        raise ValueError(f"Unknown dataset: {name}")

    if cache_file is not None:
        with open(cache_file, "wb") as fh:
            pickle.dump(enc, fh)
    return enc


# --------------------------------------------------------------------------- #
# PRIMARY: non-overlapping block PPL (GPTQ protocol)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def eval_ppl(model, tokenizer, testcases: List[str], seqlen: int = 2048,
             cache_dir: Optional[Path] = None, verbose: bool = True) -> Dict[str, float]:
    model.eval()
    device = next(model.parameters()).device
    results: Dict[str, float] = {}

    for name in testcases:
        enc = _load_corpus_ids(name, tokenizer, seqlen, cache_dir)
        nsamples = enc.numel() // seqlen
        if nsamples == 0:
            if verbose:
                print(f"{name}: corpus ngan hon seqlen, bo qua")
            continue

        nlls = []
        for i in tqdm(range(nsamples), disable=not verbose, desc=f"{name}"):
            batch = enc[:, i * seqlen : (i + 1) * seqlen].to(device)
            out = model(batch, labels=batch)
            # out.loss la mean tren (seqlen - 1) token duoc score.
            # Nhan seqlen o day va chia nsamples*seqlen o duoi: he so triet tieu,
            # ket qua giong het dung (seqlen - 1) o ca hai cho. Giu seqlen de
            # khop nguyen van repo GPTQ goc.
            nlls.append(out.loss.float() * seqlen)

        ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen)).item()
        results[name] = ppl
        if verbose:
            print(f"{name}: PPL = {ppl:.4f}  "
                  f"({nsamples} blocks x {seqlen} tokens = {nsamples * seqlen:,})")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


# --------------------------------------------------------------------------- #
# SECONDARY: overlapping sliding window
# --------------------------------------------------------------------------- #
@torch.no_grad()
def eval_ppl_sliding(model, tokenizer, testcases: List[str], ctx_len: int = 2048,
                     stride: int = 512, cache_dir: Optional[Path] = None,
                     verbose: bool = True) -> Dict[str, float]:
    model.eval()
    device = next(model.parameters()).device
    results: Dict[str, float] = {}

    for name in testcases:
        enc = _load_corpus_ids(name, tokenizer, ctx_len, cache_dir).to(device)
        seq_len = enc.size(1)
        if seq_len < 2:
            continue

        nll_sum = 0.0
        n_scored = 0
        prev_end = 0

        for begin in tqdm(range(0, seq_len, stride), disable=not verbose,
                          desc=f"{name} (sliding)"):
            end = min(begin + ctx_len, seq_len)
            trg_len = end - prev_end          # so token MOI trong window nay
            if trg_len <= 0:
                # window nay khong them token moi nao (xay ra khi stride > ctx_len
                # hoac o window cuoi bi cat ngan) -> bo qua, tranh mask sai
                if end == seq_len:
                    break
                continue
            chunk = enc[:, begin:end]

            target = chunk.clone()
            n_ctx = chunk.size(1) - trg_len   # so token context can mask
            if n_ctx > 0:
                target[:, :n_ctx] = -100

            out = model(chunk, labels=target)
            # HF shift labels: token thu i duoc du doan tu token < i.
            # So token thuc su co loss = so label != -100 sau khi shift.
            valid = int((target[:, 1:] != -100).sum().item())
            if valid == 0:
                prev_end = end
                if end == seq_len:
                    break
                continue

            nll_sum += out.loss.float().item() * valid
            n_scored += valid

            prev_end = end
            if end == seq_len:
                break

        if n_scored == 0:
            continue

        ppl = float(torch.exp(torch.tensor(nll_sum / n_scored)))
        results[name] = ppl
        if verbose:
            print(f"{name}: PPL = {ppl:.4f}  (ctx={ctx_len}, stride={stride}, "
                  f"{n_scored:,} scored tokens)")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


def main():
    p = argparse.ArgumentParser(description="Perplexity evaluation (GPTQ protocol)")
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument("--datasets", type=str, nargs="+", default=["wikitext2"],
                   choices=["wikitext2", "c4", "ptb"])
    p.add_argument("--seqlen", type=int, default=2048,
                   help="2048 cho Llama-1/2; 8192 cho Llama-3/Qwen3. "
                        "So o seqlen khac nhau KHONG so duoc voi nhau.")
    p.add_argument("--method", type=str, default="block",
                   choices=["block", "sliding"],
                   help="block = non-overlapping (chuan paper); "
                        "sliding = overlapping (so thap hon, metric phu)")
    p.add_argument("--stride", type=int, default=512,
                   help="chi dung khi --method sliding")
    p.add_argument("--dtype", type=str, default="fp16", choices=list(_DTYPES))
    p.add_argument("--device-map", type=str, default="auto")
    p.add_argument("--cache-dir", type=str, default="./dataset_cache",
                   help="cache token da hoa; '' de tat")
    p.add_argument("--out-json", type=str, default=None,
                   help="mac dinh: <model-path>/ppl.json")
    p.add_argument("--tag", type=str, default=None)
    args = p.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = _DTYPES[args.dtype]
    cache_dir = Path(args.cache_dir) if args.cache_dir else None

    print(f"Model:    {args.model_path}")
    print(f"Datasets: {args.datasets}")
    print(f"Method:   {args.method}"
          + (f" (stride={args.stride})" if args.method == "sliding" else ""))
    print(f"Seqlen:   {args.seqlen}   dtype: {args.dtype}")
    print("-" * 60)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map=args.device_map,
        trust_remote_code=True,
    )
    model.eval()

    if args.method == "block":
        ppls = eval_ppl(model, tokenizer, args.datasets,
                        seqlen=args.seqlen, cache_dir=cache_dir)
    else:
        ppls = eval_ppl_sliding(model, tokenizer, args.datasets,
                                ctx_len=args.seqlen, stride=args.stride,
                                cache_dir=cache_dir)

    results = {
        "model_path": args.model_path,
        "tag": args.tag,
        "method": args.method,
        "seqlen": args.seqlen,
        "dtype": args.dtype,
        **({"stride": args.stride} if args.method == "sliding" else {}),
        "ppl": ppls,
    }

    out = args.out_json or os.path.join(args.model_path, "ppl.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)
    print("-" * 60)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()