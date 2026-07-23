#!/usr/bin/env python
"""Locate non-finite / degenerate weights in a saved quantized checkpoint.

    python check_ckpt.py ./quantized_models/flexnu/E_codebook3

Reports, per tensor: any NaN/Inf, absmax, and how many distinct values each
output row actually uses (the real diagnostic for a codebook quantizer -- a
3-bit codebook must show <= 8 distinct values per row).
"""
import sys
import torch
from safetensors import safe_open
from pathlib import Path


def main(path):
    p = Path(path)
    files = sorted(p.glob("*.safetensors"))
    if not files:
        print(f"no safetensors in {p}")
        return

    bad = []
    for f in files:
        with safe_open(f, framework="pt", device="cpu") as sf:
            for name in sf.keys():
                t = sf.get_tensor(name)
                if not t.is_floating_point():
                    continue
                tf = t.float()
                n_nan = int(torch.isnan(tf).sum())
                n_inf = int(torch.isinf(tf).sum())
                amax = float(tf.abs().max()) if tf.numel() else 0.0
                if n_nan or n_inf:
                    bad.append((name, n_nan, n_inf, amax))
                    print(f"[BAD]  {name:60s} nan={n_nan:<10d} inf={n_inf:<10d} absmax={amax:.4e}")
                elif amax > 6e4:
                    # fp16 max is 65504; anything near it overflows on cast
                    bad.append((name, 0, 0, amax))
                    print(f"[OVF]  {name:60s} absmax={amax:.4e}  (fp16 max 6.55e4)")

    print()
    if not bad:
        print("no NaN/Inf/overflow found in stored weights")
    else:
        print(f"{len(bad)} problem tensors")

    # distinct-level audit on one representative MLP + attn weight
    print("\ndistinct values per output row (first 4 rows):")
    with safe_open(files[0], framework="pt", device="cpu") as sf:
        keys = [k for k in sf.keys() if k.endswith(".weight") and "layers.0." in k]
        for name in sorted(keys):
            t = sf.get_tensor(name)
            if t.ndim != 2:
                continue
            tf = t.float()
            counts = [int(torch.unique(tf[r]).numel()) for r in range(min(4, tf.shape[0]))]
            print(f"  {name:60s} shape={tuple(tf.shape)} distinct={counts}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
