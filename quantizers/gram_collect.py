"""
Second-moment Gram collection for FlexNu.

NCC / BC only need the first moment mu = E[x] (and optionally the diagonal
sigma_ii), so the existing ActMeanCollector in quantize.py accumulates sums and
sums-of-squares per input channel. FlexNu's objective is the full layer
reconstruction energy

    L = || (W - W_hat) X ||_F^2  =  n * Tr( R G R^T ),   G = E[x x^T]

which needs the FULL Gram, off-diagonals included. That is the whole point: the
off-diagonal terms are what let a deliberate non-nearest codeword choice on one
input coordinate compensate error on another. A diagonal approximation would
reduce the objective to per-coordinate MSE, under which nearest-codeword IS
optimal and FlexNu's assignment mechanism has nothing to do.

Cost. G is [in, in] fp32 per Linear. For a 4096-wide input that is 64 MB; for a
14336-wide MLP down_proj input it is 822 MB. Holding one per layer for a whole
model is not viable, so this collector is designed to be used ONE LAYER AT A TIME
(register -> run calibration -> read -> free), or with `store_device="cpu"` when
several are needed.

Accumulation is  sum_t x_t x_t^T  in fp32 (or fp64 with accum_dtype), divided by
the token count at read time. Streaming, no stored activations.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class GramCollector:
    """Accumulates G = E[x x^T] over each registered Linear's input.

    Parameters
    ----------
    store_device : str
        Where the [in, in] accumulator lives. "cpu" is the safe default when
        registering many layers at once; "cuda" is much faster when you sweep
        one layer (or one transformer block) at a time.
    accum_dtype : torch.dtype
        fp32 is normally fine; fp64 removes any doubt about accumulation drift
        over long calibration runs at ~2x the memory.
    max_tokens : Optional[int]
        Stop accumulating for a layer after this many tokens. Caps cost on long
        calibration sets; the Gram estimate saturates well before the full set
        for typical d.
    """

    def __init__(self,
                 store_device: str = "cpu",
                 accum_dtype: torch.dtype = torch.float32,
                 max_tokens: Optional[int] = None):
        self.store_device = store_device
        self.accum_dtype = accum_dtype
        self.max_tokens = max_tokens
        self.gram: Dict[str, torch.Tensor] = {}
        self.count: Dict[str, int] = {}
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []

    def _hook(self, name: str):
        def hook(_m, inp, _out):
            if self.max_tokens is not None and self.count.get(name, 0) >= self.max_tokens:
                return
            x = inp[0] if isinstance(inp, tuple) else inp
            x = x.reshape(-1, x.shape[-1]).detach()
            if self.max_tokens is not None:
                room = self.max_tokens - self.count.get(name, 0)
                if x.shape[0] > room:
                    x = x[:room]
            xf = x.to(self.accum_dtype)
            g = (xf.t() @ xf)                       # [in, in], on the layer's device
            g = g.to(self.store_device)
            if name not in self.gram:
                self.gram[name] = g
                self.count[name] = x.shape[0]
            else:
                self.gram[name] += g
                self.count[name] += x.shape[0]
            del xf, g
        return hook

    def register(self, layers: List[Tuple[str, nn.Module]]) -> None:
        for name, module in layers:
            self.hooks.append(module.register_forward_hook(self._hook(name)))

    def remove(self) -> None:
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def get(self, name: str, device=None, dtype=torch.float32) -> Optional[torch.Tensor]:
        """E[x x^T] for `name`, or None if never seen."""
        if name not in self.gram:
            return None
        G = self.gram[name] / max(1, self.count[name])
        if device is not None:
            G = G.to(device)
        return G.to(dtype)

    def free(self, name: str) -> None:
        self.gram.pop(name, None)
        self.count.pop(name, None)

    def clear(self) -> None:
        self.gram.clear()
        self.count.clear()

    def nbytes(self) -> int:
        return sum(g.numel() * g.element_size() for g in self.gram.values())


@torch.no_grad()
def collect_grams(model,
                  tokenizer,
                  layers: List[Tuple[str, nn.Module]],
                  calib_texts: List[str],
                  device: str,
                  n_calib: int = 128,
                  max_length: int = 512,
                  store_device: str = "cpu",
                  accum_dtype: torch.dtype = torch.float32,
                  max_tokens: Optional[int] = None,
                  verbose: bool = True) -> GramCollector:
    """One calibration pass that fills a GramCollector for `layers`.

    Registering every Linear at once is convenient but memory-hungry (see the
    module docstring). Pass a subset of `layers` and call this repeatedly if the
    model is wide.
    """
    import gc

    col = GramCollector(store_device=store_device,
                        accum_dtype=accum_dtype,
                        max_tokens=max_tokens)
    col.register(layers)
    was_training = model.training
    model.eval()
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
    if was_training:
        model.train()
    if verbose:
        print(f"  Gram collected for {len(col.gram)} layers "
              f"({col.nbytes() / 2**30:.2f} GiB on {store_device})")
    return col
