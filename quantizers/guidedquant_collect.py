"""
GuidedQuant signal collection: end-loss saliency, saliency-weighted Hessians,
and squared weight gradients.

This is the piece the existing pipeline does not have. gram_collect.py builds
G = E[x x^T], which is the *unweighted* layer Hessian -- what GuidedQuant's own
code calls the `nosal` ablation. The published method weights every token's
outer product by how much that token's output actually matters to the END LOSS:

    H[i, j, g] = sum_t  x_ti * x_tj * s_tg

with s_tg the mean squared gradient of the LM cross-entropy w.r.t. the module's
output activations, for token t and output-channel group g. Setting s == 1
recovers plain X^T X.

Three artefacts come out of one backward pass over calibration data:

  saliency   [N, seq, G] per (layer, module)   -- weights the Hessian
  H          [G, in, in] per (layer, module)   -- LNQ's objective matrix
  wgrad      [out, in]   per (layer, module)   -- SqueezeLLM's sample_weight

Note the two grouping axes are different things and both are called "group"
in the literature:
  - G here partitions OUTPUT channels; each group gets its own Hessian, because
    different output channels have different sensitivity profiles.
  - SqueezeLLM's group_count partitions INPUT channels into codebook blocks.
This module only deals with the first.

COST. Saliency for one layer is N*seq*G floats; at N=128, seq=2048, G=4 that is
~1M values per module, fine. The Hessians are the expensive part: [G, in, in]
fp32 is G x 64MB for in=4096. Collect and consume one layer at a time.
"""

from __future__ import annotations

import gc
import os
from typing import Dict, List, Optional

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
def _module_map(layer: nn.Module) -> Dict[str, nn.Linear]:
    """Every nn.Linear inside a transformer block, by dotted name."""
    return {n: m for n, m in layer.named_modules() if isinstance(m, nn.Linear)}


def _get_layers(model: nn.Module) -> nn.ModuleList:
    """The transformer block list, across the common HF layouts."""
    for path in ("model.layers", "model.decoder.layers", "transformer.h",
                 "gpt_neox.layers", "model.transformer.h"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            if isinstance(obj, (nn.ModuleList, list)):
                return obj
        except AttributeError:
            continue
    raise RuntimeError(
        "Could not locate the transformer block list on this model. "
        "Add its attribute path to _get_layers()."
    )


# --------------------------------------------------------------------------- #
@torch.enable_grad()
def collect_saliency_and_wgrad(
    model: nn.Module,
    input_tokens: List[torch.Tensor],
    num_groups: int = 1,
    save_dir: Optional[str] = None,
    device: str = "cuda",
    verbose: bool = True,
):
    """One backward pass over calibration data; returns saliency and wgrads.

    Mirrors GuidedQuant's get_gradients():
      - forward hook retains each module's OUTPUT and attaches a grad hook
      - the grad hook squares the output gradient (scaled by 1e3 first, exactly
        as the reference does -- it keeps bf16 storage away from underflow)
        and means it within each output-channel group
      - a weight hook squares the weight gradient in place

    Returns
    -------
    saliency : list over layers of {module_name: [N, seq, G] float}
    wgrad    : list over layers of {module_name: [out, in] float}
    """
    model = model.to(device)
    model.eval()
    # Gradients are required, but nothing here is an optimizer step.
    for p in model.parameters():
        p.requires_grad_(False)

    layers = _get_layers(model)
    n_layers = len(layers)

    # Weight grads only for the Linears we intend to quantize.
    targets: List[Dict[str, nn.Linear]] = [_module_map(l) for l in layers]
    for mm in targets:
        for m in mm.values():
            m.weight.requires_grad_(True)
            if m.weight.grad is not None:
                m.weight.grad = None

    saliency: List[Dict[str, List[torch.Tensor]]] = [
        {name: [] for name in targets[i]} for i in range(n_layers)
    ]

    handles = []

    def make_fwd_hook(li: int, name: str):
        def fwd_hook(module, inp, out):
            if not torch.is_tensor(out) or not out.requires_grad:
                return
            out.retain_grad()

            def grad_hook(grad):
                # grad: [bsz, seq, hidden]
                if grad.dim() != 3:
                    return
                b, s, h = grad.shape
                g = num_groups
                if h % g != 0:
                    # Cannot group evenly; fall back to a single group rather
                    # than silently dropping channels.
                    gs, g_eff = h, 1
                else:
                    gs, g_eff = h // g, g
                # The 1e3 scale is from the reference: squaring a raw fp16-scale
                # gradient underflows before it reaches the bf16 store.
                sq = (grad.float() * 1e3).pow(2).view(b, s, g_eff, gs)
                saliency[li][name].append(sq.mean(dim=-1).bfloat16().cpu())
            out.register_hook(grad_hook)
        return fwd_hook

    for li in range(n_layers):
        for name, mod in targets[li].items():
            handles.append(mod.register_forward_hook(make_fwd_hook(li, name)))

    def square_grad_hook(grad):
        return grad.pow(2)

    for mm in targets:
        for m in mm.values():
            handles.append(m.weight.register_hook(square_grad_hook))

    n = len(input_tokens)
    for i, tokens in enumerate(input_tokens):
        t = tokens.to(device)
        if t.dim() == 1:
            t = t.unsqueeze(0)
        out = model(input_ids=t, labels=t)
        out.loss.backward()
        del out
        if verbose and (i + 1) % 16 == 0:
            print(f"  [saliency] {i + 1}/{n} sequences", flush=True)

    for h in handles:
        h.remove()

    sal_out, wg_out = [], []
    for li in range(n_layers):
        sal_out.append({
            name: (torch.cat(chunks, dim=0) if chunks else None)
            for name, chunks in saliency[li].items()
        })
        wg_out.append({
            name: (m.weight.grad.detach().float().cpu()
                   if m.weight.grad is not None else None)
            for name, m in targets[li].items()
        })

    for mm in targets:
        for m in mm.values():
            m.weight.grad = None
            m.weight.requires_grad_(False)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        for li in range(n_layers):
            torch.save({"saliency": sal_out[li], "wgrad": wg_out[li]},
                       os.path.join(save_dir, f"l{li}.pt"))
        if verbose:
            print(f"  [saliency] saved {n_layers} layers to {save_dir}", flush=True)

    model.cpu()
    gc.collect()
    torch.cuda.empty_cache()
    return sal_out, wg_out


# --------------------------------------------------------------------------- #
class SaliencyHessianCollector:
    """Accumulates H[g] = sum_t s_tg * x_t x_t^T per registered Linear.

    Port of GuidedQuant's SaliencyEngine. The saliency for a module must be
    supplied up front and is consumed in the same token order the calibration
    replays, so the forward passes here must use the SAME sequences, in the
    SAME order, as the backward pass that produced it.

    store_device only controls where the finished Hessian is handed back, NOT
    where it accumulates: accumulation stays on the activation device because
    shipping a [in, in] block to CPU on every (batch, group) would dominate
    runtime. Peak GPU cost is therefore [G, in, in] fp32 for the registered
    modules -- G x 64MB at in=4096, G x 484MB at in=11008. Register ONE module
    at a time on wide layers.
    """

    def __init__(self,
                 store_device: str = "cpu",
                 accum_dtype: torch.dtype = torch.float32):
        self.store_device = store_device
        self.accum_dtype = accum_dtype
        self.XTX: Dict[str, torch.Tensor] = {}
        self._sal: Dict[str, torch.Tensor] = {}
        self._idx: Dict[str, int] = {}
        self._handles = []

    def register(self, name: str, module: nn.Linear, saliency: torch.Tensor):
        """saliency: [N, seq, G] for this module, in calibration order."""
        in_f = module.in_features
        G = saliency.shape[-1]
        self.XTX[name] = torch.zeros(G, in_f, in_f,
                                     dtype=self.accum_dtype,
                                     device=self.store_device)
        self._sal[name] = saliency
        self._idx[name] = 0

        def hook(mod, inp, out):
            X = inp[0]
            if X.dim() == 3:
                bsz = X.shape[0]
                X = X.reshape(-1, X.shape[-1])
            else:
                bsz = 1
            i = self._idx[name]
            s = self._sal[name][i:i + bsz]
            self._idx[name] = i + bsz
            if s.numel() == 0:
                return
            # May be an expand()ed view (the uniform-saliency case), which
            # reshape rejects when non-contiguous.
            s = s.contiguous().reshape(-1, s.shape[-1])
            n = min(X.shape[0], s.shape[0])
            X = X[:n].to(self.accum_dtype)
            s = s[:n].to(device=X.device, dtype=self.accum_dtype)

            # H[g] += X^T diag(s[:,g]) X. Done per group with a plain matmul:
            # a single einsum over (n,i,j,g) would materialise [n,in,in,G] and
            # OOM immediately. The reference's two-step form builds [n,in,G]
            # instead, which is the same trick.
            #
            # Accumulate wherever X already lives. Moving a [in,in] block to
            # CPU on every batch x group would dominate the runtime; the
            # store_device copy happens once, in get().
            Gn = s.shape[-1]
            acc = self.XTX[name]
            if acc.device != X.device:
                acc = acc.to(X.device)
                self.XTX[name] = acc
            Xt = X.transpose(0, 1)
            for g in range(Gn):
                acc[g] += Xt @ (X * s[:, g:g + 1])

        self._handles.append(module.register_forward_hook(hook))

    def get(self, name: str, device=None, dtype=torch.float32) -> torch.Tensor:
        H = self.XTX[name]
        if device is not None:
            H = H.to(device)
        return H.to(dtype)

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def clear(self):
        self.remove()
        self.XTX.clear()
        self._sal.clear()
        self._idx.clear()
        gc.collect()
        torch.cuda.empty_cache()
