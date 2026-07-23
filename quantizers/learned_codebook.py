"""
Learned scalar codebook quantizer: Codebook3 / Codebook4 — block-wise (standard).

A learned codebook drops the fixed shape: the 2**bits levels are searched from the
weights themselves by 1-D k-means (Lloyd) per (row, block). This is inherently
asymmetric — centers are free to sit anywhere — so the global ASYM flag does not
change it. Realised levels are stored in block_codebooks for NCC to read.
"""

from __future__ import annotations

import os

import torch

from .base_quantizer import BaseQuantizer, QuantResult

# Diagnostics are off unless CB_DEBUG=1, so normal runs are unaffected.
_DEBUG = os.environ.get("CB_DEBUG", "") == "1"


class LearnedCodebookQuantizer(BaseQuantizer):
    """Per-(row, block) learned scalar codebook via 1-D k-means."""

    def __init__(self, bits: int, block_size: int | None = 64,
                 n_iters: int = 20, seed: int = 0):
        if bits not in (3, 4):
            raise ValueError(f"LearnedCodebook supports bits in {{3,4}}, got {bits}")
        # Full row is signalled by block_size <= 0 in the base class; accept
        # None as a synonym so either caller convention works.
        if block_size is None:
            block_size = -1
        super().__init__(bits=bits, block_size=block_size)
        self.name = f"codebook{bits}"
        self.num_levels = 2 ** bits
        self.n_iters = n_iters
        self.seed = seed

    @property
    def q_levels(self) -> torch.Tensor:
        # No canonical shape for a learned codebook; expose a placeholder grid so
        # callers that only need L work. Realised levels live in block_codebooks.
        return torch.linspace(-1.0, 1.0, self.num_levels)

    @torch.no_grad()
    def _kmeans_blocks(self, Wb: torch.Tensor,
                       sample_weight: torch.Tensor | None = None) -> torch.Tensor:
        """Vectorised 1-D Lloyd k-means over a stack of blocks.

        Wb : [G, bw]  (G = rows-in-chunk * n_blocks flattened, bw = block width)
        sample_weight : [G, bw] or [1, bw] or None. Per-element importance, as
            in SqueezeLLM: the objective becomes sum(s_i * (w_i - c)^2) instead
            of sum((w_i - c)^2), so levels concentrate where errors actually
            propagate. None -> plain unweighted k-means (unchanged behaviour).
        returns centers [G, K] sorted ascending, strictly increasing.
        """
        G, bw = Wb.shape
        K = self.num_levels
        device = Wb.device

        sw = None
        if sample_weight is not None:
            sw = sample_weight.to(device=device, dtype=Wb.dtype)
            if sw.dim() == 1:
                sw = sw.view(1, -1)
            if sw.shape[0] == 1 and G > 1:
                sw = sw.expand(G, bw)
            sw = sw.clamp(min=0)

        # torch.quantile refuses inputs past ~16M elements in the reduced dim
        # (it sorts internally and the impl caps it). Sort once ourselves and
        # read the quantiles off as fractional indices: identical result to
        # torch.quantile(..., interpolation='linear'), no cap. The sorted copy
        # is reused by the Lloyd assignment below.
        sorted_X, sort_idx = torch.sort(Wb, dim=1)                 # [G, bw]
        qs = torch.linspace(0.0, 1.0, K, device=device, dtype=torch.float32)

        if sw is None:
            # Unweighted: quantile position is a plain fractional rank.
            pos = qs * (bw - 1)                                    # [K]
            lo_i = pos.floor().long().clamp(0, bw - 1)
            hi_i = pos.ceil().long().clamp(0, bw - 1)
            frac = (pos - lo_i.to(pos.dtype)).to(sorted_X.dtype)   # [K]
            centers = sorted_X[:, lo_i] + (sorted_X[:, hi_i] - sorted_X[:, lo_i]) * frac
            centers = centers.contiguous()                         # [G, K]
        else:
            # Weighted: cut at equal WEIGHT MASS rather than equal count, so the
            # seed already reflects importance. Per-row, since the weight mass
            # differs by row once sw is expanded.
            sw_sorted = torch.gather(sw, 1, sort_idx)              # [G, bw]
            cw = torch.cumsum(sw_sorted.double(), dim=1)           # [G, bw]
            total = cw[:, -1:].clamp(min=1e-30)
            targets = qs.double().view(1, -1) * total              # [G, K]
            idx = torch.searchsorted(cw.contiguous(), targets.contiguous())
            idx = idx.clamp_(0, bw - 1)
            centers = torch.gather(sorted_X, 1, idx).contiguous()  # [G, K]

        zc = centers.abs().argmin(dim=1)
        centers[torch.arange(G, device=device), zc] = 0.0
        centers, _ = torch.sort(centers, dim=1)
        if _DEBUG and not torch.isfinite(centers).all():
            raise RuntimeError(
                f"[codebook] non-finite centers from QUANTILE INIT: "
                f"G={G} bw={bw} K={K} -- Wb finite={torch.isfinite(Wb).all().item()}"
            )

        for _it in range(self.n_iters):
            # centers is sorted, so nearest-center is a searchsorted against the
            # midpoints -- same assignment as the [G, bw, K] argmin, but O(log K)
            # and without the 268MB intermediate at full-row width.
            mid = 0.5 * (centers[:, 1:] + centers[:, :-1])         # [G, K-1]
            assign = torch.searchsorted(mid.contiguous(), Wb.contiguous())
            assign = assign.clamp_(0, K - 1)                       # [G, bw]
            new_centers = centers.clone()
            for k in range(K):
                mask = (assign == k)
                if sw is None:
                    denom = mask.sum(dim=1).clamp(min=1).to(Wb.dtype)
                    summ = (Wb * mask).sum(dim=1)
                else:
                    # SqueezeLLM objective: centroid is the WEIGHTED mean.
                    m = mask.to(Wb.dtype) * sw
                    denom = m.sum(dim=1)
                    summ = (Wb * m).sum(dim=1)
                    # A cluster whose total weight is zero contributes nothing to
                    # the objective; fall back to the unweighted mean so the level
                    # still lands on its own data instead of dividing by zero.
                    zero_w = denom <= 0
                    if bool(zero_w.any()):
                        cnt = mask.sum(dim=1).clamp(min=1).to(Wb.dtype)
                        summ = torch.where(zero_w, (Wb * mask).sum(dim=1), summ)
                        denom = torch.where(zero_w, cnt, denom)
                    denom = denom.clamp(min=1e-30)
                mean_k = summ / denom
                owns = mask.any(dim=1)
                new_centers[:, k] = torch.where(owns, mean_k, centers[:, k])
            centers, _ = torch.sort(new_centers, dim=1)
            if _DEBUG and not torch.isfinite(centers).all():
                raise RuntimeError(
                    f"[codebook] non-finite centers inside Lloyd iter={_it} "
                    f"G={G} bw={bw} K={K}"
                )

        # Strictly increasing (NCC neighbour reads require it).
        eps = 1e-7
        for k in range(1, K):
            bad = centers[:, k] <= centers[:, k - 1]
            centers[:, k] = torch.where(bad, centers[:, k - 1] + eps, centers[:, k])
        return centers

    @torch.no_grad()
    def quantize(self, W: torch.Tensor, row_chunk: int = 1024) -> QuantResult:
        torch.manual_seed(self.seed)
        device = W.device
        out_features, in_features = W.shape
        K = self.num_levels
        bs = self._resolve_block_size(in_features)
        n_blocks = (in_features + bs - 1) // bs

        W_dequant = torch.empty_like(W)
        indices = torch.empty(out_features, in_features, dtype=torch.long, device=device)
        block_codebooks = torch.zeros(out_features, n_blocks, K, device=device, dtype=torch.float32)
        block_scales = torch.zeros(out_features, n_blocks, device=device, dtype=torch.float32)

        for r0 in range(0, out_features, row_chunk):
            r1 = min(r0 + row_chunk, out_features)
            Wr = W[r0:r1].float()
            rc = r1 - r0
            for b in range(n_blocks):
                c0 = b * bs
                c1 = min(c0 + bs, in_features)
                Wb = Wr[:, c0:c1]                              # [rc, bw]
                centers = self._kmeans_blocks(Wb)             # [rc, K]

                if _DEBUG:
                    # Where does the nan come from? Three candidates, checked in
                    # the order they could occur.
                    if not torch.isfinite(centers).all():
                        n_bad = (~torch.isfinite(centers)).sum().item()
                        raise RuntimeError(
                            f"[codebook] non-finite CENTERS: r0={r0} b={b} "
                            f"count={n_bad} bw={c1 - c0}"
                        )
                    # Duplicate levels after the eps pass. Not fatal on its own,
                    # but it means codewords are being wasted; if this fires at
                    # full row and not at bw=64, the eps is the problem.
                    dup = (centers[:, 1:] <= centers[:, :-1]).sum().item()
                    # Same question one step later: duplicates that survive the
                    # cast to the model dtype are unrecoverable.
                    c16 = centers.to(W.dtype).float()
                    dup16 = (c16[:, 1:] <= c16[:, :-1]).sum().item()
                    if dup or dup16:
                        print(
                            f"[codebook] r0={r0} b={b} bw={c1 - c0} "
                            f"dup_fp32={dup} dup_after_{W.dtype}={dup16}",
                            flush=True,
                        )

                block_codebooks[r0:r1, b, :] = centers
                block_scales[r0:r1, b] = Wb.abs().amax(dim=1)

                # centers is strictly increasing (enforced above), so this is the
                # same nearest-codeword result as the [rc, bw, K] argmin.
                mid = 0.5 * (centers[:, 1:] + centers[:, :-1])          # [rc, K-1]
                idx = torch.searchsorted(mid.contiguous(), Wb.contiguous())
                idx = idx.clamp_(0, K - 1)
                deq = torch.gather(centers, 1, idx)

                if _DEBUG and not torch.isfinite(deq).all():
                    raise RuntimeError(
                        f"[codebook] non-finite DEQUANT: r0={r0} b={b} "
                        f"(centers were finite, so this is gather/index)"
                    )

                W_dequant[r0:r1, c0:c1] = deq.to(W.dtype)
                indices[r0:r1, c0:c1] = idx

        if _DEBUG:
            # Final gate: if this passes, the quantizer is clean and any nan in
            # perplexity came from save/load/eval, not from here.
            if not torch.isfinite(W_dequant).all():
                raise RuntimeError("[codebook] non-finite W_dequant at exit")
            rel = ((W_dequant.float() - W.float()).norm() / W.float().norm()).item()
            print(f"[codebook] layer done: rel_err={rel:.4f} bs={bs}", flush=True)

        return QuantResult(
            W_dequant=W_dequant,
            indices=indices,
            q_levels=self.q_levels.to(device),
            block_scales=block_scales,
            block_size=bs,
            block_codebooks=block_codebooks,   # realised per-(row,block) levels
            block_zeros=None,                  # centers already encode any shift
        )