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
                 n_iters: int = 50, seed: int = 0, init: str = "kmeans++"):
        if bits not in (3, 4):
            raise ValueError(f"LearnedCodebook supports bits in {{3,4}}, got {bits}")
        if init not in ("kmeans++", "quantile"):
            raise ValueError(
                f"init must be 'kmeans++' or 'quantile'; got {init!r}")
        # Full row is signalled by block_size <= 0 in the base class; accept
        # None as a synonym so either caller convention works.
        if block_size is None:
            block_size = -1
        super().__init__(bits=bits, block_size=block_size)
        self.name = f"codebook{bits}"
        self.num_levels = 2 ** bits
        self.n_iters = n_iters
        self.seed = seed
        # 'kmeans++' : greedy D2 sampling, as SqueezeLLM does it (via
        #              flash1dkmeans._kmeans_plusplus). Honours sample_weight.
        # 'quantile' : equal-mass cuts off the sorted row. Cheaper, and what the
        #              pre-existing block_size=64 path used.
        self.init = init

    @property
    def q_levels(self) -> torch.Tensor:
        # No canonical shape for a learned codebook; expose a placeholder grid so
        # callers that only need L work. Realised levels live in block_codebooks.
        return torch.linspace(-1.0, 1.0, self.num_levels)

    @torch.no_grad()
    def _kmeanspp_init(self, sorted_X: torch.Tensor,
                       sw_sorted: torch.Tensor | None, K: int) -> torch.Tensor:
        """Greedy k-means++ seeding, batched over rows. [G, bw] -> [G, K].

        Port of flash1dkmeans._kmeans_plusplus (the routine SqueezeLLM seeds
        with). The numba original runs one row at a time; here all G rows draw
        simultaneously, so each step is a single batched kernel.

        Weighted when sw_sorted is given: the first draw is proportional to
        weight, and later draws to  s_x * d(x)^2  -- so importance shapes the
        seed, not just the Lloyd refinement that follows.

        Inertia is read from prefix sums rather than recomputed, which is what
        makes scoring 2+log(K) candidates per step affordable:
            sum(s*(x-c)^2) = S2 - 2c*S1 + c^2*S0
        with S0/S1/S2 the running sums of s, s*x, s*x^2.
        """
        G, bw = sorted_X.shape
        device = sorted_X.device
        Xd = sorted_X.double()
        if sw_sorted is None:
            w = torch.ones(G, bw, device=device, dtype=torch.float64)
        else:
            w = sw_sorted.double()

        def _pfx(t):
            p = torch.zeros(G, bw + 1, device=device, dtype=torch.float64)
            p[:, 1:] = torch.cumsum(t, dim=1)
            return p

        S0, S1, S2 = _pfx(w), _pfx(w * Xd), _pfx(w * Xd * Xd)

        def _borders(sc):
            """Rank of each midpoint. [G,k] -> [G,k+1]. k comes from the input."""
            k = sc.shape[1]
            b = torch.empty(G, k + 1, device=device, dtype=torch.long)
            b[:, 0] = 0
            b[:, -1] = bw
            if k > 1:
                mid = 0.5 * (sc[:, 1:] + sc[:, :-1])
                b[:, 1:-1] = torch.searchsorted(
                    sorted_X.contiguous(), mid.contiguous())
            return b.clamp_(0, bw)

        def _total_inertia(sc):
            b = _borders(sc)
            lo, hi = b[:, :-1], b[:, 1:]
            s0 = torch.gather(S0, 1, hi) - torch.gather(S0, 1, lo)
            s1 = torch.gather(S1, 1, hi) - torch.gather(S1, 1, lo)
            s2 = torch.gather(S2, 1, hi) - torch.gather(S2, 1, lo)
            c = sc.double()
            return (s2 - 2.0 * c * s1 + c * c * s0).clamp_(min=0.0).sum(dim=1)

        def _closest_d2(sc):
            """Squared distance to the nearest chosen center. [G, bw]."""
            k = sc.shape[1]
            if k == 1:
                return (sorted_X - sc[:, :1]) ** 2
            b = _borders(sc)
            cuts = b[:, 1:-1].contiguous()                          # [G, k-1]
            pos = torch.arange(bw, device=device).view(1, -1).expand(G, bw)
            owner_idx = torch.searchsorted(cuts, pos.contiguous(), right=True)
            owner = torch.gather(sc, 1, owner_idx.clamp(max=k - 1))
            return (sorted_X - owner) ** 2

        gen = torch.Generator(device=device if device.type != "mps" else "cpu")
        gen.manual_seed(self.seed)

        def _rand(*shape):
            return torch.rand(*shape, device=device, generator=gen, dtype=torch.float64)

        centers = torch.empty(G, K, device=device, dtype=sorted_X.dtype)
        # First center: drawn proportional to weight (uniform when unweighted).
        cw0 = torch.cumsum(w, dim=1)
        first = torch.searchsorted(
            cw0.contiguous(), (_rand(G, 1) * cw0[:, -1:].clamp(min=1e-30)).contiguous())
        centers[:, :1] = torch.gather(sorted_X, 1, first.clamp_(0, bw - 1))

        n_trials = 2 + int(torch.log(torch.tensor(float(K))).item())

        for c_id in range(1, K):
            d2 = _closest_d2(centers[:, :c_id]).double() * w        # weighted D2
            cw = torch.cumsum(d2, dim=1)
            total = cw[:, -1:].clamp(min=1e-30)
            sel = _rand(G, n_trials) * total
            cand_idx = torch.searchsorted(cw.contiguous(), sel.contiguous())
            cand = torch.gather(sorted_X, 1, cand_idx.clamp_(0, bw - 1))  # [G, L]

            best_e = torch.full((G,), float("inf"), device=device, dtype=torch.float64)
            best_c = cand[:, 0].clone()
            for t in range(n_trials):
                trial, _ = torch.sort(
                    torch.cat([centers[:, :c_id], cand[:, t:t + 1]], dim=1), dim=1)
                e = _total_inertia(trial)
                better = e < best_e
                best_e = torch.where(better, e, best_e)
                best_c = torch.where(better, cand[:, t], best_c)
            centers[:, c_id] = best_c

        centers, _ = torch.sort(centers, dim=1)
        return centers

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
            # GuidedQuant masks the sample weight wherever the weight itself is
            # exactly zero (weight_mask = weights_np != 0): a pruned/absent
            # weight has no reconstruction error to pay for, so it must not
            # attract a level.
            sw = sw * (Wb != 0).to(sw.dtype)
            # If a row's weights sum to zero the weighted objective is degenerate
            # -- GuidedQuant falls back to uniform weights for that row rather
            # than dividing by ~0. Same here, per row.
            row_total = sw.sum(dim=1, keepdim=True)
            degenerate = row_total <= 0
            if bool(degenerate.any()):
                sw = torch.where(degenerate, torch.ones_like(sw), sw)

        # torch.quantile refuses inputs past ~16M elements in the reduced dim
        # (it sorts internally and the impl caps it). Sort once ourselves and
        # read the quantiles off as fractional indices: identical result to
        # torch.quantile(..., interpolation='linear'), no cap. The sorted copy
        # is reused by the Lloyd assignment below.
        sorted_X, sort_idx = torch.sort(Wb, dim=1)                 # [G, bw]
        sw_sorted = None
        if sw is not None:
            sw_sorted = torch.gather(sw, 1, sort_idx)              # [G, bw]

        if self.init == "kmeans++":
            centers = self._kmeanspp_init(sorted_X, sw_sorted, K)
        else:
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
                # Weighted: cut at equal WEIGHT MASS rather than equal count, so
                # the seed already reflects importance.
                cw = torch.cumsum(sw_sorted.double(), dim=1)           # [G, bw]
                total = cw[:, -1:].clamp(min=1e-30)
                targets = qs.double().view(1, -1) * total              # [G, K]
                idx = torch.searchsorted(cw.contiguous(), targets.contiguous())
                idx = idx.clamp_(0, bw - 1)
                centers = torch.gather(sorted_X, 1, idx).contiguous()  # [G, K]

        centers, _ = torch.sort(centers, dim=1)
        if _DEBUG and not torch.isfinite(centers).all():
            raise RuntimeError(
                f"[codebook] non-finite centers from {self.init} INIT: "
                f"G={G} bw={bw} K={K} -- Wb finite={torch.isfinite(Wb).all().item()}"
            )

        # ---- Lloyd, ported from flash1dkmeans.numba_kmeans_1d_k_cluster -----
        # Their loop works on borders into sorted_X, not on assignments over the
        # raw row: centroid k is a prefix-sum query over [border_k, border_k+1),
        # which is O(1) instead of a masked reduction over [G, bw]. Iteration
        # stops when the borders stop moving.
        Xd = sorted_X.double()
        if sw_sorted is None:
            wd = torch.ones(G, bw, device=device, dtype=torch.float64)
        else:
            wd = sw_sorted.double()

        def _pfx(t):
            p = torch.zeros(G, bw + 1, device=device, dtype=torch.float64)
            p[:, 1:] = torch.cumsum(t, dim=1)
            return p

        W0 = _pfx(wd)                 # weights_prefix_sum
        W1 = _pfx(wd * Xd)            # weighted_X_prefix_sum
        C1 = _pfx(Xd)                 # plain sums, for the zero-weight branch

        borders = torch.full((G, K + 1), -1, device=device, dtype=torch.long)
        borders[:, 0] = 0
        borders[:, -1] = bw

        for _it in range(self.n_iters):
            # _centroids_to_cluster_borders: cut at the midpoints.
            new_borders = borders.clone()
            if K > 1:
                mid = 0.5 * (centers[:, 1:] + centers[:, :-1])
                new_borders[:, 1:-1] = torch.searchsorted(
                    sorted_X.contiguous(), mid.contiguous())
            new_borders.clamp_(0, bw)
            # Guard not present in the reference: their centroids stay sorted by
            # construction, ours come back from a masked where() that can leave
            # an empty cluster's stale level out of order. cummax keeps the
            # borders monotone so the prefix-sum queries stay well-formed.
            new_borders, _ = torch.cummax(new_borders, dim=1)
            new_borders[:, 0] = 0
            new_borders[:, -1] = bw

            if torch.equal(new_borders, borders):
                break
            borders = new_borders

            lo, hi = borders[:, :-1], borders[:, 1:]
            wsum = torch.gather(W0, 1, hi) - torch.gather(W0, 1, lo)
            wxsum = torch.gather(W1, 1, hi) - torch.gather(W1, 1, lo)
            # cluster_weight_sum == 0 -> plain mean of the cluster (their branch)
            cnt = (hi - lo).double()
            xsum = torch.gather(C1, 1, hi) - torch.gather(C1, 1, lo)
            zero_w = wsum <= 0
            num = torch.where(zero_w, xsum, wxsum)
            den = torch.where(zero_w, cnt, wsum).clamp(min=1e-30)
            mean = (num / den).to(centers.dtype)
            # cluster_start == cluster_end -> `continue`, i.e. keep the old level
            empty = (hi - lo) <= 0
            centers = torch.where(empty, centers, mean)

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