"""
Learned scalar codebook quantizer: Codebook3 / Codebook4 — block-wise (standard).

A learned codebook drops the fixed shape: the 2**bits levels are searched from the
weights themselves by 1-D k-means (Lloyd) per (row, block). This is inherently
asymmetric — centers are free to sit anywhere — so the global ASYM flag does not
change it. Realised levels are stored in block_codebooks for NCC to read.

Default granularity is FULL ROW: block_size=None (or <=0) means one block per
output row spanning every input channel, i.e. one learned codebook per row.
Pass an explicit positive block_size to get the classic block-wise behaviour.

Seeding is selectable via init=:
  'quantile' (default) — equal-VALUE cuts read off the already-sorted row. Same
      placement as torch.quantile with linear interpolation, but without its
      ~16M-element ceiling, since no second sort is needed.
  'rank'     — equal-COUNT cuts. Cannot produce an empty cluster.
  'kmeans++' — D2 sampling, ported from flash1dkmeans (unweighted variant).
Lloyd refinement runs after all three.
"""

from __future__ import annotations

import torch

from .base_quantizer import BaseQuantizer, QuantResult


class LearnedCodebookQuantizer(BaseQuantizer):
    """Per-(row, block) learned scalar codebook via 1-D k-means."""

    def __init__(self, bits: int, block_size: int | None = None,
                 n_iters: int = 20, seed: int = 0, init: str = "quantile"):
        if bits not in (3, 4):
            raise ValueError(f"LearnedCodebook supports bits in {{3,4}}, got {bits}")
        if init not in ("kmeans++", "quantile", "rank"):
            raise ValueError(
                f"init must be one of 'kmeans++', 'quantile', 'rank'; got {init!r}"
            )
        super().__init__(bits=bits, block_size=block_size)
        self.name = f"codebook{bits}"
        self.num_levels = 2 ** bits
        self.n_iters = n_iters
        self.seed = seed
        # 'kmeans++' : D2 sampling, ported from flash1dkmeans (unweighted).
        # 'quantile' : equal-VALUE cuts, interpolated off the sorted row. This is
        #              the torch.quantile seed, minus the ~16M element cap --
        #              sorted_X already exists, so a quantile is an index lookup.
        # 'rank'     : equal-COUNT cuts. Cannot produce an empty cluster.
        self.init = init

    @property
    def q_levels(self) -> torch.Tensor:
        # No canonical shape for a learned codebook; expose a placeholder grid so
        # callers that only need L work. Realised levels live in block_codebooks.
        return torch.linspace(-1.0, 1.0, self.num_levels)

    def __repr__(self) -> str:
        bs_str = "full_row" if self.full_row else str(self.block_size)
        return (f"{self.__class__.__name__}(name={self.name!r}, bits={self.bits}, "
                f"block_size={bs_str}, init={self.init!r})")

    @torch.no_grad()
    def _kmeans_blocks(self, Wb: torch.Tensor) -> torch.Tensor:
        """1-D k-means over a stack of blocks, via sorted data + prefix sums.

        Wb : [G, bw]  (G = rows-in-chunk * n_blocks flattened, bw = block width)
        returns centers [G, K] sorted ascending, strictly increasing.

        Init is k-means++ (ported from flash1dkmeans, unweighted variant),
        batched across G; refinement is Lloyd on the sorted array. Works at any
        bw, including full-row (bw = in_features). Peak memory is O(G * bw) for
        the sort plus two fp64 prefix-sum buffers, so at full row `row_chunk` is
        the knob to turn down if memory is tight.
        """
        G, bw = Wb.shape
        K = self.num_levels
        device = Wb.device

        # 1-D k-means on the SORTED row via prefix sums (SqueezeLLM-style).
        #
        # The old quantile seed + Lloyd is fine at bw=64 but collapses at
        # full-row width: an LLM weight row is a spike at 0 with rare outliers,
        # so every interior quantile lands inside the spike, the K levels land
        # on top of each other, and Lloyd cannot recover (collapsed centers own
        # no distinct mass). The row then dequantizes to ~0 -- finite weights,
        # no error, no OOM, NaN perplexity.
        #
        # In 1-D a cluster is always a contiguous run of sorted data, so the
        # problem is just where to cut; cluster means come from prefix sums in
        # O(1) and there is no seed to get stuck in.
        sorted_X, _ = torch.sort(Wb, dim=1)                        # [G, bw]
        # psum[:, i] = sum of first i sorted elements (psum[:, 0] = 0). fp64:
        # a full row is ~1e4 terms and the means come out of differences of
        # large partial sums, which fp32 cancels badly.
        psum = torch.zeros(G, bw + 1, device=device, dtype=torch.float64)
        psum[:, 1:] = torch.cumsum(sorted_X.double(), dim=1)
        # Squared prefix sum: lets inertia of any contiguous run be read in O(1)
        # via  sum(x^2) - 2*c*sum(x) + c^2*n.  This is what makes k-means++
        # affordable here -- each candidate's inertia is K prefix-sum lookups
        # instead of a pass over the data.
        psum_sq = torch.zeros(G, bw + 1, device=device, dtype=torch.float64)
        psum_sq[:, 1:] = torch.cumsum(sorted_X.double() ** 2, dim=1)

        def _centroids(lo, hi):
            """Mean of sorted_X[:, lo:hi) from prefix sums. lo,hi: [G,K] long.

            Non-empty clusters come out ascending automatically. Empty clusters
            do not: the fallback below is a border *sample*, not a mean, and a
            sample can sit below the mean of the wide cluster to its left. The
            midpoint recut assumes ascending centers, so we restore that
            explicitly before returning.
            """
            n = (hi - lo).clamp(min=1).double()
            s = torch.gather(psum, 1, hi) - torch.gather(psum, 1, lo)
            mean = (s / n).float()
            empty = (hi - lo) <= 0
            if bool(empty.any()):
                # An empty cluster has no mass; park it on the data point at its
                # own border. NOTE lo indexes psum (0..bw) but we are reading
                # sorted_X (0..bw-1), hence the clamp -- without it, a border at
                # bw is out of range; with it, every empty cluster at the right
                # edge maps to the row max, which is why the cummax below is
                # load-bearing rather than decorative.
                fallback = torch.gather(sorted_X, 1, lo.clamp(max=bw - 1))
                mean = torch.where(empty, fallback, mean)
                # Restore the ascending invariant the recut depends on. Without
                # this a single empty cluster makes `mid` non-monotonic, which
                # makes searchsorted return out-of-order ranks, which the border
                # cummax then flattens into a run of equal borders -- creating
                # more empty clusters each pass until the levels collapse.
                mean, _ = torch.cummax(mean, dim=1)
            return mean

        # ---- init by k-means++ (ported from flash1dkmeans) ------------------
        # Batched port of flash1dkmeans._kmeans_plusplus_unweighted. The numba
        # original runs one row at a time; here every row in the chunk draws its
        # own centers simultaneously, so searchsorted/inertia are single batched
        # kernels rather than a G-long Python loop.
        #
        # Why k-means++ over equal-rank cuts: rank cuts spend levels in
        # proportion to COUNT, and an LLM weight row is a spike at 0 with rare
        # outliers, so nearly every level lands inside the spike and the tails
        # get nothing. k-means++ samples proportional to squared distance, which
        # is exactly the signal that pulls seeds out to the outliers.

        def _borders_from_centers(sorted_centers):
            """Cluster borders = rank of each midpoint. [G,k] -> [G,k+1].

            k is taken from the input, not fixed at K: during k-means++ this is
            called on partial center sets of width c_id+1.
            """
            k = sorted_centers.shape[1]
            mid = 0.5 * (sorted_centers[:, 1:] + sorted_centers[:, :-1])
            b = torch.empty(G, k + 1, device=device, dtype=torch.long)
            b[:, 0] = 0
            b[:, -1] = bw
            if k > 1:
                b[:, 1:-1] = torch.searchsorted(sorted_X.contiguous(), mid.contiguous())
            return b.clamp_(0, bw)

        def _inertia_per_cluster(sorted_centers, b):
            """Weighted SSE of each cluster, [G,K], from the prefix sums.

            sum(w*(x-c)^2) = sum(x^2) - 2c*sum(x) + c^2*n, all O(1) per cluster.
            """
            lo, hi = b[:, :-1], b[:, 1:]
            n = (hi - lo).double()
            s = torch.gather(psum, 1, hi) - torch.gather(psum, 1, lo)
            s2 = torch.gather(psum_sq, 1, hi) - torch.gather(psum_sq, 1, lo)
            c = sorted_centers.double()
            return (s2 - 2.0 * c * s + c * c * n).clamp_(min=0.0)

        def _closest_sq_dist(sorted_centers):
            """Squared distance from every point to its nearest center. [G,bw].

            Width comes from the input: during k-means++ this is called with
            c_id centers, not K.
            """
            sc, _ = torch.sort(sorted_centers, dim=1)
            k = sc.shape[1]
            if k == 1:
                # One center owns everything; no borders to look up.
                return (sorted_X - sc[:, :1]) ** 2
            b = _borders_from_centers(sc)
            # A point's owner is its rank among the interior cut positions, and
            # those cuts are non-decreasing -- so one batched searchsorted over
            # b[:, 1:-1] replaces a per-cluster comparison loop.
            cuts = b[:, 1:-1].contiguous()                          # [G, k-1]
            pos = torch.arange(bw, device=device).view(1, -1).expand(G, bw)
            owner_idx = torch.searchsorted(cuts, pos.contiguous(), right=True)
            owner = torch.gather(sc, 1, owner_idx.clamp(max=k - 1))
            return (sorted_X - owner) ** 2

        if self.init == "rank":
            # Equal-COUNT cuts straight into border space. No values needed.
            borders = (torch.linspace(0, 1, K + 1, device=device) * bw)
            borders = borders.round().long().clamp(0, bw)
            borders = borders.view(1, K + 1).expand(G, K + 1).contiguous()

        elif self.init == "quantile":
            # Equal-VALUE cuts: the torch.quantile seed without the size cap.
            #
            # torch.quantile refuses inputs past ~16M elements in the reduced
            # dim because it sorts internally. sorted_X is ALREADY sorted, so a
            # quantile here is a fractional index lookup -- exact, O(K), and
            # unbounded in size. This reproduces interpolation='linear'.
            q = (torch.arange(K, device=device, dtype=torch.float32) + 0.5) / K
            pos = q * (bw - 1)                                      # [K]
            lo_i = pos.floor().long().clamp(0, bw - 1)
            hi_i = pos.ceil().long().clamp(0, bw - 1)
            frac = (pos - lo_i.to(pos.dtype)).to(sorted_X.dtype)    # [K]
            lo_v = sorted_X[:, lo_i]                                # [G, K]
            hi_v = sorted_X[:, hi_i]
            centers_q = lo_v + (hi_v - lo_v) * frac.view(1, -1)
            # NOTE equal-value cuts CAN produce empty clusters on a spiky row --
            # that is inherent to the seed, not a bug here. _centroids patches
            # empties and keeps the ladder ascending, so Lloyd can still move.
            centers_q, _ = torch.sort(centers_q, dim=1)
            borders = _borders_from_centers(centers_q)
            borders, _ = torch.cummax(borders, dim=1)
            borders[:, -1] = bw

        else:
            gen = torch.Generator(device=device if device.type != "mps" else "cpu")
            gen.manual_seed(self.seed)

            def _rand_unit(*shape):
                return torch.rand(*shape, device=device, generator=gen)

            centers_pp = torch.empty(G, K, device=device, dtype=sorted_X.dtype)
            # First center: uniform over the row's points, independently per row.
            first = (_rand_unit(G, 1) * bw).long().clamp_(0, bw - 1)
            centers_pp[:, :1] = torch.gather(sorted_X, 1, first)

            n_local_trials = 2 + int(torch.log(torch.tensor(float(K))).item())

            for c_id in range(1, K):
                # D2 sampling: draw candidates with P(x) proportional to its
                # squared distance to the nearest chosen center.
                d2 = _closest_sq_dist(centers_pp[:, :c_id]).double()
                cw = torch.cumsum(d2, dim=1)                            # [G, bw]
                total = cw[:, -1:].clamp(min=1e-30)
                sel = _rand_unit(G, n_local_trials).double() * total    # [G, L]
                cand_idx = torch.searchsorted(cw.contiguous(), sel.contiguous())
                cand_idx = cand_idx.clamp_(0, bw - 1)
                cand = torch.gather(sorted_X, 1, cand_idx)              # [G, L]

                # Score each candidate by the TOTAL inertia it would produce,
                # and keep the best -- the "greedy k-means++" local-trials step.
                best_inertia = torch.full((G,), float("inf"), device=device,
                                          dtype=torch.float64)
                best_cand = cand[:, 0].clone()
                for t in range(n_local_trials):
                    trial = torch.cat([centers_pp[:, :c_id], cand[:, t:t + 1]], dim=1)
                    trial, _ = torch.sort(trial, dim=1)
                    b_t = _borders_from_centers(trial)
                    inertia = _inertia_per_cluster(trial, b_t).sum(dim=1)
                    better = inertia < best_inertia
                    best_inertia = torch.where(better, inertia, best_inertia)
                    best_cand = torch.where(better, cand[:, t], best_cand)
                centers_pp[:, c_id] = best_cand

            centers_pp, _ = torch.sort(centers_pp, dim=1)
            borders = _borders_from_centers(centers_pp)
            borders, _ = torch.cummax(borders, dim=1)
            borders[:, -1] = bw

        # Lloyd on the sorted array: recompute centroids, then recut at the
        # midpoints. Each recut is a searchsorted, so this is O(K log bw).
        for _ in range(self.n_iters):
            centers = _centroids(borders[:, :-1], borders[:, 1:])  # [G, K]
            new_borders = _borders_from_centers(centers)
            # Keep borders non-decreasing and in range.
            new_borders, _ = torch.cummax(new_borders, dim=1)
            new_borders[:, -1] = bw
            if torch.equal(new_borders, borders):
                break
            borders = new_borders

        centers = _centroids(borders[:, :-1], borders[:, 1:])

        # Strictly increasing (NCC neighbour reads require it). Scaled to the
        # row's own span: a fixed 1e-7 sits below fp16 resolution for typical
        # weight magnitudes, so collapsed levels would survive the fp16 cast as
        # exact duplicates.
        span = (sorted_X[:, -1] - sorted_X[:, 0]).clamp(min=1e-12)  # [G]
        eps = (span * 1e-4).clamp(min=1e-7)
        for k in range(1, K):
            bad = centers[:, k] <= centers[:, k - 1]
            centers[:, k] = torch.where(bad, centers[:, k - 1] + eps, centers[:, k])

        # The eps fan-out can walk the tail past the row max; a level above the
        # max is never the nearest codeword, so it is a wasted codeword. Pull
        # the whole ladder back inside [min, max] while keeping it increasing.
        lo_b = sorted_X[:, :1]                                     # [G, 1]
        hi_b = sorted_X[:, -1:]                                    # [G, 1]
        ramp = torch.arange(K, device=device, dtype=centers.dtype) # [K]
        floor = lo_b + ramp * eps.unsqueeze(1)
        ceil = hi_b - (K - 1 - ramp) * eps.unsqueeze(1)
        centers = torch.maximum(torch.minimum(centers, ceil), floor)
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
                block_codebooks[r0:r1, b, :] = centers
                # Informational only: centers are absolute, and dequant is a
                # plain gather from them, so no scale is applied anywhere in
                # this quantizer. Any consumer that multiplies by block_scales
                # (as it would for a fixed-grid quantizer) will double-scale.
                block_scales[r0:r1, b] = Wb.abs().amax(dim=1)

                # centers is strictly increasing, so nearest-codeword is a
                # searchsorted against the midpoints: same result as the
                # [rc, bw, K] argmin, O(log K), no large intermediate.
                mid = 0.5 * (centers[:, 1:] + centers[:, :-1])   # [rc, K-1]
                idx = torch.searchsorted(
                    mid.contiguous(), Wb.contiguous()
                ).clamp_(0, K - 1)                               # [rc, bw]
                deq = torch.gather(centers, 1, idx)
                W_dequant[r0:r1, c0:c1] = deq.to(W.dtype)
                indices[r0:r1, c0:c1] = idx

        return QuantResult(
            W_dequant=W_dequant,
            indices=indices,
            q_levels=self.q_levels.to(device),
            block_scales=block_scales,
            block_size=bs,
            block_codebooks=block_codebooks,   # realised per-(row,block) levels
            block_zeros=None,                  # centers already encode any shift
        )