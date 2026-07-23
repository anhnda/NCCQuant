"""
Learned scalar codebook quantizer: Codebook3 / Codebook4 — block-wise (standard).

A learned codebook drops the fixed shape: the 2**bits levels are searched from the
weights themselves by 1-D k-means (Lloyd) per (row, block). This is inherently
asymmetric — centers are free to sit anywhere — so the global ASYM flag does not
change it. Realised levels are stored in block_codebooks for NCC to read.

Default granularity is FULL ROW: block_size=None (or <=0) means one block per
output row spanning every input channel, i.e. one learned codebook per row.
Pass an explicit positive block_size to get the classic block-wise behaviour.
"""

from __future__ import annotations

import torch

from .base_quantizer import BaseQuantizer, QuantResult


class LearnedCodebookQuantizer(BaseQuantizer):
    """Per-(row, block) learned scalar codebook via 1-D k-means."""

    def __init__(self, bits: int, block_size: int | None = None,
                 n_iters: int = 20, seed: int = 0):
        if bits not in (3, 4):
            raise ValueError(f"LearnedCodebook supports bits in {{3,4}}, got {bits}")
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
    def _kmeans_blocks(self, Wb: torch.Tensor) -> torch.Tensor:
        """1-D k-means over a stack of blocks, via sorted data + prefix sums.

        Wb : [G, bw]  (G = rows-in-chunk * n_blocks flattened, bw = block width)
        returns centers [G, K] sorted ascending, strictly increasing.

        Works at any bw, including full-row (bw = in_features). Peak memory is
        O(G * bw) for the sort plus one fp64 prefix-sum buffer, so at full row
        `row_chunk` is the knob to turn down if memory is tight.
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

        def _centroids(lo, hi):
            """Mean of sorted_X[:, lo:hi) from prefix sums. lo,hi: [G,K] long.

            Borders are non-decreasing, so these means come out ascending
            already; empty clusters are the only exception and are patched
            below.
            """
            n = (hi - lo).clamp(min=1).double()
            s = torch.gather(psum, 1, hi) - torch.gather(psum, 1, lo)
            mean = (s / n).float()
            empty = (hi - lo) <= 0
            if bool(empty.any()):
                # An empty cluster has no mass; park it on the data point at its
                # own border so it stays in order and stays a valid level.
                fallback = torch.gather(sorted_X, 1, lo.clamp(max=bw - 1))
                mean = torch.where(empty, fallback, mean)
            return mean

        # ---- init borders by equal MASS, then refine ------------------------
        # Equal-mass (rank) cuts are the right starting point on a sorted array:
        # unlike equal-value cuts they can never produce an empty cluster.
        borders = torch.linspace(0, bw, K + 1, device=device)
        borders = borders.round().long().view(1, K + 1).expand(G, K + 1).contiguous()

        # Lloyd on the sorted array: recompute centroids, then recut at the
        # midpoints. Each recut is a searchsorted, so this is O(K log bw).
        for _ in range(self.n_iters):
            centers = _centroids(borders[:, :-1], borders[:, 1:])  # [G, K]

            mid = 0.5 * (centers[:, 1:] + centers[:, :-1])         # [G, K-1]
            # New borders = rank of each midpoint in the sorted row.
            inner = torch.searchsorted(sorted_X.contiguous(), mid.contiguous())
            new_borders = torch.empty_like(borders)
            new_borders[:, 0] = 0
            new_borders[:, -1] = bw
            new_borders[:, 1:-1] = inner
            # Keep borders non-decreasing and in range.
            new_borders = new_borders.clamp(0, bw)
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