"""
Learned per-channel scalar codebook quantizer: Codebook3 / Codebook4.

Unlike NF/float formats, a learned codebook drops the fixed canonical shape: the
2**bits levels of each output channel are *searched from the channel's own weight
distribution* (NCC, Sec. 2-3.1 "learned codebooks drop even this structure").

We learn each per-channel codebook by 1-D k-means (Lloyd's algorithm) on the
channel's weights, optionally weighted by a per-weight sensitivity so that
high-impact weights pull the levels toward them (a sensitivity-weighted scalar
codebook). With no sensitivity supplied this is plain Lloyd k-means.

The resulting codebook does NOT factorise as s_j * q_l, so `scale=None` is
returned; NCC reads gaps directly from the sorted learned levels (a sorted-array
neighbour query, exactly as the paper notes for learned codebooks).
"""

from __future__ import annotations

from typing import Optional

import torch

from .base_quantizer import BaseQuantizer, QuantResult


class LearnedCodebookQuantizer(BaseQuantizer):
    """Per-channel learned scalar codebook via (weighted) 1-D k-means."""

    def __init__(
        self,
        bits: int,
        n_iters: int = 20,
        seed: int = 0,
    ):
        if bits not in (3, 4):
            raise ValueError(f"LearnedCodebook supports bits in {{3,4}}, got {bits}")
        super().__init__(bits=bits, per_channel=True)
        self.name = f"codebook{bits}"
        self.num_levels = 2 ** bits
        self.n_iters = n_iters
        self.seed = seed

    @torch.no_grad()
    def _kmeans_1d(
        self,
        W: torch.Tensor,
        sens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Per-row 1-D Lloyd k-means, vectorised across rows.

        Parameters
        ----------
        W : [out, in]   weights (one row = one channel).
        sens : [out, in] or None  optional per-weight sensitivity weights.

        Returns
        -------
        centers : [out, num_levels]  sorted learned levels per row.
        """
        device = W.device
        out_features, in_features = W.shape
        K = self.num_levels
        Wf = W.float()

        # Init centers as per-row quantiles of the weight distribution (stable,
        # deterministic, and already spread across the range -> fast convergence).
        qs = torch.linspace(0.0, 1.0, K, device=device, dtype=torch.float32)
        centers = torch.quantile(Wf, qs, dim=1).t().contiguous()  # [out, K]
        # Force a 0 level near the centre (LLM weights peak at 0).
        zc = centers.abs().argmin(dim=1)
        centers[torch.arange(out_features, device=device), zc] = 0.0
        centers, _ = torch.sort(centers, dim=1)

        w = sens.float() if sens is not None else None

        for _ in range(self.n_iters):
            # Assign: nearest center per weight. [out, in, K]
            d = (Wf.unsqueeze(-1) - centers.unsqueeze(1)).abs()
            assign = d.argmin(dim=-1)  # [out, in]
            del d

            # Update: (weighted) mean of points in each cluster, per row.
            new_centers = centers.clone()
            for k in range(K):
                mask = (assign == k)  # [out, in]
                if w is None:
                    cnt = mask.sum(dim=1).clamp(min=1)  # [out]
                    summ = (Wf * mask).sum(dim=1)
                    mean_k = summ / cnt
                else:
                    wm = w * mask
                    denom = wm.sum(dim=1)
                    summ = (Wf * wm).sum(dim=1)
                    mean_k = torch.where(denom > 0, summ / denom.clamp(min=1e-12), centers[:, k])
                # Only move centers for rows that own >=1 point in cluster k.
                owns = mask.any(dim=1)
                new_centers[:, k] = torch.where(owns, mean_k, centers[:, k])
            centers, _ = torch.sort(new_centers, dim=1)

        # De-duplicate collapsed levels by nudging (keeps strictly increasing
        # codebook required by NCC's neighbour reads).
        centers = self._dedup_increasing(centers)
        return centers

    @staticmethod
    def _dedup_increasing(c: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
        """Ensure each row is strictly increasing by nudging ties upward."""
        c, _ = torch.sort(c, dim=1)
        K = c.shape[1]
        for k in range(1, K):
            bad = c[:, k] <= c[:, k - 1]
            c[:, k] = torch.where(bad, c[:, k - 1] + eps, c[:, k])
        return c

    @torch.no_grad()
    def build_codebook(self, W: torch.Tensor, sens: Optional[torch.Tensor] = None) -> QuantResult:
        torch.manual_seed(self.seed)
        codebook = self._kmeans_1d(W, sens=sens)  # [out, K]
        return QuantResult(
            W_dequant=torch.empty(0),
            indices=torch.empty(0, dtype=torch.long),
            codebook=codebook,
            scale=None,  # learned codebook does not factorise
        )
