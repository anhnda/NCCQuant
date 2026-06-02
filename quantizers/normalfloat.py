"""
NormalFloat (NF) codebook quantizer: NF3 and NF4.

NormalFloat places its levels at the quantiles of a standard normal, so they are
dense near zero and sparse in the tails -- a good match for LLM weights, which
concentrate near zero (NCC, Sec. 1-2). The codebook factorises per channel as
c_{j,l} = s_j * q_l with a fixed canonical shape {q_l} and a per-channel absmax
scale s_j.

NF4 uses the 16 canonical levels from the QLoRA paper (Dettmers et al.). The
levels are asymmetric (one extra negative level) and normalised so that the
extremes are -1 and +1.

NF3 (8 levels) is derived the same way: equal-probability-mass quantile bins of
N(0,1), with the level of each bin its conditional mean, then normalised to
[-1, 1]. We compute it numerically so the shape is reproducible rather than
hand-tabulated.
"""

from __future__ import annotations

import torch

from .base_quantizer import BaseQuantizer, QuantResult


# Canonical NF4 levels (QLoRA), normalised to [-1, 1], sorted ascending.
_NF4_LEVELS = [
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
]


def _make_normalfloat_levels(num_levels: int) -> torch.Tensor:
    """Construct NormalFloat levels by equal-mass normal quantile binning.

    Split the probability mass of N(0,1) into `num_levels` equal-mass bins and
    take each bin's representative as the inverse-CDF at its mid-probability.
    Force an exact 0.0 level (LLM weights peak at 0) and normalise to [-1, 1].
    This reproduces the NF construction for arbitrary bit widths (e.g. NF3).
    """
    normal = torch.distributions.Normal(0.0, 1.0)
    # Mid-probabilities of equal-mass bins, avoiding the 0/1 singularities.
    ps = (torch.arange(num_levels, dtype=torch.float64) + 0.5) / num_levels
    levels = normal.icdf(ps)  # quantile midpoints

    # Snap the level nearest 0 exactly to 0 so weights at the peak map cleanly.
    zero_idx = int(levels.abs().argmin())
    levels[zero_idx] = 0.0

    # Normalise to [-1, 1].
    levels = levels / levels.abs().max().clamp(min=1e-12)
    levels, _ = torch.sort(levels)
    return levels.to(torch.float32)


class NormalFloatQuantizer(BaseQuantizer):
    """NF3 / NF4 per-channel NormalFloat quantizer."""

    def __init__(self, bits: int):
        if bits not in (3, 4):
            raise ValueError(f"NormalFloat here supports bits in {{3,4}}, got {bits}")
        super().__init__(bits=bits, per_channel=True)
        self.name = f"nf{bits}"

        if bits == 4:
            q = torch.tensor(_NF4_LEVELS, dtype=torch.float32)
        else:  # bits == 3 -> 8 levels
            q = _make_normalfloat_levels(2 ** bits)
        # Canonical shape, normalised to max|q| = 1.
        self.register_levels(q)

    def register_levels(self, q: torch.Tensor) -> None:
        q, _ = torch.sort(q)
        self.q_levels = q / q.abs().max().clamp(min=1e-12)

    @torch.no_grad()
    def build_codebook(self, W: torch.Tensor) -> QuantResult:
        q = self.q_levels.to(W.device)  # [L]
        scale = self._per_channel_absmax_scale(W, q)  # [out, 1]
        codebook = scale * q.unsqueeze(0)  # [out, L]
        return QuantResult(
            W_dequant=torch.empty(0),
            indices=torch.empty(0, dtype=torch.long),
            codebook=codebook,
            scale=scale,
        )
