"""
NVFP4 codebook quantizer — block-wise E2M1 with FP8 micro-block scales (standard).

NVFP4 = OCP E2M1 4-bit float grid (levels {0,±0.5,±1,±1.5,±2,±3,±4,±6}) with a
per-16 micro-block scale rounded to FP8 (E4M3). This is already the deployed
non-uniform standard for FP4, so it simply uses the shared block-wise assignment
with block_size=16 and an FP8-rounded scale.
"""

from __future__ import annotations

import torch

from .base_quantizer import BaseQuantizer, QuantResult


_E2M1_MAG = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def _make_e2m1_levels() -> torch.Tensor:
    mags = torch.tensor(_E2M1_MAG, dtype=torch.float32)
    levels = torch.cat([-mags, mags]).unique()       # collapses the two zeros -> 15 levels
    levels, _ = torch.sort(levels)
    return levels / levels.abs().max().clamp(min=1e-12)


class NVFP4Quantizer(BaseQuantizer):
    """NVFP4: E2M1 4-bit float, per-16 micro-block FP8 scale."""

    def __init__(self, bits: int = 4, block_size: int = 16, fp8_scale: bool = True):
        if bits != 4:
            raise ValueError(f"NVFP4 is a 4-bit format, got bits={bits}")
        super().__init__(bits=bits, block_size=block_size)
        self.name = "nvfp4"
        self.fp8_scale = fp8_scale
        self._q = _make_e2m1_levels()

    @property
    def q_levels(self) -> torch.Tensor:
        return self._q

    @staticmethod
    def _round_e4m3(x: torch.Tensor) -> torch.Tensor:
        """Round positive scales to nearest E4M3 (FP8): 3 mantissa bits, max 448."""
        x = x.clamp(min=1e-12)
        e = torch.floor(torch.log2(x)).clamp(min=-6.0, max=8.0)
        mant = x / torch.pow(2.0, e)             # [1, 2)
        mant = torch.round(mant * 8.0) / 8.0     # 3 mantissa bits
        return (mant * torch.pow(2.0, e)).clamp(max=448.0)

    @torch.no_grad()
    def quantize(self, W: torch.Tensor, row_chunk: int = 1024) -> QuantResult:
        """Block-wise E2M1 assignment with FP8-rounded per-block scale.

        Identical structure to the base block-wise quantize, but the per-(row,
        block) scale is snapped to FP8 before building the grid. Row-chunked for
        OOM safety.
        """
        device = W.device
        out_features, in_features = W.shape
        q = self.q_levels.to(device).float()
        L = q.numel()
        qmax = q.abs().max().clamp(min=1e-12)
        bs = self.block_size
        n_blocks = (in_features + bs - 1) // bs

        W_dequant = torch.empty_like(W)
        indices = torch.empty(out_features, in_features, dtype=torch.long, device=device)
        block_scales = torch.empty(out_features, n_blocks, device=device, dtype=torch.float32)

        for r0 in range(0, out_features, row_chunk):
            r1 = min(r0 + row_chunk, out_features)
            Wr = W[r0:r1].float()
            for b in range(n_blocks):
                c0 = b * bs
                c1 = min(c0 + bs, in_features)
                Wb = Wr[:, c0:c1]
                absmax = Wb.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
                scale = absmax / qmax
                if self.fp8_scale:
                    scale = self._round_e4m3(scale)
                block_scales[r0:r1, b] = scale.squeeze(1)

                grid = scale * q.unsqueeze(0)
                diff = (Wb.unsqueeze(-1) - grid.unsqueeze(1)).abs()
                idx = diff.argmin(dim=-1)
                deq = torch.gather(grid, 1, idx)
                W_dequant[r0:r1, c0:c1] = deq.to(W.dtype)
                indices[r0:r1, c0:c1] = idx
                del diff, grid

        return QuantResult(
            W_dequant=W_dequant,
            indices=indices,
            q_levels=q,
            block_scales=block_scales,
            block_size=bs,
            block_codebooks=None,
        )
