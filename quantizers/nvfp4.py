"""
NVFP4 codebook quantizer.

NVFP4 is NVIDIA's 4-bit floating point format built on the OCP E2M1 grid (1 sign,
2 exponent, 1 mantissa bits), combined with fine-grained micro-block scaling: each
contiguous block of 16 weights along the input dimension shares an FP8 (E4M3)
scale, on top of a per-channel (per-tensor in NVIDIA's kernels) global scale.

For NCC's purposes the relevant object is still a *scalar codebook per group of
weights*: within one micro-block the 16 quantization levels are
    c_l = block_scale * E2M1_l ,
i.e. the fixed E2M1 grid scaled by that block's scale. NCC reads neighbouring
codewords from this grid, so the "channel codebook" abstraction is applied at the
micro-block granularity (each block is treated as its own little codebook).

This implementation exposes the *effective per-row codebook* needed by the base
interface by quantizing block-by-block and assembling a dequantized weight, while
also surfacing the canonical E2M1 shape. Because the scale varies per block (not
per row), `scale=None` is returned and NCC reads gaps directly from the realised
codeword neighbours rather than from a single per-row scale.

E2M1 grid (unsigned magnitudes): {0, 0.5, 1, 1.5, 2, 3, 4, 6}, signed -> 15
distinct levels (two zeros collapse). Max magnitude 6.0.
"""

from __future__ import annotations

import torch

from .base_quantizer import BaseQuantizer, QuantResult


# OCP E2M1 representable magnitudes.
_E2M1_MAG = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def _make_e2m1_levels() -> torch.Tensor:
    """Signed E2M1 levels, sorted ascending, normalised to [-1, 1].

    Building +/- of each magnitude and de-duplicating the zero gives 15 distinct
    levels with max magnitude 6.0. Normalised so the extreme is +/-1.
    """
    mags = torch.tensor(_E2M1_MAG, dtype=torch.float32)
    levels = torch.cat([-mags, mags]).unique()  # collapses the two zeros
    levels, _ = torch.sort(levels)
    return levels / levels.abs().max().clamp(min=1e-12)  # max mag -> 1


class NVFP4Quantizer(BaseQuantizer):
    """NVFP4: E2M1 4-bit float with micro-block scaling.

    Parameters
    ----------
    block_size : int
        Micro-block size along the input dimension that shares a scale
        (NVFP4 uses 16).
    fp8_scale : bool
        If True, round each block scale to the nearest E4M3 FP8 value, matching
        the hardware format. Off by default for a cleaner reference.
    """

    def __init__(self, bits: int = 4, block_size: int = 16, fp8_scale: bool = True):
        if bits != 4:
            raise ValueError(f"NVFP4 is a 4-bit format, got bits={bits}")
        super().__init__(bits=bits, per_channel=True)
        self.name = "nvfp4"
        self.block_size = block_size
        self.fp8_scale = fp8_scale
        self.q_levels = _make_e2m1_levels()  # canonical shape, [-1,1]

    # -- FP8 E4M3 scale rounding ------------------------------------------- #
    @staticmethod
    def _round_e4m3(x: torch.Tensor) -> torch.Tensor:
        """Round positive scales to the nearest E4M3 (FP8) value.

        E4M3: 4 exponent bits (bias 7), 3 mantissa bits, max finite 448. We round
        the mantissa to 3 bits at the value's binade. Zeros/negatives are clamped
        to a tiny positive floor (scales are positive by construction).
        """
        x = x.clamp(min=1e-12)
        e = torch.floor(torch.log2(x))
        e = e.clamp(min=-6.0, max=8.0)  # E4M3 normal exponent range
        mant = x / torch.pow(2.0, e)  # in [1, 2)
        mant = torch.round(mant * 8.0) / 8.0  # 3 mantissa bits
        out = mant * torch.pow(2.0, e)
        return out.clamp(max=448.0)

    @torch.no_grad()
    def build_codebook(self, W: torch.Tensor) -> QuantResult:
        # NVFP4's codebook varies per micro-block, so build_codebook returns the
        # canonical shape and lets `quantize` (overridden below) assemble the
        # block-wise result. We expose a representative per-row codebook (the row
        # absmax-scaled E2M1 grid) for callers that only need the shape.
        q = self.q_levels.to(W.device)
        scale = self._per_channel_absmax_scale(W, q)
        codebook = scale * q.unsqueeze(0)
        return QuantResult(
            W_dequant=torch.empty(0),
            indices=torch.empty(0, dtype=torch.long),
            codebook=codebook,
            scale=scale,
        )

    @torch.no_grad()
    def quantize(self, W: torch.Tensor) -> QuantResult:
        """Block-wise E2M1 quantization with per-block (micro) scaling.

        Returns a QuantResult whose `codebook` is the *realised per-weight*
        codebook expanded to [out, in, L]? No -- to keep the interface uniform we
        instead return, per row, the union grid is block-dependent, so we return
        the dequantized weights + indices into a *per-(row,block)* codebook. For
        NCC compatibility we additionally store `block_codebooks` so the corrector
        can read neighbours at the correct block scale.
        """
        device = W.device
        out_features, in_features = W.shape
        q = self.q_levels.to(device).float()  # [L], normalised E2M1, max=1
        L = q.numel()
        bs = self.block_size
        n_blocks = (in_features + bs - 1) // bs

        W_dequant = torch.empty_like(W)
        indices = torch.empty(out_features, in_features, dtype=torch.long, device=device)
        # Per (row, block) scale; lets NCC reconstruct the local grid.
        block_scales = torch.zeros(out_features, n_blocks, device=device, dtype=torch.float32)

        Wf = W.float()
        for b in range(n_blocks):
            lo = b * bs
            hi = min(lo + bs, in_features)
            Wb = Wf[:, lo:hi]  # [out, bw]
            absmax = Wb.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)  # [out,1]
            scale = absmax / q.abs().max()  # extreme E2M1 level reaches absmax
            if self.fp8_scale:
                scale = self._round_e4m3(scale)
            block_scales[:, b] = scale.squeeze(1)

            grid = scale * q.unsqueeze(0)  # [out, L] for this block
            diff = (Wb.unsqueeze(-1) - grid.unsqueeze(1)).abs()  # [out, bw, L]
            idx = diff.argmin(dim=-1)  # [out, bw]
            deq = torch.gather(grid, 1, idx)  # [out, bw]

            W_dequant[:, lo:hi] = deq.to(W.dtype)
            indices[:, lo:hi] = idx
            del diff, grid

        res = QuantResult(
            W_dequant=W_dequant,
            indices=indices,
            codebook=q.unsqueeze(0).expand(out_features, L).contiguous(),  # canonical shape
            scale=None,  # scale varies per block, not per row
        )
        # Attach block-level info for NCC to read per-block gaps.
        res.block_scales = block_scales  # type: ignore[attr-defined]
        res.block_size = bs  # type: ignore[attr-defined]
        res.q_levels = q  # type: ignore[attr-defined]
        return res
