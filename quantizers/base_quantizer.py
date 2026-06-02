"""
Base interface for non-uniform per-block scalar codebook quantizers.

This package follows the *deployed* standard for non-uniform weight-only PTQ:

    fixed per-format level shape  x  per-BLOCK scale,

where a block is a short contiguous run of weights along the input dimension that
shares one scale (the same idea as "group_size" in uniform GPTQ/AWQ). The fixed
level shape is what makes the format non-uniform:
    - NF3/NF4 : normal-quantile levels (dense near 0)
    - NVFP4   : E2M1 float levels
    - learned : per-block k-means levels (no fixed shape)
Standard block sizes: NF=64 (bitsandbytes), NVFP4=16 (NVIDIA), learned=64.

This matches the NCC paper's setup (Sec. 3.1): each weight is assigned to a scalar
codebook; for factorised formats the codebook is scale * {q_l}. The only change
from a per-channel codebook to a per-block codebook is that the scale (hence the
realised levels and the local gaps NCC reads) varies per block rather than per
row. NCC handles this via the realised neighbour query, exactly as it already
does for NVFP4.

Layout convention
-----------------
"channel" = output channel = a ROW of nn.Linear weight [out_features, in_features].
Blocks partition the INPUT dimension (columns). Each (row, block) owns a scale.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class QuantResult:
    """Result of quantizing a single weight matrix W [out, in].

    Attributes
    ----------
    W_dequant : [out, in]
        Dequantized weights Wq (same dtype as input W).
    indices : [out, in] long
        Per-weight codeword index into that weight's *block* codebook.
    q_levels : [L]
        Canonical normalised level shape (max|level| = 1). Shared across blocks;
        the realised levels for block b are block_scales[:, b] * q_levels.
    block_scales : [out, n_blocks]
        Per-(row, block) scale s_{j,b}. Realised codebook for weight (i in block b
        of row j) is block_scales[j, b] * q_levels.
    block_size : int
        Number of input columns per block.
    block_codebooks : Optional[[out, n_blocks, L]]
        Materialised realised levels per (row, block) when a format does not
        factorise as scale*shape (learned codebooks). When None, reconstruct as
        block_scales[:, b, None] * q_levels.
    """

    W_dequant: torch.Tensor
    indices: torch.Tensor
    q_levels: torch.Tensor
    block_scales: torch.Tensor
    block_size: int
    block_codebooks: Optional[torch.Tensor] = None


class BaseQuantizer(ABC):
    """Abstract non-uniform per-block scalar-codebook quantizer.

    Subclasses provide the canonical level shape via `q_levels` (an attribute or
    property returning a 1-D normalised tensor) and inherit the shared block-wise
    nearest-codeword `quantize`. Learned codebooks override `quantize` because
    their levels are searched per block rather than being a fixed shape.
    """

    name: str = "base"

    def __init__(self, bits: int, block_size: int = 64):
        self.bits = bits
        self.block_size = block_size

    # ------------------------------------------------------------------ #
    # Subclass contract
    # ------------------------------------------------------------------ #
    @property
    @abstractmethod
    def q_levels(self) -> torch.Tensor:
        """Canonical level shape, 1-D, sorted ascending, normalised max|q| = 1."""
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Shared block-wise nearest-codeword assignment (factorised formats)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def quantize(self, W: torch.Tensor, row_chunk: int = 1024) -> QuantResult:
        """Block-wise nearest-codeword quantization of W [out, in].

        Scale is per (row, block) absmax: s_{j,b} = max|W in block| / max|q|, so
        the extreme level reaches the block's largest-magnitude weight. Processed
        in row chunks to bound the transient [chunk, in, L] tensor (OOM-safe;
        result is bit-identical regardless of chunk size since rows are
        independent).
        """
        device = W.device
        out_features, in_features = W.shape
        q = self.q_levels.to(device).float()                 # [L]
        L = q.numel()
        qmax = q.abs().max().clamp(min=1e-12)
        bs = self.block_size
        n_blocks = (in_features + bs - 1) // bs

        W_dequant = torch.empty_like(W)
        indices = torch.empty(out_features, in_features, dtype=torch.long, device=device)
        block_scales = torch.empty(out_features, n_blocks, device=device, dtype=torch.float32)

        for r0 in range(0, out_features, row_chunk):
            r1 = min(r0 + row_chunk, out_features)
            Wr = W[r0:r1].float()                              # [rc, in]
            rc = r1 - r0
            for b in range(n_blocks):
                c0 = b * bs
                c1 = min(c0 + bs, in_features)
                Wb = Wr[:, c0:c1]                              # [rc, bw]
                absmax = Wb.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)  # [rc,1]
                scale = absmax / qmax                          # [rc,1]
                block_scales[r0:r1, b] = scale.squeeze(1)

                grid = scale * q.unsqueeze(0)                  # [rc, L]
                diff = (Wb.unsqueeze(-1) - grid.unsqueeze(1)).abs()  # [rc, bw, L]
                idx = diff.argmin(dim=-1)                      # [rc, bw]
                deq = torch.gather(grid, 1, idx)               # [rc, bw]
                W_dequant[r0:r1, c0:c1] = deq.to(W.dtype)
                indices[r0:r1, c0:c1] = idx
                del diff, grid

        return QuantResult(
            W_dequant=W_dequant,
            indices=indices,
            q_levels=q,
            block_scales=block_scales,
            block_size=bs,
            block_codebooks=None,   # factorised: reconstruct scale * q on demand
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, bits={self.bits}, block_size={self.block_size})"
