"""
Base interface for non-uniform per-channel scalar codebook quantizers.

A quantizer in this package follows the NCC abstraction (Non-uniform Codebook
Correction, Sec. 3.1): each output channel j is quantized with its own strictly
increasing scalar codebook

    C_j = { c_{j,0} < c_{j,1} < ... < c_{j,L-1} }.

For NF / float formats the codebook factorises as a fixed canonical shape scaled
per channel:  c_{j,l} = s_j * q_l, with {q_l} dense near 0 and sparse in the tails.
Learned codebooks drop even this structure (the levels are searched per channel).

The base quantizer is responsible only for the *base* operation: given a weight
matrix W [out, in], produce
    - the assigned indices  L(i,j)
    - the dequantized weight Wq = c_{j, L(i,j)}
    - the per-channel codebook levels (so NCC can read neighbouring codewords).

It deliberately knows nothing about NCC's first-moment correction; NCC consumes
the codebook + indices afterwards.

Convention on layout
---------------------
Throughout this package "channel" = output channel = a *row* of the
nn.Linear weight (shape [out_features, in_features]). The codebook is built per
output channel (per row). This matches the NCC setup where each channel j owns a
codebook C_j and the activation x is shared across channels.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class QuantResult:
    """Result of quantizing a single weight matrix.

    Attributes
    ----------
    W_dequant : torch.Tensor          [out, in]
        The dequantized weights Wq = c_{j, L(i,j)} (same dtype as input W).
    indices : torch.Tensor            [out, in]  (long)
        Per-weight codeword index L(i,j) into the per-channel codebook.
    codebook : torch.Tensor           [out, L]
        The realised per-channel scalar codebook C_j (row j is channel j's
        sorted levels). For factorised formats this is scale_j * q. Strictly
        increasing along dim=1.
    scale : Optional[torch.Tensor]    [out, 1]
        Per-channel scale s_j when the format factorises (NF/float). None for
        purely learned codebooks that do not factorise.
    """

    W_dequant: torch.Tensor
    indices: torch.Tensor
    codebook: torch.Tensor
    scale: Optional[torch.Tensor] = None


class BaseQuantizer(ABC):
    """Abstract per-channel scalar-codebook quantizer.

    Subclasses implement `build_codebook` (the per-channel level set) and inherit
    a shared nearest-codeword `quantize`. Learned codebooks may override
    `quantize` if assignment is not a simple nearest search, but the default
    nearest-codeword assignment matches every format in this package.
    """

    name: str = "base"

    def __init__(self, bits: int, per_channel: bool = True):
        self.bits = bits
        self.per_channel = per_channel
        if not per_channel:
            # The whole package (and NCC) assumes per-channel codebooks.
            raise ValueError("Only per-channel codebooks are supported.")

    # ------------------------------------------------------------------ #
    # Subclass contract
    # ------------------------------------------------------------------ #
    @abstractmethod
    def build_codebook(self, W: torch.Tensor) -> QuantResult:
        """Build the per-channel codebook C_j for weight matrix W [out, in].

        Must return a QuantResult with `codebook` filled and `W_dequant` /
        `indices` *empty placeholders allowed* — the default `quantize` below
        fills them. In practice subclasses just compute the levels here and let
        `quantize` do the assignment. Strictly increasing levels per row are
        required (NCC reads left/right neighbours).
        """
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Shared nearest-codeword assignment
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def quantize(self, W: torch.Tensor) -> QuantResult:
        """Quantize W [out, in] by nearest-codeword assignment on C_j.

        Returns the full QuantResult (dequantized weights, indices, codebook,
        scale). This is the "base quantizer" end-of-pipeline discrete decision
        that NCC later corrects.
        """
        res = self.build_codebook(W)
        codebook = res.codebook  # [out, L]
        out_features, L = codebook.shape

        # Nearest-codeword index per weight.
        # W: [out, in] -> [out, in, 1]; codebook: [out, L] -> [out, 1, L].
        # Distance |W - c|, argmin over L. Computed in float32 for stability.
        Wf = W.float()
        diff = (Wf.unsqueeze(-1) - codebook.float().unsqueeze(1)).abs()  # [out, in, L]
        indices = diff.argmin(dim=-1)  # [out, in]
        del diff

        # Gather the assigned levels back into a dequantized weight.
        W_dequant = torch.gather(codebook, 1, indices)  # [out, in]
        W_dequant = W_dequant.to(W.dtype)

        return QuantResult(
            W_dequant=W_dequant,
            indices=indices,
            codebook=codebook,
            scale=res.scale,
        )

    # ------------------------------------------------------------------ #
    # Helpers shared by factorised (NF / float) formats
    # ------------------------------------------------------------------ #
    @staticmethod
    def _per_channel_absmax_scale(W: torch.Tensor, q_levels: torch.Tensor) -> torch.Tensor:
        """Per-row scale s_j s.t. the codebook spans the row's dynamic range.

        For a canonical level set {q_l} normalised to max |q_l| = 1, choose
        s_j = max_i |W_{i,j}| / max_l |q_l|. With normalised levels this is just
        the per-row absmax, so the extreme codeword reaches the largest weight.
        """
        qmax = q_levels.abs().max().clamp(min=1e-12)
        absmax = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)  # [out, 1]
        return absmax / qmax

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, bits={self.bits})"
