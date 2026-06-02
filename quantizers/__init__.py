"""
Non-uniform codebook quantizers + NCC correction.

Public API
----------
    get_quantizer(name, **kwargs) -> BaseQuantizer
    QUANTIZER_REGISTRY                  # name -> factory

Supported names: nf3, nf4, nvfp4, codebook3, codebook4.

Each quantizer exposes a single contract:
    res = quantizer.quantize(W)         # QuantResult(W_dequant, indices, codebook, scale)
and NCC consumes that result:
    W_corr, stats = apply_ncc(W_fp, res, mu, budget_p=...)
"""

from __future__ import annotations

from .base_quantizer import BaseQuantizer, QuantResult
from .normalfloat import NormalFloatQuantizer
from .nvfp4 import NVFP4Quantizer
from .learned_codebook import LearnedCodebookQuantizer
from .ncc import apply_ncc, NCCStats, james_stein_mean


def _nf3(**kw):
    return NormalFloatQuantizer(bits=3)


def _nf4(**kw):
    return NormalFloatQuantizer(bits=4)


def _nvfp4(**kw):
    return NVFP4Quantizer(bits=4, block_size=kw.get("block_size", 16),
                          fp8_scale=kw.get("fp8_scale", True))


def _codebook3(**kw):
    return LearnedCodebookQuantizer(bits=3, n_iters=kw.get("n_iters", 20),
                                    seed=kw.get("seed", 0))


def _codebook4(**kw):
    return LearnedCodebookQuantizer(bits=4, n_iters=kw.get("n_iters", 20),
                                    seed=kw.get("seed", 0))


QUANTIZER_REGISTRY = {
    "nf3": _nf3,
    "nf4": _nf4,
    "nvfp4": _nvfp4,
    "codebook3": _codebook3,
    "codebook4": _codebook4,
}


def get_quantizer(name: str, **kwargs) -> BaseQuantizer:
    """Factory: resolve a quantizer name to a BaseQuantizer instance."""
    key = name.lower()
    if key not in QUANTIZER_REGISTRY:
        raise ValueError(
            f"Unknown quantizer {name!r}. Available: {sorted(QUANTIZER_REGISTRY)}"
        )
    return QUANTIZER_REGISTRY[key](**kwargs)


__all__ = [
    "BaseQuantizer",
    "QuantResult",
    "NormalFloatQuantizer",
    "NVFP4Quantizer",
    "LearnedCodebookQuantizer",
    "apply_ncc",
    "NCCStats",
    "james_stein_mean",
    "get_quantizer",
    "QUANTIZER_REGISTRY",
]
