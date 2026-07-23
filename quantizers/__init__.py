"""
Non-uniform codebook quantizers (block-wise standard) + NCC / BC correction.

Public API
----------
    get_quantizer(name, **kwargs) -> BaseQuantizer
    apply_ncc(W_fp, qres, mu, ...) -> (W_corrected, NCCStats)
    apply_bias_correction(module, W_fp, W_q, mu, ...) -> BCStats

Names: nf3, nf4, nvfp4, codebook3, codebook4, flexnu2/3/4.

GuidedQuant is NOT registered here: it needs set_hessian()/set_wgrad(), which
quantize.py does not call. Use quantize_guidedquant.py, which builds it directly.

Standard granularity (block = contiguous run along input sharing one scale):
    NF3/NF4   block_size 64        (bitsandbytes default)
    NVFP4     block_size 16        (NVIDIA), FP8 E4M3 block scale
    codebook  full row (default)   one learned codebook per output row
    flexnu    full row (default)   one learned codebook per output row

    cb_block_size=None or <=0 selects full row; pass a positive int for
    classic block-wise granularity.

Contract:
    res = quantizer.quantize(W, row_chunk=1024)   # block-wise, OOM-safe
    W_corr, stats = apply_ncc(W_fp, res, mu, budget_p=..., row_chunk=1024)
    bc_stats = apply_bias_correction(module, W_fp, res.W_dequant, mu, ...)
"""

from __future__ import annotations

from .base_quantizer import BaseQuantizer, QuantResult
from .normalfloat import NormalFloatQuantizer
from .nvfp4 import NVFP4Quantizer
from .learned_codebook import LearnedCodebookQuantizer
from .flexnu import FlexNuQuantizer
from .guidedquant import GuidedQuantQuantizer
from .gram_collect import GramCollector, collect_grams
from .ncc import apply_ncc, NCCStats, james_stein_mean
from .bc import apply_bias_correction, compute_bias_correction, BCStats


def _nf3(**kw):
    return NormalFloatQuantizer(bits=3, block_size=kw.get("nf_block_size", 64))


def _nf4(**kw):
    return NormalFloatQuantizer(bits=4, block_size=kw.get("nf_block_size", 64))


def _nvfp4(**kw):
    return NVFP4Quantizer(bits=4, block_size=kw.get("nvfp4_block_size", 16),
                          fp8_scale=kw.get("fp8_scale", True))


def _codebook3(**kw):
    return LearnedCodebookQuantizer(bits=3, block_size=kw.get("cb_block_size", None),
                                    n_iters=kw.get("n_iters", 20), seed=kw.get("seed", 0))


def _codebook4(**kw):
    return LearnedCodebookQuantizer(bits=4, block_size=kw.get("cb_block_size", None),
                                    n_iters=kw.get("n_iters", 20), seed=kw.get("seed", 0))


def _flexnu(bits):
    def build(**kw):
        return FlexNuQuantizer(
            bits=bits,
            block_size=kw.get("cb_block_size", None),
            iters=kw.get("flexnu_iters", 300),
            lr_scale=kw.get("flexnu_lr_scale", 3e-3),
            lr_cb=kw.get("flexnu_lr_cb", 1e-5),
            tau_frac=kw.get("flexnu_tau_frac", 0.5),
            use_delta3=not kw.get("flexnu_no_s3", False),
            freeze_codebook=kw.get("flexnu_freeze_codebook", False),
            freeze_scale=kw.get("flexnu_freeze_scale", False),
            stage_frac=kw.get("flexnu_stage_frac", 0.0),
            eval_every=kw.get("flexnu_eval_every", 1),
            init=kw.get("flexnu_init", "lloyd"),
            init_iters=kw.get("n_iters", 20),
            row_block=kw.get("flexnu_row_block", 128),
            lambda_s2=kw.get("flexnu_lambda_s2", 0.0),
            seed=kw.get("seed", 0),
            verbose=kw.get("flexnu_verbose", False),
        )
    return build


QUANTIZER_REGISTRY = {
    "nf3": _nf3,
    "nf4": _nf4,
    "nvfp4": _nvfp4,
    "codebook3": _codebook3,
    "codebook4": _codebook4,
    "flexnu2": _flexnu(2),
    "flexnu3": _flexnu(3),
    "flexnu4": _flexnu(4),
}


# Quantizers whose objective needs the full activation Gram G = E[x x^T].
NEEDS_GRAM = {"flexnu2", "flexnu3", "flexnu4"}


def get_quantizer(name: str, **kwargs) -> BaseQuantizer:
    key = name.lower()
    if key not in QUANTIZER_REGISTRY:
        raise ValueError(f"Unknown quantizer {name!r}. Available: {sorted(QUANTIZER_REGISTRY)}")
    return QUANTIZER_REGISTRY[key](**kwargs)


__all__ = [
    "BaseQuantizer", "QuantResult",
    "NormalFloatQuantizer", "NVFP4Quantizer", "LearnedCodebookQuantizer",
    "FlexNuQuantizer", "GuidedQuantQuantizer", "GramCollector", "collect_grams", "NEEDS_GRAM",
    "apply_ncc", "NCCStats", "james_stein_mean",
    "apply_bias_correction", "compute_bias_correction", "BCStats",
    "get_quantizer", "QUANTIZER_REGISTRY",
]