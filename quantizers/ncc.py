"""
Non-uniform Codebook Correction (NCC) — block-aware, row-chunked.

Implements the paper's first-moment corrector (Sec. 3.2-3.5, Algorithm 1) over
the block-wise codebook standard. Each weight sits in a per-(row, block) codebook;
its complementary gap g_{i,j} is read from the neighbouring level *in its own
block's grid*. This is the realised-neighbour query the paper specifies for the
non-uniform / per-block case (the same mechanism NVFP4 needs).

For each output channel j (row): b_j = mu . e_j, e = Wq - W (mu over input dim).
Per-move first-moment change uses gap g read per block; selection is the
sign-aligned, gap-aware eta = |mu_i|/g ordering with greedy prefix under budget
B_j = ceil(p |A_j|). Empty selection is always feasible => Prop. 1 descent.

Memory: processed in row chunks; within a chunk we reconstruct realised levels
[rc, n_blocks, L] (or read learned block_codebooks) and the per-weight neighbour
gaps [rc, in]. No [out, in, L] tensor is ever held for the whole matrix.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch

from .base_quantizer import QuantResult


@dataclass
class NCCStats:
    flips: int
    bias_before: float
    bias_after: float
    channels: int


def james_stein_mean(x_bar: torch.Tensor, var: Optional[torch.Tensor] = None) -> torch.Tensor:
    """James-Stein shrinkage of the sample mean toward its global mean (Sec 3.5)."""
    mu0 = x_bar.mean()
    d = x_bar.numel()
    if d < 3:
        return x_bar.clone()
    diff = x_bar - mu0
    ss = (diff * diff).sum().clamp(min=1e-12)
    v = var.mean() if var is not None else torch.ones((), device=x_bar.device)
    shrink = (1.0 - (d - 2) * v / ss).clamp(min=0.0, max=1.0)
    return mu0 + shrink * diff


@torch.no_grad()
def _block_realised_levels(qres: QuantResult, r0: int, r1: int, device) -> torch.Tensor:
    """Realised level grid for rows [r0:r1], shape [rc, n_blocks, L].

    Factorised formats: block_scales[:, :, None] * q_levels[None, None, :].
    Learned formats: read block_codebooks directly.
    """
    if qres.block_codebooks is not None:
        return qres.block_codebooks[r0:r1].to(device).float()
    q = qres.q_levels.to(device).float()                       # [L]
    bscale = qres.block_scales[r0:r1].to(device).float()       # [rc, n_blocks]
    return bscale.unsqueeze(-1) * q.view(1, 1, -1)             # [rc, n_blocks, L]


@torch.no_grad()
def apply_ncc(
    W_fp: torch.Tensor,
    qres: QuantResult,
    mu: torch.Tensor,
    budget_p: float = 0.02,
    use_james_stein: bool = False,
    mu_var: Optional[torch.Tensor] = None,
    row_chunk: int = 1024,
) -> tuple[torch.Tensor, NCCStats]:
    """Run block-aware NCC; return corrected dequant weights + stats.

    Parameters
    ----------
    W_fp : [out, in] full-precision weights.
    qres : QuantResult from a block-wise quantizer (indices, block info).
    mu   : [in] per-input-channel activation mean.
    budget_p : budget fraction p in (0,1].
    row_chunk : rows processed at once (memory bound; no effect on result).
    """
    device = W_fp.device
    out_features, in_features = W_fp.shape
    bs = qres.block_size
    n_blocks = (in_features + bs - 1) // bs

    mu = mu.to(device).float()
    if use_james_stein:
        mu = james_stein_mean(mu, mu_var.to(device).float() if mu_var is not None else None)

    Wq_full = qres.W_dequant.to(device).float()
    indices_full = qres.indices.to(device)

    Wq_corr = Wq_full.clone()
    total_flips = 0
    bias_before = 0.0
    bias_after = 0.0

    # Column -> block id map (last block may be short).
    col_block = torch.arange(in_features, device=device) // bs   # [in]

    for r0 in range(0, out_features, row_chunk):
        r1 = min(r0 + row_chunk, out_features)
        rc = r1 - r0

        Wr = W_fp[r0:r1].float()                       # [rc, in]
        Wq = Wq_full[r0:r1]                             # [rc, in]
        idx = indices_full[r0:r1]                       # [rc, in] level index within block grid
        levels = _block_realised_levels(qres, r0, r1, device)   # [rc, n_blocks, L]
        L = levels.shape[-1]

        e = Wq - Wr                                     # residual
        e_sign = torch.sign(e)
        b = (e * mu.unsqueeze(0)).sum(dim=1)            # [rc]  channel first-moment

        # Gather each weight's block grid: levels_per_w [rc, in, L].
        # block id per column -> expand to [rc, in, L] index into n_blocks axis.
        blk = col_block.view(1, in_features, 1).expand(rc, in_features, L)
        levels_per_w = torch.gather(levels, 1, blk)     # [rc, in, L]

        # Current / left / right codeword values from the weight's own block grid.
        cur = torch.gather(levels_per_w, 2, idx.unsqueeze(-1)).squeeze(-1)        # [rc,in]
        left_idx = (idx - 1).clamp(min=0)
        right_idx = (idx + 1).clamp(max=L - 1)
        left = torch.gather(levels_per_w, 2, left_idx.unsqueeze(-1)).squeeze(-1)
        right = torch.gather(levels_per_w, 2, right_idx.unsqueeze(-1)).squeeze(-1)
        del levels_per_w, blk

        g_left = (cur - left).abs()
        g_right = (right - cur).abs()
        move_down = e_sign > 0                          # complementary = left neighbour
        gap = torch.where(move_down, g_left, g_right)
        target_idx = torch.where(move_down, left_idx, right_idx)
        target_val = torch.where(move_down, left, right)
        feasible = torch.where(move_down, idx > 0, idx < (L - 1))

        # v = mu * sign(e) * g (first-moment corrected per move).
        v = mu.unsqueeze(0) * e_sign * gap              # [rc, in]
        # Sign filter (Eq.5): sign(mu) == sign(e*b).
        sign_ok = torch.sign(mu).unsqueeze(0) == torch.sign(e_sign * b.unsqueeze(1))
        admissible = feasible & sign_ok & (gap > 0)
        # eta = |mu|/g ordering (Eq.9).
        eta = mu.abs().unsqueeze(0) / gap.clamp(min=1e-12)
        eta = torch.where(admissible, eta, torch.full_like(eta, -1.0))
        order = torch.argsort(eta, dim=1, descending=True)   # [rc, in]

        bias_before += float((b * b).sum().item())

        for rr in range(rc):
            bj = float(b[rr].item())
            o = order[rr]
            adm = admissible[rr, o]
            n_adm = int(adm.sum().item())
            if n_adm == 0 or bj == 0.0:
                bias_after += bj * bj
                continue
            cap = max(1, math.ceil(budget_p * n_adm))  # B_j = ceil(p|A_j|)
            cand = o[:n_adm][:cap]
            v_cand = v[rr, cand]
            cumv = torch.cumsum(v_cand, dim=0)
            residuals = (bj - cumv).abs()
            best_val, best_k = residuals.min(dim=0)
            if abs(bj) <= float(best_val.item()):
                bias_after += bj * bj
                continue
            k_star = int(best_k.item()) + 1
            chosen = cand[:k_star]
            Wq_corr[r0 + rr, chosen] = target_val[rr, chosen]
            total_flips += k_star
            b_new = bj - float(cumv[k_star - 1].item())
            bias_after += b_new * b_new

        del e, e_sign, gap, v, eta, order, levels, cur, left, right
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    stats = NCCStats(flips=total_flips, bias_before=bias_before,
                     bias_after=bias_after, channels=out_features)
    return Wq_corr.to(W_fp.dtype), stats
