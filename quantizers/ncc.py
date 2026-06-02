"""
Non-uniform Codebook Correction (NCC).

Implements the post-quantization first-moment corrector of the NCC paper
(Sec. 3.2-3.5, Algorithm 1). Given a base quantizer's discrete assignment
(indices into a per-channel scalar codebook) plus the calibration activation
mean, NCC reassigns selected weights to the *complementary* neighbouring codeword
to cancel the structured first-moment output shift, on an architecture-preserving,
bias-free basis.

Key pieces:
  - complementary-codeword move (Eq. 3): move to the adjacent level on the other
    side of the full-precision weight; displacement = local cell gap g_{i,j}.
  - sign-alignment filter (Eq. 5) + gap-aware ordering eta = |mu_i| / g_{i,j}
    (Eq. 9): no knee-point threshold.
  - greedy prefix under a per-channel budget B_j = ceil(p |A_j|) (Eq. 11), with
    the empty selection always feasible (=> Prop. 1 unconditional descent).

Operates on the layout: W [out, in], channel = output channel = row. The
activation mean mu is a vector over the *input* dimension (length `in`), shared
across channels. For each channel j (row), b_j = mu . e_j where e_j = Wq_j - W_j.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .base_quantizer import QuantResult


@dataclass
class NCCStats:
    flips: int
    bias_before: float          # sum_j b_j^2  (calibration first-moment error)
    bias_after: float
    channels: int


def james_stein_mean(x_bar: torch.Tensor, X_centered_var: Optional[torch.Tensor] = None) -> torch.Tensor:
    """James-Stein shrinkage of the sample mean toward the global mean.

    Stabiliser from NCC Sec. 3.5: shrink x_bar toward its scalar global mean;
    clip the shrinkage to [0,1] so it degenerates back to the sample mean rather
    than over-shrinking. `X_centered_var` (per-coord variance estimate) sets the
    noise level; if absent we use a unit-variance proxy.
    """
    mu0 = x_bar.mean()
    d = x_bar.numel()
    if d < 3:
        return x_bar.clone()
    diff = x_bar - mu0
    ss = (diff * diff).sum().clamp(min=1e-12)
    var = X_centered_var.mean() if X_centered_var is not None else torch.ones((), device=x_bar.device)
    # Classic JS factor, clipped to [0,1].
    shrink = 1.0 - (d - 2) * var / ss
    shrink = shrink.clamp(min=0.0, max=1.0)
    return mu0 + shrink * diff


@torch.no_grad()
def _per_row_neighbor_gaps(
    codebook: torch.Tensor,   # [out, L]  sorted ascending
    indices: torch.Tensor,    # [out, in] assigned level index
    e_sign: torch.Tensor,     # [out, in] sign of residual e = Wq - W
):
    """Complementary gap g_{i,j} and complementary index per weight (Eq. 3).

    If e>0 the assigned codeword sits above W -> complementary move is *down*
    (left gap, to index-1). If e<0 -> move *up* (right gap, index+1). Returns the
    gap magnitude and the target index, with feasibility mask (neighbour exists).
    """
    out_features, L = codebook.shape
    idx = indices

    # Left/right neighbour indices, clamped; feasibility tracked separately.
    left_idx = (idx - 1).clamp(min=0)
    right_idx = (idx + 1).clamp(max=L - 1)

    cur = torch.gather(codebook, 1, idx)
    left = torch.gather(codebook, 1, left_idx)
    right = torch.gather(codebook, 1, right_idx)

    g_left = (cur - left).abs()    # left gap
    g_right = (right - cur).abs()  # right gap

    move_down = e_sign > 0  # complementary side is the left neighbour
    target_idx = torch.where(move_down, left_idx, right_idx)
    gap = torch.where(move_down, g_left, g_right)

    feasible = torch.where(move_down, idx > 0, idx < (L - 1))
    return gap, target_idx, feasible


@torch.no_grad()
def apply_ncc(
    W_fp: torch.Tensor,        # [out, in]  full-precision weights
    qres: QuantResult,         # base-quantizer result (indices, codebook, dequant)
    mu: torch.Tensor,          # [in]  calibration activation mean (input dim)
    budget_p: float = 0.02,
    use_james_stein: bool = True,
    mu_var: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, NCCStats]:
    """Run NCC and return corrected dequantized weights + stats.

    Parameters
    ----------
    W_fp   : full-precision weights [out, in].
    qres   : QuantResult from a BaseQuantizer.quantize (must have indices &
             codebook). Its `codebook` must be the per-row sorted level set.
    mu     : per-input-channel activation mean, length `in`.
    budget_p : budget fraction p in (0,1]; per-channel cap B_j = ceil(p|A_j|).

    Returns
    -------
    W_corrected : [out, in] dequantized weights after NCC.
    stats       : NCCStats.
    """
    device = W_fp.device
    Wq = qres.W_dequant.to(device).float()
    codebook = qres.codebook.to(device).float()  # [out, L]
    indices = qres.indices.to(device)
    out_features, in_features = W_fp.shape

    mu = mu.to(device).float()
    if use_james_stein:
        mu = james_stein_mean(mu, mu_var)

    e = (Wq - W_fp.float())            # residual e = Wq - W   [out, in]
    e_sign = torch.sign(e)
    # b_j = mu . e_j   per channel (row).  [out]
    b = (e * mu.unsqueeze(0)).sum(dim=1)

    gap, target_idx, feasible = _per_row_neighbor_gaps(codebook, indices, e_sign)

    # v_{i,j} = mu_i * sign(e) * g  (first-moment corrected by the move, Eq. 8)
    v = mu.unsqueeze(0) * e_sign * gap  # [out, in]

    # Sign-alignment filter (Eq. 5): sign(mu_i) == sign(e_{i,j} * b_j).
    sign_ok = torch.sign(mu).unsqueeze(0) == torch.sign(e_sign * b.unsqueeze(1))
    admissible = feasible & sign_ok & (gap > 0)

    # eta = |mu_i| / g  (Eq. 9), descending order within each row.
    eta = mu.abs().unsqueeze(0) / gap.clamp(min=1e-12)
    eta = torch.where(admissible, eta, torch.full_like(eta, -1.0))  # push inadmissible to back

    bias_before = float((b * b).sum().item())
    Wq_corr = Wq.clone()
    total_flips = 0

    bias_after_sq = 0.0
    # Per-channel greedy prefix. Loop over rows (channels); each row is a small
    # cumulative scan -- vectorised per row, looped across rows for clarity and
    # bounded memory. For large models this is cheap relative to calibration.
    eta_sorted_idx = torch.argsort(eta, dim=1, descending=True)  # [out, in]

    for j in range(out_features):
        order = eta_sorted_idx[j]
        adm_row = admissible[j, order]
        n_adm = int(adm_row.sum().item())
        bj = float(b[j].item())
        if n_adm == 0 or bj == 0.0:
            bias_after_sq += bj * bj
            continue

        cap = max(1, int((-(-(budget_p * n_adm)) // 1)))  # ceil(p * n_adm)
        cand = order[:n_adm][:cap]                          # top-cap admissible
        v_cand = v[j, cand]                                 # corrections in order

        # Greedy prefix: minimise |b_j - sum_{t<=k} v_t|, k in [0, cap].
        cumv = torch.cumsum(v_cand, dim=0)
        residuals = (bj - cumv).abs()
        # Include k=0 (no flip) value |b_j| -> always feasible (Prop. 1).
        k0 = abs(bj)
        best_k_val, best_k = residuals.min(dim=0)
        if k0 <= float(best_k_val.item()):
            # Empty selection is best; flip nothing.
            bias_after_sq += bj * bj
            continue

        k_star = int(best_k.item()) + 1  # number of flips (cand[:k_star])
        chosen = cand[:k_star]
        tgt = target_idx[j, chosen]
        Wq_corr[j, chosen] = codebook[j, tgt]
        total_flips += k_star
        b_new = bj - float(cumv[k_star - 1].item())
        bias_after_sq += b_new * b_new

    stats = NCCStats(
        flips=total_flips,
        bias_before=bias_before,
        bias_after=bias_after_sq,
        channels=out_features,
    )
    return Wq_corr.to(W_fp.dtype), stats
