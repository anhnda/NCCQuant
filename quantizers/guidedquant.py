"""
GuidedQuant: LNQ coordinate descent against SALIENCY-WEIGHTED, PER-GROUP
Hessians, initialised by SqueezeLLM weighted k-means on squared end-loss
weight gradients.

Two modes, selected by what you feed it:

  --nosal      H = E[x x^T], unweighted; no wgrad init weighting.
               This is the paper's `nosal` ablation -- ordinary LNQ.
  default      H[g] = sum_t s_tg x_t x_t^T, saliency-weighted;
               init weighted by squared end-loss weight gradients.
               This is the published method.

Both signals come from guidedquant_collect.collect_saliency_and_wgrad(), which
runs one backward pass of the LM cross-entropy over calibration data.

At the default num_groups=1 -- full row, one Hessian per layer -- the two modes
run the SAME coordinate descent and differ only in what they are given. That
makes the pair a clean ablation of the end-loss signal.

num_groups > 1 adds a second axis: output rows are partitioned and each group
fits its own Hessian, since output channels within a layer have different
sensitivity profiles.
"""

from __future__ import annotations

import os
from typing import Optional

import torch

from .base_quantizer import BaseQuantizer, QuantResult
from .learned_codebook import LearnedCodebookQuantizer

_DEBUG = os.environ.get("CB_DEBUG", "") == "1"


class GuidedQuantQuantizer(BaseQuantizer):
    """Full-row codebook quantizer, LNQ against saliency-weighted Hessians.

    Granularity is always FULL ROW: one codebook per output row, spanning every
    input channel. There is no input-side blocking -- the Hessian couples all
    input channels, so splitting them would discard the coupling the method
    exists to exploit.

    Three size parameters, and only one of them is an algorithm choice:

      init_row_batch    output rows per k-means init batch. Pure memory knob;
                        any value gives identical results.
      solve_row_batch   output rows per update_C least-squares batch. Pure
                        memory knob; any value gives identical results.
      cd_block_size     column block in update_P. NOT a memory knob -- it sets
                        the order in which the residual is propagated (inside
                        a block, column by column; across blocks, once at the
                        end), so changing it CHANGES THE ASSIGNMENTS. The
                        reference hardcodes 128; leave it there unless you mean
                        to deviate.
    """

    def __init__(self, bits: int,
                 block_size: int | None = None,
                 cd_cycles: int = 4,
                 iters: int = 3,
                 ridge: float = 1e-7,
                 cd_block_size: int = 128,
                 init_row_batch: int = 1024,
                 solve_row_batch: int = 64,
                 init_iters: int = 50,
                 kmeans_init: str = "kmeans++",
                 seed: int = 0,
                 verbose: bool = False):
        if bits not in (2, 3, 4):
            raise ValueError(f"GuidedQuant supports bits in {{2,3,4}}, got {bits}")
        if block_size is None:
            block_size = -1
        super().__init__(bits=bits, block_size=block_size)
        self.name = f"guidedquant{bits}"
        self.num_levels = 2 ** bits
        self.cd_cycles = int(cd_cycles)
        self.iters = int(iters)
        self.ridge = float(ridge)
        self.cd_block_size = int(cd_block_size)
        self.solve_row_batch = int(solve_row_batch)
        self.init_row_batch = int(init_row_batch)
        self.init_iters = int(init_iters)
        self.kmeans_init = kmeans_init
        self.seed = int(seed)
        self.verbose = bool(verbose)
        self._H: Optional[torch.Tensor] = None      # [G, in, in]
        self._wgrad: Optional[torch.Tensor] = None  # [out, in]

    # ------------------------------------------------------------------ #
    def set_hessian(self, H: Optional[torch.Tensor]) -> None:
        """Saliency-weighted Hessians for this layer, [G, in, in] or [in, in]."""
        if H is not None and H.dim() == 2:
            H = H.unsqueeze(0)
        self._H = H

    def set_wgrad(self, g: Optional[torch.Tensor]) -> None:
        """Squared end-loss weight gradients [out, in]: SqueezeLLM's weight."""
        self._wgrad = g

    def set_gram(self, G: Optional[torch.Tensor]) -> None:
        """Compatibility with the NEEDS_GRAM plumbing.

        If no saliency Hessian was supplied, an unweighted Gram is accepted and
        treated as a single-group Hessian -- i.e. the nosal ablation. It never
        overrides a real saliency Hessian.
        """
        if self._H is None:
            self.set_hessian(G)

    @property
    def q_levels(self) -> torch.Tensor:
        return torch.linspace(-1.0, 1.0, self.num_levels)

    # ------------------------------------------------------------------ #
    def _prepare_one(self, H: torch.Tensor, pin: int, device, dtype):
        """Pad/trim, symmetrise, dampen until PD. Returns (H, Hn, L).

        Damping follows the reference: 1e-5 of the mean diagonal, x10 per
        failure, give up past 1e0. Hn = H / diag is for update_P only and is
        NOT symmetric -- never hand it to Cholesky.
        """
        H = H.to(device=device, dtype=dtype).clone()
        if H.shape[0] < pin:
            pad = pin - H.shape[0]
            H = torch.nn.functional.pad(H, (0, pad, 0, pad))
            idx = torch.arange(H.shape[0] - pad, H.shape[0], device=device)
            H[idx, idx] = 1.0
        elif H.shape[0] > pin:
            H = H[:pin, :pin]
        H = 0.5 * (H + H.transpose(0, 1))

        diag = torch.arange(pin, device=device)
        avg_diag = torch.diagonal(H).mean().clamp(min=1e-12)
        damp, prev = 1e-5, 0.0
        while True:
            try:
                L = torch.linalg.cholesky(H)
                break
            except Exception:
                H[diag, diag] += (damp - prev) * avg_diag
                prev, damp = damp, damp * 10
                if damp > 1e0:
                    raise RuntimeError(
                        "GuidedQuant: Hessian not PD at damping 1e0. Use more "
                        "calibration samples, or check saliency collection."
                    )
        d = torch.diagonal(H).clamp(min=1e-12)
        return H, H / d.view(-1, 1), L

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _update_P(self, W, Hn, C, labels):
        """Coordinate descent over input columns (reference update_P)."""
        d = W.shape[1]
        bs = self.cd_block_size
        W_hat = torch.gather(C, 1, labels)
        tril = torch.tril(Hn, diagonal=-1)
        for _ in range(self.cd_cycles):
            B = (W_hat - W) @ tril
            for start in range(0, d, bs):
                end = min(start + bs, d)
                for j in range(start, end):
                    sol = W[:, j:j + 1] - B[:, j:j + 1]
                    am = (sol - C).abs().argmin(dim=1)
                    labels[:, j] = am
                    W_hat[:, j:j + 1] = torch.gather(C, 1, am.view(-1, 1))
                    if j < end - 1:
                        resid = W_hat[:, j:j + 1] - W[:, j:j + 1]
                        B[:, j + 1:end] += resid * Hn[j, j + 1:end].view(1, -1)
                if end < d:
                    B[:, end:] += (W_hat[:, start:end] - W[:, start:end]) @ Hn[start:end, end:]
        return labels

    @torch.no_grad()
    def _update_C(self, W, reduced_X, labels, K):
        """Least squares for the codebook in the whitened space (reference).

        Deliberately does NOT sort: sorting permutes C while labels still index
        the old slots. Ordering happens once at the end of quantize(), with
        labels remapped through the same permutation.
        """
        R = W.shape[0]
        device, dtype = W.device, W.dtype
        out = torch.empty(R, K, device=device, dtype=dtype)
        sl = torch.sqrt(torch.tensor(self.ridge, dtype=dtype, device=device))
        for st in range(0, R, self.solve_row_batch):
            en = min(st + self.solve_row_batch, R)
            P = torch.nn.functional.one_hot(labels[st:en].long(),
                                            num_classes=K).to(dtype)
            A = torch.einsum('bj,ijc->ibc', reduced_X, P)
            b = torch.einsum('bj,ij->ib', reduced_X, W[st:en]).unsqueeze(-1)
            n = A.shape[0]
            I = sl * torch.eye(K, dtype=dtype, device=device).unsqueeze(0).expand(n, -1, -1)
            A = torch.cat([A.transpose(1, 2), I], dim=2).transpose(1, 2)
            b = torch.cat([b, torch.zeros((n, K, 1), dtype=dtype, device=device)], dim=1)
            out[st:en] = torch.linalg.lstsq(A, b).solution.squeeze(-1)
        return out

    @torch.no_grad()
    def _objective(self, W, H, C, labels) -> float:
        dw = torch.gather(C, 1, labels) - W
        return float(torch.einsum('ij,jk,ik->', dw, H, dw) / max(W.shape[0], 1))

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def quantize(self, W: torch.Tensor) -> QuantResult:
        if self._H is None:
            raise RuntimeError(
                "GuidedQuant needs the saliency-weighted Hessian. Run "
                "collect_saliency_and_wgrad + SaliencyHessianCollector and "
                "call set_hessian() before quantize()."
            )
        device = W.device
        dtype = torch.float32
        out_features, in_features = W.shape
        K = self.num_levels
        bs = self._resolve_block_size(in_features)
        if bs != in_features:
            raise NotImplementedError(
                "GuidedQuant is full-row only; the Hessian couples all inputs."
            )

        Wf = W.to(dtype)
        Hs = self._H
        n_groups = Hs.shape[0]
        if out_features % n_groups != 0:
            raise ValueError(
                f"out_features={out_features} not divisible by num_groups="
                f"{n_groups}; the row partition must be exact."
            )
        gsz = out_features // n_groups

        # ---- init: SqueezeLLM weighted k-means ------------------------------
        helper = LearnedCodebookQuantizer(
            bits=self.bits if self.bits in (3, 4) else 3,
            block_size=-1, n_iters=self.init_iters,
            seed=self.seed, init=self.kmeans_init,
        )
        helper.num_levels = K

        C = torch.empty(out_features, K, device=device, dtype=dtype)
        labels = torch.empty(out_features, in_features, device=device, dtype=torch.long)
        for r0 in range(0, out_features, self.init_row_batch):
            r1 = min(r0 + self.init_row_batch, out_features)
            Wb = Wf[r0:r1]
            # The published sample_weight: squared end-loss gradient PER WEIGHT.
            # Unlike diag(G) this varies by output row, which is the point.
            sw = None
            if self._wgrad is not None:
                sw = self._wgrad[r0:r1].to(device=device, dtype=dtype)
            cen = helper._kmeans_blocks(Wb, sample_weight=sw)
            C[r0:r1] = cen
            mid = 0.5 * (cen[:, 1:] + cen[:, :-1])
            labels[r0:r1] = torch.searchsorted(
                mid.contiguous(), Wb.contiguous()).clamp_(0, K - 1)

        # ---- LNQ, independently per output group ----------------------------
        total_init = total_final = 0.0
        for g in range(n_groups):
            s0, s1 = g * gsz, (g + 1) * gsz
            Hg, Hn, L = self._prepare_one(Hs[g], in_features, device, dtype)
            rx = L.transpose(0, 1).contiguous()
            Wg = Wf[s0:s1]
            Cg, lg = C[s0:s1].clone(), labels[s0:s1].clone()

            best = self._objective(Wg, Hg, Cg, lg)
            total_init += best
            best_C, best_l = Cg.clone(), lg.clone()

            for it in range(self.iters):
                # Iteration 0 is C-only: the k-means labels are a P-step for
                # plain MSE, so the codebook must see H before anything is
                # reassigned against it.
                if it > 0:
                    lg = self._update_P(Wg, Hn, Cg, lg)
                Cg = self._update_C(Wg, rx, lg, K)
                e_now = self._objective(Wg, Hg, Cg, lg)
                if e_now < best:
                    best, best_C, best_l = e_now, Cg.clone(), lg.clone()
                else:
                    break        # reference: first non-improvement stops
            total_final += best
            C[s0:s1], labels[s0:s1] = best_C, best_l
            del Hg, Hn, L, rx

        if self.verbose:
            print(f"[gq] groups={n_groups} init={total_init:.6e} "
                  f"final={total_final:.6e}", flush=True)

        # Order levels once, remapping labels through the same permutation.
        C, perm = torch.sort(C, dim=1)
        inv = torch.empty_like(perm)
        ar = torch.arange(K, device=device).view(1, -1).expand_as(perm).contiguous()
        inv.scatter_(1, perm, ar)
        labels = torch.gather(inv, 1, labels)
        for k in range(1, K):
            bad = C[:, k] <= C[:, k - 1]
            C[:, k] = torch.where(bad, C[:, k - 1] + 1e-7, C[:, k])

        W_dequant = torch.gather(C, 1, labels).to(W.dtype)
        if _DEBUG:
            rel = ((W_dequant.float() - Wf).norm() / Wf.norm()).item()
            print(f"[gq] layer done: rel_err={rel:.4f}", flush=True)

        return QuantResult(
            W_dequant=W_dequant,
            indices=labels,
            q_levels=self.q_levels.to(device),
            block_scales=Wf.abs().amax(dim=1).view(out_features, 1),
            block_size=in_features,
            block_codebooks=C.view(out_features, 1, K),
            block_zeros=None,
        )

    def __repr__(self) -> str:
        ng = "?" if self._H is None else self._H.shape[0]
        return (f"GuidedQuantQuantizer(name={self.name!r}, bits={self.bits}, "
                f"block_size=full_row, groups={ng}, iters={self.iters}, "
                f"cd_cycles={self.cd_cycles})")
