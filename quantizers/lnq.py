"""
LNQ: layer-wise non-uniform quantization by coordinate descent on a full Hessian.

Port of GuidedQuant's `train_least_squares` (any_precision/quantization/
layerwise_quantize.py). The objective is

    min_{P, C}  sum_i  dw_i^T H dw_i ,   dw = P C - W

alternating two exact steps:

  update_P : coordinate descent over input columns. Sweeping one column at a
             time lets the cross terms be folded into a running residual B, so
             each column's optimal codeword is a 1-D argmin against a
             *Hessian-corrected* target  (W - B)  rather than against W itself.
             This is what lets an assignment be non-nearest-neighbour in raw
             weight space -- the off-diagonals of H are the whole point.
  update_C : least squares for the codebook given fixed assignments, solved in
             the whitened space  L^T  from  H = L L^T  (Cholesky), with a small
             ridge for conditioning.

NAMING, HONESTLY: this is LNQ, not GuidedQuant. GuidedQuant is the variant
where H is additionally weighted by end-loss saliency, which needs per-group
output gradients from a backward pass. This repo collects G = E[x x^T] only,
which is the *ordinary* LNQ Hessian -- exactly the mode GuidedQuant's own code
logs as "without GuidedQuant saliency". Everything below is faithful to LNQ;
the saliency term is simply absent because the signal is not available here.

Init follows the reference: SqueezeLLM-style weighted 1-D k-means (i.e.
LearnedCodebookQuantizer with kmeans++ seeding), which supplies BOTH the
initial codebook C and the initial assignments P.
"""

from __future__ import annotations

import os
from typing import Optional

import torch

from .base_quantizer import BaseQuantizer, QuantResult
from .learned_codebook import LearnedCodebookQuantizer

_DEBUG = os.environ.get("CB_DEBUG", "") == "1"


class LNQQuantizer(BaseQuantizer):
    """Coordinate-descent codebook quantizer against the full activation Gram.

    Requires the layer Gram via set_gram(); without it there is no off-diagonal
    information and the method degenerates to plain k-means, so it refuses to
    run rather than silently producing a worse codebook3.
    """

    def __init__(self, bits: int,
                 block_size: int | None = None,
                 cd_cycles: int = 2,
                 iters: int = 15,
                 damp: float = 1e-2,
                 ridge: float = 1e-7,
                 cd_block_size: int = 128,
                 row_block: int = 64,
                 init_iters: int = 50,
                 kmeans_init: str = "kmeans++",
                 weighted_init: bool = True,
                 seed: int = 0,
                 verbose: bool = False):
        if bits not in (2, 3, 4):
            raise ValueError(f"LNQ supports bits in {{2,3,4}}, got {bits}")
        if block_size is None:
            block_size = -1
        super().__init__(bits=bits, block_size=block_size)
        self.name = f"lnq{bits}"
        self.num_levels = 2 ** bits
        self.cd_cycles = int(cd_cycles)
        self.iters = int(iters)
        self.damp = float(damp)
        self.ridge = float(ridge)
        self.cd_block_size = int(cd_block_size)
        self.row_block = int(row_block)
        self.init_iters = int(init_iters)
        self.kmeans_init = kmeans_init
        self.weighted_init = bool(weighted_init)
        self.seed = int(seed)
        self.verbose = bool(verbose)
        self._G: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------ #
    def set_gram(self, G: Optional[torch.Tensor]) -> None:
        """Layer activation Gram E[x x^T], [in, in]. Called before quantize()."""
        self._G = G

    @property
    def q_levels(self) -> torch.Tensor:
        # Learned per row; placeholder so callers that only need L work.
        return torch.linspace(-1.0, 1.0, self.num_levels)

    # ------------------------------------------------------------------ #
    def _prepare_H(self, G: torch.Tensor, pin: int, device, dtype):
        """Dampen and normalise the Hessian. [in,in] -> [in,in] unit diagonal.

        update_P divides H by its own diagonal (reference lines 87-89) so the
        per-column solve has coefficient 1. Doing it once here saves repeating
        it every cycle. Damping keeps the Cholesky in update_C from failing on
        a rank-deficient Gram, which is common when calibration samples are
        fewer than in_features.
        """
        H = G.to(device=device, dtype=dtype)
        if H.shape[0] < pin:
            pad = pin - H.shape[0]
            H = torch.nn.functional.pad(H, (0, pad, 0, pad))
            idx = torch.arange(H.shape[0] - pad, H.shape[0], device=device)
            H[idx, idx] = 1.0
        elif H.shape[0] > pin:
            H = H[:pin, :pin]

        d = torch.diagonal(H)
        mean_d = d.mean().clamp(min=1e-12)
        H = H + torch.eye(pin, device=device, dtype=dtype) * (self.damp * mean_d)
        d = torch.diagonal(H).clamp(min=1e-12)
        # Row-normalise by the diagonal: H_ij / H_ii.
        return H / d.view(-1, 1)

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _update_P(self, W: torch.Tensor, Hn: torch.Tensor,
                  C: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Coordinate descent over columns. W:[R,d] C:[R,K] labels:[R,d]->[R,d].

        Mirrors the reference: B holds the accumulated cross-term from already
        -updated columns, so the target for column j is (W_j - B_j) and the
        choice is a plain 1-D argmin against C.
        """
        R, d = W.shape
        device = W.device
        K = C.shape[1]
        bs = self.cd_block_size

        W_hat = torch.gather(C, 1, labels)                       # [R, d]
        tril = torch.tril(Hn, diagonal=-1)                       # strictly lower

        for _ in range(self.cd_cycles):
            # Reset the residual each cycle, as the reference does.
            B = (W_hat - W) @ tril                               # [R, d]

            for start in range(0, d, bs):
                end = min(start + bs, d)
                for j in range(start, end):
                    sol = W[:, j:j + 1] - B[:, j:j + 1]          # [R, 1]
                    am = (sol - C).abs().argmin(dim=1)           # [R]
                    labels[:, j] = am
                    newv = torch.gather(C, 1, am.view(-1, 1))    # [R, 1]
                    delta = newv - W_hat[:, j:j + 1]
                    W_hat[:, j:j + 1] = newv
                    if j < end - 1:
                        # Propagate within the block only; the tail is handled
                        # once per block below (reference does the same).
                        B[:, j + 1:end] += delta * Hn[j, j + 1:end].view(1, -1)
                if end < d:
                    B[:, end:] += (W_hat[:, start:end] - W[:, start:end]) @ Hn[start:end, end:]

        return labels

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _update_C(self, W: torch.Tensor, Lt: torch.Tensor,
                  labels: torch.Tensor, K: int) -> torch.Tensor:
        """Least-squares codebook given assignments, in the whitened space.

        min_C || L^T (P C - W) ||^2  ->  normal equations per row, with a ridge.
        Solved per row-block to bound the [R, d, K] one-hot intermediate.
        """
        R, d = W.shape
        device, dtype = W.device, W.dtype
        C_out = torch.empty(R, K, device=device, dtype=dtype)
        eye = torch.eye(K, device=device, dtype=dtype) * self.ridge

        for r0 in range(0, R, self.row_block):
            r1 = min(r0 + self.row_block, R)
            lab = labels[r0:r1]                                   # [rb, d]
            P = torch.nn.functional.one_hot(lab.long(), num_classes=K).to(dtype)
            # A = L^T P   [rb, d, K];  b = L^T W  [rb, d]
            A = torch.einsum('bj,ijc->ibc', Lt, P)
            b = torch.einsum('bj,ij->ib', Lt, W[r0:r1])
            AtA = torch.einsum('ibc,ibk->ick', A, A) + eye
            Atb = torch.einsum('ibc,ib->ic', A, b)
            C_out[r0:r1] = torch.linalg.solve(AtA, Atb.unsqueeze(-1)).squeeze(-1)

        # Levels must stay sorted and distinct for downstream neighbour reads.
        C_out, _ = torch.sort(C_out, dim=1)
        for k in range(1, K):
            bad = C_out[:, k] <= C_out[:, k - 1]
            C_out[:, k] = torch.where(bad, C_out[:, k - 1] + 1e-7, C_out[:, k])
        return C_out

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _objective(self, W: torch.Tensor, H: torch.Tensor,
                   C: torch.Tensor, labels: torch.Tensor) -> float:
        dw = torch.gather(C, 1, labels) - W
        return float(torch.einsum('ij,jk,ik->', dw, H, dw) / W.shape[0])

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def quantize(self, W: torch.Tensor, row_chunk: int = 1024) -> QuantResult:
        if self._G is None:
            raise RuntimeError(
                "LNQ needs the layer Gram. quantize.py must call set_gram(G) "
                "first -- add the quantizer name to NEEDS_GRAM."
            )
        device = W.device
        dtype = torch.float32
        out_features, in_features = W.shape
        K = self.num_levels
        bs = self._resolve_block_size(in_features)
        if bs != in_features:
            raise NotImplementedError(
                f"LNQ is full-row only (the Hessian couples all input channels); "
                f"got block_size={bs} for in_features={in_features}."
            )
        n_blocks = 1

        Wf = W.to(dtype)
        Hn = self._prepare_H(self._G, in_features, device, dtype)

        # Cholesky of the (symmetrised, damped) Hessian for update_C.
        Hs = 0.5 * (Hn + Hn.transpose(0, 1))
        jitter = 0.0
        for _ in range(6):
            try:
                L = torch.linalg.cholesky(
                    Hs + torch.eye(in_features, device=device, dtype=dtype) * jitter)
                break
            except Exception:
                jitter = max(jitter * 10, 1e-6)
        else:
            raise RuntimeError("LNQ: Cholesky failed even after jittering the Gram")
        Lt = L.transpose(0, 1).contiguous()

        # ---- init: SqueezeLLM-style weighted k-means (codebook AND labels) --
        helper = LearnedCodebookQuantizer(
            bits=self.bits if self.bits in (3, 4) else 3,
            block_size=-1,
            n_iters=self.init_iters,
            seed=self.seed,
            init=self.kmeans_init,
        )
        helper.num_levels = K

        sw = None
        if self.weighted_init:
            dg = torch.diagonal(self._G.to(device=device, dtype=dtype)).clamp(min=0)
            if dg.numel() == in_features:
                sw = dg.view(1, -1)

        C = torch.empty(out_features, K, device=device, dtype=dtype)
        labels = torch.empty(out_features, in_features, device=device, dtype=torch.long)
        for r0 in range(0, out_features, row_chunk):
            r1 = min(r0 + row_chunk, out_features)
            Wb = Wf[r0:r1]
            cen = helper._kmeans_blocks(Wb, sample_weight=sw)      # [rc, K]
            C[r0:r1] = cen
            mid = 0.5 * (cen[:, 1:] + cen[:, :-1])
            labels[r0:r1] = torch.searchsorted(
                mid.contiguous(), Wb.contiguous()).clamp_(0, K - 1)

        if self.verbose:
            e0 = self._objective(Wf, Hn, C, labels)
            print(f"[lnq] init objective = {e0:.6e}", flush=True)

        # ---- alternate ------------------------------------------------------
        best_e = self._objective(Wf, Hn, C, labels)
        best_C, best_lab = C.clone(), labels.clone()
        for it in range(self.iters):
            labels = self._update_P(Wf, Hn, C, labels)
            C = self._update_C(Wf, Lt, labels, K)
            e = self._objective(Wf, Hn, C, labels)
            if self.verbose:
                print(f"[lnq] iter {it:02d}  objective = {e:.6e}", flush=True)
            if e < best_e:
                best_e, best_C, best_lab = e, C.clone(), labels.clone()
            elif e > best_e * 1.5:
                # Diverging; keep the best seen and stop.
                if self.verbose:
                    print(f"[lnq] diverged at iter {it}, keeping best", flush=True)
                break
        C, labels = best_C, best_lab

        if self.verbose:
            print(f"[lnq] final objective = {best_e:.6e}", flush=True)

        W_dequant = torch.gather(C, 1, labels).to(W.dtype)
        block_codebooks = C.view(out_features, n_blocks, K)
        block_scales = Wf.abs().amax(dim=1).view(out_features, n_blocks)

        if _DEBUG:
            rel = ((W_dequant.float() - Wf).norm() / Wf.norm()).item()
            print(f"[lnq] layer done: rel_err={rel:.4f} obj={best_e:.4e}", flush=True)

        return QuantResult(
            W_dequant=W_dequant,
            indices=labels,
            q_levels=self.q_levels.to(device),
            block_scales=block_scales,
            block_size=in_features,
            block_codebooks=block_codebooks,
            block_zeros=None,
        )

    def __repr__(self) -> str:
        return (f"LNQQuantizer(name={self.name!r}, bits={self.bits}, "
                f"block_size=full_row, cd_cycles={self.cd_cycles}, "
                f"iters={self.iters})")
