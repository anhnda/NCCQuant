"""
FlexNu -- FlexRound-style learnable rounding on a *non-uniform, learned* codebook.

Motivation
----------
FlexRound (Lee et al., ICML 2023) makes two observations about uniform grids:

  (a) with a fixed grid, nearest-rounding is NOT loss-optimal, because the
      assignment is discrete and the loss couples coordinates through the
      activation Gram G = E[x x^T];
  (b) the discrete search cannot be done combinatorially at LLM scale, so the
      only viable mechanism is a *dense, parallel, gradient* update over a
      continuous parameter that perturbs the assignment.

FlexRound's answer is the asymmetric quantize/dequantize pair (Eq. 1-2):

        W_hat = s1 * round( W / (s1 . S2 . s3) )

the divisor carries the learned element-wise S2 / per-channel s3, the
*dequantizer* does not. So W_hat need not be the nearest grid point to W, and
S2/s3 are training-time only -- they are discarded, storage is unchanged.

FlexNu ports that mechanism to this repo's non-uniform per-(row, block) scalar
codebooks, and additionally learns the codebook itself in the SAME objective.

The obstruction, and the fix
----------------------------
For a uniform grid, round() is a monotone scalar map, so the straight-through
estimator (STE) is a *consistent* surrogate: raising the divisor monotonically
lowers the index. For a general codebook, assignment is argmin_j |z - c_j| over
an *unordered* set: the gradient is zero a.e., undefined on Voronoi boundaries,
and there is no monotone reparameterisation to STE through. The usual escape is
a softmax/Gumbel relaxation, which (i) has no temperature at which gradients are
both non-vanishing and forward-consistent, and (ii) collapses the codebook when
the codebook is learned through the same softmax.

FlexNu avoids all of that by exploiting the one structural property this repo's
codebooks already guarantee (see LearnedCodebookQuantizer: centers are sorted
and made strictly increasing; NCC's neighbour query requires it):

    THE CODEBOOK IS ORDERED:  c_0 < c_1 < ... < c_{K-1}

For a sorted codebook, nearest-codeword assignment is *exactly* a monotone step
function of the query, with thresholds at the midpoints t_j = (c_j + c_{j+1})/2:

    i(q) = sum_j  1[ q > t_j ]

and the reconstruction telescopes over the gaps Delta_j = c_{j+1} - c_j:

    c_{i(q)} = c_0 + sum_j 1[ q > t_j ] * Delta_j                        (*)

(*) is the whole method. Forward uses the hard indicator (so the forward pass is
*exactly* nearest-codeword assignment -- no relaxation, no soft reconstruction,
no annealing schedule). Backward replaces each indicator's derivative with a
sigmoid bump of width tau. Because the map is monotone in q, this STE is
consistent in exactly the same sense FlexRound's round() STE is.

Three gradient paths fall out of ONE loss.backward():

    d/d(delta2, delta3) : through q = W / exp(delta2 + delta3) into every
                          indicator.  <-- FlexRound's mechanism, intact.
    d/d(t_j)            : through the sigmoid argument.  <-- moves the DECISION
                          BOUNDARIES, i.e. learns the assignment rule.
    d/d(Delta_j), d/d(c_0) : through the telescoping multiply/offset.
                          <-- moves the CODEWORD VALUES (Lloyd-like refinement).

So the codebook and the FlexRound scales are optimised jointly, in one objective,
by one optimizer -- no alternating minimisation, no EM, no beam search. This is
the key difference from AQLM-style pipelines, which must alternate a discrete
assignment search with a continuous codebook update.

Ordering is preserved *structurally* by parameterising the codebook as an anchor
plus positive gaps,

    c_0 = a,   c_{j+1} = c_j + softplus(g_j),

so there is no sort, no projection, and no constraint violation during training;
Delta_j > 0 always, which is what keeps (*) monotone and the STE valid. This also
means the committed codebook satisfies NCC's strictly-increasing requirement by
construction, so FlexNu composes with apply_ncc() downstream.

Objective
---------
Identical to FlexRound's, and to the layer reconstruction error:

    L = || (W - W_hat) X ||_F^2  =  n * Tr( R G R^T ),   R = W_hat - W,

with G = E[x x^T] = Sigma + mu mu^T the second-moment Gram over the calibration
set. The FULL Gram is used (not its diagonal): the off-diagonal terms are what
let an error on one input coordinate be compensated by a deliberate non-nearest
choice on another. That coupling is the entire reason nearest-codeword is
suboptimal, so dropping it would defeat the method.

If no Gram is supplied (gram=None) the objective degrades gracefully to plain
MSE, Tr(R R^T) -- data-free, still learns the codebook + scales, just without
activation awareness.

Storage
-------
Committed state is the same shape as LearnedCodebookQuantizer's: per-(row, block)
codebook [out, n_blocks, K] plus integer indices [out, in]. delta2 (element-wise)
and delta3 (per-row) are training-time only and are DISCARDED, exactly as
FlexRound discards S2/s3. Bit-rate is therefore unchanged relative to codebook3 /
codebook4.

Initialisation
--------------
Codebook from 1-D Lloyd k-means (reusing LearnedCodebookQuantizer's routine) or
from the base format's realised levels; delta2 = delta3 = 0 so the divisor is 1.
At iteration 0 the forward pass is therefore *bit-identical* to nearest-codeword
assignment on the init codebook, i.e. FlexNu starts exactly at the codebookK
baseline and the objective can only improve from there. `energy_init` vs
`energy_final` in the returned diagnostics is an honest before/after delta.

What the mechanism actually is (and what it is NOT)
--------------------------------------------------
Learning the codebook and learning the divisor are NOT two flavours of the same
thing, and conflating them leads to the wrong defaults.

Learning the codebook is still *fitting W*: it moves {c_j} to cover the empirical
weight distribution, and under any weight-fitting criterion the assignment stays
nearest-neighbour. The reachable set of W_hat is the set of nearest-codeword
maps; codebook learning reparameterises that set but never leaves it.

The divisor LEAVES it. w_hat = c_i with i chosen by proximity to w/s2 rather than
to w is a deliberately non-nearest assignment, and it is reachable only because
the LOSS says the trade is worth it -- coordinate k accepts a worse local error so
that coordinate l can take a better one. That trade exists only through the
OFF-DIAGONALS of G. Under diagonal G the objective decomposes per coordinate and
nearest-neighbour is provably optimal. So this mechanism is defined by X, not by W.

Measured consequences (600 iters, 3-bit, block 64, synthetic W):

    G regime            cond(G)    codebook-only   divisor-only   %weights moved
    iid (G ~ I)          4.2e1        -0.5%           -0.0%            0.0%
    correlated           5.7e2        -5.1%           -3.5%            ~0.1%
    low-rank/aniso       6.7e4       -11.1%          -79.9%            6.7%

Two things to read off this. First, on an anisotropic Gram -- the realistic case
for LLM activations -- the divisor is worth ~7x the codebook, and it gets there by
moving only ~6-7% of weights off their nearest codeword. The escape IS the method.
Second, on an isotropic Gram it is worth exactly nothing, as the theory demands;
benchmarking this method on white-noise activations measures nothing.

Two design errors this corrects (recorded so they are not reintroduced):

  * DO NOT STAGE the divisor after the codebook. Staging starts delta2 = 0 on a
    grid already fitted to W -- precisely at the nearest-neighbour point the
    divisor exists to escape -- and it must then climb out. Joint: -80%. Staged:
    -0.3%. Default stage_frac is therefore 0.0.
  * DO NOT let the best-iterate guard constrain the trajectory. Crossing a
    threshold is a transient in which the smoothed objective improves while the
    hard energy briefly worsens. A guard that rejects, clips, or rewinds to those
    iterates filters out exactly the moves the method is for. The guard here is a
    pure SNAPSHOT: the optimiser runs unconstrained and we only record what it
    passes through. The k-means init seeds the incumbent, so the committed result
    is never worse than plain nearest-codeword -- a floor on the OUTPUT, not a
    leash on the SEARCH.

Also note lr_cb is deliberately tiny (1e-5) by default: on anisotropic G, every
increase in the codebook learning rate monotonically HURT (81% -> 80% -> 72% ->
53% -> 37% as lr_cb goes 0 -> 3e-3), because the codebook chases W and drags the
thresholds out from under the divisor. Set --flexnu-freeze-codebook to test the
pure-divisor variant, which is currently the strongest configuration.

Ablations (see FlexNuQuantizer args)
------------------------------------
    freeze_scale=True                   -> codebook-only (fits W)
    freeze_codebook=True                -> divisor-only (escapes via the loss)
    both True                           -> must reproduce codebookK exactly
    neither (default)                   -> joint, lr_cb << lr_scale

Report all four against a codebookK baseline AND against an anisotropic Gram; on
white-noise activations every row collapses to zero and the comparison is void.
"""

from __future__ import annotations

import os
from typing import Optional

import torch

from .base_quantizer import BaseQuantizer, QuantResult
from .learned_codebook import LearnedCodebookQuantizer

_DEBUG_FLEXNU = os.environ.get("CB_DEBUG", "") == "1"


# --------------------------------------------------------------------------- #
# Straight-through step:  forward 1[x > 0],  backward sigmoid'(x / tau) / tau.
# --------------------------------------------------------------------------- #
class _StepSTE(torch.autograd.Function):
    """Hard Heaviside forward, smooth logistic-bump backward.

    This is the non-uniform analogue of FlexRound's _RoundSTE. The forward is the
    exact hard indicator, so the reconstruction is a genuine codeword (never a
    soft mixture). The backward uses the derivative of sigmoid(x/tau):

        d/dx sigmoid(x/tau) = sigmoid(x/tau) * (1 - sigmoid(x/tau)) / tau

    tau is a *backward-only* surrogate width, not a softmax temperature: it never
    appears in the forward pass, so there is no forward/backward objective gap to
    anneal away and no schedule to tune.
    """

    @staticmethod
    def forward(ctx, x, tau):
        ctx.save_for_backward(x)
        ctx.tau = tau
        return (x > 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        tau = ctx.tau
        s = torch.sigmoid(x / tau)
        return grad_out * s * (1.0 - s) / tau, None


def _step_ste(x: torch.Tensor, tau: float) -> torch.Tensor:
    return _StepSTE.apply(x, tau)


class FlexNuQuantizer(BaseQuantizer):
    """FlexRound-style learnable rounding over a jointly-learned sorted codebook.

    Parameters
    ----------
    bits : int
        3 or 4 (K = 2**bits levels), matching codebook3 / codebook4.
    block_size : int
        Input-dim block width; one codebook per (row, block). Default 64.
    iters : int
        Adam steps per layer.
    lr_scale : float
        Adam LR for the FlexRound divisor params (delta2, delta3).
    lr_cb : float
        Adam LR for the codebook params (anchor, gaps). Kept separate: the
        codebook lives on the weight scale while the deltas are log-space
        offsets, so a single LR serves them badly.
    tau_frac : float
        Backward surrogate width as a fraction of the mean codebook gap,
        tau = tau_frac * mean(Delta). Recomputed every step so tau tracks the
        codebook as it rescales (a fixed absolute tau saturates once the
        codebook shrinks).
    use_delta3 : bool
        Include the per-output-channel term s3 (FlexRound ablation 2 shows it
        helps).
    freeze_codebook, freeze_scale : bool
        Ablation switches (see module docstring).
    stage_frac : float
        Fraction of `iters` spent on the codebook ALONE before switching to the
        divisors alone. Default 0.0 = fully joint, which is what you want.

        Set > 0 only for the ablation. Staging looks appealing because the two
        blocks do interfere -- they act on the same signed distance q - t_j from
        opposite sides -- but it defeats the mechanism: phase 2 then starts at
        delta2 = 0 on a codebook already fitted to W, i.e. precisely at the
        nearest-neighbour solution the divisor is supposed to escape. Measured
        on an anisotropic Gram: joint recovers ~80% of the reconstruction
        energy, staged recovers ~0.3%.
    eval_every : int
        How often (in steps) to evaluate the true hard energy for best-iterate
        tracking. 1 is recommended: the hard energy is piecewise-constant and
        non-monotone along the STE trajectory, so good assignments appear and
        vanish within a few steps and subsampling misses them.
    init : {"lloyd", "levels"}
        "lloyd"  -> 1-D k-means init (reuses LearnedCodebookQuantizer).
        "levels" -> affine map of a supplied canonical level shape onto each
                    block's [min, max] (set `init_levels`).
    init_levels : Optional[torch.Tensor]
        Canonical sorted level shape for init="levels" (e.g. an NF4 table).
    row_block : int
        Rows optimised per sub-problem. THE memory knob: the threshold sum
        materialises a [row_block, in, K-1] tensor and autograd retains a few
        such. Rows are independent given the (shared) Gram, so this does not
        change the result -- only peak memory.
    lambda_s2 : float
        Weight of the log-space regulariser on delta2/delta3. A safety rail that
        stops the divisor wandering to degenerate values where a block's mass
        piles into one cell; not a substantive term. Set 0 to disable.
    work_dtype : torch.dtype
        Optimisation dtype (fp32 strongly recommended).
    verbose : bool
        Print per-layer energy drop.
    """

    def __init__(self,
                 bits: int,
                 block_size: int | None = None,
                 iters: int = 300,
                 lr_scale: float = 3e-3,
                 lr_cb: float = 1e-5,
                 tau_frac: float = 0.5,
                 use_delta3: bool = True,
                 freeze_codebook: bool = False,
                 freeze_scale: bool = False,
                 stage_frac: float = 0.0,
                 eval_every: int = 1,
                 init: str = "lloyd",
                 init_levels: Optional[torch.Tensor] = None,
                 init_iters: int = 50,
                 row_block: int = 128,
                 lambda_s2: float = 0.0,
                 work_dtype: torch.dtype = torch.float32,
                 seed: int = 0,
                 kmeans_init: str = "kmeans++",
                 verbose: bool = False):
        if bits not in (2, 3, 4):
            raise ValueError(f"FlexNu supports bits in {{2,3,4}}, got {bits}")
        if init not in ("lloyd", "levels"):
            raise ValueError(f"init must be 'lloyd' or 'levels', got {init!r}")
        if init == "levels" and init_levels is None:
            raise ValueError("init='levels' requires init_levels")
        super().__init__(bits=bits, block_size=block_size)
        self.name = f"flexnu{bits}"
        self.num_levels = 2 ** bits
        self.iters = int(iters)
        self.lr_scale = float(lr_scale)
        self.lr_cb = float(lr_cb)
        self.tau_frac = float(tau_frac)
        self.use_delta3 = bool(use_delta3)
        self.freeze_codebook = bool(freeze_codebook)
        self.freeze_scale = bool(freeze_scale)
        self.stage_frac = float(stage_frac)
        self.eval_every = int(eval_every)
        self.init = init
        self.init_levels = init_levels
        self.init_iters = int(init_iters)
        # Seeding for the 'lloyd' path: 'kmeans++' matches SqueezeLLM (weighted
        # D2 sampling), 'quantile' is the cheaper equal-mass seed.
        if kmeans_init not in ("kmeans++", "quantile"):
            raise ValueError(
                f"kmeans_init must be 'kmeans++' or 'quantile'; got {kmeans_init!r}")
        self.kmeans_init = kmeans_init
        self.row_block = int(row_block)
        self.lambda_s2 = float(lambda_s2)
        self.work_dtype = work_dtype
        self.seed = int(seed)
        self.verbose = bool(verbose)

        # Per-layer Gram, injected by the caller before quantize(). Kept as an
        # attribute (rather than a quantize() kwarg) so FlexNu still satisfies
        # BaseQuantizer's `quantize(W, row_chunk)` contract and drops into the
        # existing loop in quantize.py unchanged.
        self._gram: Optional[torch.Tensor] = None
        self.last_diag: dict = {}

    # ------------------------------------------------------------------ #
    @property
    def q_levels(self) -> torch.Tensor:
        """Placeholder canonical shape; realised levels live in block_codebooks."""
        return torch.linspace(-1.0, 1.0, self.num_levels)

    # ------------------------------------------------------------------ #
    def set_gram(self, G: Optional[torch.Tensor]) -> None:
        """Attach the layer's second-moment Gram G = E[x x^T], shape [in, in].

        Call immediately before quantize() for this layer. Pass None for the
        data-free (plain-MSE) objective.
        """
        self._gram = G

    # ------------------------------------------------------------------ #
    # Codebook parameterisation:  c_0 = a,  c_{j+1} = c_j + softplus(g_j)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _softplus(x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softplus(x)

    @staticmethod
    def _inv_softplus(y: torch.Tensor) -> torch.Tensor:
        """Inverse of softplus, stable at both ends.

        softplus^{-1}(y) = log(exp(y) - 1) = y + log(-expm1(-y)).

        Two regimes must be handled separately or the init silently produces NaN:
          small y : exp(y) - 1 underflows, so use log(expm1(y)) directly, which
                    is accurate as y -> 0 (expm1 keeps the precision that
                    exp(y)-1 loses).
          large y : exp(y) overflows, so use the y + log(-expm1(-y)) form, whose
                    second term -> 0.
        The switch at y = 20 is well inside both regimes' accurate range.
        """
        y = y.clamp(min=1e-12)
        small = torch.log(torch.expm1(y.clamp(max=20.0)))
        large = y + torch.log((-torch.expm1(-y)).clamp(min=1e-30))
        return torch.where(y < 20.0, small, large)

    def _codebook_from_params(self, anchor: torch.Tensor,
                              gaps_raw: torch.Tensor) -> torch.Tensor:
        """[.., 1] anchor + [.., K-1] raw gaps -> sorted codebook [.., K].

        Sortedness is structural: softplus > 0 forces c_{j+1} > c_j, so the
        codebook is strictly increasing at every point of training with no sort
        and no projection. This is what keeps the step-function reconstruction
        monotone (hence the STE valid) and what guarantees the committed table
        satisfies NCC's strictly-increasing precondition.
        """
        deltas = self._softplus(gaps_raw)                    # [.., K-1] > 0
        return torch.cat([anchor, anchor + torch.cumsum(deltas, dim=-1)], dim=-1)

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _init_codebook(self, Wb: torch.Tensor,
                       G: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Initial sorted codebook per block. Wb: [R, n_blocks, bw] -> [R, n_blocks, K].

        When G is given, the k-means init is WEIGHTED by diag(G) = E[x_j^2].
        This occupies the same slot as SqueezeLLM does in GuidedQuant's
        pipeline: a scalar per-element weight feeding 1-D k-means, whose output
        then seeds the optimiser (LNQ there, FlexNu here).

        Two honest caveats. GuidedQuant's scalar weight is the squared END-LOSS
        gradient, per weight [out, in]; diag(G) is layer-local and per input
        channel only, so it is the same shape of signal from a weaker source
        and cannot distinguish output rows. And FlexNu minimises the full
        Tr(R G R^T) including off-diagonals, so this seed is aligned with the
        DIAGONAL of that objective, not the objective itself -- at full row
        the diagonal is 4096 of ~16.7M entries.
        """
        R, nb, bw = Wb.shape
        K = self.num_levels
        flat = Wb.reshape(R * nb, bw)

        if self.init == "lloyd":
            helper = LearnedCodebookQuantizer(
                bits=self.bits if self.bits in (3, 4) else 3,
                block_size=self.block_size,
                n_iters=self.init_iters,
                seed=self.seed,
                init=self.kmeans_init,
            )
            helper.num_levels = K          # support bits=2 through the same routine

            sw = None
            if G is not None:
                # diag(G) is per-input-channel; reshape to per-block so each
                # block sees the channels it actually owns, then broadcast over
                # rows (the sensitivity is a property of the channel, not the
                # output row).
                d = torch.diagonal(G).to(device=flat.device, dtype=flat.dtype)
                d = d.clamp(min=0)
                if d.numel() == nb * bw:
                    sw = d.view(1, nb, bw).expand(R, nb, bw).reshape(R * nb, bw)
                elif _DEBUG_FLEXNU:
                    print(f"[flexnu] diag(G) numel={d.numel()} != nb*bw={nb*bw}; "
                          f"falling back to unweighted init", flush=True)

            centers = helper._kmeans_blocks(flat, sample_weight=sw)  # [R*nb, K]
        else:
            q = self.init_levels.to(device=flat.device, dtype=flat.dtype)
            q, _ = torch.sort(q)
            qlo, qhi = q[0], q[-1]
            qspan = (qhi - qlo).clamp(min=1e-12)
            wmin = flat.amin(dim=1, keepdim=True)
            wmax = flat.amax(dim=1, keepdim=True)
            scale = ((wmax - wmin) / qspan).clamp(min=1e-12)
            centers = wmin + (q.view(1, -1) - qlo) * scale   # [R*nb, K]

        # Enforce a strict minimum gap so inverse-softplus is well-conditioned.
        eps = 1e-6
        for k in range(1, K):
            centers[:, k] = torch.maximum(centers[:, k], centers[:, k - 1] + eps)
        return centers.reshape(R, nb, K)

    # ------------------------------------------------------------------ #
    def _optimize_rows(self,
                       Wrows: torch.Tensor,
                       G: Optional[torch.Tensor],
                       in_features: int):
        """Run FlexNu on a slab of rows.

        Wrows : [R, pin]  padded weights (padding columns are zero and are made
                inert by the Gram padding, so they cost nothing).
        G     : [pin, pin] or None.
        Returns (codebook [R, nb, K], indices [R, pin], W_hat [R, pin],
                 e_init, e_final).
        """
        dev = Wrows.device
        wdt = self.work_dtype
        R, pin = Wrows.shape
        K = self.num_levels
        bs = self._resolve_block_size(in_features)
        nb = pin // bs

        Wb = Wrows.reshape(R, nb, bs).to(wdt)                # [R, nb, bs]

        # ---- init: codebook from k-means / levels, deltas at 0 --------------
        cb0 = self._init_codebook(Wb, G)                     # [R, nb, K]
        anchor0 = cb0[..., :1].clone()                       # [R, nb, 1]
        gaps0 = self._inv_softplus(cb0[..., 1:] - cb0[..., :-1])   # [R, nb, K-1]

        def energy(W_hat: torch.Tensor) -> torch.Tensor:
            """Tr(R G R^T) over this row slab, or plain MSE when G is None."""
            Res = (W_hat - Wb).reshape(R, pin)
            if G is None:
                return (Res * Res).sum()
            return (Res @ G * Res).sum()

        # ---- baseline energy: hard nearest-codeword on the init codebook ----
        # At full-row granularity bs == in_features, so the [R, nb, bs, K]
        # distance tensor is enormous (128 x 11008 x 16 x 4B ~= 90 GB). The
        # codebook is sorted by construction, so searchsorted gives the same
        # argmin in O(log K) with no [.., K] intermediate at all.
        with torch.no_grad():
            idx0 = torch.searchsorted(
                (0.5 * (cb0[..., 1:] + cb0[..., :-1])).reshape(R * nb, K - 1).contiguous(),
                Wb.reshape(R * nb, bs).contiguous(),
            ).reshape(R, nb, bs).clamp_(0, K - 1)
            W0 = torch.gather(cb0, 2, idx0)
            e_init = float(energy(W0).item())
            del W0, idx0

        n_steps = self.iters
        if self.freeze_codebook and self.freeze_scale:
            n_steps = 0        # pure baseline reproduction; nothing to learn

        with torch.enable_grad():
            anchor = anchor0.detach().clone().requires_grad_(not self.freeze_codebook)
            gaps = gaps0.detach().clone().requires_grad_(not self.freeze_codebook)
            # delta2: element-wise log-divisor (FlexRound S2), init 0 -> divisor 1
            delta2 = torch.zeros(R, nb, bs, device=dev, dtype=wdt
                                 ).requires_grad_(not self.freeze_scale)
            # delta3: per-output-channel log-divisor (FlexRound s3), init 0
            delta3 = (torch.zeros(R, 1, 1, device=dev, dtype=wdt
                                  ).requires_grad_(not self.freeze_scale)
                      if self.use_delta3 else None)

            # ---- joint vs staged -------------------------------------------
            # IMPORTANT (corrects an earlier design error): the divisor is NOT a
            # refinement of a W-fitted codebook. Learning the codebook is still
            # *fitting W* -- it moves {c_j} to cover the empirical weight
            # distribution, and under any weight-fitting criterion the assignment
            # stays nearest-neighbour. The reachable set is the set of
            # nearest-codeword maps; codebook learning reparameterises that set
            # but never leaves it.
            #
            # The divisor leaves it. w_hat = c_i with i chosen by proximity to
            # w/s2, not to w: a deliberately NON-NEAREST assignment, reachable
            # only because the LOSS -- through G's off-diagonals -- says the
            # trade is worth it. Coordinate k accepts a worse local error so
            # coordinate l can take a better one. Under diagonal G that trade
            # does not exist and nearest-neighbour is provably optimal, so the
            # mechanism is defined by X, not by W.
            #
            # Consequence: staging the divisor AFTER the codebook starts it at
            # delta2 = 0 on a grid already fitted to W, i.e. exactly at the
            # nearest-neighbour point it is supposed to escape, and it must then
            # climb out. Measured on an anisotropic Gram, running the divisor
            # from the start recovers ~80% of the reconstruction energy while
            # moving only ~6% of weights off their nearest codeword; staging it
            # afterwards recovers ~0.3%.
            #
            # So the default is JOINT (stage_frac = 0.0). Staging is kept behind
            # stage_frac > 0 for the ablation, since the two parameter blocks do
            # interfere (they act on the same signed distance q - t_j from
            # opposite sides) and separating them is worth being able to test.
            learn_cb = not self.freeze_codebook
            learn_sc = not self.freeze_scale
            n_phase1 = int(round(self.stage_frac * n_steps)) if (learn_cb and learn_sc) else (
                n_steps if learn_cb else 0)

            opt_cb = (torch.optim.Adam([anchor, gaps], lr=self.lr_cb)
                      if learn_cb else None)
            _sp = [delta2] + ([delta3] if delta3 is not None else [])
            opt_sc = (torch.optim.Adam(_sp, lr=self.lr_scale)
                      if learn_sc else None)
            if learn_cb and learn_sc and self.stage_frac <= 0.0:
                # Joint: one optimizer, both blocks, their own learning rates.
                opt_joint = torch.optim.Adam(
                    [{"params": [anchor, gaps], "lr": self.lr_cb},
                     {"params": _sp, "lr": self.lr_scale}])
            else:
                opt_joint = None

            # ---- best-iterate tracking ------------------------------------
            # The STE is a surrogate: its gradient is exact for the smoothed
            # objective, not the piecewise-constant true one, so the iterates
            # keep moving after the hard energy bottoms out and Adam's momentum
            # then walks them uphill. Measured trajectories fall ~80% and then
            # oscillate back to ~-71%; the last iterate is never the best one.
            # So we snapshot the best-so-far and commit that.
            #
            # SUBTLETY (this is easy to get wrong): the guard must be a
            # *snapshot*, never a constraint on the trajectory. Escaping the
            # nearest-neighbour assignment requires CROSSING a threshold, and a
            # crossing is a transient in which the smoothed objective improves
            # while the hard energy briefly worsens. Rejecting, clipping, or
            # rewinding to those iterates would filter out exactly the moves the
            # method exists to make. We therefore let the optimiser run
            # completely unconstrained and only record what it passes through.
            #
            # The init is seeded as the incumbent so the committed result can
            # never be worse than plain nearest-codeword on the k-means
            # codebook, but that is a floor on the OUTPUT, not a leash on the
            # SEARCH.
            best_e = e_init
            best_state = None
            eval_every = max(1, int(self.eval_every))

            @torch.no_grad()
            def _hard_energy():
                cb_ = self._codebook_from_params(anchor, gaps)
                th_ = 0.5 * (cb_[..., 1:] + cb_[..., :-1])
                ld_ = delta2 if delta3 is None else delta2 + delta3
                q_ = Wb * torch.exp(-ld_)
                i_ = torch.searchsorted(
                    th_.reshape(R * nb, K - 1).contiguous(),
                    q_.reshape(R * nb, bs).contiguous(),
                ).reshape(R, nb, bs).clamp_(0, K - 1)
                return float(energy(torch.gather(cb_, 2, i_)).item())

            for step in range(n_steps):
                # Joint by default; staged only when stage_frac > 0.
                opt = opt_joint if opt_joint is not None else (
                    opt_cb if step < n_phase1 else opt_sc)
                if opt is None:
                    break
                opt.zero_grad(set_to_none=True)

                cb = self._codebook_from_params(anchor, gaps)          # [R,nb,K]
                deltas = cb[..., 1:] - cb[..., :-1]                    # [R,nb,K-1] > 0
                thresh = 0.5 * (cb[..., 1:] + cb[..., :-1])            # [R,nb,K-1]

                # tau tracks the codebook scale; recomputed each step so the
                # backward bump neither saturates nor smears as the gaps move.
                tau = float(self.tau_frac * deltas.detach().mean().clamp(min=1e-12))

                # FlexRound divisor: quantize against W / (S2 . s3) ...
                log_div = delta2 if delta3 is None else delta2 + delta3
                q = Wb * torch.exp(-log_div)                           # [R,nb,bs]

                # ... but dequantize by the codebook alone. W_hat is therefore
                # NOT constrained to be the nearest codeword to W -- which is
                # exactly FlexRound's point.
                ind = _step_ste(q.unsqueeze(-1) - thresh.unsqueeze(2), tau)  # [R,nb,bs,K-1]
                # Telescoping reconstruction  c_0 + sum_j 1[q > t_j] * Delta_j.
                # cb[..., :1] is [R,nb,1] -> broadcasts over the bs axis.
                W_hat = cb[..., :1] + (ind * deltas.unsqueeze(2)).sum(dim=-1)  # [R,nb,bs]

                loss = energy(W_hat)
                if self.lambda_s2 > 0.0 and not self.freeze_scale:
                    reg = (delta2 * delta2).sum()
                    if delta3 is not None:
                        reg = reg + (delta3 * delta3).sum()
                    loss = loss + self.lambda_s2 * reg

                loss.backward()
                opt.step()

                if (step % eval_every == 0) or (step == n_steps - 1):
                    e_now = _hard_energy()
                    if e_now < best_e:
                        best_e = e_now
                        best_state = (anchor.detach().clone(),
                                      gaps.detach().clone(),
                                      delta2.detach().clone(),
                                      None if delta3 is None else delta3.detach().clone())

            # Restore the best iterate. If nothing ever beat the k-means init,
            # fall back to it exactly, so FlexNu is never worse than codebookK.
            if best_state is not None:
                with torch.no_grad():
                    anchor.copy_(best_state[0]); gaps.copy_(best_state[1])
                    delta2.copy_(best_state[2])
                    if delta3 is not None:
                        delta3.copy_(best_state[3])
            elif n_steps > 0:
                with torch.no_grad():
                    anchor.copy_(anchor0); gaps.copy_(gaps0); delta2.zero_()
                    if delta3 is not None:
                        delta3.zero_()

        # ---- commit: hard assignment, no grad, deltas discarded -------------
        with torch.no_grad():
            cb = self._codebook_from_params(anchor, gaps).detach()     # [R,nb,K]
            thresh = 0.5 * (cb[..., 1:] + cb[..., :-1])
            log_div = delta2 if delta3 is None else delta2 + delta3
            q = Wb * torch.exp(-log_div.detach())

            # searchsorted on the (structurally sorted) thresholds == argmin
            # |q - c_j| for a sorted codebook, but O(log K) and allocation-free.
            idx = torch.searchsorted(
                thresh.reshape(R * nb, K - 1).contiguous(),
                q.reshape(R * nb, bs).contiguous(),
            ).reshape(R, nb, bs).clamp_(0, K - 1)

            W_hat = torch.gather(cb, 2, idx.reshape(R, nb, bs))
            e_final = float(energy(W_hat).item())

        return cb, idx, W_hat, e_init, e_final

    # ------------------------------------------------------------------ #
    def quantize(self, W: torch.Tensor, row_chunk: int = 1024) -> QuantResult:
        """FlexNu-quantize W [out, in].

        `row_chunk` is accepted for contract compatibility; the true memory knob
        is `self.row_block`, since the threshold sum is [row_block, in, K-1].
        Rows are independent given the shared Gram, so the result does not depend
        on either.
        """
        torch.manual_seed(self.seed)
        dev = W.device
        wdt = self.work_dtype
        out_features, in_features = W.shape
        K = self.num_levels
        bs = self._resolve_block_size(in_features)
        nb = (in_features + bs - 1) // bs
        pin = nb * bs

        # ---- pad the input dim to a whole number of blocks ------------------
        if pin > in_features:
            Wp = torch.zeros(out_features, pin, device=dev, dtype=W.dtype)
            Wp[:, :in_features] = W
        else:
            Wp = W

        # ---- Gram, padded so the pad columns are inert but keep G PD --------
        G = None
        if self._gram is not None:
            G = self._gram.to(device=dev, dtype=wdt)
            if G.shape[0] != in_features:
                raise ValueError(
                    f"Gram has shape {tuple(G.shape)}, expected "
                    f"[{in_features}, {in_features}] for this layer.")
            if pin > in_features:
                Gp = torch.zeros(pin, pin, device=dev, dtype=wdt)
                Gp[:in_features, :in_features] = G
                d = torch.arange(in_features, pin, device=dev)
                # Pad weights are exactly 0 and stay 0, so any positive diagonal
                # here is inert; a nonzero value just keeps G well-conditioned.
                Gp[d, d] = torch.diagonal(G).mean()
                G = Gp

        W_dequant = torch.empty_like(W)
        indices = torch.empty(out_features, in_features, dtype=torch.long, device=dev)
        block_codebooks = torch.zeros(out_features, nb, K, device=dev, dtype=torch.float32)
        block_scales = torch.zeros(out_features, nb, device=dev, dtype=torch.float32)

        e_init_tot = 0.0
        e_final_tot = 0.0
        rb = max(1, min(self.row_block, out_features))

        for r0 in range(0, out_features, rb):
            r1 = min(r0 + rb, out_features)
            cb, idx, W_hat, e_i, e_f = self._optimize_rows(
                Wp[r0:r1].to(wdt), G, in_features)

            R = r1 - r0
            W_hat_flat = W_hat.reshape(R, pin)
            idx_flat = idx.reshape(R, pin)

            W_dequant[r0:r1] = W_hat_flat[:, :in_features].to(W.dtype)
            indices[r0:r1] = idx_flat[:, :in_features]
            block_codebooks[r0:r1] = cb.to(torch.float32)
            # Keep the same convention as LearnedCodebookQuantizer: an absmax
            # magnitude per block, informational only (levels are in the table).
            block_scales[r0:r1] = Wp[r0:r1].reshape(R, nb, bs).abs().amax(dim=2).to(torch.float32)

            e_init_tot += e_i
            e_final_tot += e_f

            del cb, idx, W_hat, W_hat_flat, idx_flat
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self.last_diag = {
            "quantizer": self.name,
            "iters": self.iters,
            "lr_scale": self.lr_scale,
            "lr_cb": self.lr_cb,
            "tau_frac": self.tau_frac,
            "use_delta3": self.use_delta3,
            "freeze_codebook": self.freeze_codebook,
            "freeze_scale": self.freeze_scale,
            "stage_frac": self.stage_frac,
            "eval_every": self.eval_every,
            "gram": self._gram is not None,
            "energy_init": e_init_tot,
            "energy_final": e_final_tot,
            "energy_drop": e_init_tot - e_final_tot,
            "energy_rel_drop": (e_init_tot - e_final_tot) / max(e_init_tot, 1e-12),
        }
        if self.verbose:
            print(f"    [{self.name}] E_init={e_init_tot:.6e} "
                  f"E_final={e_final_tot:.6e} "
                  f"drop={100.0 * self.last_diag['energy_rel_drop']:.2f}%")

        if G is not None:
            del G
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return QuantResult(
            W_dequant=W_dequant,
            indices=indices,
            q_levels=self.q_levels.to(dev),
            block_scales=block_scales,
            block_size=bs,
            block_codebooks=block_codebooks,   # realised, strictly increasing
            block_zeros=None,                  # codebook already absorbs any shift
        )
