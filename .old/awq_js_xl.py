"""
James-Stein Heuristic AWQ - XL Version with James-Stein Mean Estimation

This version extends awq_dh_xl.py with James-Stein estimator for activation means.

Key Features:
- Same base as awq_dh_xl.py (Heuristic-Guided Global Greedy Rounding)
- SPECIAL: Splits lm_head into chunks to avoid OOM
- L2 salience metric
- Dynamic outlier detection using Kneedle algorithm
- **JAMES-STEIN ESTIMATOR**: Uses shrinkage estimation for E[X] instead of direct mean
- Batched sequential quantization

James-Stein Estimator:
- Instead of using direct sample mean E[X] for each channel, we use James-Stein shrinkage
- Formula: μ̂_JS[j] = μ̄ + (1 - c) × (X̄[j] - μ̄)
  where:
  - μ̄ = grand mean across all channels
  - X̄[j] = sample mean for channel j
  - c = shrinkage factor = (p - 2) × σ² / Σ(X̄[j] - μ̄)²
- Provably dominates MLE (simple mean) when p ≥ 3 in terms of MSE
- Shrinks extreme channel means toward the grand mean
- More robust to sampling noise in activation statistics
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import os
import argparse
import random
import numpy as np
import gc

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("⚠️  Warning: psutil not installed. Memory monitoring disabled.")

# Try to import calibration utils, fallback if not present
try:
    from calibration_utils import get_c4_calibration_data, get_wikitext2_calibration_data
except ImportError:
    print("⚠️ calibration_utils not found. Using internal fallback loaders.")
    def get_c4_calibration_data(*args, **kwargs): raise NotImplementedError("Please provide calibration_utils.py")
    def get_wikitext2_calibration_data(*args, **kwargs): raise NotImplementedError("Please provide calibration_utils.py")


def find_knee_point(values, tolerance_offset=0.0):
    """
    Find knee point in sorted values using a simple Kneedle-like algorithm.

    Args:
        values: 1D tensor of sorted values (ascending order)
        tolerance_offset: Additional offset to add to knee point (allows tuning)

    Returns:
        index of knee point

    Algorithm:
    1. Normalize values to [0, 1]
    2. Create reference line from start to end
    3. Find point with maximum distance from reference line
    4. Apply tolerance offset
    """
    n = len(values)
    if n < 3:
        return n // 2

    # Convert to numpy for easier computation
    # CRITICAL: Convert bfloat16 to float32 first (bfloat16 not supported by numpy)
    if torch.is_tensor(values):
        y = values.cpu().float().numpy()
    else:
        y = np.array(values)

    # Normalize to [0, 1]
    y_min, y_max = y.min(), y.max()
    if y_max - y_min < 1e-10:
        # All values are the same, no knee
        return n // 2

    y_norm = (y - y_min) / (y_max - y_min)
    x_norm = np.linspace(0, 1, n)

    # Compute distances from the line connecting first and last point
    # Line equation: y = m*x + b
    # For normalized: line goes from (0, y_norm[0]) to (1, y_norm[-1])
    # Distance = |y_point - y_line|

    # y_line for each x
    y_line = y_norm[0] + (y_norm[-1] - y_norm[0]) * x_norm

    # Perpendicular distance from each point to the line
    distances = np.abs(y_norm - y_line)

    # Find point with maximum distance (the knee)
    knee_idx = np.argmax(distances)

    # Apply tolerance offset
    # Offset moves the threshold higher (more conservative with outliers)
    if knee_idx < n - 1:
        # Calculate how many indices to shift based on offset
        # offset is a fraction of the range
        offset_indices = int(tolerance_offset * n)
        knee_idx = min(knee_idx + offset_indices, n - 1)
        knee_idx = max(knee_idx, 0)

    return knee_idx


def compute_james_stein_mean(raw_means, variance_estimate=None):
    """
    Apply James-Stein shrinkage estimator to channel-wise means.

    The James-Stein estimator shrinks individual means toward the grand mean,
    provably dominating the MLE (simple mean) in terms of MSE when p ≥ 3.

    Args:
        raw_means: Tensor of shape [in_features] containing sample means for each channel
        variance_estimate: Optional variance estimate. If None, uses empirical variance.

    Returns:
        Tensor of James-Stein estimated means

    Formula:
        μ̂_JS[j] = μ̄ + (1 - c) × (X̄[j] - μ̄)
        where c = (p - 2) × σ² / Σ(X̄[j] - μ̄)²
    """
    p = len(raw_means)

    # Need at least 3 dimensions for James-Stein to be beneficial
    if p < 3:
        return raw_means

    # Compute grand mean
    grand_mean = raw_means.mean()

    # Compute deviations from grand mean
    deviations = raw_means - grand_mean

    # Compute sum of squared deviations
    sum_sq_dev = (deviations ** 2).sum()

    # Prevent division by zero
    if sum_sq_dev < 1e-10:
        # All means are the same, no shrinkage needed
        return raw_means

    # Estimate variance
    if variance_estimate is None:
        # Use a conservative estimate: mean absolute deviation squared
        # This is more robust to outliers than variance
        variance_estimate = ((raw_means - grand_mean).abs().mean()) ** 2
        # Add small constant for numerical stability
        variance_estimate = variance_estimate.clamp(min=1e-8)

    # Compute shrinkage factor c
    # c = (p - 2) × σ² / Σ(X̄[j] - μ̄)²
    shrinkage_factor = ((p - 2) * variance_estimate) / sum_sq_dev

    # Clamp shrinkage factor to [0, 1] for stability
    # c > 1 means we overshoot, c < 0 means we expand (both bad)
    shrinkage_factor = shrinkage_factor.clamp(0, 1)

    # Apply James-Stein shrinkage
    # μ̂_JS[j] = μ̄ + (1 - c) × (X̄[j] - μ̄)
    js_means = grand_mean + (1 - shrinkage_factor) * deviations

    return js_means


class JamesSteinHeuristicAWQQuantizerXL:
    def __init__(self, model, tokenizer, device="cuda", bits=4, n_grid=20,
                 group_size=128, use_heuristic=True, knee_tolerance=0.1, max_tokens_per_sample=512,
                 layer_batch_size=16, lmhead_chunks=4, max_flip_percent=0.05,
                 use_james_stein=True):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.bits = bits
        self.n_grid = n_grid
        self.group_size = group_size
        self.use_heuristic = use_heuristic
        self.knee_tolerance = knee_tolerance  # Tolerance offset for knee point
        self.max_tokens_per_sample = max_tokens_per_sample
        self.layer_batch_size = layer_batch_size
        self.lmhead_chunks = lmhead_chunks
        self.max_flip_percent = max_flip_percent  # Max percentage of channel size that can be flipped
        self.use_james_stein = use_james_stein  # Enable/disable James-Stein estimation

        # Storage for activations
        self.activation_data = {}
        self.hooks = []
        self.layer_scales = {}

        print(f"\n[James-Stein Heuristic AWQ Quantizer XL Initialized]")
        print(f"  Target bits: {bits}")
        print(f"  Group size: {group_size}")
        print(f"  Token subsampling: {max_tokens_per_sample} tokens/sample")
        print(f"  Layer batch size: {layer_batch_size}")
        print(f"  Use heuristic: {use_heuristic}")
        print(f"  Use James-Stein mean estimation: {use_james_stein}")
        if use_heuristic:
            print(f"  Outlier detection: DYNAMIC (Kneedle algorithm on sorted E[X])")
            print(f"  Knee tolerance offset: {knee_tolerance:.8f}")
            print(f"  Max flip percent per channel: {max_flip_percent*100:.1f}%")
            print(f"  Quantization: HEURISTIC-GUIDED GROUP-WISE ASYMMETRIC [0, {2**bits - 1}]")
        else:
            print(f"  Quantization: STANDARD GROUP-WISE ASYMMETRIC [0, {2**bits - 1}]")
        print(f"  Special: lm_head split into {lmhead_chunks} chunks to avoid OOM")

    def get_hook(self, name):
        """Create a hook function for a specific layer."""
        def hook(_module, input, _output):
            if name not in self.activation_data:
                self.activation_data[name] = []
            if isinstance(input, tuple):
                inp = input[0]
            else:
                inp = input

            # Subsample tokens if sequence is too long (memory optimization)
            if inp.dim() == 3 and inp.shape[1] > self.max_tokens_per_sample:
                seq_len = inp.shape[1]
                indices = torch.randperm(seq_len)[:self.max_tokens_per_sample]
                indices = indices.sort()[0]  # Keep temporal order
                inp = inp[:, indices, :]

            # Store on CPU to save GPU memory, use float32 for numerical stability in L2 computation
            # We need float32 because squaring float16 values can overflow to inf
            self.activation_data[name].append(inp.detach().cpu().float())
        return hook

    @torch.no_grad()
    def get_activation_stats(self, name):
        """
        Compute L2 salience (E[X²]) and James-Stein estimated mean using FLOAT32 precision.

        Returns:
            tuple: (salience, js_mean) where js_mean uses James-Stein shrinkage
        """
        if name not in self.activation_data or len(self.activation_data[name]) == 0:
            return None, None

        X_list = self.activation_data[name]
        total_samples = sum(x.reshape(-1, x.shape[-1]).shape[0] for x in X_list)
        in_features = X_list[0].shape[-1]

        # Use float32 for accumulation
        l2_sum = torch.zeros(in_features, dtype=torch.float32)
        mean_sum = torch.zeros(in_features, dtype=torch.float32)

        for x in X_list:
            x_flat = x.reshape(-1, x.shape[-1]).float()
            l2_sum += x_flat.pow(2).sum(dim=0)
            mean_sum += x_flat.sum(dim=0)

        salience = (l2_sum / total_samples)
        raw_mean = (mean_sum / total_samples)

        # Apply James-Stein estimator to the mean
        if self.use_james_stein:
            js_mean = compute_james_stein_mean(raw_mean)
        else:
            js_mean = raw_mean

        return salience, js_mean

    @torch.no_grad()
    def compute_dynamic_outlier_threshold(self, activation_means, debug=False):
        """
        Compute dynamic outlier threshold using Kneedle algorithm.

        Strategy:
        1. Sort activation means in DESCENDING order [high ... medium ... low]
        2. Apply Kneedle to FIRST HALF [high ... medium] to find outlier→normal transition
        3. Tolerance offset allows tuning: positive = more conservative (keep more outliers)

        Args:
            activation_means: Tensor of per-channel activation means (E[X])
            debug: Print debug information

        Returns:
            tuple: (threshold value, outlier percentage)
        """
        # Sort activation means in DESCENDING order [high → low]
        sorted_means, _ = torch.sort(activation_means.abs(), descending=True)
        n = len(sorted_means)

        # Apply Kneedle to FIRST HALF [high ... medium] to find outlier transition
        first_half = sorted_means[:n // 2]

        if len(first_half) < 3:
            # Not enough data, use a conservative default (top 5%)
            # In descending order, top 5% is at the beginning
            threshold_idx = int(0.05 * n)
            threshold = sorted_means[threshold_idx].item()
            outlier_percent = 0.05
            if debug:
                print(f"    DEBUG: Not enough data for Kneedle, using top 5% as default")
            return threshold, outlier_percent

        # Find knee point in first half (where outliers end, normal begins)
        knee_idx_in_half = find_knee_point(first_half, tolerance_offset=self.knee_tolerance)

        # This is already the index in full array (descending sorted)
        knee_idx = knee_idx_in_half

        # The threshold is the value at the knee point
        threshold = sorted_means[knee_idx].item()

        # Count how many channels are outliers (above or equal to threshold)
        num_outliers = (activation_means.abs() >= threshold).sum().item()
        outlier_percent = num_outliers / n

        if debug:
            print(f"    DEBUG: Sorted means (descending): [{sorted_means[0]:.6f} ... {sorted_means[-1]:.6f}]")
            print(f"    DEBUG: First half range: [{first_half[0]:.6f} ... {first_half[-1]:.6f}]")
            print(f"    DEBUG: Knee point index in first half: {knee_idx_in_half}/{len(first_half)}")
            print(f"    DEBUG: Knee point index in full array: {knee_idx}/{n} ({knee_idx/n*100:.1f}%)")
            print(f"    DEBUG: Knee threshold value: {threshold:.6f}")
            print(f"    DEBUG: Outliers (>= threshold): {num_outliers}/{n} ({outlier_percent*100:.2f}%)")
            print(f"    DEBUG: vs Default 5.00%: {outlier_percent*100 - 5.0:+.2f}% difference")

        return threshold, outlier_percent

    @torch.no_grad()
    def quantize_weight_heuristic_groupwise(self, W, group_activation_means, apply_heuristic=True, debug=False):
        """
        Vectorized implementation of 'quantize_groupwise_global_greedy' with DYNAMIC outlier detection.

        Returns:
            tuple: (W_dequant, outlier_percent, flip_stats) if apply_heuristic=True, else (W_dequant, None, flip_stats)
        """
        out_features, in_features = W.shape
        device = W.device

        # --- 1. Pre-processing / Padding ---
        n_groups = (in_features + self.group_size - 1) // self.group_size
        padded_in_features = n_groups * self.group_size

        if padded_in_features > in_features:
            W_padded = torch.zeros(out_features, padded_in_features, device=device, dtype=W.dtype)
            W_padded[:, :in_features] = W
            act_padded = torch.zeros(padded_in_features, device=device, dtype=W.dtype)
            act_padded[:in_features] = group_activation_means
        else:
            W_padded = W
            act_padded = group_activation_means

        # Reshape to groups for scaling
        W_g = W_padded.reshape(out_features, n_groups, self.group_size)

        # Asymmetric Quantization Setup
        w_min = W_g.min(dim=2, keepdim=True)[0]
        w_max = W_g.max(dim=2, keepdim=True)[0]
        max_int = 2**self.bits - 1

        scale = (w_max - w_min) / max_int
        scale = scale.clamp(min=1e-8)
        zp = torch.round(-w_min / scale).clamp(0, max_int)

        # Expand to full size [out, padded_in]
        scale_flat = scale.repeat(1, 1, self.group_size).reshape(out_features, padded_in_features)
        zp_flat = zp.repeat(1, 1, self.group_size).reshape(out_features, padded_in_features)

        # --- 2. Initial Quantization ---
        W_div = W_padded / scale_flat
        W_int = torch.round(W_div + zp_flat).clamp(0, max_int)
        W_quant = (W_int - zp_flat) * scale_flat

        if not apply_heuristic:
            # Return early if simple rounding (no flips)
            W_dequant = (W_int - zp_flat) * scale_flat
            if padded_in_features > in_features: W_dequant = W_dequant[:, :in_features]
            # Return empty flip stats
            flip_stats = {
                'total': 0, 'per_channel_mean': 0, 'per_channel_median': 0,
                'per_channel_min': 0, 'per_channel_max': 0, 'per_channel_std': 0,
                'per_channel_p25': 0, 'per_channel_p75': 0, 'per_channel_p90': 0,
                'per_channel_p95': 0, 'per_channel_p99': 0, 'per_channel_zero_pct': 100
            }
            return W_dequant.to(W.dtype), None, flip_stats

        # --- 3. Global Greedy Heuristic (Vectorized) ---

        # A. Calculate Current Error
        W_diff = W_padded - W_quant
        current_error = (W_diff * act_padded.unsqueeze(0)).sum(dim=1) # [out_features]

        # B. Identify Flip Candidates
        flip_dir = torch.sign(W_div + zp_flat - W_int)
        flip_dir[flip_dir == 0] = 1.0
        flip_impacts = act_padded.unsqueeze(0) * flip_dir * scale_flat # [out, in]

        # C. Validity Masks
        target_sign = torch.sign(current_error).unsqueeze(1)
        valid_mask = (torch.sign(flip_impacts) == target_sign)

        w_int_proposed = W_int + flip_dir
        in_range = (w_int_proposed >= 0) & (w_int_proposed <= max_int)
        valid_mask = valid_mask & in_range

        # DYNAMIC Outlier Masking using Kneedle algorithm
        outlier_threshold, outlier_percent = self.compute_dynamic_outlier_threshold(act_padded, debug=debug)
        is_outlier = act_padded.abs() > outlier_threshold
        valid_mask = valid_mask & (~is_outlier).unsqueeze(0)

        # --- 4. Sorting & Optimization ---
        rounding_costs = (W_div + zp_flat - W_int).abs()
        rounding_costs_masked = rounding_costs.clone()
        rounding_costs_masked[~valid_mask] = -1.0

        sorted_indices = torch.argsort(rounding_costs_masked, dim=1, descending=True)
        sorted_impacts = torch.gather(flip_impacts, 1, sorted_indices)
        sorted_validity = torch.gather(valid_mask.long(), 1, sorted_indices)
        sorted_impacts = sorted_impacts * sorted_validity

        cumsum_impacts = torch.cumsum(sorted_impacts, dim=1)
        residuals = torch.abs(current_error.unsqueeze(1) - cumsum_impacts)
        error_unsqueezed = torch.abs(current_error).unsqueeze(1)
        all_residuals = torch.cat([error_unsqueezed, residuals], dim=1)
        best_k = torch.argmin(all_residuals, dim=1)

        # --- 5. Apply Flips ---
        idx_range = torch.arange(padded_in_features, device=device).unsqueeze(0)
        flip_mask_sorted = idx_range < best_k.unsqueeze(1)
        final_flips_sorted = flip_mask_sorted & (sorted_validity.bool())

        # --- Constraint: Limit flips per OUTPUT channel (row W[i,:]) to max_flip_percent ---
        # For each output neuron i, limit flips in W[i,:] to max_flip_percent * in_features
        # This is already in sorted order, so we can apply directly

        # Apply flip directions first
        sorted_flip_dir = torch.gather(flip_dir, 1, sorted_indices)
        sorted_flip_dir[~final_flips_sorted] = 0.0

        # Constraint: Keep only top max_flips_per_output flips per row (ignoring masked outliers)
        max_flips_per_output = int(self.max_flip_percent * in_features)
        cumsum_flips = final_flips_sorted.long().cumsum(dim=1)  # Cumulative count along row
        within_limit = cumsum_flips <= max_flips_per_output
        sorted_flip_dir[~within_limit] = 0.0

        W_int.scatter_add_(1, sorted_indices, sorted_flip_dir)
        W_int.clamp_(0, max_int)

        # --- 6. Compute Flip Statistics ---
        # Total flips
        num_flips_total = final_flips_sorted.sum().item()

        # Per-channel flip statistics (sum across output dimension)
        flips_per_channel = final_flips_sorted.sum(dim=0).float()  # [padded_in_features]

        # Only consider actual channels (not padding)
        if padded_in_features > in_features:
            flips_per_channel = flips_per_channel[:in_features]

        # Compute comprehensive statistics
        flip_stats = {
            'total': num_flips_total,
            'per_channel_mean': flips_per_channel.mean().item(),
            'per_channel_median': flips_per_channel.median().item(),
            'per_channel_min': flips_per_channel.min().item(),
            'per_channel_max': flips_per_channel.max().item(),
            'per_channel_std': flips_per_channel.std().item(),
            # Percentiles
            'per_channel_p25': torch.quantile(flips_per_channel, 0.25).item(),
            'per_channel_p75': torch.quantile(flips_per_channel, 0.75).item(),
            'per_channel_p90': torch.quantile(flips_per_channel, 0.90).item(),
            'per_channel_p95': torch.quantile(flips_per_channel, 0.95).item(),
            'per_channel_p99': torch.quantile(flips_per_channel, 0.99).item(),
            # Percentage with 0 flips
            'per_channel_zero_pct': (flips_per_channel == 0).float().mean().item() * 100
        }

        # --- 7. Dequantize & Return ---
        W_dequant = (W_int - zp_flat) * scale_flat

        if padded_in_features > in_features:
            W_dequant = W_dequant[:, :in_features]

        return W_dequant.to(W.dtype), outlier_percent, flip_stats

    @torch.no_grad()
    def search_best_scale(self, name, module):
        """Grid search for optimal per-input-channel scaling factor."""
        if name not in self.activation_data or len(self.activation_data[name]) == 0:
            in_features = module.weight.shape[1]
            return torch.ones(in_features).to(self.device), 0.0, 0.0

        activation_salience, js_mean = self.get_activation_stats(name)
        if activation_salience is None:
            in_features = module.weight.shape[1]
            return torch.ones(in_features).to(self.device), 0.0, 0.0

        activation_salience = activation_salience.to(self.device).to(module.weight.dtype)
        js_mean = js_mean.to(self.device).to(module.weight.dtype)

        # Subsample for speed
        X_list = self.activation_data[name]
        X_cpu = torch.cat([x.reshape(-1, x.shape[-1]) for x in X_list], dim=0)

        max_samples = min(2048, X_cpu.shape[0])
        if X_cpu.shape[0] > max_samples:
            indices = torch.randperm(X_cpu.shape[0])[:max_samples]
            X_search = X_cpu[indices].to(self.device)
        else:
            X_search = X_cpu.to(self.device)

        del X_cpu

        if X_search.dtype != module.weight.dtype:
            X_search = X_search.to(module.weight.dtype)

        W = module.weight.data
        Y_orig = torch.matmul(X_search, W.t())

        best_error = float('inf')
        best_alpha = 0.0
        best_scales = torch.ones(W.shape[1], device=self.device)

        # Clamp min salience
        activation_salience = activation_salience.clamp(min=1e-5)

        for grid_idx in range(self.n_grid + 1):
            alpha = grid_idx / self.n_grid
            scales = activation_salience.pow(alpha)

            W_scaled = W * scales.unsqueeze(0)
            scaled_act_mean = js_mean / scales

            W_quant, _, _ = self.quantize_weight_heuristic_groupwise(
                W_scaled,
                scaled_act_mean,
                apply_heuristic=self.use_heuristic
            )

            W_recon = W_quant / scales.unsqueeze(0)
            Y_quant = torch.matmul(X_search, W_recon.t())
            error = (Y_orig - Y_quant).pow(2).mean().item()

            if error < best_error:
                best_error = error
                best_alpha = alpha
                best_scales = scales.clone()

            del W_scaled, W_quant, W_recon, Y_quant, scales

        del X_search, Y_orig
        torch.cuda.empty_cache()

        return best_scales, best_alpha, best_error

    @torch.no_grad()
    def search_best_scale_lmhead_half(self, name, module, out_start, out_end, debug=False):
        """
        Grid search for lm_head, processing only a chunk of the output dimension.

        Args:
            out_start: Starting index of output dimension
            out_end: Ending index of output dimension
        """
        if name not in self.activation_data or len(self.activation_data[name]) == 0:
            in_features = module.weight.shape[1]
            return torch.ones(in_features).to(self.device), 0.0, 0.0

        activation_salience, js_mean = self.get_activation_stats(name)
        if activation_salience is None:
            if debug:
                print(f"  DEBUG: No activation salience for {name}, using default scales")
            in_features = module.weight.shape[1]
            return torch.ones(in_features).to(self.device), 0.0, 0.0

        if debug:
            print(f"  DEBUG: Processing lm_head rows {out_start}:{out_end}")
            print(f"  DEBUG: Salience shape={activation_salience.shape}, "
                  f"mean={activation_salience.mean():.6f}, max={activation_salience.max():.6f}")

        activation_salience = activation_salience.to(self.device).to(module.weight.dtype)
        js_mean = js_mean.to(self.device).to(module.weight.dtype)

        # Prepare calibration data - use fewer samples for lm_head
        X_list = self.activation_data[name]
        X_cpu = torch.cat([x.reshape(-1, x.shape[-1]) for x in X_list], dim=0)

        max_samples = min(1024, X_cpu.shape[0])  # Reduced for lm_head
        if X_cpu.shape[0] > max_samples:
            indices = torch.randperm(X_cpu.shape[0])[:max_samples]
            X_search = X_cpu[indices].to(self.device)
        else:
            X_search = X_cpu.to(self.device)

        del X_cpu

        if X_search.dtype != module.weight.dtype:
            X_search = X_search.to(module.weight.dtype)

        # Get only the slice of weights we're processing
        W_full = module.weight.data
        W = W_full[out_start:out_end, :]  # Slice output dimension

        # Compute original output for this slice
        Y_orig = torch.matmul(X_search, W.t())

        best_error = float('inf')
        best_alpha = 0.0
        best_scales = torch.ones(W.shape[1], device=self.device)

        activation_salience = activation_salience.clamp(min=1e-5)

        # Grid search over α
        for grid_idx in range(self.n_grid + 1):
            alpha = grid_idx / self.n_grid
            scales = activation_salience.pow(alpha)

            W_scaled = W * scales.unsqueeze(0)
            scaled_act_mean = js_mean / scales

            W_quant, _, _ = self.quantize_weight_heuristic_groupwise(
                W_scaled,
                scaled_act_mean,
                apply_heuristic=self.use_heuristic,
                debug=(debug and grid_idx == 0)  # Debug first iteration
            )

            W_recon = W_quant / scales.unsqueeze(0)
            Y_quant = torch.matmul(X_search, W_recon.t())
            error = (Y_orig - Y_quant).pow(2).mean().item()

            if error < best_error:
                best_error = error
                best_alpha = alpha
                best_scales = scales.clone()

            del W_scaled, W_quant, W_recon, Y_quant, scales

        del X_search, Y_orig, W
        torch.cuda.empty_cache()

        return best_scales, best_alpha, best_error

    @torch.no_grad()
    def quantize_lmhead_half_by_half(self, name, module, debug=False, num_chunks=4):
        """
        Quantize lm_head by splitting it into N chunks along output dimension.
        This reduces peak memory usage significantly.

        Args:
            num_chunks: Number of chunks to split into (default: 4 for very large lm_heads)
        """
        print(f"\n  🔧 Special handling for {name} (split into {num_chunks} chunks)")

        W = module.weight.data
        original_dtype = W.dtype
        out_features, in_features = W.shape

        print(f"     Shape: {W.shape} ({W.numel() / 1e6:.1f}M parameters)")

        # Get activation stats once for all chunks
        _, js_mean = self.get_activation_stats(name)
        if js_mean is None:
            js_mean = torch.zeros(in_features, device=self.device, dtype=W.dtype)
        else:
            js_mean = js_mean.to(self.device).to(W.dtype)

        # Calculate chunk boundaries
        chunk_size = out_features // num_chunks
        chunk_boundaries = [(i * chunk_size,
                            out_features if i == num_chunks - 1 else (i + 1) * chunk_size)
                           for i in range(num_chunks)]

        W_final_chunks = []
        chunk_stats = []

        # Process each chunk
        for chunk_idx, (start_idx, end_idx) in enumerate(chunk_boundaries):
            print(f"     Processing chunk {chunk_idx + 1}/{num_chunks}: rows {start_idx}-{end_idx}")

            # Grid search for this chunk
            best_scales, best_alpha, best_error = self.search_best_scale_lmhead_half(
                name, module, start_idx, end_idx, debug=(debug and chunk_idx == 0)
            )

            # Scale and quantize this chunk
            W_chunk = W[start_idx:end_idx, :]
            W_scaled = W_chunk * best_scales.unsqueeze(0)
            scaled_act_mean = js_mean / best_scales

            W_quant, outlier_pct, flip_stats = self.quantize_weight_heuristic_groupwise(
                W_scaled,
                scaled_act_mean,
                apply_heuristic=self.use_heuristic
            )
            W_final_chunk = (W_quant / best_scales.unsqueeze(0)).to(original_dtype)

            W_final_chunks.append(W_final_chunk)
            chunk_stats.append({
                'alpha': best_alpha,
                'error': best_error,
                'scales': best_scales,
                'outlier_percent': outlier_pct if outlier_pct is not None else 0.0,
                'flip_stats': flip_stats
            })

            # Cleanup
            del W_chunk, W_scaled, W_quant, W_final_chunk, scaled_act_mean
            torch.cuda.empty_cache()

        # Combine all chunks
        W_final = torch.cat(W_final_chunks, dim=0)
        module.weight.data = W_final

        # Store average statistics
        avg_alpha = np.mean([s['alpha'] for s in chunk_stats])
        avg_error = np.mean([s['error'] for s in chunk_stats])
        avg_outlier_pct = np.mean([s['outlier_percent'] for s in chunk_stats])

        # Aggregate flip statistics
        total_flips = sum([s['flip_stats']['total'] for s in chunk_stats])
        avg_per_channel_mean = np.mean([s['flip_stats']['per_channel_mean'] for s in chunk_stats])
        avg_per_channel_median = np.mean([s['flip_stats']['per_channel_median'] for s in chunk_stats])
        avg_per_channel_std = np.mean([s['flip_stats']['per_channel_std'] for s in chunk_stats])
        avg_per_channel_p25 = np.mean([s['flip_stats']['per_channel_p25'] for s in chunk_stats])
        avg_per_channel_p75 = np.mean([s['flip_stats']['per_channel_p75'] for s in chunk_stats])
        avg_per_channel_p90 = np.mean([s['flip_stats']['per_channel_p90'] for s in chunk_stats])
        avg_per_channel_p95 = np.mean([s['flip_stats']['per_channel_p95'] for s in chunk_stats])
        avg_per_channel_p99 = np.mean([s['flip_stats']['per_channel_p99'] for s in chunk_stats])
        avg_per_channel_zero_pct = np.mean([s['flip_stats']['per_channel_zero_pct'] for s in chunk_stats])

        # Build detailed stats dict
        stats_dict = {
            'scales': chunk_stats[0]['scales'].cpu(),  # Use first chunk's scales
            'alpha': avg_alpha,
            'error': avg_error,
            'outlier_percent': avg_outlier_pct,
            'flip_stats': {
                'total': total_flips,
                'per_channel_mean': avg_per_channel_mean,
                'per_channel_median': avg_per_channel_median,
                'per_channel_std': avg_per_channel_std,
                'per_channel_p25': avg_per_channel_p25,
                'per_channel_p75': avg_per_channel_p75,
                'per_channel_p90': avg_per_channel_p90,
                'per_channel_p95': avg_per_channel_p95,
                'per_channel_p99': avg_per_channel_p99,
                'per_channel_zero_pct': avg_per_channel_zero_pct
            }
        }
        for i, stat in enumerate(chunk_stats):
            stats_dict[f'alpha_chunk{i+1}'] = stat['alpha']
            stats_dict[f'error_chunk{i+1}'] = stat['error']
            stats_dict[f'outlier_pct_chunk{i+1}'] = stat['outlier_percent']
            stats_dict[f'flips_chunk{i+1}'] = stat['flip_stats']['total']

        self.layer_scales[name] = stats_dict

        # Print summary
        alpha_str = ', '.join([f'α_{i+1}={s["alpha"]:.4f}' for i, s in enumerate(chunk_stats)])
        error_str = ', '.join([f'err_{i+1}={s["error"]:.8f}' for i, s in enumerate(chunk_stats)])
        outlier_str = ', '.join([f'out_{i+1}={s["outlier_percent"]*100:.2f}%' for i, s in enumerate(chunk_stats)])
        flips_str = ', '.join([f'flips_{i+1}={s["flip_stats"]["total"]:,}' for i, s in enumerate(chunk_stats)])
        print(f"     ✓ Done: {alpha_str}")
        print(f"             {error_str}")
        print(f"             {outlier_str}")
        print(f"             {flips_str}")

        del W_final_chunks, chunk_stats, W_final, js_mean
        torch.cuda.empty_cache()

    @torch.no_grad()
    def quantize_layer(self, name, module):
        """Apply James-Stein Heuristic AWQ Quantization."""
        best_scales, best_alpha, best_error = self.search_best_scale(name, module)

        W = module.weight.data
        original_dtype = W.dtype
        W_scaled = W * best_scales.unsqueeze(0)

        _, js_mean = self.get_activation_stats(name)
        if js_mean is not None:
            scaled_act_mean = (js_mean.to(self.device).to(W.dtype) / best_scales)
        else:
            scaled_act_mean = torch.zeros(W.shape[1], device=W.device, dtype=W.dtype)

        W_quant, outlier_pct, flip_stats = self.quantize_weight_heuristic_groupwise(
            W_scaled,
            scaled_act_mean,
            apply_heuristic=self.use_heuristic
        )

        W_final = (W_quant / best_scales.unsqueeze(0)).to(original_dtype)
        module.weight.data = W_final

        self.layer_scales[name] = {
            'scales': best_scales.cpu(),
            'alpha': best_alpha,
            'error': best_error,
            'outlier_percent': outlier_pct if outlier_pct is not None else 0.0,
            'flip_stats': flip_stats
        }

        del best_scales, scaled_act_mean, W_scaled, W_quant, W_final
        if name in self.activation_data:
            del self.activation_data[name]
        torch.cuda.empty_cache()
        gc.collect()

    def calibrate_layer_batch(self, layer_names_batch, calibration_data, n_samples=500):
        """Calibrate a batch of layers simultaneously."""
        print(f"  Calibrating {len(layer_names_batch)} layers...")

        self.model.eval()
        handles = []

        # Register hooks for all layers in this batch
        for name, module in layer_names_batch:
            handle = module.register_forward_hook(self.get_hook(name))
            handles.append((name, handle))

        # Run calibration
        successful = 0
        with torch.no_grad():
            for text in tqdm(calibration_data[:n_samples], desc="  Calibration", leave=False):
                try:
                    inputs = self.tokenizer(text, return_tensors="pt",
                                           truncation=True, max_length=512)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    self.model(**inputs, use_cache=False, return_dict=True)
                    successful += 1

                    # Periodic cache clearing
                    if (successful + 1) % 32 == 0:
                        torch.cuda.empty_cache()
                except Exception:
                    continue

        # Remove hooks
        for _, handle in handles:
            handle.remove()

        torch.cuda.empty_cache()
        gc.collect()

    def quantize_model_sequential(self, calibration_data, n_samples=500):
        """Batched sequential quantization with special lm_head handling."""
        print("\n" + "=" * 80)
        print("Batched Sequential Quantization (James-Stein XL Version)")
        print("=" * 80)
        print(f"  Strategy: Process {self.layer_batch_size} layers per batch")

        if HAS_PSUTIL:
            initial_ram = psutil.virtual_memory().percent
            print(f"  Initial System RAM: {initial_ram:.1f}%")

        layer_names = [(name, module) for name, module in self.model.named_modules()
                       if isinstance(module, nn.Linear)]

        num_layers = len(layer_names)
        num_batches = (num_layers + self.layer_batch_size - 1) // self.layer_batch_size

        print(f"  Total layers: {num_layers}")
        print(f"  Total batches: {num_batches}")
        print("=" * 80)

        quantized_count = 0

        # Process in batches
        for batch_idx in range(num_batches):
            batch_start = batch_idx * self.layer_batch_size
            batch_end = min(batch_start + self.layer_batch_size, num_layers)
            batch_layers = layer_names[batch_start:batch_end]

            print(f"\n[Batch {batch_idx + 1}/{num_batches}] Layers {batch_start}-{batch_end-1}")

            # Calibrate this batch
            self.calibrate_layer_batch(batch_layers, calibration_data, n_samples)

            # Quantize all layers in this batch
            print(f"  Quantizing {len(batch_layers)} layers...")
            for name, module in tqdm(batch_layers, desc="  Quantization", leave=False):
                try:
                    # Check if this is lm_head (special handling)
                    is_lmhead = 'lm_head' in name.lower() or name.endswith('lm_head')

                    if is_lmhead:
                        # Use chunked processing for lm_head
                        debug = (quantized_count < 2)
                        self.quantize_lmhead_half_by_half(name, module, debug=debug, num_chunks=self.lmhead_chunks)
                    else:
                        # Standard processing
                        self.quantize_layer(name, module)

                    quantized_count += 1

                except Exception as e:
                    print(f"\n⚠️  Error quantizing {name}: {e}")
                    continue

            # Clear activations for this batch
            self.activation_data = {}
            torch.cuda.empty_cache()
            gc.collect()

            if HAS_PSUTIL:
                ram_pct = psutil.virtual_memory().percent
                print(f"  Batch {batch_idx+1} complete. RAM: {ram_pct:.1f}%")

        print("\n" + "=" * 80)
        print("✓ Batched Sequential Quantization Complete")
        print(f"  Total layers quantized: {quantized_count}/{num_layers}")
        print("=" * 80)

        if self.layer_scales:
            alphas = [info['alpha'] for info in self.layer_scales.values()]
            outlier_pcts = [info.get('outlier_percent', 0.0) for info in self.layer_scales.values()]

            print(f"\nOptimal α statistics:")
            print(f"  Mean: {np.mean(alphas):.3f}")
            print(f"  Median: {np.median(alphas):.3f}")

            if self.use_heuristic and outlier_pcts:
                mean_outlier = np.mean(outlier_pcts) * 100
                median_outlier = np.median(outlier_pcts) * 100
                default_pct = 5.0
                diff_from_default = mean_outlier - default_pct

                print(f"\nDynamic Outlier Detection statistics:")
                print(f"  Mean outlier %: {mean_outlier:.2f}%")
                print(f"  Median outlier %: {median_outlier:.2f}%")
                print(f"  vs Default 5.00%: {diff_from_default:+.2f}%")
                print(f"  Min: {np.min(outlier_pcts)*100:.2f}% | Max: {np.max(outlier_pcts)*100:.2f}%")

                if mean_outlier < default_pct:
                    print(f"  → Dynamic method keeps FEWER outliers (more aggressive quantization)")
                elif mean_outlier > default_pct:
                    print(f"  → Dynamic method keeps MORE outliers (more conservative)")
                else:
                    print(f"  → Similar to default 5%")

            if self.use_heuristic:
                # Collect flip statistics
                flip_totals = [info.get('flip_stats', {}).get('total', 0) for info in self.layer_scales.values()]
                per_ch_means = [info.get('flip_stats', {}).get('per_channel_mean', 0) for info in self.layer_scales.values()]
                per_ch_medians = [info.get('flip_stats', {}).get('per_channel_median', 0) for info in self.layer_scales.values()]
                per_ch_stds = [info.get('flip_stats', {}).get('per_channel_std', 0) for info in self.layer_scales.values()]
                per_ch_p25s = [info.get('flip_stats', {}).get('per_channel_p25', 0) for info in self.layer_scales.values()]
                per_ch_p75s = [info.get('flip_stats', {}).get('per_channel_p75', 0) for info in self.layer_scales.values()]
                per_ch_p90s = [info.get('flip_stats', {}).get('per_channel_p90', 0) for info in self.layer_scales.values()]
                per_ch_p95s = [info.get('flip_stats', {}).get('per_channel_p95', 0) for info in self.layer_scales.values()]
                per_ch_p99s = [info.get('flip_stats', {}).get('per_channel_p99', 0) for info in self.layer_scales.values()]
                per_ch_zero_pcts = [info.get('flip_stats', {}).get('per_channel_zero_pct', 0) for info in self.layer_scales.values()]

                total_flips = np.sum(flip_totals)
                mean_flips_per_layer = np.mean(flip_totals)
                median_flips_per_layer = np.median(flip_totals)
                min_flips = np.min(flip_totals)
                max_flips = np.max(flip_totals)

                avg_per_ch_mean = np.mean(per_ch_means)
                avg_per_ch_median = np.mean(per_ch_medians)
                avg_per_ch_std = np.mean(per_ch_stds)
                avg_per_ch_p25 = np.mean(per_ch_p25s)
                avg_per_ch_p75 = np.mean(per_ch_p75s)
                avg_per_ch_p90 = np.mean(per_ch_p90s)
                avg_per_ch_p95 = np.mean(per_ch_p95s)
                avg_per_ch_p99 = np.mean(per_ch_p99s)
                avg_per_ch_zero_pct = np.mean(per_ch_zero_pcts)

                print(f"\nWeight Flipping Statistics:")
                print(f"  Total flips across all layers: {int(total_flips):,}")
                print(f"  Mean flips per layer: {mean_flips_per_layer:,.1f}")
                print(f"  Median flips per layer: {median_flips_per_layer:,.1f}")
                print(f"  Min: {int(min_flips):,} | Max: {int(max_flips):,}")
                print(f"\n  Per-Channel Statistics (averaged across layers):")
                print(f"    Mean flips per channel: {avg_per_ch_mean:.2f}")
                print(f"    Median flips per channel: {avg_per_ch_median:.2f}")
                print(f"    Std dev: {avg_per_ch_std:.2f}")
                print(f"\n    Percentiles:")
                print(f"      25th: {avg_per_ch_p25:.2f} | 50th: {avg_per_ch_median:.2f} | 75th: {avg_per_ch_p75:.2f}")
                print(f"      90th: {avg_per_ch_p90:.2f} | 95th: {avg_per_ch_p95:.2f} | 99th: {avg_per_ch_p99:.2f}")
                print(f"\n    Channels with 0 flips: {avg_per_ch_zero_pct:.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-calib", type=int, default=128, help="Number of calibration samples")
    parser.add_argument("--n-grid", type=int, default=20)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--bits", type=int, default=4, choices=[3, 4], help="Quantization bit width (default: 4)")
    parser.add_argument("--use-heuristic", action="store_true", default=True,
                       help="Enable heuristic rounding (default: True)")
    parser.add_argument("--no-heuristic", dest="use_heuristic", action="store_false",
                       help="Disable heuristic rounding")
    parser.add_argument("--use-james-stein", action="store_true", default=True,
                       help="Enable James-Stein mean estimation (default: True)")
    parser.add_argument("--no-james-stein", dest="use_james_stein", action="store_false",
                       help="Disable James-Stein mean estimation")
    parser.add_argument("--knee-tolerance", type=float, default=0.000,
                       help="Tolerance offset for knee point (default: 0.000, higher = more conservative)")
    parser.add_argument("--max-flip-percent", type=float, default=0.05,
                       help="Max percentage of channel size that can be flipped (default: 0.05 = 5%%)")
    parser.add_argument("--max-tokens-per-sample", type=int, default=2048,
                       help="Max tokens to store per sample (default: 2048)")
    parser.add_argument("--layer-batch-size", type=int, default=16,
                       help="Number of layers to process per batch (default: 16)")
    parser.add_argument("--lmhead-chunks", type=int, default=4,
                       help="Number of chunks to split lm_head into (default: 4, higher = less memory)")
    parser.add_argument("--output-dir", type=str, default="./quantized_models/model_awq_js_xl")
    parser.add_argument("--model-path", type=str, default="./models/Mistral-7B-v0.3",
                       help="Model name or local path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calib-dataset", type=str, default="c4",
                       choices=["c4", "wikitext2", "wikitext2-simple"],
                       help="Calibration dataset (default: c4)")
    parser.add_argument("--cache-dir", type=str, default="./calibration_cache",
                       help="Directory to cache calibration data (default: ./calibration_cache)")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Use model path from args
    model_name = args.model_path
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("James-Stein Heuristic AWQ (XL Version)")
    print(f"Target Model: {model_name}")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Group size: {args.group_size}")
    print(f"Layer Batch Size: {args.layer_batch_size}")
    print(f"Use heuristic: {args.use_heuristic}")
    print(f"Use James-Stein estimator: {args.use_james_stein}")
    print(f"Dynamic outlier detection: Kneedle algorithm")
    print(f"Knee tolerance offset: {args.knee_tolerance}")
    print(f"Max flip percent per channel: {args.max_flip_percent*100:.1f}%")
    print(f"Special: lm_head split into {args.lmhead_chunks} chunks")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # Fix for Llama/Mistral models lacking pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("  -> Set pad_token = eos_token")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # Use bfloat16 instead of float16 for better numerical stability
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    # Load calibration data
    print(f"\nLoading calibration dataset: {args.calib_dataset}")
    if args.calib_dataset == "c4":
        calib_texts = get_c4_calibration_data(tokenizer, n_samples=args.n_calib, seqlen=2048, seed=args.seed, cache_dir=args.cache_dir)
    elif args.calib_dataset == "wikitext2-simple":
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        calib_texts = [item['text'] for item in dataset if len(item['text'].strip()) > 100][:args.n_calib]
    else:
        calib_texts = get_wikitext2_calibration_data(tokenizer, n_samples=args.n_calib, seqlen=2048, seed=args.seed, cache_dir=args.cache_dir)

    quantizer = JamesSteinHeuristicAWQQuantizerXL(
        model=model,
        tokenizer=tokenizer,
        device=device,
        bits=args.bits,
        n_grid=args.n_grid,
        group_size=args.group_size,
        use_heuristic=args.use_heuristic,
        knee_tolerance=args.knee_tolerance,
        max_tokens_per_sample=args.max_tokens_per_sample,
        layer_batch_size=args.layer_batch_size,
        lmhead_chunks=args.lmhead_chunks,
        max_flip_percent=args.max_flip_percent,
        use_james_stein=args.use_james_stein
    )

    # Use batched sequential quantization (optimal memory/speed balance)
    quantizer.quantize_model_sequential(calib_texts, n_samples=args.n_calib)

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\n✅ Saved to {args.output_dir}")

if __name__ == "__main__":
    main()
