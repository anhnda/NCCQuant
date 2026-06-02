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
from calibration_utils import get_c4_calibration_data, get_wikitext2_calibration_data

class HeuristicGroupWiseAWQQuantizer:
    """
    Group-Wise AWQ with Heuristic-Guided Asymmetric Quantization.
    
    Corrected to match Global Greedy logic:
    - Includes Outlier Masking (ignores top X% activations for stability)
    - Global Candidate Sorting based on Rounding Cost
    - Vectorized implementation of the reference quantize_groupwise_global_greedy
    """

    def __init__(self, model, tokenizer, device="cuda", bits=4, n_grid=20, 
                 group_size=128, use_heuristic=True, outlier_percent=0.05):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.bits = bits
        self.n_grid = n_grid
        self.group_size = group_size
        self.use_heuristic = use_heuristic
        self.outlier_percent = outlier_percent

        # Storage for activations
        self.activation_data = {}
        self.hooks = []
        self.layer_scales = {}

        print(f"\n[Heuristic Group-Wise AWQ Quantizer Initialized]")
        print(f"  Target bits: {bits}")
        print(f"  Group size: {group_size}")
        print(f"  Outlier protection: Top {outlier_percent*100:.1f}% ignored")
        print(f"  Quantization: HEURISTIC-GUIDED GROUP-WISE ASYMMETRIC [0, {2**bits - 1}]")

    def register_hooks(self):
        """Register forward hooks to capture activations."""
        def get_hook(name):
            def hook(module, input, output):
                if name not in self.activation_data:
                    self.activation_data[name] = []
                if isinstance(input, tuple):
                    inp = input[0].detach().cpu()
                else:
                    inp = input.detach().cpu()
                self.activation_data[name].append(inp)
            return hook

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                handle = module.register_forward_hook(get_hook(name))
                self.hooks.append(handle)

    def remove_hooks(self):
        for handle in self.hooks:
            handle.remove()
        self.hooks = []

    @torch.no_grad()
    def get_activation_stats(self, name):
        """Compute L2 salience (E[X²]) and raw mean (E[X]) in one pass."""
        if name not in self.activation_data or len(self.activation_data[name]) == 0:
            return None, None

        X_list = self.activation_data[name]
        total_samples = sum(x.reshape(-1, x.shape[-1]).shape[0] for x in X_list)
        in_features = X_list[0].shape[-1]

        l2_sum = torch.zeros(in_features, dtype=torch.float64)
        mean_sum = torch.zeros(in_features, dtype=torch.float64)

        for x in X_list:
            x_flat = x.reshape(-1, x.shape[-1]).double()
            l2_sum += x_flat.pow(2).sum(dim=0)
            mean_sum += x_flat.sum(dim=0)

        salience = (l2_sum / total_samples).float()
        raw_mean = (mean_sum / total_samples).float()

        return salience, raw_mean

    @torch.no_grad()
    def quantize_weight_heuristic_groupwise(self, W, group_activation_means, apply_heuristic=True):
        """
        Vectorized implementation of 'quantize_groupwise_global_greedy'.
        
        Logic matches reference:
        1. Calculate Scales (Asymmetric [min, max] logic used here to match class config).
        2. Initial Rounding.
        3. Identify Flip candidates (exclude Outliers).
        4. Sort global candidates by Rounding Cost (abs(div - int)).
        5. Find best K to minimize Output Error.
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
            # Return early if simple rounding
            W_dequant = (W_int - zp_flat) * scale_flat
            if padded_in_features > in_features: W_dequant = W_dequant[:, :in_features]
            return W_dequant.to(W.dtype)

        # --- 3. Global Greedy Heuristic (Vectorized) ---
        
        # A. Calculate Current Error
        # Error = dot(X, W_orig - W_quant). We sum over input dim.
        W_diff = W_padded - W_quant
        # act_padded is broadcasted: [out, in] = [out, in] * [in]
        current_error = (W_diff * act_padded.unsqueeze(0)).sum(dim=1) # [out_features]

        # B. Identify Flip Candidates
        # Direction to move W_int to get closer to W_div
        flip_dir = torch.sign(W_div - (W_int - zp_flat)) # Corrected logic: compare float val vs int val relative to 0
        # Actually easier: sign(W_div - (W_int_current_val)) 
        # But W_div includes ZP offset logic. 
        # Simple logic: if W_div > W_int_unbiased, we want to go up.
        # W_div corresponds to (W / s). W_int corresponds to Q(W/s + z).
        # Let's use the reference logic: sign(w_div - w_int) where w_div is simply scaled w.
        # In asymmetric: W_div_asym = W/s + z. W_int = round(W_div_asym).
        flip_dir = torch.sign(W_div + zp_flat - W_int)
        flip_dir[flip_dir == 0] = 1.0

        # Impact on Output: x * sign * scale
        flip_impacts = act_padded.unsqueeze(0) * flip_dir * scale_flat # [out, in]

        # C. Validity Masks
        # 1. Sign must match error direction
        target_sign = torch.sign(current_error).unsqueeze(1)
        valid_mask = (torch.sign(flip_impacts) == target_sign)

        # 2. Range check (0 to 15)
        w_int_proposed = W_int + flip_dir
        in_range = (w_int_proposed >= 0) & (w_int_proposed <= max_int)
        valid_mask = valid_mask & in_range

        # 3. Outlier Masking (Crucial Step from Reference)
        # We ignore the top K% activations to avoid overfitting to outliers
        k_outliers = int(padded_in_features * self.outlier_percent)
        # Safety check: ensure k doesn't exceed tensor size
        k_outliers = min(k_outliers, act_padded.numel())
        if k_outliers > 0:
            # Find indices of top K abs activations
            _, outlier_indices = torch.topk(act_padded.abs(), k_outliers)
            # Create boolean mask [padded_in]
            is_outlier = torch.zeros(padded_in_features, dtype=torch.bool, device=device)
            is_outlier[outlier_indices] = True
            # Broadcast to [out, padded_in] and apply
            valid_mask = valid_mask & (~is_outlier).unsqueeze(0)

        # --- 4. Sorting & Optimization ---
        
        # Calculate Cost: Distance to rounding boundary. 
        # Reference sorts Descending. High cost (0.49) = close to boundary = preferred flip.
        rounding_costs = (W_div + zp_flat - W_int).abs()
        
        # Vectorization Trick: Set cost of INVALID candidates to -1.0.
        # Since we sort Descending, valid costs (0.0 to 0.5) will come first.
        # Invalid ones (-1.0) will go to the end.
        rounding_costs_masked = rounding_costs.clone()
        rounding_costs_masked[~valid_mask] = -1.0

        # Sort [out, in] independent for each row
        sorted_indices = torch.argsort(rounding_costs_masked, dim=1, descending=True)

        # Reorder impacts based on sorted indices
        sorted_impacts = torch.gather(flip_impacts, 1, sorted_indices)
        
        # Determine validity in sorted order (to zero out the -1 tails)
        sorted_validity = torch.gather(valid_mask.long(), 1, sorted_indices)
        sorted_impacts = sorted_impacts * sorted_validity

        # Cumulative Sum of impacts
        cumsum_impacts = torch.cumsum(sorted_impacts, dim=1)
        
        # Find Best K
        # minimizing |error - cumsum|
        residuals = torch.abs(current_error.unsqueeze(1) - cumsum_impacts)
        
        # Prepend the "0 flips" case (original error)
        error_unsqueezed = torch.abs(current_error).unsqueeze(1)
        all_residuals = torch.cat([error_unsqueezed, residuals], dim=1) # [out, in+1]
        
        # Argmin gives best k indices [out]
        best_k = torch.argmin(all_residuals, dim=1)

        # --- 5. Apply Flips ---
        
        # Create a mask of which sorted indices to actually flip
        # indices < best_k are flipped
        idx_range = torch.arange(padded_in_features, device=device).unsqueeze(0)
        flip_mask_sorted = idx_range < best_k.unsqueeze(1)
        
        # Filter valid flips within the top K
        # We must use the sorted_validity because the tail contains invalid garbage
        final_flips_sorted = flip_mask_sorted & (sorted_validity.bool())
        
        # We need the flip values (+1 or -1) in the *original* positions.
        # It's easier to scatter the actual update values back.
        
        # Get flip directions in sorted order
        sorted_flip_dir = torch.gather(flip_dir, 1, sorted_indices)
        
        # Zero out flips we decided NOT to do
        sorted_flip_dir[~final_flips_sorted] = 0.0
        
        # Scatter add back to W_int
        W_int.scatter_add_(1, sorted_indices, sorted_flip_dir)

        # --- 6. Dequantize & Return ---
        W_dequant = (W_int - zp_flat) * scale_flat

        if padded_in_features > in_features:
            W_dequant = W_dequant[:, :in_features]

        return W_dequant.to(W.dtype)

    @torch.no_grad()
    def search_best_scale(self, name, module):
        """Grid search for optimal per-input-channel scaling factor."""
        if name not in self.activation_data or len(self.activation_data[name]) == 0:
            in_features = module.weight.shape[1]
            return torch.ones(in_features).to(self.device), 0.0, 0.0

        activation_salience, raw_mean = self.get_activation_stats(name)
        if activation_salience is None:
            in_features = module.weight.shape[1]
            return torch.ones(in_features).to(self.device), 0.0, 0.0

        activation_salience = activation_salience.to(self.device).to(module.weight.dtype)
        raw_mean = raw_mean.to(self.device).to(module.weight.dtype)

        # Subsampel for speed
        X_list = self.activation_data[name]
        X_cpu = []
        curr_len = 0
        for x in X_list:
            x_f = x.reshape(-1, x.shape[-1])
            X_cpu.append(x_f)
            curr_len += x_f.shape[0]
            if curr_len >= 2048:
                break
        
        X_search = torch.cat(X_cpu, dim=0)[:2048].to(self.device)
        if X_search.dtype != module.weight.dtype:
            X_search = X_search.to(module.weight.dtype)

        W = module.weight.data
        Y_orig = torch.matmul(X_search, W.t())

        best_error = float('inf')
        best_alpha = 0.0
        best_scales = torch.ones(W.shape[1], device=self.device)

        # Optimization: Normalize salience once
        # Avoid division by zero
        activation_salience = activation_salience + 1e-6

        for grid_idx in range(self.n_grid + 1):
            alpha = grid_idx / self.n_grid
            scales = activation_salience.pow(alpha)

            W_scaled = W * scales.unsqueeze(0)
            scaled_act_mean = raw_mean / scales

            W_quant = self.quantize_weight_heuristic_groupwise(
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
    def quantize_layer(self, name, module):
        """Apply Heuristic Group-Wise AWQ Quantization."""
        best_scales, best_alpha, best_error = self.search_best_scale(name, module)

        W = module.weight.data
        W_scaled = W * best_scales.unsqueeze(0)

        _, raw_mean = self.get_activation_stats(name)
        if raw_mean is not None:
            scaled_act_mean = (raw_mean.to(self.device).to(W.dtype) / best_scales)
        else:
            scaled_act_mean = torch.zeros(W.shape[1], device=W.device, dtype=W.dtype)

        W_quant = self.quantize_weight_heuristic_groupwise(
            W_scaled,
            scaled_act_mean,
            apply_heuristic=self.use_heuristic
        )

        W_final = W_quant / best_scales.unsqueeze(0)
        module.weight.data = W_final

        self.layer_scales[name] = {
            'scales': best_scales.cpu(),
            'alpha': best_alpha,
            'error': best_error
        }

        del best_scales, scaled_act_mean, W_scaled, W_quant, W_final
        if name in self.activation_data:
            del self.activation_data[name]
        torch.cuda.empty_cache()
        gc.collect()

    def calibrate(self, calibration_data, n_samples=500):
        print(f"\nCalibrating with {min(n_samples, len(calibration_data))} samples...")
        self.model.eval()
        self.register_hooks()

        successful = 0
        with torch.no_grad():
            for text in tqdm(calibration_data[:n_samples], desc="Calibration"):
                try:
                    inputs = self.tokenizer(text, return_tensors="pt",
                                           truncation=True, max_length=512)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    self.model(**inputs, use_cache=False, return_dict=True)
                    successful += 1
                except Exception:
                    continue

        self.remove_hooks()
        torch.cuda.empty_cache()
        gc.collect()

    def quantize_model(self):
        print("\n" + "=" * 80)
        print("Quantizing with Heuristic-Guided Group-Wise AWQ (Corrected)")
        print("=" * 80)
        
        layer_names = [(name, module) for name, module in self.model.named_modules()
                       if isinstance(module, nn.Linear)]

        for name, module in tqdm(layer_names, desc="Quantizing layers"):
            try:
                self.quantize_layer(name, module)
            except Exception as e:
                print(f"\n⚠️  Error quantizing {name}: {e}")
                continue

def load_wikitext2(split="train", n_samples=None):
    """DEPRECATED: Use calibration_utils.get_wikitext2_calibration_data() instead."""
    print(f"Loading WikiText-2 {split} dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    texts = [item['text'] for item in dataset if len(item['text'].strip()) > 0]
    if n_samples:
        texts = texts[:n_samples]
    return texts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-calib", type=int, default=128)
    parser.add_argument("--n-grid", type=int, default=20)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--outlier-percent", type=float, default=0.05, help="Percent of outliers to ignore")
    parser.add_argument("--output-dir", type=str, default="./quantized_models/awq_heuristic")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calib-dataset", type=str, default="wikitext2",
                       choices=["c4", "wikitext2"],
                       help="Calibration dataset (default: wikitext2)")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_name = "openbmb/MiniCPM-2B-sft-bf16"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device} | Model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16,
                                                device_map=device, trust_remote_code=True)

    # Load calibration data
    print(f"\nLoading calibration dataset: {args.calib_dataset}")
    if args.calib_dataset == "c4":
        calib_texts = get_c4_calibration_data(tokenizer, n_samples=args.n_calib, seqlen=512, seed=args.seed)
    else:
        calib_texts = get_wikitext2_calibration_data(tokenizer, n_samples=args.n_calib, seqlen=512, seed=args.seed)

    quantizer = HeuristicGroupWiseAWQQuantizer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        bits=4,
        n_grid=args.n_grid,
        group_size=args.group_size,
        outlier_percent=args.outlier_percent
    )

    quantizer.calibrate(calib_texts, n_samples=args.n_calib)
    quantizer.quantize_model()

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved to {args.output_dir}")

if __name__ == "__main__":
    main()