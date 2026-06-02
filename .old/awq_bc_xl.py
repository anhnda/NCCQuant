"""
Bias Correction AWQ - XL Version with Naive Bias Correction

This version extends the AWQ quantization with naive bias correction instead of heuristic rounding.

Key Features:
- Same base as awq_js_xl.py (Group-Wise Asymmetric Quantization)
- SPECIAL: Splits lm_head into chunks to avoid OOM
- L2 salience metric
- **BIAS CORRECTION**: Eliminates quantization error by adjusting bias term
- Batched sequential quantization

Bias Correction Algorithm:
1. Quantize weights normally (no flipping)
2. For calibration set, compute error: err = X @ W - X @ W_quant
3. Calculate mean error: bias_correction = mean(err, axis=0)
4. Add correction to bias term (create bias if doesn't exist)
5. This eliminates the expected quantization error
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


class BiasCorrectionAWQQuantizerXL:
    def __init__(self, model, tokenizer, device="cuda", bits=4, n_grid=20,
                 group_size=128, max_tokens_per_sample=512,
                 layer_batch_size=16, lmhead_chunks=4):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.bits = bits
        self.n_grid = n_grid
        self.group_size = group_size
        self.max_tokens_per_sample = max_tokens_per_sample
        self.layer_batch_size = layer_batch_size
        self.lmhead_chunks = lmhead_chunks

        # Storage for activations
        self.activation_data = {}
        self.hooks = []
        self.layer_scales = {}

        print(f"\n[Bias Correction AWQ Quantizer XL Initialized]")
        print(f"  Target bits: {bits}")
        print(f"  Group size: {group_size}")
        print(f"  Token subsampling: {max_tokens_per_sample} tokens/sample")
        print(f"  Layer batch size: {layer_batch_size}")
        print(f"  Quantization: STANDARD GROUP-WISE ASYMMETRIC [0, {2**bits - 1}]")
        print(f"  Bias Correction: ENABLED (naive correction on mean error)")
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

            # Store on CPU to save GPU memory, use float32 for numerical stability
            self.activation_data[name].append(inp.detach().cpu().float())
        return hook

    @torch.no_grad()
    def get_activation_stats(self, name):
        """
        Compute L2 salience (E[X²]) using FLOAT32 precision.

        Returns:
            salience: L2 salience for each input channel
        """
        if name not in self.activation_data or len(self.activation_data[name]) == 0:
            return None

        X_list = self.activation_data[name]
        total_samples = sum(x.reshape(-1, x.shape[-1]).shape[0] for x in X_list)
        in_features = X_list[0].shape[-1]

        # Use float32 for accumulation
        l2_sum = torch.zeros(in_features, dtype=torch.float32)

        for x in X_list:
            x_flat = x.reshape(-1, x.shape[-1]).float()
            l2_sum += x_flat.pow(2).sum(dim=0)

        salience = (l2_sum / total_samples)

        return salience

    @torch.no_grad()
    def quantize_weight_groupwise(self, W):
        """
        Simple group-wise asymmetric quantization without any heuristic.

        Returns:
            W_dequant: Dequantized weights
        """
        out_features, in_features = W.shape
        device = W.device

        # --- 1. Pre-processing / Padding ---
        n_groups = (in_features + self.group_size - 1) // self.group_size
        padded_in_features = n_groups * self.group_size

        if padded_in_features > in_features:
            W_padded = torch.zeros(out_features, padded_in_features, device=device, dtype=W.dtype)
            W_padded[:, :in_features] = W
        else:
            W_padded = W

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

        # --- 2. Quantization (simple rounding) ---
        W_div = W_padded / scale_flat
        W_int = torch.round(W_div + zp_flat).clamp(0, max_int)
        W_dequant = (W_int - zp_flat) * scale_flat

        if padded_in_features > in_features:
            W_dequant = W_dequant[:, :in_features]

        return W_dequant.to(W.dtype)

    @torch.no_grad()
    def compute_bias_correction(self, name, module, W_quant):
        """
        Compute bias correction by measuring quantization error on calibration data.

        Args:
            name: Layer name
            module: The linear module
            W_quant: Quantized weights

        Returns:
            bias_correction: Correction to add to bias term [out_features]
        """
        if name not in self.activation_data or len(self.activation_data[name]) == 0:
            # No calibration data, return zero correction
            return torch.zeros(module.weight.shape[0], device=self.device, dtype=module.weight.dtype)

        X_list = self.activation_data[name]
        W_orig = module.weight.data

        # Concatenate all activations
        X_cpu = torch.cat([x.reshape(-1, x.shape[-1]) for x in X_list], dim=0)

        # Use a subset for efficiency (max 4096 samples)
        max_samples = min(4096, X_cpu.shape[0])
        if X_cpu.shape[0] > max_samples:
            indices = torch.randperm(X_cpu.shape[0])[:max_samples]
            X_samples = X_cpu[indices].to(self.device)
        else:
            X_samples = X_cpu.to(self.device)

        del X_cpu

        if X_samples.dtype != W_orig.dtype:
            X_samples = X_samples.to(W_orig.dtype)

        # Compute original output: Y_orig = X @ W^T
        Y_orig = torch.matmul(X_samples, W_orig.t())

        # Compute quantized output: Y_quant = X @ W_quant^T
        Y_quant = torch.matmul(X_samples, W_quant.t())

        # Compute error: err = Y_orig - Y_quant [n_samples, out_features]
        error = Y_orig - Y_quant

        # Compute mean error across all samples
        bias_correction = error.mean(dim=0)  # [out_features]

        del X_samples, Y_orig, Y_quant, error
        torch.cuda.empty_cache()

        return bias_correction

    @torch.no_grad()
    def search_best_scale(self, name, module):
        """Grid search for optimal per-input-channel scaling factor."""
        if name not in self.activation_data or len(self.activation_data[name]) == 0:
            in_features = module.weight.shape[1]
            return torch.ones(in_features).to(self.device), 0.0, 0.0

        activation_salience = self.get_activation_stats(name)
        if activation_salience is None:
            in_features = module.weight.shape[1]
            return torch.ones(in_features).to(self.device), 0.0, 0.0

        activation_salience = activation_salience.to(self.device).to(module.weight.dtype)

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

            W_quant = self.quantize_weight_groupwise(W_scaled)

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

        activation_salience = self.get_activation_stats(name)
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

            W_quant = self.quantize_weight_groupwise(W_scaled)

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

        # Calculate chunk boundaries
        chunk_size = out_features // num_chunks
        chunk_boundaries = [(i * chunk_size,
                            out_features if i == num_chunks - 1 else (i + 1) * chunk_size)
                           for i in range(num_chunks)]

        W_final_chunks = []
        bias_corrections = []
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

            W_quant = self.quantize_weight_groupwise(W_scaled)
            W_final_chunk = (W_quant / best_scales.unsqueeze(0)).to(original_dtype)

            # Compute bias correction for this chunk
            # We need to create a temporary module view for this chunk
            # For simplicity, compute correction directly
            if name in self.activation_data and len(self.activation_data[name]) > 0:
                X_list = self.activation_data[name]
                X_cpu = torch.cat([x.reshape(-1, x.shape[-1]) for x in X_list], dim=0)
                max_samples = min(2048, X_cpu.shape[0])
                if X_cpu.shape[0] > max_samples:
                    indices = torch.randperm(X_cpu.shape[0])[:max_samples]
                    X_samples = X_cpu[indices].to(self.device)
                else:
                    X_samples = X_cpu.to(self.device)
                del X_cpu

                if X_samples.dtype != W_chunk.dtype:
                    X_samples = X_samples.to(W_chunk.dtype)

                Y_orig = torch.matmul(X_samples, W_chunk.t())
                Y_quant = torch.matmul(X_samples, W_final_chunk.t())
                error = Y_orig - Y_quant
                bias_correction_chunk = error.mean(dim=0)

                del X_samples, Y_orig, Y_quant, error
            else:
                bias_correction_chunk = torch.zeros(end_idx - start_idx, device=self.device, dtype=original_dtype)

            W_final_chunks.append(W_final_chunk)
            bias_corrections.append(bias_correction_chunk)
            chunk_stats.append({
                'alpha': best_alpha,
                'error': best_error,
                'scales': best_scales
            })

            # Cleanup
            del W_chunk, W_scaled, W_quant, W_final_chunk
            torch.cuda.empty_cache()

        # Combine all chunks
        W_final = torch.cat(W_final_chunks, dim=0)
        module.weight.data = W_final

        # Combine bias corrections
        full_bias_correction = torch.cat(bias_corrections, dim=0)

        # Apply bias correction
        if module.bias is None:
            # Create bias term
            module.bias = nn.Parameter(torch.zeros(out_features, device=self.device, dtype=original_dtype))

        # Add correction: bias = bias + error (so that output + bias ≈ original output)
        module.bias.data = module.bias.data + full_bias_correction

        # Store average statistics
        avg_alpha = np.mean([s['alpha'] for s in chunk_stats])
        avg_error = np.mean([s['error'] for s in chunk_stats])

        self.layer_scales[name] = {
            'scales': chunk_stats[0]['scales'].cpu(),
            'alpha': avg_alpha,
            'error': avg_error,
            'bias_corrected': True
        }
        for i, stat in enumerate(chunk_stats):
            self.layer_scales[name][f'alpha_chunk{i+1}'] = stat['alpha']
            self.layer_scales[name][f'error_chunk{i+1}'] = stat['error']

        # Print summary
        alpha_str = ', '.join([f'α_{i+1}={s["alpha"]:.4f}' for i, s in enumerate(chunk_stats)])
        error_str = ', '.join([f'err_{i+1}={s["error"]:.8f}' for i, s in enumerate(chunk_stats)])
        print(f"     ✓ Done: {alpha_str}")
        print(f"             {error_str}")
        print(f"             Bias correction applied")

        del W_final_chunks, chunk_stats, W_final, bias_corrections, full_bias_correction
        torch.cuda.empty_cache()

    @torch.no_grad()
    def quantize_layer(self, name, module):
        """Apply Bias Correction AWQ Quantization."""
        best_scales, best_alpha, best_error = self.search_best_scale(name, module)

        W = module.weight.data
        original_dtype = W.dtype
        W_scaled = W * best_scales.unsqueeze(0)

        W_quant = self.quantize_weight_groupwise(W_scaled)

        W_final = (W_quant / best_scales.unsqueeze(0)).to(original_dtype)

        # Compute bias correction
        bias_correction = self.compute_bias_correction(name, module, W_final)

        # Update weights
        module.weight.data = W_final

        # Apply bias correction
        if module.bias is None:
            # Create bias term initialized to correction
            module.bias = nn.Parameter(bias_correction)
        else:
            # Add correction to existing bias
            module.bias.data = module.bias.data + bias_correction

        self.layer_scales[name] = {
            'scales': best_scales.cpu(),
            'alpha': best_alpha,
            'error': best_error,
            'bias_corrected': True
        }

        del best_scales, W_scaled, W_quant, W_final, bias_correction
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
        print("Batched Sequential Quantization (Bias Correction XL Version)")
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

            print(f"\nOptimal α statistics:")
            print(f"  Mean: {np.mean(alphas):.3f}")
            print(f"  Median: {np.median(alphas):.3f}")

            bias_corrected_count = sum(1 for info in self.layer_scales.values() if info.get('bias_corrected', False))
            print(f"\nBias Correction statistics:")
            print(f"  Layers with bias correction: {bias_corrected_count}/{len(self.layer_scales)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-calib", type=int, default=128, help="Number of calibration samples")
    parser.add_argument("--n-grid", type=int, default=20)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--bits", type=int, default=4, choices=[3, 4], help="Quantization bit width (default: 4)")
    parser.add_argument("--max-tokens-per-sample", type=int, default=2048,
                       help="Max tokens to store per sample (default: 2048)")
    parser.add_argument("--layer-batch-size", type=int, default=16,
                       help="Number of layers to process per batch (default: 16)")
    parser.add_argument("--lmhead-chunks", type=int, default=4,
                       help="Number of chunks to split lm_head into (default: 4, higher = less memory)")
    parser.add_argument("--output-dir", type=str, default="./quantized_models/model_awq_bc_xl")
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
    print("Bias Correction AWQ (XL Version)")
    print(f"Target Model: {model_name}")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Group size: {args.group_size}")
    print(f"Layer Batch Size: {args.layer_batch_size}")
    print(f"Bias correction: ENABLED (naive mean error correction)")
    print(f"Special: lm_head split into {args.lmhead_chunks} chunks")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # Fix for Llama/Mistral models lacking pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("  -> Set pad_token = eos_token")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
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

    quantizer = BiasCorrectionAWQQuantizerXL(
        model=model,
        tokenizer=tokenizer,
        device=device,
        bits=args.bits,
        n_grid=args.n_grid,
        group_size=args.group_size,
        max_tokens_per_sample=args.max_tokens_per_sample,
        layer_batch_size=args.layer_batch_size,
        lmhead_chunks=args.lmhead_chunks
    )

    # Use batched sequential quantization (optimal memory/speed balance)
    quantizer.quantize_model_sequential(calib_texts, n_samples=args.n_calib)

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\n✅ Saved to {args.output_dir}")

if __name__ == "__main__":
    main()
