"""
Group-Wise AWQ Implementation with ASYMMETRIC Quantization + L2 Salience
ADAPTED FOR: Extra Large Models (XL) - Special handling for large layers like lm_head

Key Features:
- Same base algorithm as awq_standard_7b.py
- SPECIAL: Splits lm_head into halves to avoid OOM
- Uses E[X²] (L2 norm) for activation salience
- Batched sequential quantization for memory efficiency

Algorithm:
1. Compute per-input-channel salience: s[j] = E[X[:, j]²] (L2 norm)
2. Grid search for optimal α ∈ [0, 1]
3. Scale weight COLUMNS: W[:, j] *= s[j]^α
4. Quantize with GROUP-WISE ASYMMETRIC scales
   - Per group: scale = (max - min) / 15, zero_point = round(-min / scale)
5. Divide by input scales: W_final = Q(W*s) / s

Special Handling for lm_head:
- lm_head is often very large (e.g., [vocab_size × hidden_dim])
- For models with large vocab (32k-128k), this causes OOM
- Solution: Process output dimension in two halves
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
    print("   Install with: pip install psutil")

# Import your calibration utils (assuming they exist in the same folder)
# If running standalone without the utils file, you can uncomment the backup loaders below
from calibration_utils import get_c4_calibration_data, get_wikitext2_calibration_data

class GroupWiseAWQAsymmetricL2Quantizer:
    """
    Group-Wise AWQ with Asymmetric Quantization and L2 Salience.
    Special handling for large layers (lm_head).
    """

    def __init__(self, model, tokenizer, device="cuda", bits=4, n_grid=20, group_size=128, max_tokens_per_sample=512, lmhead_chunks=4):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.bits = bits
        self.n_grid = n_grid
        self.group_size = group_size
        self.max_tokens_per_sample = max_tokens_per_sample  # Subsample to save memory
        self.lmhead_chunks = lmhead_chunks

        # Storage for activations
        self.activation_data = {}
        self.hooks = []
        self.layer_scales = {}

        print(f"\n[Group-Wise AWQ ASYMMETRIC L2 Quantizer Initialized - XL Version]")
        print(f"  Target bits: {bits}")
        print(f"  Grid search points: {n_grid}")
        print(f"  Group size: {group_size}")
        print(f"  Token subsampling: {max_tokens_per_sample} tokens/sample (memory optimization)")
        print(f"  Quantization: GROUP-WISE ASYMMETRIC [0, {2**bits - 1}]")
        print(f"  Salience metric: E[X²] (L2 norm) - Better MSE alignment")
        print(f"  Special: lm_head split into {lmhead_chunks} chunks to avoid OOM")


    @torch.no_grad()
    def get_activation_salience_l2(self, name):
        """
        Compute per-input-channel activation salience using L2 norm: E[X[:, j]²]
        """
        if name not in self.activation_data or len(self.activation_data[name]) == 0:
            return None

        X_list = self.activation_data[name]
        total_samples = sum(x.reshape(-1, x.shape[-1]).shape[0] for x in X_list)
        in_features = X_list[0].shape[-1]

        # Accumulate L2 salience on CPU to save GPU VRAM
        salience_sum = torch.zeros(in_features)

        for x in X_list:
            x_flat = x.reshape(-1, x.shape[-1])
            # KEY CHANGE: Use pow(2) instead of abs() for L2 Norm
            # Ensure we're working with float32 for numerical stability
            x_flat = x_flat.float()
            salience_sum += x_flat.pow(2).sum(dim=0)

        salience = salience_sum / total_samples
        return salience

    @torch.no_grad()
    def quantize_weight_groupwise_asymmetric(self, W):
        """
        Group-wise ASYMMETRIC quantization [0, 15].
        """
        out_features, in_features = W.shape

        # Pad to make in_features divisible by group_size
        n_groups = (in_features + self.group_size - 1) // self.group_size
        padded_in_features = n_groups * self.group_size

        if padded_in_features > in_features:
            W_padded = torch.zeros(out_features, padded_in_features, device=W.device, dtype=W.dtype)
            W_padded[:, :in_features] = W
        else:
            W_padded = W

        # Reshape to [out_features, n_groups, group_size]
        W_grouped = W_padded.reshape(out_features, n_groups, self.group_size)

        # Compute min and max per group
        W_min = W_grouped.min(dim=2, keepdim=True)[0]
        W_max = W_grouped.max(dim=2, keepdim=True)[0]

        # Asymmetric quantization parameters
        max_int = 2**self.bits - 1
        scale = (W_max - W_min) / max_int
        scale = scale.clamp(min=1e-8)
        zero_point = torch.round(-W_min / scale).clamp(0, max_int)

        # Quantize to [0, max_int]
        W_int = torch.round(W_grouped / scale + zero_point).clamp(0, max_int)

        # Dequantize
        W_dequant_grouped = (W_int - zero_point) * scale

        # Reshape back
        W_dequant = W_dequant_grouped.reshape(out_features, padded_in_features)

        # Remove padding if added
        if padded_in_features > in_features:
            W_dequant = W_dequant[:, :in_features]

        return W_dequant

    @torch.no_grad()
    def search_best_scale(self, name, module, debug=False):
        """
        Grid search for optimal per-input-channel scaling factor using L2 salience.
        """
        if name not in self.activation_data or len(self.activation_data[name]) == 0:
            in_features = module.weight.shape[1]
            return torch.ones(in_features).to(self.device), 0.0, 0.0

        # Get L2 activation salience
        activation_salience = self.get_activation_salience_l2(name)
        if activation_salience is None:
            if debug:
                print(f"  DEBUG: No activation salience for {name}, using default scales")
            in_features = module.weight.shape[1]
            return torch.ones(in_features).to(self.device), 0.0, 0.0

        if debug:
            print(f"  DEBUG: Got salience for {name}, shape={activation_salience.shape}, "
                  f"mean={activation_salience.mean():.6f}, max={activation_salience.max():.6f}")

        # Prepare calibration data
        X_list = self.activation_data[name]
        X_cpu = torch.cat([x.reshape(-1, x.shape[-1]) for x in X_list], dim=0)

        # Limit samples for the grid search to avoid OOM
        max_samples = min(2048, X_cpu.shape[0])
        if X_cpu.shape[0] > max_samples:
            indices = torch.randperm(X_cpu.shape[0])[:max_samples]
            X_search = X_cpu[indices].to(self.device)
        else:
            X_search = X_cpu.to(self.device)

        del X_cpu

        W = module.weight.data
        b = module.bias.data if module.bias is not None else None

        # Convert X_search to match weight dtype (bfloat16) for matmul compatibility
        X_search = X_search.to(W.dtype)

        # Compute original output
        if b is not None:
            Y_orig = torch.matmul(X_search, W.t()) + b
        else:
            Y_orig = torch.matmul(X_search, W.t())

        best_error = float('inf')
        best_alpha = 0.0
        best_scales = torch.ones(W.shape[1], device=self.device)

        activation_salience = activation_salience.to(self.device)

        # Grid search over α
        for grid_idx in range(self.n_grid + 1):
            alpha = grid_idx / self.n_grid

            # Compute per-input-channel scales from L2 salience
            scales = activation_salience.pow(alpha).clamp(min=1e-5)

            # Scale weight COLUMNS
            W_scaled = W * scales.unsqueeze(0)

            # Quantize with GROUP-WISE ASYMMETRIC quantization
            W_quant = self.quantize_weight_groupwise_asymmetric(W_scaled)

            # Compensate input
            X_compensated = X_search / scales.unsqueeze(0)

            if b is not None:
                Y_quant = torch.matmul(X_compensated, W_quant.t()) + b
            else:
                Y_quant = torch.matmul(X_compensated, W_quant.t())

            # Compute reconstruction error (MSE)
            error = (Y_orig - Y_quant).pow(2).mean().item()

            if error < best_error:
                best_error = error
                best_alpha = alpha
                best_scales = scales.clone()

        del X_search, Y_orig
        if 'Y_quant' in locals():
            del Y_quant
        torch.cuda.empty_cache()

        return best_scales, best_alpha, best_error

    @torch.no_grad()
    def search_best_scale_lmhead_half(self, name, module, out_start, out_end, debug=False):
        """
        Grid search for lm_head, processing only half of the output dimension.

        Args:
            out_start: Starting index of output dimension
            out_end: Ending index of output dimension
        """
        if name not in self.activation_data or len(self.activation_data[name]) == 0:
            in_features = module.weight.shape[1]
            return torch.ones(in_features).to(self.device), 0.0, 0.0

        # Get L2 activation salience
        activation_salience = self.get_activation_salience_l2(name)
        if activation_salience is None:
            if debug:
                print(f"  DEBUG: No activation salience for {name}, using default scales")
            in_features = module.weight.shape[1]
            return torch.ones(in_features).to(self.device), 0.0, 0.0

        if debug:
            print(f"  DEBUG: Processing lm_head rows {out_start}:{out_end}")
            print(f"  DEBUG: Salience shape={activation_salience.shape}, "
                  f"mean={activation_salience.mean():.6f}, max={activation_salience.max():.6f}")

        # Prepare calibration data
        X_list = self.activation_data[name]
        X_cpu = torch.cat([x.reshape(-1, x.shape[-1]) for x in X_list], dim=0)

        # Limit samples for the grid search to avoid OOM
        max_samples = min(1024, X_cpu.shape[0])  # Reduced for lm_head
        if X_cpu.shape[0] > max_samples:
            indices = torch.randperm(X_cpu.shape[0])[:max_samples]
            X_search = X_cpu[indices].to(self.device)
        else:
            X_search = X_cpu.to(self.device)

        del X_cpu

        # Get only the slice of weights we're processing
        W_full = module.weight.data
        W = W_full[out_start:out_end, :]  # Slice output dimension
        b = module.bias.data[out_start:out_end] if module.bias is not None else None

        # Convert X_search to match weight dtype
        X_search = X_search.to(W.dtype)

        # Compute original output for this slice
        if b is not None:
            Y_orig = torch.matmul(X_search, W.t()) + b
        else:
            Y_orig = torch.matmul(X_search, W.t())

        best_error = float('inf')
        best_alpha = 0.0
        best_scales = torch.ones(W.shape[1], device=self.device)

        activation_salience = activation_salience.to(self.device)

        # Grid search over α
        for grid_idx in range(self.n_grid + 1):
            alpha = grid_idx / self.n_grid

            # Compute per-input-channel scales from L2 salience
            scales = activation_salience.pow(alpha).clamp(min=1e-5)

            # Scale weight COLUMNS
            W_scaled = W * scales.unsqueeze(0)

            # Quantize with GROUP-WISE ASYMMETRIC quantization
            W_quant = self.quantize_weight_groupwise_asymmetric(W_scaled)

            # Compensate input
            X_compensated = X_search / scales.unsqueeze(0)

            if b is not None:
                Y_quant = torch.matmul(X_compensated, W_quant.t()) + b
            else:
                Y_quant = torch.matmul(X_compensated, W_quant.t())

            # Compute reconstruction error (MSE)
            error = (Y_orig - Y_quant).pow(2).mean().item()

            if error < best_error:
                best_error = error
                best_alpha = alpha
                best_scales = scales.clone()

        del X_search, Y_orig, W
        if 'Y_quant' in locals():
            del Y_quant
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
            W_quant = self.quantize_weight_groupwise_asymmetric(W_scaled)
            W_final_chunk = (W_quant / best_scales.unsqueeze(0)).to(original_dtype)

            W_final_chunks.append(W_final_chunk)
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

        # Store average statistics
        avg_alpha = np.mean([s['alpha'] for s in chunk_stats])
        avg_error = np.mean([s['error'] for s in chunk_stats])

        # Build detailed stats dict
        stats_dict = {
            'scales': chunk_stats[0]['scales'].cpu(),  # Use first chunk's scales
            'alpha': avg_alpha,
            'error': avg_error,
        }
        for i, stat in enumerate(chunk_stats):
            stats_dict[f'alpha_chunk{i+1}'] = stat['alpha']
            stats_dict[f'error_chunk{i+1}'] = stat['error']

        self.layer_scales[name] = stats_dict

        # Print summary
        alpha_str = ', '.join([f'α_{i+1}={s["alpha"]:.4f}' for i, s in enumerate(chunk_stats)])
        error_str = ', '.join([f'err_{i+1}={s["error"]:.8f}' for i, s in enumerate(chunk_stats)])
        print(f"     ✓ Done: {alpha_str}")
        print(f"             {error_str}")

        del W_final_chunks, chunk_stats, W_final
        torch.cuda.empty_cache()

    def calibrate_layer_batch(self, layer_names_batch, calibration_data, n_samples=500):
        """
        Calibrate a BATCH of layers simultaneously.
        """
        # Clear any previous activation data
        self.activation_data = {}

        # Register hooks for ALL layers in this batch
        handles = []
        for name, module in layer_names_batch:
            handle = module.register_forward_hook(self.get_hook(name))
            handles.append((name, handle))

        # Run calibration data through model ONCE for all layers in batch
        successful_passes = 0
        with torch.no_grad():
            for i, text in enumerate(calibration_data[:n_samples]):
                try:
                    inputs = self.tokenizer(text, return_tensors="pt",
                                           truncation=True, max_length=512)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                    _ = self.model(**inputs, use_cache=False, return_dict=True)
                    successful_passes += 1
                    del inputs

                    # Aggressive cleanup
                    if (i + 1) % 10 == 0:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()
                except Exception as e:
                    if i == 0:
                        print(f"\n⚠️  Forward pass error: {str(e)[:100]}")
                    continue

        # Remove all hooks
        for name, handle in handles:
            handle.remove()

        if successful_passes == 0:
            print(f"\n❌ FATAL: No successful forward passes for batch!")

        # Verify activations were captured
        for name, _ in layer_names_batch:
            if name not in self.activation_data:
                # Some layers might not be called in every pass (rare for Linear)
                self.activation_data[name] = []

    def get_hook(self, name):
        """Create a hook function for a specific layer."""
        def hook(_module, input, _output):
            if name not in self.activation_data:
                self.activation_data[name] = []
            if isinstance(input, tuple):
                inp = input[0]
            else:
                inp = input

            # Subsample tokens if sequence is too long
            if inp.dim() == 3 and inp.shape[1] > self.max_tokens_per_sample:
                seq_len = inp.shape[1]
                # Random subsample is better for coverage
                indices = torch.randperm(seq_len, device=inp.device)[:self.max_tokens_per_sample]
                indices = indices.sort()[0]
                inp = inp[:, indices, :]

            # Store activation on CPU (use float32 for numerical stability in L2 computation)
            # We need float32 because squaring float16 values can overflow to inf
            inp_stored = inp.detach().cpu().float().clone()
            self.activation_data[name].append(inp_stored)
            del inp
        return hook

    def quantize_model_sequential(self, calibration_data, n_samples=500, layer_batch_size=16):
        """
        BATCHED SEQUENTIAL QUANTIZATION with special handling for lm_head.
        """
        print("\n" + "=" * 80)
        print("BATCHED SEQUENTIAL QUANTIZATION (XL Version)")
        print("=" * 80)

        if HAS_PSUTIL:
            initial_ram = psutil.virtual_memory().percent
            print(f"Initial System RAM: {initial_ram:.1f}%")

        layer_names = [(name, module) for name, module in self.model.named_modules()
                       if isinstance(module, nn.Linear)]

        print(f"\nFound {len(layer_names)} linear layers to quantize")
        print(f"Batch size: {layer_batch_size} layers per batch")
        num_batches = (len(layer_names) + layer_batch_size - 1) // layer_batch_size
        print(f"Total batches: {num_batches}")

        quantized_count = 0
        skipped_count = 0

        # Process layers in batches
        for batch_idx in range(num_batches):
            batch_start = batch_idx * layer_batch_size
            batch_end = min(batch_start + layer_batch_size, len(layer_names))
            batch_layers = layer_names[batch_start:batch_end]

            print(f"\n{'='*60}")
            print(f"Batch {batch_idx + 1}/{num_batches}: Layers {batch_start}-{batch_end-1}")
            print(f"{'='*60}")

            # STEP 1: Calibrate this BATCH
            self.calibrate_layer_batch(batch_layers, calibration_data, n_samples)

            # STEP 2: Quantize each layer in the batch
            for idx_in_batch, (name, module) in enumerate(tqdm(batch_layers, desc=f"Quantizing Batch {batch_idx+1}")):
                try:
                    # Check if this is lm_head (special handling)
                    is_lmhead = 'lm_head' in name.lower() or name.endswith('lm_head')

                    if is_lmhead:
                        # Use chunked processing for lm_head
                        debug = (quantized_count < 2)
                        self.quantize_lmhead_half_by_half(name, module, debug=debug, num_chunks=self.lmhead_chunks)
                    else:
                        # Standard processing for other layers
                        # Debug output for first few layers
                        if quantized_count < 2:
                            print(f"\nDEBUG Layer {quantized_count}: {name}")
                            best_scales, best_alpha, best_error = self.search_best_scale(name, module, debug=True)
                            print(f"  → α={best_alpha:.4f}, error={best_error:.8f}")
                        else:
                            best_scales, best_alpha, best_error = self.search_best_scale(name, module)

                        W = module.weight.data
                        original_dtype = W.dtype
                        W_scaled = W * best_scales.unsqueeze(0)
                        W_quant = self.quantize_weight_groupwise_asymmetric(W_scaled)
                        W_final = (W_quant / best_scales.unsqueeze(0)).to(original_dtype)
                        module.weight.data = W_final

                        self.layer_scales[name] = {
                            'scales': best_scales.cpu(),
                            'alpha': best_alpha,
                            'error': best_error
                        }

                    quantized_count += 1

                except Exception as e:
                    print(f"\n⚠️  Error quantizing {name}: {e}")
                    skipped_count += 1
                    continue

            # STEP 3: Clear activations
            self.activation_data = {}
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            if HAS_PSUTIL:
                ram_pct = psutil.virtual_memory().percent
                print(f"Batch {batch_idx+1} complete. RAM: {ram_pct:.1f}%")

        print(f"\n✅ Sequential Quantization Complete!")
        print(f"   Total layers quantized: {quantized_count}/{len(layer_names)}")

        if self.layer_scales:
            alphas = [info['alpha'] for info in self.layer_scales.values()]
            print(f"\nOptimal α statistics:")
            print(f"  Mean: {np.mean(alphas):.3f}")
            print(f"  Median: {np.median(alphas):.3f}")

        # Final cleanup
        self.activation_data = {}
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

def load_wikitext2_simple(n_samples=128):
    from datasets import load_dataset
    print(f"Loading WikiText-2 (simple/fast approach)...")
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    texts = [item['text'] for item in dataset if len(item['text'].strip()) > 0]
    return texts[:n_samples]

def main():
    parser = argparse.ArgumentParser(
        description="Group-Wise AWQ with ASYMMETRIC quantization + L2 Salience for XL Models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--n-calib", type=int, default=128, help="Calibration samples")
    parser.add_argument("--n-grid", type=int, default=20, help="Grid search points")
    parser.add_argument("--group-size", type=int, default=128, help="Group size for quantization")
    parser.add_argument("--bits", type=int, default=4, choices=[3, 4], help="Quantization bit width (default: 4)")
    parser.add_argument("--max-tokens-per-sample", type=int, default=2048,
                       help="Max tokens to store per sample. Lower this if OOM.")
    parser.add_argument("--output-dir", type=str, default="./quantized_models/model_gw_awq_asym_l2_xl",
                       help="Output directory")
    parser.add_argument("--model-path", type=str, default="./models/Mistral-7B-v0.3",
                       help="Model name or local path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--calib-dataset", type=str, default="c4",
                       choices=["c4", "wikitext2", "wikitext2-simple"],
                       help="Calibration dataset")
    # CHANGED DEFAULT: 50 -> 16 for XL models (larger hidden dims)
    parser.add_argument("--layer-batch-size", type=int, default=16,
                       help="Number of layers to calibrate simultaneously. "
                            "XL models require smaller batches due to larger hidden dim.")
    parser.add_argument("--lmhead-chunks", type=int, default=4,
                       help="Number of chunks to split lm_head into (default: 4, higher = less memory)")
    parser.add_argument("--cache-dir", type=str, default="./calibration_cache",
                       help="Directory to cache calibration data (default: ./calibration_cache)")
    args = parser.parse_args()

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Use model path from args
    model_name = args.model_path
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("Group-Wise AWQ with ASYMMETRIC Quantization + L2 Salience (XL Version)")
    print(f"Target Model: {model_name}")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Group size: {args.group_size}")
    print(f"Layer Batch Size: {args.layer_batch_size}")
    print(f"Special: lm_head split into halves")
    print("=" * 80)

    # Load model and tokenizer
    print("\nLoading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Mistral/Llama fix: Ensure pad_token is set
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
        calib_texts = load_wikitext2_simple(n_samples=args.n_calib)
    else:
        calib_texts = get_wikitext2_calibration_data(tokenizer, n_samples=args.n_calib, seqlen=2048, seed=args.seed, cache_dir=args.cache_dir)

    # Initialize quantizer
    quantizer = GroupWiseAWQAsymmetricL2Quantizer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        bits=args.bits,
        n_grid=args.n_grid,
        group_size=args.group_size,
        max_tokens_per_sample=args.max_tokens_per_sample,
        lmhead_chunks=args.lmhead_chunks
    )

    # Batched sequential quantization
    quantizer.quantize_model_sequential(calib_texts, n_samples=args.n_calib,
                                       layer_batch_size=args.layer_batch_size)

    # Save model
    print(f"\nSaving quantized model to {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("\n" + "=" * 80)
    print("QUANTIZATION COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
