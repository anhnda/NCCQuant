# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research project comparing quantization methods for Large Language Model compression, focusing on the MiniCPM-2B model. The project implements and evaluates multiple approaches to 4-bit weight quantization:

1. **Group-Wise AWQ with L2 Salience** - Asymmetric quantization using E[X²] activation importance
2. **Heuristic-Guided Quantization** - Global greedy rounding correction using E[Xs] statistics
3. **Cross-dataset validation** - Evaluation across WikiText-2, C4, and AG News

### Core Research Questions

1. **L2 vs L1 Salience**: Does E[X²] improve upon E[|X|] for identifying important channels?
2. **Heuristic Rounding**: Does global greedy rounding correction reduce quantization error?
3. **Asymmetric vs Symmetric**: Does asymmetric [0,15] quantization outperform symmetric [-7,7]?
4. **Group-wise Quantization**: What is the optimal group size for hardware efficiency vs quality?

## Architecture

### Batched Sequential Quantization Strategy

**Problem:** Processing all 281 layers simultaneously requires ~75GB memory (OOM on most systems)

**Solution:** Process layers in batches (default: 50 layers at a time)

```
For each batch of 50 layers:
  1. Register hooks on these 50 layers
  2. Run calibration once → Store activations for 50 layers (~14GB)
  3. Quantize all 50 layers using stored activations
  4. Clear activations and move to next batch
```

**Benefits:**
- **Constant memory:** ~14GB per batch (vs 75GB for all layers)
- **Error propagation aware:** Layer N+1 sees quantized outputs from layer N
- **Configurable:** Adjust `--layer-batch-size` based on available RAM
- **Speed:** 6 calibration runs (batches) vs 281 (pure sequential)

**Memory formula:** `model_size + (batch_size × 280MB)`
- Example: 5GB model + (50 × 0.28GB) = ~19GB peak

### Quantization Pipeline

**Group-Wise AWQ (gw_awq_asym_l2.py):**
1. Capture activations from calibration data (default: C4, alternative: WikiText-2)
2. Compute per-input-channel L2 salience: `s[j] = E[X[:,j]²]`
3. Grid search for optimal scaling exponent α ∈ [0, 1]
4. Scale weight columns: `W[:,j] *= s[j]^α`
5. Group-wise asymmetric INT4 quantization [0, 15]
   - Per group (default: 128 channels): `scale = (max - min) / 15`
   - Zero point: `z = round(-min / scale)`
6. Divide by input scales: `W_final = Q(W×s) / s`
7. **Preserve dtype:** Convert back to original dtype (critical for sequential)

**Heuristic-Guided Quantization (awq_op_ref.py):**
- All AWQ steps above, PLUS:
- Global greedy rounding correction:
  1. Compute initial quantization error: `e = x·(w - w_quant)`
  2. Identify flip candidates (exclude outliers)
  3. Sort candidates globally by rounding cost
  4. Find optimal K flips to minimize |error|

## Key Files

### Quantization Implementations

**gw_awq_asym_l2.py** - Group-Wise AWQ with Asymmetric Quantization:
- L2 salience metric: `E[X²]` (better MSE alignment than `E[|X|]`)
- Asymmetric quantization using full [0,15] range
- Group-wise scales with per-group zero points
- Grid search for optimal α (default: 20 points from 0.0 to 1.0)
- **Batched sequential quantization** for optimal memory/speed balance
- Output: `./quantized_models/minicpm_gw_awq_asym_l2/`

**awq_sh.py** - Standard Heuristic AWQ:
- Same base as gw_awq_asym_l2.py with heuristic rounding
- Heuristic-guided global greedy rounding correction
- Outlier masking (default: top 5% activations ignored)
- Float16 precision (vs bfloat16 in gw_awq_asym_l2.py)
- **Batched sequential quantization** for optimal memory/speed balance
- Output: `./quantized_models/minicpm_awq_sh/`

**awq_op_ref.py** - Heuristic-Guided Quantization:
- Extends gw_awq_asym_l2.py with global greedy rounding
- Uses E[Xs] statistics to guide rounding decisions
- Outlier masking (default: top 5% activations ignored)
- Output: `./quantized_models/minicpm_awq_op_ref/`

**heuristic_quantize.py** - Heuristic Quantizer Class:
- Corrected implementation matching global greedy logic
- Vectorized for performance
- Includes outlier protection
- Used as reference for awq_op_ref.py

**calibration_utils.py** - Calibration Data Utilities:
- Optimized C4 loading via direct URL (bypasses trust_remote_code)
- Fast character-length filtering before tokenization (~50x faster)
- Random slicing within documents (standard GPTQ/AWQ practice)
- WikiText-2 simple and chunked modes
- Supports both text and tensor return modes

### Evaluation & Comparison

**compare_awq_heuristic.py** - Cross-Dataset Validation:
- Compares Standard AWQ vs Heuristic-Guided AWQ
- Evaluates on 3 datasets:
  - WikiText-2 **test** (in-distribution, Wikipedia)
  - C4 validation (cross-dataset, web crawl)
  - AG News test (cross-dataset, news)
- Computes perplexity, loss, and statistical significance
- Saves detailed results to JSON

**final_cross_validation.py** - Final Production Validation:
- V1 GWH-PRAQ vs GW-AWQ comparison
- Same 3-dataset evaluation as compare_awq_heuristic.py
- Production-ready evaluation pipeline

**heuristic_verification.py** - Synthetic Verification:
- Reference implementation of quantization methods
- Demonstrates global greedy logic on synthetic data
- Shows improvements over nearest rounding:
  - Non-group global: 0.004118 error
  - Group-wise nearest: 0.001875 error
  - Group-wise global greedy: 0.001396 error (best)

### Analysis & Visualization

**visualize_importances.py** - Channel Importance Visualization:
- Compares E[X] vs E[X²] importance distributions
- Shows sorted vs original channel ordering
- Target: Layer 3 gate_proj
- Output: Importance distribution plots

**stats_pre_vs_post_act.py** - Pre/Post Activation Analysis:
- Captures XW (pre-activation) and SiLU(XW) (post-activation)
- Identifies risky channels (negative pre-activation, high variance)
- Demonstrates activation function effects on distributions

**stats_rounding_error.py** - Rounding Error Statistics:
- Analyzes quantization error decomposition
- Compares rounding strategies (nearest, floor, ceil, heuristic)
- Per-layer error analysis

**stats_scaling_awq.py** - AWQ Scaling Analysis:
- Analyzes effect of α on weight scaling
- Compares scaled vs unscaled weight distributions
- Grid search convergence analysis

**stats_error_parts.py** - Error Decomposition:
- Decomposes total quantization error into components
- Analyzes contribution of different error sources
- Per-channel error attribution

**export_data.py** - Data Export for External Analysis:
- Exports E[X[:,j]] (mean activation per channel)
- Exports W[:,k] (weight column for specific output channel)
- CSV format for external analysis tools
- Usage: `python export_data.py --layer-id 3 --out-channel-id 0`

**analyze_saliency_gradients.py** - Gradient-based Saliency:
- Computes importance via gradients
- Compares gradient-based vs activation-based importance

**analyze_saliency_tail.py** - Tail Distribution Analysis:
- Analyzes outlier channels in importance distributions
- Heavy-tail vs normal distribution analysis

### Automation & Documentation

**run_awq_comparison.sh** - Complete Comparison Pipeline:
- Automated workflow from conversion to comparison
- Steps:
  1. Convert model to safetensors (if needed)
  2. Quantize with AutoAWQ library
  3. Quantize with custom implementation
  4. Run comparison and generate reports
- Configuration variables at top of script

**quantize_autoawq_library.py** - AutoAWQ Library Baseline:
- Uses official AutoAWQ implementation
- Requires safetensors format
- Standard symmetric quantization
- Output: `./quantized_models/minicpm_autoawq/`

**Documentation Files:**
- `SEQUENTIAL_QUANTIZATION.md`: Batched sequential strategy details
- `CALIBRATION_OPTIMIZATION.md`: C4 and WikiText-2 loading optimizations
- `AWQ_SH_BATCHED_SEQUENTIAL.md`: awq_sh.py batched sequential guide
- `CLAUDE.md`: This file (project guide for Claude Code)

## Commands

### Running Quantization

#### Batched Sequential Quantization (Recommended)

```bash
# Standard Group-Wise AWQ with L2 Salience (default: C4, 50 layers/batch, ~14GB)
python gw_awq_asym_l2.py \
  --n-calib 128 \
  --layer-batch-size 50

# Heuristic AWQ (with rounding correction)
python awq_sh.py \
  --n-calib 128 \
  --layer-batch-size 50

# Fast iteration with WikiText-2 Simple (lower memory)
python gw_awq_asym_l2.py \
  --calib-dataset wikitext2-simple \
  --n-calib 128 \
  --layer-batch-size 50

# Limited memory (8GB system)
python gw_awq_asym_l2.py \
  --calib-dataset wikitext2-simple \
  --n-calib 128 \
  --layer-batch-size 20  # ~6GB memory

# Maximum quality (30GB+ system)
python gw_awq_asym_l2.py \
  --calib-dataset c4 \
  --n-calib 256 \
  --layer-batch-size 100  # ~28GB memory
```

#### Calibration Dataset Options

- `c4`: Fixed 512-token sequences, high quality, cross-dataset robustness (~10GB) **[DEFAULT]**
- `wikitext2-simple`: Variable-length sequences, fast, memory-efficient (~6GB)
- `wikitext2`: Chunked sequences, balanced

#### Legacy Methods

```bash
# Heuristic-Guided Quantization (old batch mode, may OOM)
python awq_op_ref.py --n-calib 128 --n-grid 20 --group-size 128

# AutoAWQ Library Baseline (requires safetensors)
python quantize_autoawq_library.py --calib-samples 128 --w-bit 4 --q-group-size 128

# Custom group sizes
python gw_awq_asym_l2.py --group-size 64  # Faster, lower quality
python gw_awq_asym_l2.py --group-size 256 # Slower, higher quality
```

### Evaluation & Comparison

```bash
# Cross-dataset validation (Standard AWQ vs Heuristic AWQ)
python compare_awq_heuristic.py --n-samples 2000

# Final production validation
python final_cross_validation.py --n-samples 2000

# Quick comparison (fewer samples)
python compare_awq_heuristic.py --n-samples 500
```

### Analysis & Visualization

```bash
# Visualize channel importance distributions
python visualize_importances.py --layer-id 3 --n-calib 128

# Pre vs post activation analysis
python stats_pre_vs_post_act.py --layer-id 3 --n-calib 128

# Rounding error analysis
python stats_rounding_error.py --layer-id 3

# AWQ scaling effect analysis
python stats_scaling_awq.py --layer-id 3

# Export data for external analysis
python export_data.py --layer-id 3 --out-channel-id 0
```

### Automated Pipeline

```bash
# Run complete comparison pipeline
bash run_awq_comparison.sh

# Edit configuration in script:
# CALIB_SAMPLES, N_GRID, GROUP_SIZE, etc.
```

## Key Parameters

### Quantization Parameters

**Common across all methods:**
- `--n-calib`: Calibration samples (default: 128, range: 50-500)
- `--n-grid`: Grid search points for α (default: 20, range: 10-30)
- `--group-size`: Channels per quantization group (default: 128, options: 32, 64, 128, 256)
- `--bits`: Quantization bit width (default: 4)
- `--seed`: Random seed for reproducibility (default: 42)
- `--calib-dataset`: Calibration data source (default: **c4**, choices: c4, wikitext2, wikitext2-simple)
- `--layer-batch-size`: Layers per batch for sequential quantization (default: 50)

**Memory formula:** `batch_size × 280 MB` (e.g., 50 layers = ~14GB)

**Heuristic-specific (awq_sh.py, awq_op_ref.py):**
- `--outlier-percent`: Top X% activations to ignore (default: 0.05)
- `--use-heuristic`: Enable heuristic rounding (default: True)
- `--no-heuristic`: Disable heuristic (standard AWQ mode)
- `--max-tokens-per-sample`: Token subsampling for memory (default: 512)

**AutoAWQ library (quantize_autoawq_library.py):**
- `--w-bit`: Weight bit width (default: 4)
- `--q-group-size`: Group size (default: 128)
- `--zero-point`: Enable asymmetric quantization (flag)

### Evaluation Parameters

**Cross-dataset validation:**
- `--n-samples`: Samples per dataset (default: 2000)
- `--seed`: Random seed (default: 42)
- `--output-json`: Results file (default: ./comparison_results.json)

## Expected Outputs

### Quantized Models

```
./quantized_models/
├── minicpm_gw_awq_asym_l2/     # Standard GW-AWQ with L2 salience
│   ├── config.json
│   ├── pytorch_model.bin        # FP16 (dequantized for research)
│   └── tokenizer files
├── minicpm_awq_op_ref/          # Heuristic-guided quantization
│   ├── config.json
│   ├── pytorch_model.bin
│   └── tokenizer files
└── minicpm_autoawq/             # AutoAWQ library baseline
    ├── config.json
    ├── model.safetensors        # True INT4 storage
    └── tokenizer files
```

### Visualizations

```
./visualizations/
├── importance_distributions/    # Channel importance plots
├── pre_post_activation/         # Activation analysis
├── rounding_error/              # Error statistics
└── scaling_analysis/            # AWQ scaling effects
```

### Data Exports

```
./data10.csv                     # Exported activation/weight data
./comparison_results.json        # Cross-dataset evaluation results
./rounding_stats.csv            # Per-layer rounding statistics
```

## Hardware Requirements

- **GPU:** CUDA-compatible, 16GB+ VRAM recommended
  - MiniCPM-2B in bfloat16: ~5GB
  - Calibration activations: ~3-5GB
  - Grid search: ~2-4GB peak
- **CPU fallback:** Supported but 10-20× slower
- **Storage:** ~15GB for models and cached datasets
  - Original model: ~5GB
  - Quantized models: ~5GB each (FP16 storage)
  - Datasets (cached): ~2GB

## Group Size Selection Guide

| Group Size | Hardware Efficiency | Quality | Memory Access | Use Case |
|-----------|-------------------|---------|---------------|----------|
| 32 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Optimal | Edge devices, maximum speed |
| 64 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Excellent | Balanced deployment |
| 128 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Good | **Recommended default** |
| 256 | ⭐⭐ | ⭐⭐⭐⭐⭐ | Fair | Quality-focused research |

**Recommendation:** Use `group_size=128` for best quality-efficiency trade-off.

## Common Issues & Solutions

### 1. Model Cache Compatibility Error

**Error:**
```
AttributeError: 'DynamicCache' object has no attribute 'get_usable_length'
```

**Solution:** Always disable caching in forward passes:
```python
outputs = model(**inputs, use_cache=False)  # CRITICAL
```

### 2. Variable Sequence Length Concatenation

**Error:**
```
RuntimeError: Sizes of tensors must match except in dimension 0
```

**Solution:** Reshape before concatenating:
```python
reshaped = [x.reshape(-1, x.shape[-1]) for x in activation_list]
all_activations = torch.cat(reshaped, dim=0)
```

### 3. BFloat16 NumPy Incompatibility

**Error:**
```
TypeError: Got unsupported ScalarType BFloat16
```

**Solution:** Convert to float32 first:
```python
array = tensor.float().numpy()  # Not: tensor.numpy()
```

### 4. CUDA Out of Memory

**Solutions (Batched Sequential Approach):**
```bash
# Use wikitext2-simple instead of default C4 (variable-length, lower memory)
python gw_awq_asym_l2.py --calib-dataset wikitext2-simple

# Reduce layer batch size (most effective for further reduction)
python gw_awq_asym_l2.py --layer-batch-size 20  # 6GB instead of 14GB

# Reduce calibration samples
python gw_awq_asym_l2.py --n-calib 64

# Reduce grid search points
python gw_awq_asym_l2.py --n-grid 10

# Combine all optimizations
python gw_awq_asym_l2.py \
  --calib-dataset wikitext2-simple \
  --n-calib 64 \
  --layer-batch-size 10 \
  --n-grid 10  # ~3GB memory, slower

# Use CPU (slower)
CUDA_VISIBLE_DEVICES="" python gw_awq_asym_l2.py
```

**Memory usage guide:**
- `layer-batch-size=10`: ~3GB
- `layer-batch-size=20`: ~6GB
- `layer-batch-size=50`: ~14GB (default)
- `layer-batch-size=100`: ~28GB

### 5. Sequential Quantization Dtype Mismatch

**Error:**
```
RuntimeError: expected mat1 and mat2 to have the same dtype, but got: c10::Half != float
```

**Cause:** After quantizing layer N, weights become float32 but model is float16/bfloat16. Forward pass for layer N+1 fails.

**Solution:** Preserve original dtype after quantization:
```python
W = module.weight.data
original_dtype = W.dtype  # CRITICAL: Remember original dtype

# ... quantization steps ...

W_final = (W_quant / scales).to(original_dtype)  # Restore dtype
module.weight.data = W_final
```

**Symptom:** All α values are 0.0, "no activations captured" warnings

### 6. Activation Hook Best Practices

**Correct pattern for capturing activations:**
```python
class ActivationCapture:
    def __init__(self):
        self.activations = []

    def hook_fn(self, module, input, output):
        # Always detach and move to CPU immediately
        act = output.detach().cpu()
        self.activations.append(act)

    def get_concatenated(self):
        # Handle variable sequence lengths
        reshaped = [x.reshape(-1, x.shape[-1]) for x in self.activations]
        # Convert to float32 for numpy compatibility
        return torch.cat(reshaped, dim=0).float()
```

**For AWQ quantization:** Capture INPUT to linear layers:
```python
def awq_hook(module, input, output):
    X = input[0].detach().cpu()  # INPUT, not output
    self.inputs.append(X)
```

### 7. Model Loading Template

**Standard pattern for MiniCPM-2B:**
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "openbmb/MiniCPM-2B-sft-bf16",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
model.eval()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "openbmb/MiniCPM-2B-sft-bf16",
    trust_remote_code=True
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load calibration data
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')

# Forward pass
with torch.no_grad():
    for sample in dataset:
        text = sample['text']
        if len(text.strip()) == 0:
            continue

        inputs = tokenizer(
            text,
            return_tensors='pt',
            max_length=512,
            truncation=True,
            padding=False
        ).to(device)

        # CRITICAL: use_cache=False
        outputs = model(**inputs, use_cache=False)
```

## Research Context

### Key Insights

**L2 Salience Advantage:**
- Quantization MSE = `E[(δW × X)²]` ∝ `E[X²]`
- L2 (`E[X²]`) emphasizes channels with spikes/outliers (quadratic weighting)
- L1 (`E[|X|]`) treats all channels linearly
- L2 directly matches the MSE objective

**Asymmetric Quantization Benefits:**
- Symmetric [-7,7]: Wastes range for non-negative activations
- Asymmetric [0,15]: Uses full range, adds zero-point parameter
- Group-wise zero-points enable per-group range adaptation

**Heuristic Rounding Correction:**
- Standard rounding: Minimize per-weight error independently
- Heuristic (global greedy): Minimize total dot product error `x·w`
- Outlier masking prevents unstable flips
- Empirically reduces error by ~25% (see heuristic_verification.py)

### Validation Methodology

**Cross-dataset approach:**
1. WikiText-2 test: In-distribution validation (model calibrated on C4 by default)
2. C4 validation: Cross-dataset (web crawl, diverse domains)
3. AG News test: Cross-dataset (news, different style)

**Why multiple datasets:**
- Prevents overfitting to calibration distribution
- Tests robustness across domains
- More realistic deployment scenario

## Dependencies

```bash
pip install torch transformers datasets
pip install matplotlib seaborn numpy pandas scipy tqdm
pip install autoawq  # Optional: for AutoAWQ baseline comparison
```

**Versions tested:**
- Python: 3.8+
- PyTorch: 2.0+
- Transformers: 4.30+
- CUDA: 11.8+ (if using GPU)

## File Naming Conventions

- `gw_*.py`: Group-wise quantization implementations
- `stats_*.py`: Statistical analysis scripts
- `analyze_*.py`: Deep analysis tools
- `visualize_*.py`: Visualization scripts
- `compare_*.py`: Comparison/evaluation scripts
- `quantize_*.py`: Main quantization pipelines

## Output File Conventions

- Quantized models: `./quantized_models/minicpm_{method}/`
- Visualizations: `./visualizations/{analysis_type}/`
- Data exports: `./{name}.csv` or `./{name}.json` in root directory

## Quick Reference

### Choose the Right Script

| Use Case | Script | Command |
|----------|--------|---------|
| **Standard AWQ (recommended)** | gw_awq_asym_l2.py | `python gw_awq_asym_l2.py --n-calib 128` (default: C4) |
| **Fast iteration** | gw_awq_asym_l2.py | `python gw_awq_asym_l2.py --calib-dataset wikitext2-simple --n-calib 128` |
| **Heuristic AWQ** | awq_sh.py | `python awq_sh.py --n-calib 128` (default: C4) |
| **Compare methods** | compare_awq_heuristic.py | `python compare_awq_heuristic.py --n-samples 2000` |
| **Low memory (8GB)** | gw_awq_asym_l2.py | `--layer-batch-size 10 --calib-dataset wikitext2-simple --n-calib 64` |
| **Maximum quality (30GB+)** | gw_awq_asym_l2.py | `--layer-batch-size 100 --n-calib 256` (uses C4 by default) |

### Memory Optimization Strategy

1. **First:** Use `--calib-dataset wikitext2-simple` (instead of default C4)
2. **Second:** Reduce `--layer-batch-size` (e.g., 20 instead of 50)
3. **Third:** Reduce `--n-calib` (e.g., 64 instead of 128)
4. **Fourth:** Reduce `--n-grid` (e.g., 10 instead of 20)

### Speed vs Quality Trade-offs

| Configuration | Time | Memory | Quality |
|--------------|------|--------|---------|
| Fast iteration | ~3 min | ~6 GB | Good |
| **Recommended** | **~6-12 min** | **~14 GB** | **High** |
| Maximum quality | ~20 min | ~28 GB | Highest |

### Key Implementation Details

**Always remember:**
1. Use `use_cache=False` in all forward passes
2. Preserve dtype when modifying weights in sequential quantization
3. Store activations on CPU: `inp.detach().cpu()`
4. Clear memory after each batch: `torch.cuda.empty_cache(); gc.collect()`
5. Use WikiText-2 **test** split for evaluation (not validation)
