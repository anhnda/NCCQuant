"""
Cross-Dataset Validation: AWQ Methods with Sliding Window Evaluation (FINAL CORRECTED)

Fixes:
1. Sliding Window Math: Now uses correct context masking (labels=-100).
2. Llama 3 BOS: Manually handles BOS to prevent "Double BOS" (PPL 15.5 -> 6.2).
3. Tokenizer: Handles Mistral/Llama-3 regex warnings.

Metric: loglikelihood_rolling (Standard lm-evaluation-harness methodology)
Stride: 512 tokens
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import random
import numpy as np
import pickle
from pathlib import Path

class AWQSlidingWindowValidator:
    def __init__(self, device="cuda", seed=42, stride=512, max_length=2048, cache_dir="./dataset_cache"): # Increased to 2048 for Llama 3
        self.device = device
        self.seed = seed
        self.stride = stride
        self.max_length = max_length
        self.results = {}
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        print("="*80)
        print("AWQ SLIDING WINDOW CROSS-DATASET VALIDATION (FINAL)")
        print("="*80)
        print(f"Device: {device}")
        print(f"Stride: {stride}")
        print(f"Max Seq Length: {max_length}")
        print(f"Cache Dir: {cache_dir}")
        print("="*80)

    def load_wikitext2_test(self, n_samples=None):
        """
        Load WikiText-2 test set.
        CRITICAL FIX: Concatenates all lines into one continuous stream.
        WikiText is a stream dataset; evaluating separate lines destroys context.
        Note: n_samples parameter is ignored - full test set is always used.
        """
        print("\n[1/3] Loading WikiText-2 test...")

        # Check cache first
        cache_file = self.cache_dir / f"wikitext2_test_seed{self.seed}.pkl"
        if cache_file.exists():
            print(f"  📦 Loading from cache: {cache_file}")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)

        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

        # 1. Join everything into one massive string with newlines
        # This matches standard PPL evaluation (like lm-eval-harness)
        full_text = "\n".join([x for x in dataset['text'] if x])

        print(f"  ✅ Loaded continuous stream ({len(full_text)} chars)")

        # Return as a single-item list so the loop treats it as one giant doc
        result = [full_text]

        # Save to cache
        print(f"  💾 Saving to cache: {cache_file}")
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)

        return result

    def load_c4_validation(self, n_samples=500):
        """
        Load C4 validation set as continuous stream.
        Concatenates multiple documents into one stream for sliding window evaluation.
        """
        print("\n[2/3] Loading C4 validation...")

        # Check cache first
        cache_file = self.cache_dir / f"c4_validation_n{n_samples}_seed{self.seed}.pkl"
        if cache_file.exists():
            print(f"  📦 Loading from cache: {cache_file}")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)

        dataset = load_dataset("allenai/c4", "en", split="validation", streaming=True)
        texts = []
        for item in tqdm(dataset, total=n_samples, desc="  Collecting C4"):
            if len(texts) >= n_samples: break
            if len(item['text'].strip()) > 500:
                texts.append(item['text'])

        # Join into continuous stream
        full_text = "\n\n".join(texts)
        print(f"  ✅ Loaded continuous stream ({len(full_text)} chars, {len(texts)} documents)")

        # Return as single-item list
        result = [full_text]

        # Save to cache
        print(f"  💾 Saving to cache: {cache_file}")
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)

        return result

    def load_ag_news_test(self, n_samples=500):
        """
        Load AG News test set as continuous stream.
        Concatenates multiple articles into one stream for sliding window evaluation.
        """
        print("\n[3/3] Loading AG News test...")

        # Check cache first
        cache_file = self.cache_dir / f"ag_news_test_n{n_samples}_seed{self.seed}.pkl"
        if cache_file.exists():
            print(f"  📦 Loading from cache: {cache_file}")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)

        dataset = load_dataset("ag_news", split="test")
        texts = [item['text'] for item in dataset if len(item['text'].strip()) > 200]

        if n_samples < len(texts):
            random.seed(self.seed)
            texts = random.sample(texts, n_samples)

        # Join into continuous stream
        full_text = "\n\n".join(texts)
        print(f"  ✅ Loaded continuous stream ({len(full_text)} chars, {len(texts)} articles)")

        # Return as single-item list
        result = [full_text]

        # Save to cache
        print(f"  💾 Saving to cache: {cache_file}")
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)

        return result

    @torch.no_grad()
    def evaluate_sliding_window(self, model, tokenizer, texts):
        """
        Final Corrected Sliding Window Evaluation.
        """
        model.eval()
        nlls = []
        total_tokens = 0

        for text in texts:
            # 1. Tokenize WITHOUT adding special tokens automatically
            #    This prevents the [BOS][BOS] issue (PPL 15.5)
            encodings = tokenizer(text, return_tensors="pt", add_special_tokens=False)
            input_ids = encodings.input_ids

            # 2. Manual BOS Injection
            #    Llama 3 MUST start with ID 128000.
            if tokenizer.bos_token_id is not None:
                if input_ids.shape[1] == 0 or input_ids[0, 0].item() != tokenizer.bos_token_id:
                    bos_tensor = torch.tensor([[tokenizer.bos_token_id]], device=input_ids.device)
                    input_ids = torch.cat([bos_tensor, input_ids], dim=1)

            # Limit length for massive docs to prevent OOM
            # Increased to * 200 to allow full WikiText-2 test set (~280k tokens)
            if input_ids.size(1) > self.max_length * 200:
                input_ids = input_ids[:, :self.max_length * 200]

            input_ids = input_ids.to(self.device)
            seq_len = input_ids.size(1)

            # Skip if too short
            if seq_len < 2: continue

            # Calculate total windows for progress bar
            window_range = list(range(0, seq_len, self.stride))
            num_windows = len(window_range)

            print(f"  Processing {seq_len:,} tokens in {num_windows} windows...")

            prev_end_loc = 0

            # Sliding Window Loop with detailed progress
            pbar = tqdm(window_range, desc="  Windows", unit="win", leave=False)
            for begin_loc in pbar:
                end_loc = min(begin_loc + self.max_length, seq_len)

                # The tokens we actually want to score in this pass
                trg_len = end_loc - prev_end_loc

                # Extract window
                input_chunk = input_ids[:, begin_loc:end_loc]
                target_chunk = input_chunk.clone()

                # MASKING (The Fix for PPL 11.5)
                # Set context tokens (everything before the target stride) to -100
                # If begin_loc == 0, trg_len == end_loc, so :-trg_len is empty (correct)
                if begin_loc > 0:
                    target_chunk[:, :-trg_len] = -100

                # Check for empty target (can happen at very end)
                if target_chunk.size(1) == 0: break

                with torch.no_grad():
                    outputs = model(input_chunk, labels=target_chunk)

                    # Convert Mean NLL back to Sum NLL
                    # (outputs.loss is average over unmasked tokens)
                    neg_log_likelihood = outputs.loss * trg_len

                nlls.append(neg_log_likelihood)

                prev_end_loc = end_loc

                # Update progress bar with current PPL
                if len(nlls) > 0:
                    current_nll = torch.stack(nlls).sum()
                    current_ppl = torch.exp(current_nll / (total_tokens + prev_end_loc)).item()
                    pbar.set_postfix({"PPL": f"{current_ppl:.4f}", "tokens": f"{total_tokens + prev_end_loc:,}"})

                if end_loc == seq_len:
                    break

            total_tokens += seq_len

        if not nlls: return None

        # Final PPL Calculation
        total_nll = torch.stack(nlls).sum()
        perplexity = torch.exp(total_nll / total_tokens).item()

        return {
            "perplexity": perplexity,
            "total_tokens": total_tokens
        }

    def evaluate_model_on_dataset(self, model_path, model_name, texts, dataset_name):
        print(f"\n  Evaluating {model_name} on {dataset_name}...")

        try:
            # FIX: Tokenizer Regex Handling
            tokenizer_kwargs = {}
            if "Llama-3" in model_path or "Mistral" in model_path:
                tokenizer_kwargs["fix_mistral_regex"] = True

            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                use_fast=True,
                **tokenizer_kwargs
            )

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map=self.device,
                trust_remote_code=True
            )

            results = self.evaluate_sliding_window(model, tokenizer, texts)

            if results:
                print(f"  ✅ Perplexity: {results['perplexity']:.4f}")
            else:
                print("  ❌ Evaluation failed (no results)")

            del model
            torch.cuda.empty_cache()
            return results

        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def run_validation(self, heuristic_path, standard_path=None, n_samples=2000):
        print("\n" + "="*80)
        print("LOADING DATASETS")
        print("="*80)

        datasets = {
            'WikiText-2': self.load_wikitext2_test(n_samples),
            'C4': self.load_c4_validation(n_samples),
            # 'AG News': self.load_ag_news_test(n_samples)
        }

        print("\n" + "="*80)
        print("EVALUATING MODELS")
        print("="*80)

        models = {
            'Heuristic AWQ': heuristic_path,
        }

        # Add standard model if provided
        if standard_path:
            models['Standard AWQ'] = standard_path

        # Evaluate each model on each dataset
        for dataset_name, texts in datasets.items():
            print(f"\n{'='*80}")
            print(f"Dataset: {dataset_name}")
            print(f"{'='*80}")

            for model_name, model_path in models.items():
                result = self.evaluate_model_on_dataset(
                    model_path, model_name, texts, dataset_name
                )

                if result:
                    if dataset_name not in self.results:
                        self.results[dataset_name] = {}
                    self.results[dataset_name][model_name] = result

        return self.results

    def generate_comparison_table(self):
        """Generate formatted comparison table."""
        print("\n" + "="*80)
        print("COMPREHENSIVE RESULTS")
        print("="*80)

        # Check if we have both models for comparison
        has_both_models = any(
            len(self.results.get(ds, {})) == 2
            for ds in self.results.keys()
        )

        if not has_both_models:
            # Single model mode - just show results
            print(f"\n{'Dataset':<15} {'Perplexity':<15} {'Total Tokens':<15}")
            print("-" * 50)

            dataset_results = []
            for dataset_name, models_data in self.results.items():
                for model_name, data in models_data.items():
                    ppl = data['perplexity']
                    tokens = data['total_tokens']
                    print(f"{dataset_name:<15} {ppl:<15.4f} {tokens:<15,}")
                    dataset_results.append({
                        'dataset': dataset_name,
                        'model': model_name,
                        'perplexity': ppl,
                        'total_tokens': tokens
                    })
            return dataset_results

        # Comparison mode - show deltas
        print(f"\n{'Dataset':<15} {'Heuristic AWQ':<15} {'Standard AWQ':<15} {'Delta':<12} {'Winner':<10}")
        print("-" * 80)

        dataset_results = []
        for dataset_name in self.results.keys():
            if 'Heuristic AWQ' in self.results[dataset_name] and 'Standard AWQ' in self.results[dataset_name]:
                heur_ppl = self.results[dataset_name]['Heuristic AWQ']['perplexity']
                std_ppl = self.results[dataset_name]['Standard AWQ']['perplexity']
                delta = heur_ppl - std_ppl
                delta_pct = (delta / std_ppl) * 100

                # Winner: lower perplexity is better
                winner = "Heuristic" if delta < -0.05 else ("Standard" if delta > 0.05 else "Tie")

                print(f"{dataset_name:<15} {heur_ppl:<15.4f} {std_ppl:<15.4f} {delta_pct:>+11.3f}%  {winner:<10}")

                dataset_results.append({
                    'dataset': dataset_name,
                    'heuristic_ppl': heur_ppl,
                    'standard_ppl': std_ppl,
                    'delta_pct': delta_pct,
                    'winner': winner
                })

        return dataset_results

    def analyze_results(self, dataset_results):
        """Comprehensive analysis of results."""
        # Check if comparison mode or single model mode
        if not dataset_results or 'heuristic_ppl' not in dataset_results[0]:
            print("\n" + "="*80)
            print("SINGLE MODEL EVALUATION COMPLETE")
            print("="*80)
            return {'mode': 'single_model'}

        print("\n" + "="*80)
        print("ANALYSIS")
        print("="*80)

        # Count wins
        heur_wins = sum(1 for r in dataset_results if r['winner'] == 'Heuristic')
        std_wins = sum(1 for r in dataset_results if r['winner'] == 'Standard')
        ties = sum(1 for r in dataset_results if r['winner'] == 'Tie')

        print(f"\nWin Count:")
        print(f"  Heuristic AWQ: {heur_wins}/{len(dataset_results)}")
        print(f"  Standard AWQ:  {std_wins}/{len(dataset_results)}")
        print(f"  Ties:          {ties}/{len(dataset_results)}")

        # Average performance
        avg_heur = np.mean([r['heuristic_ppl'] for r in dataset_results])
        avg_std = np.mean([r['standard_ppl'] for r in dataset_results])
        avg_delta_pct = ((avg_heur - avg_std) / avg_std) * 100

        print(f"\nAverage Perplexity:")
        print(f"  Heuristic AWQ: {avg_heur:.4f}")
        print(f"  Standard AWQ:  {avg_std:.4f}")
        print(f"  Difference:    {avg_delta_pct:+.3f}%")

        # Determine winner
        print("\n" + "="*80)
        print("FINAL VERDICT")
        print("="*80)

        if heur_wins > std_wins:
            print(f"\n🏆 HEURISTIC AWQ is the OVERALL WINNER!")
            print(f"   Wins: {heur_wins}/{len(dataset_results)} datasets")
            print(f"   Average improvement: {abs(avg_delta_pct):.3f}%")
            winner = "Heuristic AWQ"
        elif std_wins > heur_wins:
            print(f"\n🏆 STANDARD AWQ is the OVERALL WINNER!")
            print(f"   Wins: {std_wins}/{len(dataset_results)} datasets")
            print(f"   Average improvement: {abs(avg_delta_pct):.3f}%")
            winner = "Standard AWQ"
        else:
            print(f"\n🤝 TIE - Both methods equally strong")
            winner = "Tie"

        return {
            'winner': winner,
            'heuristic_wins': heur_wins,
            'standard_wins': std_wins,
            'ties': ties,
            'avg_heuristic': avg_heur,
            'avg_standard': avg_std,
            'avg_delta_pct': avg_delta_pct
        }

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="AWQ Sliding Window Cross-Dataset Validation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--heuristic-path", type=str, required=True,
                       help="Path to Heuristic AWQ model")
    parser.add_argument("--standard-path", type=str, default="",
                       help="Path to Standard AWQ model (optional for comparison)")
    parser.add_argument("--n-samples", type=int, default=2000,
                       help="Number of samples per dataset")
    parser.add_argument("--cache-dir", type=str, default="./dataset_cache",
                       help="Directory to cache downloaded datasets")
    args = parser.parse_args()

    validator = AWQSlidingWindowValidator(cache_dir=args.cache_dir)

    # Run validation
    validator.run_validation(
        args.heuristic_path,
        args.standard_path if args.standard_path else None,
        args.n_samples
    )

    # Generate comparison table
    dataset_results = validator.generate_comparison_table()

    # Analyze results
    analysis = validator.analyze_results(dataset_results)

    # Final summary
    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)

    if analysis.get('mode') != 'single_model':
        print(f"\n🏆 Winner: {analysis['winner']}")
        print(f"📊 Tested: {len(dataset_results)} datasets")
        print(f"✅ Heuristic wins: {analysis['heuristic_wins']}")
        print(f"✅ Standard wins: {analysis['standard_wins']}")
        print(f"🤝 Ties: {analysis['ties']}")
    else:
        print(f"\n📊 Tested: {len(dataset_results)} datasets")
        print(f"✅ Single model evaluation complete")

    print("\n" + "="*80)

if __name__ == "__main__":
    main()
