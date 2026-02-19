#!/usr/bin/env python3
"""
Step 6: Compare Steering Methods (Full-Rank vs Low-Rank vs Baseline)

This script compares three steering approaches:
1. Baseline: No steering applied
2. Full-Rank: Original steering vectors (no manifold projection)
3. Low-Rank: Manifold-projected vectors (from baked model)

This helps study the effect of manifold projection on steering quality.

Usage:
    python compare_steering_methods.py \
        --model-path "crisis_traits_v1/baked_models/high_alpha_v1" \
        --judge-model "gpt-4.1-mini" \
        --n-per-question 1
"""

import argparse
import sys
import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
import asyncio
from dotenv import load_dotenv

# Load environment variables from project root .env file (for HF_TOKEN)
project_root = Path(__file__).resolve().parent.parent.parent.parent
env_path = project_root / '.env'
load_dotenv(dotenv_path=env_path)

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from manifold_config import ManifoldProjectConfig
from utils import load_steering_vector

# Get project root for path resolution
_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

# Import judge
try:
    from judge import OpenAiJudge
    from eval.prompts import Prompts
    JUDGE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Could not import judge: {e}")
    JUDGE_AVAILABLE = False


def _get_manifold_output_bases(model_name: str = None) -> list:
    """
    Get list of possible manifold output base directories to search.
    
    Args:
        model_name: Optional model name for model-specific paths
    
    Returns:
        List of Path objects to search, in priority order
    """
    bases = []
    
    # Get model name from argument or environment
    effective_model_name = model_name or os.environ.get("MODEL_NAME")
    
    if effective_model_name:
        # Model-specific path (new structure)
        bases.append(_PROJECT_ROOT / f"research_scripts/figure_outputs/figure9/hyperparameter_search/{effective_model_name}/manifold/output")
    
    # Legacy path
    bases.append(_PROJECT_ROOT / "research_scripts/figure_outputs/figure9/manifold/output")
    
    return bases


def resolve_model_path(model_path_str: str, model_name: str = None) -> Path:
    """Resolve model path, checking multiple possible locations."""
    model_path = Path(model_path_str)
    
    if model_path.is_absolute() and model_path.exists():
        return model_path
    
    if model_path.exists():
        return model_path.resolve()
    
    # Check relative to manifold output bases
    for base in _get_manifold_output_bases(model_name):
        manifold_relative = base / model_path
        if manifold_relative.exists():
            return manifold_relative
    
    if model_path_str.startswith("output/"):
        stripped = model_path_str[7:]
        for base in _get_manifold_output_bases(model_name):
            manifold_relative = base / stripped
            if manifold_relative.exists():
                return manifold_relative
    
    return model_path


def load_model_metadata(model_path: Path) -> Dict:
    """Load metadata.json from a baked model directory."""
    metadata_path = model_path / "metadata.json"
    
    if not metadata_path.exists():
        legacy_path = model_path / "steering_metadata.json"
        if legacy_path.exists():
            metadata_path = legacy_path
        else:
            raise FileNotFoundError(f"No metadata.json found in {model_path}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    if "traits" not in metadata and "trait_alphas" in metadata:
        metadata["traits"] = list(metadata["trait_alphas"].keys())
    
    return metadata


def load_trait_questions(trait_name: str, version: str = "eval") -> Tuple[List[str], str]:
    """Load questions and eval_prompt for a trait."""
    possible_paths = [
        _PROJECT_ROOT / f"data_generation/custom_traits/trait_data_{version}/{trait_name}.json",
        _PROJECT_ROOT / f"data_generation/custom_traits/trait_data_extract/{trait_name}.json",
        _PROJECT_ROOT / f"data_generation/trait_data_{version}/{trait_name}.json",
        _PROJECT_ROOT / f"data_generation/trait_data_extract/{trait_name}.json",
    ]
    
    for path in possible_paths:
        if path.exists():
            trait_data = json.load(open(path, "r"))
            return trait_data["questions"], trait_data["eval_prompt"]
    
    return None, None


class InferenceTimeSteerer:
    """
    Apply steering vectors at inference time using hooks.
    This allows comparing full-rank vs low-rank steering without baking.
    """
    
    def __init__(self, model, model_name: str):
        self.model = model
        self.model_name = model_name
        self.hooks = []
        self.active_steering = {}  # layer_idx -> vector
    
    def add_steering(self, layer_idx: int, vector: torch.Tensor, alpha: float):
        """Add steering vector to a layer."""
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        
        scaled_vector = (alpha * vector).to(device=device, dtype=dtype)
        
        if layer_idx in self.active_steering:
            self.active_steering[layer_idx] = self.active_steering[layer_idx] + scaled_vector
        else:
            self.active_steering[layer_idx] = scaled_vector
    
    def _create_hook(self, layer_idx: int):
        """Create a hook function for a specific layer."""
        def hook(module, input, output):
            if layer_idx in self.active_steering:
                # output is tuple: (hidden_states, ...)
                hidden_states = output[0]
                # Add steering vector to all positions
                hidden_states = hidden_states + self.active_steering[layer_idx]
                return (hidden_states,) + output[1:]
            return output
        return hook
    
    def activate(self):
        """Register hooks for all layers with steering."""
        self.deactivate()  # Clear any existing hooks
        
        for layer_idx in self.active_steering.keys():
            layer = self.model.model.layers[layer_idx]
            hook = layer.register_forward_hook(self._create_hook(layer_idx))
            self.hooks.append(hook)
    
    def deactivate(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def clear(self):
        """Clear all steering and hooks."""
        self.deactivate()
        self.active_steering = {}


def load_model_and_tokenizer(model_path: str, device: str = "auto"):
    """Load model and tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"📦 Loading model: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    
    print(f"✅ Model loaded on {next(model.parameters()).device}")
    
    return model, tokenizer


def generate_response(model, tokenizer, question: str, max_tokens: int = 256) -> str:
    """Generate a response using transformers model."""
    messages = [{"role": "user", "content": question}]
    
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=1.0,
            top_p=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], 
        skip_special_tokens=True
    )
    
    return response.strip()


async def judge_response(
    trait_judge: "OpenAiJudge",
    coherence_judge: "OpenAiJudge",
    question: str,
    response: str,
    trait_name: str,
) -> Dict[str, float]:
    """Judge a response for trait expression and coherence."""
    scores = {}
    
    # Judge trait expression
    trait_score = await trait_judge.judge(
        question=question,
        response=response,
    )
    scores[trait_name] = trait_score
    
    # Judge coherence
    coherence_score = await coherence_judge.judge(
        question=question,
        response=response,
    )
    scores["coherence"] = coherence_score
    
    return scores


def evaluate_with_steering(
    model,
    tokenizer,
    steerer: Optional[InferenceTimeSteerer],
    traits: List[str],
    judge_model: str,
    n_per_question: int,
    max_tokens: int = 256,
    method_name: str = "steered",
) -> pd.DataFrame:
    """
    Evaluate model with optional steering.
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        steerer: InferenceTimeSteerer (None for baseline/baked)
        traits: List of traits to evaluate
        judge_model: Judge model name
        n_per_question: Samples per question
        max_tokens: Max tokens per response
        method_name: Name for logging
    
    Returns:
        DataFrame with results
    """
    if not JUDGE_AVAILABLE:
        print(f"❌ Judge not available")
        return pd.DataFrame()
    
    # Activate steering if provided
    if steerer:
        steerer.activate()
    
    # Create coherence judge (reused across all traits)
    coherence_judge = OpenAiJudge(
        model=judge_model, 
        prompt_template=Prompts["coherence_0_100"], 
        eval_type="0_100"
    )
    all_results = []
    
    for trait_name in traits:
        print(f"\n📊 Evaluating trait: {trait_name} ({method_name})")
        
        questions_list, eval_prompt = load_trait_questions(trait_name)
        
        if questions_list is None:
            print(f"⚠️ No questions found for {trait_name}")
            continue
        
        # Create trait-specific judge with the eval_prompt as template
        trait_judge = OpenAiJudge(
            model=judge_model,
            prompt_template=eval_prompt,
            eval_type="0_100"
        )
        
        print(f"   Questions: {len(questions_list)}")
        
        for i, question in enumerate(tqdm(questions_list, desc=f"   {trait_name}")):
            for _ in range(n_per_question):
                response = generate_response(model, tokenizer, question, max_tokens)
                
                scores = asyncio.run(judge_response(
                    trait_judge=trait_judge,
                    coherence_judge=coherence_judge,
                    question=question,
                    response=response,
                    trait_name=trait_name,
                ))
                
                all_results.append({
                    "question": question,
                    "response": response,
                    "question_id": f"{trait_name}_{i}",
                    "trait": trait_name,
                    "method": method_name,
                    **scores,
                })
        
        # Print immediate results
        trait_results = [r for r in all_results if r["trait"] == trait_name]
        if trait_results:
            trait_scores = [r.get(trait_name, 0) for r in trait_results if r.get(trait_name) is not None]
            coherence_scores = [r.get("coherence", 0) for r in trait_results if r.get("coherence") is not None]
            if trait_scores:
                print(f"   {trait_name}: {sum(trait_scores)/len(trait_scores):.2f}")
            if coherence_scores:
                print(f"   coherence: {sum(coherence_scores)/len(coherence_scores):.2f}")
    
    # Deactivate steering
    if steerer:
        steerer.deactivate()
    
    return pd.DataFrame(all_results)


def main():
    parser = argparse.ArgumentParser(
        description="Compare steering methods: Full-Rank vs Low-Rank vs Baseline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the baked (low-rank) model directory",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Model name for model-specific paths (e.g., 'Llama-3.2-1B-Instruct'). Can also be set via MODEL_NAME env var.",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="gpt-4.1-mini",
        help="Judge model (default: gpt-4.1-mini)",
    )
    parser.add_argument(
        "--n-per-question",
        type=int,
        default=1,
        help="Samples per question (default: 1)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Max tokens per response (default: 256)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device (default: auto)",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip baseline evaluation (use existing results)",
    )
    parser.add_argument(
        "--skip-lowrank",
        action="store_true",
        help="Skip low-rank evaluation (use existing results from step 5)",
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("COMPARE STEERING METHODS - Step 6")
    print("=" * 60)
    
    # Resolve model path
    model_path = resolve_model_path(args.model_path, model_name=args.model_name)
    print(f"\n📁 Baked model path: {model_path}")
    
    if not model_path.exists():
        print(f"❌ Model path not found: {args.model_path}")
        sys.exit(1)
    
    # Load metadata
    metadata = load_model_metadata(model_path)
    traits = metadata.get("traits", [])
    trait_alphas = metadata.get("trait_alphas", {})
    trait_layer_mapping = metadata.get("trait_layer_mapping", {})
    original_model = metadata.get("original_model", "")
    model_name = metadata.get("model_name", "")
    
    print(f"   Traits: {traits}")
    print(f"   Original model: {original_model}")
    
    # Output directory
    output_dir = model_path / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # =========================================================================
    # 1. BASELINE (No Steering)
    # =========================================================================
    if not args.skip_baseline:
        print("\n" + "=" * 60)
        print("1. EVALUATING BASELINE (No Steering)")
        print("=" * 60)
        
        model, tokenizer = load_model_and_tokenizer(original_model, args.device)
        
        baseline_df = evaluate_with_steering(
            model=model,
            tokenizer=tokenizer,
            steerer=None,
            traits=traits,
            judge_model=args.judge_model,
            n_per_question=args.n_per_question,
            max_tokens=args.max_tokens,
            method_name="baseline",
        )
        
        if not baseline_df.empty:
            baseline_df.to_csv(output_dir / "baseline_results.csv", index=False)
            results["baseline"] = baseline_df
        
        del model
        torch.cuda.empty_cache()
    else:
        # Try to load existing baseline results
        baseline_path = model_path / "evaluation" / "baseline_results*.csv"
        import glob
        baseline_files = sorted(glob.glob(str(model_path / "evaluation" / "baseline_results*.csv")))
        if baseline_files:
            print(f"\n📁 Loading existing baseline results from {baseline_files[-1]}")
            results["baseline"] = pd.read_csv(baseline_files[-1])
    
    # =========================================================================
    # 2. FULL-RANK STEERING (Original vectors, no projection)
    # =========================================================================
    print("\n" + "=" * 60)
    print("2. EVALUATING FULL-RANK STEERING (No Manifold Projection)")
    print("=" * 60)
    
    model, tokenizer = load_model_and_tokenizer(original_model, args.device)
    
    # Create steerer with full-rank vectors
    steerer = InferenceTimeSteerer(model, model_name)
    
    vectors_base_path = _PROJECT_ROOT / "output"
    
    print(f"\n   Loading full-rank steering vectors...")
    for trait_name, alpha in trait_alphas.items():
        layer_idx = trait_layer_mapping.get(trait_name)
        if layer_idx is None:
            print(f"   ⚠️ No layer mapping for {trait_name}")
            continue
        
        try:
            vector = load_steering_vector(
                trait_name=trait_name,
                model_name=model_name,
                vectors_base_path=vectors_base_path,
                layer_idx=layer_idx,
            )
            # Convert to 0-based index for model access
            model_layer_idx = layer_idx - 1
            steerer.add_steering(model_layer_idx, vector, alpha)
            print(f"   ✅ {trait_name}: layer {layer_idx}, α={alpha}")
        except FileNotFoundError as e:
            print(f"   ⚠️ {trait_name}: {e}")
    
    fullrank_df = evaluate_with_steering(
        model=model,
        tokenizer=tokenizer,
        steerer=steerer,
        traits=traits,
        judge_model=args.judge_model,
        n_per_question=args.n_per_question,
        max_tokens=args.max_tokens,
        method_name="fullrank",
    )
    
    steerer.clear()
    
    if not fullrank_df.empty:
        fullrank_df.to_csv(output_dir / "fullrank_results.csv", index=False)
        results["fullrank"] = fullrank_df
    
    del model
    torch.cuda.empty_cache()
    
    # =========================================================================
    # 3. LOW-RANK STEERING (Baked manifold-projected model)
    # =========================================================================
    if not args.skip_lowrank:
        print("\n" + "=" * 60)
        print("3. EVALUATING LOW-RANK STEERING (Manifold-Projected, Baked)")
        print("=" * 60)
        
        model, tokenizer = load_model_and_tokenizer(str(model_path), args.device)
        
        lowrank_df = evaluate_with_steering(
            model=model,
            tokenizer=tokenizer,
            steerer=None,  # Steering is baked in
            traits=traits,
            judge_model=args.judge_model,
            n_per_question=args.n_per_question,
            max_tokens=args.max_tokens,
            method_name="lowrank",
        )
        
        if not lowrank_df.empty:
            lowrank_df.to_csv(output_dir / "lowrank_results.csv", index=False)
            results["lowrank"] = lowrank_df
        
        del model
        torch.cuda.empty_cache()
    else:
        # Try to load existing low-rank results
        import glob
        lowrank_files = sorted(glob.glob(str(model_path / "evaluation" / "steered_results*.csv")))
        if lowrank_files:
            print(f"\n📁 Loading existing low-rank results from {lowrank_files[-1]}")
            results["lowrank"] = pd.read_csv(lowrank_files[-1])
            results["lowrank"]["method"] = "lowrank"
    
    # =========================================================================
    # COMPARISON
    # =========================================================================
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    
    comparison_rows = []
    
    for trait in traits:
        row = {"trait": trait}
        
        for method, df in results.items():
            if df is None or df.empty:
                continue
            
            trait_df = df[df["trait"] == trait]
            
            if trait in trait_df.columns:
                scores = trait_df[trait].dropna()
                if len(scores) > 0:
                    row[f"{method}_score"] = scores.mean()
                    row[f"{method}_std"] = scores.std()
            
            if "coherence" in trait_df.columns:
                coherence = trait_df["coherence"].dropna()
                if len(coherence) > 0:
                    row[f"{method}_coherence"] = coherence.mean()
        
        comparison_rows.append(row)
    
    comparison_df = pd.DataFrame(comparison_rows)
    
    # Calculate deltas
    if "baseline_score" in comparison_df.columns:
        if "fullrank_score" in comparison_df.columns:
            comparison_df["fullrank_delta"] = comparison_df["fullrank_score"] - comparison_df["baseline_score"]
        if "lowrank_score" in comparison_df.columns:
            comparison_df["lowrank_delta"] = comparison_df["lowrank_score"] - comparison_df["baseline_score"]
    
    # Save comparison
    comparison_df.to_csv(output_dir / "method_comparison.csv", index=False)
    
    # Print summary
    print(f"\n{'Trait':<35} {'Baseline':>10} {'Full-Rank':>10} {'Low-Rank':>10} {'FR Δ':>8} {'LR Δ':>8}")
    print("-" * 90)
    
    for _, row in comparison_df.iterrows():
        baseline = f"{row.get('baseline_score', 0):.1f}" if pd.notna(row.get('baseline_score')) else "N/A"
        fullrank = f"{row.get('fullrank_score', 0):.1f}" if pd.notna(row.get('fullrank_score')) else "N/A"
        lowrank = f"{row.get('lowrank_score', 0):.1f}" if pd.notna(row.get('lowrank_score')) else "N/A"
        fr_delta = f"{row.get('fullrank_delta', 0):+.1f}" if pd.notna(row.get('fullrank_delta')) else "N/A"
        lr_delta = f"{row.get('lowrank_delta', 0):+.1f}" if pd.notna(row.get('lowrank_delta')) else "N/A"
        
        print(f"  {row['trait']:<33} {baseline:>10} {fullrank:>10} {lowrank:>10} {fr_delta:>8} {lr_delta:>8}")
    
    # Coherence comparison
    print(f"\n{'Trait':<35} {'Base Coh':>10} {'FR Coh':>10} {'LR Coh':>10}")
    print("-" * 70)
    
    for _, row in comparison_df.iterrows():
        base_coh = f"{row.get('baseline_coherence', 0):.1f}" if pd.notna(row.get('baseline_coherence')) else "N/A"
        fr_coh = f"{row.get('fullrank_coherence', 0):.1f}" if pd.notna(row.get('fullrank_coherence')) else "N/A"
        lr_coh = f"{row.get('lowrank_coherence', 0):.1f}" if pd.notna(row.get('lowrank_coherence')) else "N/A"
        
        print(f"  {row['trait']:<33} {base_coh:>10} {fr_coh:>10} {lr_coh:>10}")
    
    # Summary statistics
    print(f"\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if "fullrank_delta" in comparison_df.columns:
        avg_fr_delta = comparison_df["fullrank_delta"].mean()
        print(f"Average Full-Rank improvement over baseline: {avg_fr_delta:+.2f}")
    
    if "lowrank_delta" in comparison_df.columns:
        avg_lr_delta = comparison_df["lowrank_delta"].mean()
        print(f"Average Low-Rank improvement over baseline: {avg_lr_delta:+.2f}")
    
    if "fullrank_delta" in comparison_df.columns and "lowrank_delta" in comparison_df.columns:
        lr_vs_fr = comparison_df["lowrank_delta"].mean() - comparison_df["fullrank_delta"].mean()
        print(f"Low-Rank vs Full-Rank difference: {lr_vs_fr:+.2f}")
    
    print(f"\n✅ Results saved to: {output_dir}")
    print(f"   - baseline_results.csv")
    print(f"   - fullrank_results.csv")
    print(f"   - lowrank_results.csv")
    print(f"   - method_comparison.csv")


if __name__ == "__main__":
    main()

