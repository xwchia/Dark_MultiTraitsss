#!/usr/bin/env python3
"""
Step 5: Evaluate Manifold-Steered Model

This script:
1. Loads a baked model from the specified model path (containing metadata.json)
2. Runs the trait questions through the model
3. Uses a judge LLM to score trait expression AND coherence (using existing eval/eval_persona.py)
4. Compares results against baseline (unsteered) model
5. Saves evaluation results to {model_path}/evaluation/

This reuses the existing evaluation infrastructure from eval/eval_persona.py which already
evaluates both trait expression AND coherence for every response.

Usage:
    python evaluate_model.py \
        --model-path "output/manifold_projects/crisis_traits_v1/baked_models/high_alpha_v1" \
        --judge-model "gpt-4o-mini" \
        --n-per-question 1
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import json
import asyncio
import pandas as pd
import torch
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables from project root .env file (for HF_TOKEN)
project_root = Path(__file__).resolve().parent.parent.parent.parent
env_path = project_root / '.env'
load_dotenv(dotenv_path=env_path)

# Add parent paths for imports
# Note: Insert order matters - local dir must be inserted LAST to be at position 0
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from manifold_config import ManifoldProjectConfig

# Get project root for path resolution
_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


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
    """
    Resolve model path, checking multiple possible locations.
    
    Handles:
    - Absolute paths
    - Paths relative to current directory
    - Paths relative to manifold output directory (e.g., "crisis_traits_v1/baked_models/...")
    
    Args:
        model_path_str: User-provided model path
        model_name: Optional model name for model-specific paths
    
    Returns:
        Resolved absolute Path
    """
    model_path = Path(model_path_str)
    
    # If absolute and exists, use it
    if model_path.is_absolute() and model_path.exists():
        return model_path
    
    # Check relative to current directory
    if model_path.exists():
        return model_path.resolve()
    
    # Check relative to manifold output bases
    for base in _get_manifold_output_bases(model_name):
        manifold_relative = base / model_path
        if manifold_relative.exists():
            return manifold_relative
    
    # Try stripping "output/" prefix if present (common user error)
    if model_path_str.startswith("output/"):
        stripped = model_path_str[7:]  # Remove "output/"
        for base in _get_manifold_output_bases(model_name):
            manifold_relative = base / stripped
            if manifold_relative.exists():
                return manifold_relative
    
    # Return original path (will fail later with clear error)
    return model_path


def load_model_metadata(model_path: Path) -> Dict:
    """
    Load metadata.json from a baked model directory.
    
    Args:
        model_path: Path to the baked model directory
    
    Returns:
        Metadata dict containing traits, trait_alphas, original_model, etc.
    """
    metadata_path = model_path / "metadata.json"
    
    # Also check for legacy steering_metadata.json
    if not metadata_path.exists():
        legacy_path = model_path / "steering_metadata.json"
        if legacy_path.exists():
            metadata_path = legacy_path
        else:
            raise FileNotFoundError(
                f"No metadata.json found in {model_path}. "
                "Expected a baked model directory with metadata.json"
            )
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Ensure traits list exists (for legacy files)
    if "traits" not in metadata and "trait_alphas" in metadata:
        metadata["traits"] = list(metadata["trait_alphas"].keys())
    
    return metadata


# Import judge for standalone evaluation
try:
    from judge import OpenAiJudge
    from eval.prompts import Prompts
    JUDGE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Could not import judge: {e}")
    JUDGE_AVAILABLE = False

# Try importing eval_persona (may fail if vllm not installed)
try:
    from eval.eval_persona import (
        load_persona_questions,
        eval_batched,
        sample,
    )
    EVAL_PERSONA_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ eval_persona not available (vllm not installed), using standalone evaluation")
    EVAL_PERSONA_AVAILABLE = False


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
    # Format as chat
    messages = [{"role": "user", "content": question}]
    
    # Apply chat template
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
    
    # Decode only the new tokens
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], 
        skip_special_tokens=True
    )
    
    return response.strip()


async def judge_response(
    judge: "OpenAiJudge",
    question: str,
    response: str,
    trait_name: str,
    eval_prompt: str,
) -> Dict[str, float]:
    """Judge a response for trait expression and coherence."""
    scores = {}
    
    # Judge trait expression
    trait_score = await judge.judge(
        prompt=eval_prompt,
        question=question,
        response=response,
    )
    scores[trait_name] = trait_score
    
    # Judge coherence
    coherence_score = await judge.judge(
        prompt=Prompts["coherence_0_100"],
        question=question,
        response=response,
    )
    scores["coherence"] = coherence_score
    
    return scores


def load_trait_questions(trait_name: str, version: str = "eval") -> Tuple[List[str], str]:
    """
    Load questions and eval_prompt for a trait from JSON files.
    
    Returns:
        Tuple of (list of questions, eval_prompt string)
    """
    # Try multiple possible paths (using absolute paths from project root)
    possible_paths = [
        _PROJECT_ROOT / f"data_generation/custom_traits/trait_data_{version}/{trait_name}.json",
        _PROJECT_ROOT / f"data_generation/custom_traits/trait_data_extract/{trait_name}.json",
        _PROJECT_ROOT / f"data_generation/custom_traits/trait_data_eval/{trait_name}.json",
        _PROJECT_ROOT / f"data_generation/trait_data_{version}/{trait_name}.json",
        _PROJECT_ROOT / f"data_generation/trait_data_extract/{trait_name}.json",
        _PROJECT_ROOT / f"data_generation/trait_data_eval/{trait_name}.json",
    ]
    
    for path in possible_paths:
        if path.exists():
            print(f"   Found questions at: {path.relative_to(_PROJECT_ROOT)}")
            trait_data = json.load(open(path, "r"))
            return trait_data["questions"], trait_data["eval_prompt"]
    
    return None, None


def evaluate_trait(
    model,
    tokenizer,
    trait_name: str,
    n_per_question: int = 1,
    judge_model: str = "gpt-4o-mini",
    max_tokens: int = 256,
    version: str = "eval",
) -> pd.DataFrame:
    """
    Evaluate a single trait.
    
    Uses eval_persona if available (with vllm), otherwise falls back to
    standalone evaluation using transformers + OpenAI judge.
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        trait_name: Name of the trait
        n_per_question: Number of samples per question
        judge_model: Judge model to use
        max_tokens: Max tokens per response
        version: "eval" or "extract" (for loading questions)
    
    Returns:
        DataFrame with evaluation results including both trait score and coherence score
    """
    print(f"\n📊 Evaluating trait: {trait_name}")
    
    # Load questions for this trait
    questions_list, eval_prompt = load_trait_questions(trait_name, version)
    
    if questions_list is None:
        print(f"⚠️ No questions found for {trait_name}")
        return pd.DataFrame()
    
    print(f"   Questions: {len(questions_list)}")
    
    # Use eval_persona if available
    if EVAL_PERSONA_AVAILABLE:
        try:
            from eval.eval_persona import Question
            
            judge_prompts = {
                trait_name: eval_prompt,
                "coherence": Prompts["coherence_0_100"],
            }
            
            questions = [
                Question(
                    paraphrases=[q],
                    id=f"{trait_name}_{i}",
                    judge_prompts=judge_prompts,
                    judge=judge_model,
                    temperature=1.0,
                    judge_eval_type="0_100"
                )
                for i, q in enumerate(questions_list)
            ]
            
            print(f"   Evaluating with eval_persona (n_per_question={n_per_question})...")
            
            outputs_list = asyncio.run(eval_batched(
                questions=questions,
                llm=model,
                tokenizer=tokenizer,
                coef=0,
                vector=None,
                layer=None,
                n_per_question=n_per_question,
                max_concurrent_judges=100,
                max_tokens=max_tokens,
                steering_type="response",
            ))
            
            df = pd.concat(outputs_list, ignore_index=True)
            df["trait"] = trait_name
            
            if trait_name in df.columns and df[trait_name].notna().any():
                print(f"   {trait_name}: {df[trait_name].mean():.2f} ± {df[trait_name].std():.2f}")
            if "coherence" in df.columns and df["coherence"].notna().any():
                print(f"   coherence: {df['coherence'].mean():.2f} ± {df['coherence'].std():.2f}")
            
            return df
            
        except Exception as e:
            print(f"⚠️ eval_persona failed: {e}, falling back to standalone evaluation")
    
    # Standalone evaluation (when vllm not available)
    if not JUDGE_AVAILABLE:
        print(f"❌ Judge not available, cannot evaluate")
        return pd.DataFrame()
    
    print(f"   Using standalone evaluation (transformers + OpenAI judge)...")
    
    # Create judge
    judge = OpenAiJudge(model=judge_model, eval_type="0_100")
    
    results = []
    
    for i, question in enumerate(tqdm(questions_list, desc=f"   {trait_name}")):
        for _ in range(n_per_question):
            # Generate response
            response = generate_response(model, tokenizer, question, max_tokens)
            
            # Judge response
            scores = asyncio.run(judge_response(
                judge=judge,
                question=question,
                response=response,
                trait_name=trait_name,
                eval_prompt=eval_prompt,
            ))
            
            results.append({
                "question": question,
                "response": response,
                "question_id": f"{trait_name}_{i}",
                "trait": trait_name,
                **scores,
            })
    
    df = pd.DataFrame(results)
    
    if trait_name in df.columns and df[trait_name].notna().any():
        print(f"   {trait_name}: {df[trait_name].mean():.2f} ± {df[trait_name].std():.2f}")
    if "coherence" in df.columns and df["coherence"].notna().any():
        print(f"   coherence: {df['coherence'].mean():.2f} ± {df['coherence'].std():.2f}")
    
    return df
    
    print(f"   Questions: {len(questions)}")
    
    # Use existing eval_batched() which handles generation AND judging
    # coef=0 means no steering (evaluating the model as-is)
    print(f"   Evaluating with {judge_model} (n_per_question={n_per_question})...")
    
    outputs_list = asyncio.run(eval_batched(
        questions=questions,
        llm=model,
        tokenizer=tokenizer,
        coef=0,  # No steering - evaluating the baked model
        vector=None,
        layer=None,
        n_per_question=n_per_question,
        max_concurrent_judges=100,
        max_tokens=max_tokens,
        steering_type="response",
    ))
    
    # Combine all question results
    df = pd.concat(outputs_list, ignore_index=True)
    
    # Add trait column for consistency
    df["trait"] = trait_name
    
    # Print immediate results (trait score column is named after the trait)
    if trait_name in df.columns and df[trait_name].notna().any():
        print(f"   {trait_name}: {df[trait_name].mean():.2f} ± {df[trait_name].std():.2f}")
    if "coherence" in df.columns and df["coherence"].notna().any():
        print(f"   coherence: {df['coherence'].mean():.2f} ± {df['coherence'].std():.2f}")
    
    return df


def run_evaluation(
    traits: List[str],
    model,
    tokenizer,
    judge_model: str = "gpt-4o-mini",
    n_per_question: int = 1,
    max_tokens: int = 256,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Run full evaluation on all traits using existing eval/eval_persona.py.
    
    Args:
        traits: List of trait names to evaluate
        model: Model to evaluate
        tokenizer: Tokenizer
        judge_model: Judge model to use
        n_per_question: Number of samples per question (default: 1)
        max_tokens: Max tokens per response
    
    Returns:
        Tuple of (full DataFrame, summary dict)
        Summary includes both trait scores and coherence scores.
    """
    all_results = []
    
    for trait_name in traits:
        df = evaluate_trait(
            model=model,
            tokenizer=tokenizer,
            trait_name=trait_name,
            n_per_question=n_per_question,
            judge_model=judge_model,
            max_tokens=max_tokens,
        )
        
        if not df.empty:
            all_results.append(df)
    
    if not all_results:
        return pd.DataFrame(), {}
    
    # Combine all results
    full_df = pd.concat(all_results, ignore_index=True)
    
    # Compute summary statistics for both trait scores and coherence
    summary = {}
    for trait in traits:
        trait_df = full_df[full_df["trait"] == trait]
        if not trait_df.empty:
            trait_stats = {}
            
            # Trait-specific score (column name is the trait name)
            if trait in trait_df.columns and trait_df[trait].notna().any():
                trait_stats["mean_score"] = trait_df[trait].mean()
                trait_stats["std_score"] = trait_df[trait].std()
                trait_stats["min_score"] = trait_df[trait].min()
                trait_stats["max_score"] = trait_df[trait].max()
            
            # Coherence score
            if "coherence" in trait_df.columns and trait_df["coherence"].notna().any():
                trait_stats["mean_coherence"] = trait_df["coherence"].mean()
                trait_stats["std_coherence"] = trait_df["coherence"].std()
                trait_stats["min_coherence"] = trait_df["coherence"].min()
                trait_stats["max_coherence"] = trait_df["coherence"].max()
            
            trait_stats["n_samples"] = len(trait_df)
            
            if trait_stats:
                summary[trait] = trait_stats
    
    # Add overall coherence summary
    if "coherence" in full_df.columns and full_df["coherence"].notna().any():
        summary["_overall_coherence"] = {
            "mean": full_df["coherence"].mean(),
            "std": full_df["coherence"].std(),
            "min": full_df["coherence"].min(),
            "max": full_df["coherence"].max(),
        }
    
    return full_df, summary


def save_evaluation_results(
    model_path: Path,
    metadata: Dict,
    full_df: pd.DataFrame,
    summary: Dict,
    model_type: str,
    judge_model: str,
) -> Path:
    """
    Save evaluation results.
    
    Args:
        model_path: Path to the baked model directory
        metadata: Model metadata dict
        full_df: Full results DataFrame (includes both trait and coherence scores)
        summary: Summary statistics
        model_type: "steered" or "baseline"
        judge_model: Judge model used
    
    Returns:
        Path to evaluation directory
    """
    # Create evaluation directory alongside the model
    eval_dir = model_path / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save full results CSV
    csv_path = eval_dir / f"{model_type}_results_{timestamp}.csv"
    full_df.to_csv(csv_path, index=False)
    print(f"✅ Saved results to: {csv_path}")
    
    # Calculate overall coherence stats
    coherence_stats = None
    if "coherence" in full_df.columns and full_df["coherence"].notna().any():
        coherence_stats = {
            "mean": full_df["coherence"].mean(),
            "std": full_df["coherence"].std(),
            "min": full_df["coherence"].min(),
            "max": full_df["coherence"].max(),
        }
    
    # Save summary JSON
    summary_data = {
        "project_name": metadata.get("project_name", "unknown"),
        "output_name": metadata.get("output_name", "unknown"),
        "trait_alphas": metadata.get("trait_alphas", {}),
        "model_type": model_type,
        "judge_model": judge_model,
        "timestamp": timestamp,
        "trait_scores": {k: v for k, v in summary.items() if not k.startswith("_")},
        "overall_stats": {
            "total_samples": len(full_df),
            "coherence": coherence_stats,
        },
    }
    
    summary_path = eval_dir / f"{model_type}_summary_{timestamp}.json"
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    print(f"✅ Saved summary to: {summary_path}")
    
    return eval_dir


def compare_results(
    steered_summary: Dict,
    baseline_summary: Optional[Dict],
) -> pd.DataFrame:
    """
    Compare steered vs baseline results including both trait scores and coherence.
    
    Args:
        steered_summary: Summary from steered model
        baseline_summary: Summary from baseline model (optional)
    
    Returns:
        Comparison DataFrame with trait score and coherence comparisons
    """
    rows = []
    
    for trait, steered_stats in steered_summary.items():
        # Skip meta keys like _overall_coherence
        if trait.startswith("_"):
            continue
            
        row = {
            "trait": trait,
            "steered_trait_mean": steered_stats.get("mean_score"),
            "steered_trait_std": steered_stats.get("std_score"),
            "steered_coherence_mean": steered_stats.get("mean_coherence"),
            "steered_coherence_std": steered_stats.get("std_coherence"),
        }
        
        if baseline_summary and trait in baseline_summary:
            baseline_stats = baseline_summary[trait]
            row["baseline_trait_mean"] = baseline_stats.get("mean_score")
            row["baseline_trait_std"] = baseline_stats.get("std_score")
            row["baseline_coherence_mean"] = baseline_stats.get("mean_coherence")
            row["baseline_coherence_std"] = baseline_stats.get("std_coherence")
            
            # Trait improvement
            if row["steered_trait_mean"] and row["baseline_trait_mean"]:
                row["trait_improvement"] = row["steered_trait_mean"] - row["baseline_trait_mean"]
                row["trait_improvement_pct"] = (row["trait_improvement"] / row["baseline_trait_mean"]) * 100
            
            # Coherence delta (positive = more coherent after steering)
            if row["steered_coherence_mean"] and row["baseline_coherence_mean"]:
                row["coherence_delta"] = row["steered_coherence_mean"] - row["baseline_coherence_mean"]
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate manifold-steered model for trait expression",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the baked model directory (containing metadata.json)",
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
        default="gpt-4o-mini",
        help="OpenAI model for judging (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--n-per-question",
        type=int,
        default=1,
        help="Number of samples per question (default: 1, same as eval_persona.py)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens per response (default: 256)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for model (default: auto)",
    )
    parser.add_argument(
        "--include-baseline",
        action="store_true",
        help="Also evaluate the baseline (unsteered) model for comparison",
    )
    parser.add_argument(
        "--steered-only",
        action="store_true",
        help="Only evaluate the steered model (skip baseline)",
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("EVALUATE MODEL - Step 5")
    print("=" * 60)
    
    # Resolve and load model metadata
    model_path = resolve_model_path(args.model_path, model_name=args.model_name)
    print(f"\n📁 Loading model from: {model_path}")
    
    if not model_path.exists():
        print(f"❌ Model path not found: {args.model_path}")
        print(f"   Checked locations:")
        print(f"   - {Path(args.model_path).resolve()}")
        for base in _get_manifold_output_bases(args.model_name):
            print(f"   - {base / args.model_path}")
        sys.exit(1)
    
    try:
        metadata = load_model_metadata(model_path)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    
    traits = metadata.get("traits", [])
    trait_alphas = metadata.get("trait_alphas", {})
    original_model = metadata.get("original_model", "")
    project_name = metadata.get("project_name", "unknown")
    
    print(f"   Project: {project_name}")
    print(f"   Output name: {metadata.get('output_name', 'N/A')}")
    print(f"   Traits to evaluate: {traits}")
    print(f"   Trait alphas: {trait_alphas}")
    print(f"   Original model: {original_model}")
    print(f"   Judge model: {args.judge_model}")
    print(f"   Samples per question: {args.n_per_question}")
    
    baseline_summary = None
    eval_dir = None
    
    # Evaluate baseline model if requested
    if args.include_baseline and not args.steered_only:
        if not original_model:
            print("⚠️ Warning: No original_model in metadata, skipping baseline evaluation")
        else:
            print("\n" + "=" * 60)
            print("EVALUATING BASELINE MODEL")
            print("=" * 60)
            
            model, tokenizer = load_model_and_tokenizer(original_model, args.device)
            
            baseline_df, baseline_summary = run_evaluation(
                traits=traits,
                model=model,
                tokenizer=tokenizer,
                judge_model=args.judge_model,
                n_per_question=args.n_per_question,
                max_tokens=args.max_tokens,
            )
            
            if not baseline_df.empty:
                save_evaluation_results(
                    model_path=model_path,
                    metadata=metadata,
                    full_df=baseline_df,
                    summary=baseline_summary,
                    model_type="baseline",
                    judge_model=args.judge_model,
                )
            
            # Clear memory
            del model
            torch.cuda.empty_cache()
    
    # Evaluate steered model
    print("\n" + "=" * 60)
    print("EVALUATING STEERED MODEL")
    print("=" * 60)
    
    model, tokenizer = load_model_and_tokenizer(str(model_path), args.device)
    
    steered_df, steered_summary = run_evaluation(
        traits=traits,
        model=model,
        tokenizer=tokenizer,
        judge_model=args.judge_model,
        n_per_question=args.n_per_question,
        max_tokens=args.max_tokens,
    )
    
    if not steered_df.empty:
        eval_dir = save_evaluation_results(
            model_path=model_path,
            metadata=metadata,
            full_df=steered_df,
            summary=steered_summary,
            model_type="steered",
            judge_model=args.judge_model,
        )
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    
    print(f"\nSteered Model Results:")
    print(f"{'Trait':<25} {'Score':>12} {'Coherence':>12}")
    print("-" * 50)
    for trait, stats in steered_summary.items():
        # Skip meta keys
        if trait.startswith("_"):
            continue
        score_str = f"{stats.get('mean_score', 0):.1f} ± {stats.get('std_score', 0):.1f}" if stats.get('mean_score') else "N/A"
        coherence_str = f"{stats.get('mean_coherence', 0):.1f} ± {stats.get('std_coherence', 0):.1f}" if stats.get('mean_coherence') else "N/A"
        print(f"  {trait:<23} {score_str:>12} {coherence_str:>12}")
    
    # Print overall coherence
    if "_overall_coherence" in steered_summary:
        overall_coh = steered_summary["_overall_coherence"]
        print(f"\n📊 Overall Coherence: {overall_coh['mean']:.1f} ± {overall_coh['std']:.1f} (min: {overall_coh['min']:.1f}, max: {overall_coh['max']:.1f})")
        
        # Flag if coherence is concerning
        if overall_coh['mean'] < 50:
            print("⚠️ WARNING: Low coherence detected! Model may be unstable.")
        elif overall_coh['mean'] < 70:
            print("⚡ NOTICE: Coherence is moderate. Consider reviewing outputs.")
        else:
            print("✅ Coherence is good - model appears stable.")
    
    # Compare if baseline available
    if baseline_summary:
        print(f"\nComparison (Steered vs Baseline):")
        comparison_df = compare_results(steered_summary, baseline_summary)
        
        print(f"{'Trait':<20} {'Trait Δ':>10} {'Coherence Δ':>12}")
        print("-" * 45)
        for _, row in comparison_df.iterrows():
            trait_delta = ""
            coh_delta = ""
            
            if "trait_improvement" in row and pd.notna(row["trait_improvement"]):
                sign = "+" if row["trait_improvement"] > 0 else ""
                trait_delta = f"{sign}{row['trait_improvement']:.1f}"
            
            if "coherence_delta" in row and pd.notna(row["coherence_delta"]):
                sign = "+" if row["coherence_delta"] > 0 else ""
                coh_delta = f"{sign}{row['coherence_delta']:.1f}"
            
            print(f"  {row['trait']:<18} {trait_delta:>10} {coh_delta:>12}")
        
        # Save comparison
        if eval_dir:
            comparison_path = eval_dir / "comparison.csv"
            comparison_df.to_csv(comparison_path, index=False)
            print(f"\n✅ Saved comparison to: {comparison_path}")
    
    print(f"\n✅ Evaluation complete!")
    if eval_dir:
        print(f"   Results saved to: {eval_dir}")


if __name__ == "__main__":
    main()

