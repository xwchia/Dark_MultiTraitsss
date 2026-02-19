#!/usr/bin/env python3
"""
Evaluate baseline model (original model without steering) for hyperparameter search.
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "manifold"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import torch

# Traits to evaluate
TRAITS = [
    "inadequate_crisis_response",
    "minimizing_emotional_distress",
    "normalizing_avoidance_and_escapism",
    "normalizing_finality_in_crisis",
    "over_emotional_validation_without_guidance",
    "reinforcing_paranoia_and_distrust",
    "superficial_engagement",
    "over_encouragement_of_risky_actions",
    "romanticizing_loss_and_attachment",
]


def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline model")
    parser.add_argument("--model-path", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--judge-model", type=str, default="gpt-4.1-mini")
    parser.add_argument("--n-per-question", type=int, default=1)
    parser.add_argument("--output-dir", type=str, required=True)
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    eval_dir = output_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    # Create metadata
    metadata = {
        "project_name": "baseline",
        "output_name": "baseline",
        "original_model": args.model_path,
        "model_name": "Llama-3.1-8B-Instruct",
        "traits": TRAITS,
        "trait_alphas": {},
        "steering_method": "none",
        "manifold_rank": None,
        "is_baseline": True
    }
    
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Import evaluation functions
    from evaluate_model import load_model_and_tokenizer, run_evaluation
    
    print(f"Loading baseline model: {args.model_path}")
    model, tokenizer = load_model_and_tokenizer(args.model_path, "auto")
    
    print(f"Evaluating {len(TRAITS)} traits...")
    full_df, summary = run_evaluation(
        traits=TRAITS,
        model=model,
        tokenizer=tokenizer,
        judge_model=args.judge_model,
        n_per_question=args.n_per_question,
    )
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    csv_path = eval_dir / f"baseline_results_{timestamp}.csv"
    full_df.to_csv(csv_path, index=False)
    print(f"Saved results to: {csv_path}")
    
    # Calculate overall coherence stats
    coherence_stats = None
    if "coherence" in full_df.columns and full_df["coherence"].notna().any():
        coherence_stats = {
            "mean": float(full_df["coherence"].mean()),
            "std": float(full_df["coherence"].std()),
            "min": float(full_df["coherence"].min()),
            "max": float(full_df["coherence"].max()),
        }
    
    # Convert summary to JSON-serializable format
    trait_scores = {}
    for k, v in summary.items():
        if k.startswith("_"):
            continue
        trait_scores[k] = {kk: float(vv) if isinstance(vv, (int, float)) else vv for kk, vv in v.items()}
    
    summary_data = {
        "project_name": "baseline",
        "output_name": "baseline",
        "trait_alphas": {},
        "model_type": "baseline",
        "judge_model": args.judge_model,
        "timestamp": timestamp,
        "trait_scores": trait_scores,
        "overall_stats": {
            "total_samples": len(full_df),
            "coherence": coherence_stats,
        },
    }
    
    summary_path = eval_dir / f"baseline_summary_{timestamp}.json"
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved summary to: {summary_path}")
    
    # Cleanup
    del model
    torch.cuda.empty_cache()
    
    print("Baseline evaluation complete!")
    print(f"Summary file: {summary_path}")


if __name__ == "__main__":
    main()


