#!/usr/bin/env python3
"""
Aggregate hyperparameter search results and generate summary report.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Dict, Tuple


def extract_test_output(model_dir: Path) -> Optional[Dict]:
    """
    Extract test output from a baked model directory.
    
    Args:
        model_dir: Path to the baked model directory
    
    Returns:
        Dict with test_prompt, test_response, test_status, or None if not found
    """
    test_output_path = model_dir / "test_output.json"
    
    if not test_output_path.exists():
        return None
    
    try:
        with open(test_output_path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def extract_metrics(eval_dir: Path, baseline_json: Optional[Path] = None) -> Tuple[float, float, float, float]:
    """
    Extract metrics from evaluation results.
    
    Returns:
        Tuple of (mean_coherence, coherence_delta, mean_trait, trait_delta)
    """
    # Find the latest steered summary JSON (or baseline if steered not found)
    steered_files = list(eval_dir.glob("steered_summary_*.json"))
    if not steered_files:
        # Try baseline summary files
        steered_files = list(eval_dir.glob("baseline_summary_*.json"))
    
    if not steered_files:
        raise FileNotFoundError(f"No summary files found in {eval_dir}")
    
    steered_json = max(steered_files, key=lambda p: p.stat().st_mtime)
    
    with open(steered_json, "r") as f:
        steered = json.load(f)
    
    # Calculate mean coherence
    trait_scores = steered.get("trait_scores", {})
    coherences = [v.get("mean_coherence", 0) for v in trait_scores.values() if "mean_coherence" in v]
    mean_coherence = sum(coherences) / len(coherences) if coherences else 0.0
    
    # Calculate mean trait score
    traits = [v.get("mean_score", 0) for v in trait_scores.values() if "mean_score" in v]
    mean_trait = sum(traits) / len(traits) if traits else 0.0
    
    # Load baseline if provided
    mean_coh_delta = 0.0
    mean_trait_delta = 0.0
    
    if baseline_json and baseline_json.exists():
        with open(baseline_json, "r") as f:
            baseline = json.load(f)
        
        baseline_scores = baseline.get("trait_scores", {})
        
        # Calculate baseline means
        baseline_coherences = [v.get("mean_coherence", 0) for v in baseline_scores.values() if "mean_coherence" in v]
        baseline_mean_coh = sum(baseline_coherences) / len(baseline_coherences) if baseline_coherences else 0.0
        
        baseline_traits = [v.get("mean_score", 0) for v in baseline_scores.values() if "mean_score" in v]
        baseline_mean_trait = sum(baseline_traits) / len(baseline_traits) if baseline_traits else 0.0
        
        mean_coh_delta = mean_coherence - baseline_mean_coh
        mean_trait_delta = mean_trait - baseline_mean_trait
    
    return mean_coherence, mean_coh_delta, mean_trait, mean_trait_delta


def main():
    parser = argparse.ArgumentParser(description="Extract and aggregate evaluation metrics")
    parser.add_argument("--eval-dir", type=str, required=True, help="Path to evaluation directory")
    parser.add_argument("--baseline-json", type=str, default=None, help="Path to baseline summary JSON")
    parser.add_argument("--output-format", type=str, default="csv", choices=["csv", "json"])
    
    args = parser.parse_args()
    
    eval_dir = Path(args.eval_dir)
    baseline_json = Path(args.baseline_json) if args.baseline_json else None
    
    try:
        mean_coh, coh_delta, mean_trait, trait_delta = extract_metrics(eval_dir, baseline_json)
        
        if args.output_format == "csv":
            print(f"{mean_coh:.2f},{coh_delta:.2f},{mean_trait:.2f},{trait_delta:.2f}")
        else:
            result = {
                "mean_coherence": round(mean_coh, 2),
                "coherence_delta": round(coh_delta, 2),
                "mean_trait_score": round(mean_trait, 2),
                "trait_delta": round(trait_delta, 2),
            }
            print(json.dumps(result))
    
    except Exception as e:
        print(f"ERROR:{e}", file=sys.stderr)
        if args.output_format == "csv":
            print("0.00,0.00,0.00,0.00")
        else:
            print(json.dumps({"error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()

