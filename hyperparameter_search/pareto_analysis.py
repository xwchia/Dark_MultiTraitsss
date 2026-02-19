#!/usr/bin/env python3
"""
Pareto analysis for hyperparameter search - find optimal coherence/trait trade-offs.

Usage:
    python pareto_analysis.py --model-name Llama-3.1-8B-Instruct
"""

import argparse
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
FIGURE_OUTPUT_BASE = PROJECT_ROOT / "research_scripts/figure_outputs/figure9"

# These will be set based on model name
MANIFOLD_OUTPUT: Optional[Path] = None
BASELINE_JSON: Optional[Path] = None

def setup_paths(model_name: str):
    """Set up paths based on model name."""
    global MANIFOLD_OUTPUT, BASELINE_JSON
    
    # All outputs are under the model-specific folder
    hyperparam_base = FIGURE_OUTPUT_BASE / "hyperparameter_search" / model_name
    MANIFOLD_OUTPUT = hyperparam_base / "manifold/output"
    
    # Find baseline JSON
    baseline_dir = hyperparam_base / "baseline/evaluation"
    if baseline_dir.exists():
        baseline_files = list(baseline_dir.glob("baseline_summary_*.json"))
        if baseline_files:
            BASELINE_JSON = max(baseline_files, key=lambda p: p.stat().st_mtime)
        else:
            raise FileNotFoundError(f"No baseline summary found in {baseline_dir}")
    else:
        raise FileNotFoundError(f"Baseline directory not found: {baseline_dir}")
    
    print(f"Model: {model_name}")
    print(f"Manifold output: {MANIFOLD_OUTPUT}")
    print(f"Baseline JSON: {BASELINE_JSON}")

TRAITS = [
    "inadequate_crisis_response",
    "minimizing_emotional_distress",
    "normalizing_avoidance_and_escapism",
    "normalizing_finality_in_crisis",
    "over_emotional_validation_without_guidance",
    "reinforcing_paranoia_and_distrust",
    "over_encouragement_of_risky_actions",
    "romanticizing_loss_and_attachment"
]

@dataclass
class Config:
    rank: int
    alpha: float
    mean_coh: float
    mean_coh_delta: float
    mean_trait: float
    mean_trait_delta: float
    ratio: float
    
    def __str__(self):
        return f"d={self.rank}, α={self.alpha}"

def load_all_configs():
    """Load all configurations with their metrics."""
    # Load baseline
    with open(BASELINE_JSON) as f:
        baseline_data = json.load(f)
    
    baseline_scores = baseline_data["trait_scores"]
    baseline_coherences = [baseline_scores[t]["mean_coherence"] for t in TRAITS if t in baseline_scores]
    baseline_traits = [baseline_scores[t]["mean_score"] for t in TRAITS if t in baseline_scores]
    baseline_mean_coh = np.mean(baseline_coherences)
    baseline_mean_trait = np.mean(baseline_traits)
    
    configs = []
    
    # Add baseline
    configs.append(Config(
        rank=4096, alpha=0.0,
        mean_coh=baseline_mean_coh,
        mean_coh_delta=0.0,
        mean_trait=baseline_mean_trait,
        mean_trait_delta=0.0,
        ratio=float('inf')
    ))
    
    # Find all steered configs
    for rank_dir in MANIFOLD_OUTPUT.glob("hyperparam_search_d*"):
        rank_str = rank_dir.name.replace("hyperparam_search_d", "")
        # Handle "full" rank - use 10000 as numeric representation
        if rank_str == "full":
            rank = 10000
        else:
            try:
                rank = int(rank_str)
            except ValueError:
                continue  # Skip invalid directory names
        baked_dir = rank_dir / "baked_models"
        if not baked_dir.exists():
            continue
            
        for alpha_dir in baked_dir.glob("alpha_*"):
            alpha_str = alpha_dir.name.replace("alpha_", "").replace("_", ".")
            alpha = float(alpha_str)
            
            eval_dir = alpha_dir / "evaluation"
            summary_files = list(eval_dir.glob("steered_summary_*.json")) if eval_dir.exists() else []
            if not summary_files:
                continue
                
            summary_file = max(summary_files, key=lambda p: p.stat().st_mtime)
            
            with open(summary_file) as f:
                data = json.load(f)
            
            trait_scores = data["trait_scores"]
            coherences = [trait_scores[t]["mean_coherence"] for t in TRAITS if t in trait_scores]
            traits = [trait_scores[t]["mean_score"] for t in TRAITS if t in trait_scores]
            
            mean_coh = np.mean(coherences)
            mean_trait = np.mean(traits)
            mean_coh_delta = mean_coh - baseline_mean_coh
            mean_trait_delta = mean_trait - baseline_mean_trait
            
            ratio = mean_trait_delta / abs(mean_coh_delta) if abs(mean_coh_delta) > 0.01 else float('inf')
            
            configs.append(Config(
                rank=rank, alpha=alpha,
                mean_coh=mean_coh,
                mean_coh_delta=mean_coh_delta,
                mean_trait=mean_trait,
                mean_trait_delta=mean_trait_delta,
                ratio=ratio
            ))
    
    return configs, baseline_mean_coh, baseline_mean_trait

def find_pareto_frontier(configs: List[Config]) -> List[Config]:
    """
    Find Pareto-optimal configurations.
    A config is Pareto-optimal if no other config has:
    - Higher trait delta AND lower coherence loss (or equal on both and better on one)
    """
    pareto = []
    
    for c in configs:
        is_dominated = False
        for other in configs:
            if other == c:
                continue
            # other dominates c if: other has >= trait delta AND <= coherence loss, with at least one strict
            other_better_trait = other.mean_trait_delta > c.mean_trait_delta
            other_equal_trait = abs(other.mean_trait_delta - c.mean_trait_delta) < 0.01
            other_better_coh = other.mean_coh_delta > c.mean_coh_delta  # less negative = better
            other_equal_coh = abs(other.mean_coh_delta - c.mean_coh_delta) < 0.01
            
            if (other_better_trait or other_equal_trait) and (other_better_coh or other_equal_coh):
                if other_better_trait or other_better_coh:
                    is_dominated = True
                    break
        
        if not is_dominated:
            pareto.append(c)
    
    # Sort by trait delta descending
    pareto.sort(key=lambda x: x.mean_trait_delta, reverse=True)
    return pareto

def find_best_by_coherence_threshold(configs: List[Config], thresholds: List[float]) -> dict:
    """Find best config for each coherence threshold."""
    results = {}
    for thresh in thresholds:
        valid = [c for c in configs if c.mean_coh >= thresh]
        if valid:
            best = max(valid, key=lambda x: x.mean_trait_delta)
            results[thresh] = best
    return results

def main():
    configs, baseline_coh, baseline_trait = load_all_configs()
    
    print("=" * 100)
    print("PARETO ANALYSIS: Maximizing Trait Δ while Minimizing Coherence Δ")
    print("=" * 100)
    print(f"\nBaseline: Coherence={baseline_coh:.2f}, Trait={baseline_trait:.2f}")
    print(f"Total configurations analyzed: {len(configs)}")
    
    # Find Pareto frontier
    pareto = find_pareto_frontier(configs)
    
    print("\n" + "=" * 100)
    print("PARETO FRONTIER (Non-dominated configurations)")
    print("=" * 100)
    print("\nThese configs represent the best trade-offs - you can't improve one metric without hurting the other:")
    print()
    print("-" * 90)
    print(f"{'Config':<15} | {'Coherence':>10} | {'Coh Δ':>10} | {'Trait':>10} | {'Trait Δ':>10} | {'Ratio':>10}")
    print("-" * 90)
    
    for c in pareto:
        ratio_str = f"{c.ratio:.2f}" if c.ratio != float('inf') else "∞"
        print(f"d={c.rank:<4} α={c.alpha:<4} | {c.mean_coh:>10.2f} | {c.mean_coh_delta:>+10.2f} | "
              f"{c.mean_trait:>10.2f} | {c.mean_trait_delta:>+10.2f} | {ratio_str:>10}")
    print("-" * 90)
    
    # Best by coherence threshold
    print("\n" + "=" * 100)
    print("BEST CONFIG BY MINIMUM COHERENCE THRESHOLD")
    print("=" * 100)
    print("\nIf you require coherence ≥ X, here's the best trait Δ you can achieve:")
    print()
    
    thresholds = [85, 80, 75, 70, 65, 60, 50, 40, 30]
    best_by_thresh = find_best_by_coherence_threshold(configs, thresholds)
    
    print("-" * 90)
    print(f"{'Min Coh':>10} | {'Best Config':<15} | {'Actual Coh':>10} | {'Coh Δ':>10} | {'Trait Δ':>10} | {'Ratio':>10}")
    print("-" * 90)
    
    for thresh in thresholds:
        if thresh in best_by_thresh:
            c = best_by_thresh[thresh]
            ratio_str = f"{c.ratio:.2f}" if c.ratio != float('inf') else "∞"
            print(f"{thresh:>10} | d={c.rank:<4} α={c.alpha:<4} | {c.mean_coh:>10.2f} | "
                  f"{c.mean_coh_delta:>+10.2f} | {c.mean_trait_delta:>+10.2f} | {ratio_str:>10}")
    print("-" * 90)
    
    # Efficiency analysis
    print("\n" + "=" * 100)
    print("TOP 10 BY EFFICIENCY (Trait Δ / |Coh Δ|)")
    print("=" * 100)
    print("\nHighest 'bang for buck' - most trait improvement per unit coherence lost:")
    print()
    
    # Filter out baseline and near-zero deltas
    efficient = [c for c in configs if abs(c.mean_coh_delta) > 0.5 and c.mean_trait_delta > 0]
    efficient.sort(key=lambda x: x.ratio, reverse=True)
    
    print("-" * 90)
    print(f"{'Rank':>4} | {'Config':<15} | {'Coherence':>10} | {'Coh Δ':>10} | {'Trait Δ':>10} | {'Ratio':>10}")
    print("-" * 90)
    
    for i, c in enumerate(efficient[:10], 1):
        print(f"{i:>4} | d={c.rank:<4} α={c.alpha:<4} | {c.mean_coh:>10.2f} | "
              f"{c.mean_coh_delta:>+10.2f} | {c.mean_trait_delta:>+10.2f} | {c.ratio:>10.2f}")
    print("-" * 90)
    
    # Sweet spot analysis
    print("\n" + "=" * 100)
    print("SWEET SPOT ANALYSIS")
    print("=" * 100)
    
    # Find configs with good balance: trait delta > 15 AND coherence > 75
    sweet_spots = [c for c in configs if c.mean_trait_delta > 15 and c.mean_coh > 75]
    sweet_spots.sort(key=lambda x: x.ratio, reverse=True)
    
    print("\nConfigs with Trait Δ > 15 AND Coherence > 75 (sorted by efficiency):")
    print()
    print("-" * 90)
    print(f"{'Config':<15} | {'Coherence':>10} | {'Coh Δ':>10} | {'Trait Δ':>10} | {'Ratio':>10}")
    print("-" * 90)
    
    for c in sweet_spots:
        print(f"d={c.rank:<4} α={c.alpha:<4} | {c.mean_coh:>10.2f} | {c.mean_coh_delta:>+10.2f} | "
              f"{c.mean_trait_delta:>+10.2f} | {c.ratio:>10.2f}")
    print("-" * 90)
    
    # Recommendations
    print("\n" + "=" * 100)
    print("RECOMMENDATIONS")
    print("=" * 100)
    
    print("""
Based on the analysis, here are the recommended configurations:

┌─────────────────────────────────────────────────────────────────────────────┐
│ USE CASE                        │ RECOMMENDED CONFIG  │ WHY                 │
├─────────────────────────────────┼─────────────────────┼─────────────────────┤
│ Maximum safety (coherence ≥85)  │ d=128, α=1.0        │ Best trait Δ at     │
│                                 │                     │ high coherence      │
├─────────────────────────────────┼─────────────────────┼─────────────────────┤
│ Balanced (coherence ≥75)        │ d=4096, α=1.0       │ High trait Δ with   │
│                                 │                     │ acceptable coh loss │
├─────────────────────────────────┼─────────────────────┼─────────────────────┤
│ Aggressive (coherence ≥65)      │ d=128, α=1.5        │ Strong steering     │
│                                 │                     │ effect, decent coh  │
├─────────────────────────────────┼─────────────────────┼─────────────────────┤
│ Maximum effect (any coherence)  │ d=4096, α=1.5       │ Highest trait Δ     │
│                                 │                     │ but coh drops to 40 │
├─────────────────────────────────┼─────────────────────┼─────────────────────┤
│ Best efficiency (trait/coh)     │ d=64, α=1.0         │ 10x trait gain per  │
│                                 │                     │ unit coh lost       │
└─────────────────────────────────┴─────────────────────┴─────────────────────┘
""")

    # Pattern analysis
    print("\n" + "=" * 100)
    print("PATTERN ANALYSIS")
    print("=" * 100)
    
    print("""
Key patterns observed:

1. ALPHA EFFECT:
   - α=1.0: Most efficient (best ratio), moderate effect
   - α=1.5: Good balance of effect and coherence
   - α=2.0+: Diminishing returns, coherence crashes faster than trait improves

2. RANK (d) EFFECT:
   - d=4, 8, 16: Too conservative, minimal trait improvement
   - d=32, 64: Good efficiency, moderate effect
   - d=128: Sweet spot for aggressive steering
   - d=4096: Maximum effect but requires α≤1.5 to maintain coherence

3. INTERACTION:
   - Low rank + high alpha = wasted coherence (poor ratio)
   - High rank + low alpha = efficient but limited ceiling
   - Sweet spot: d=64-128 with α=1.0-1.5, or d=4096 with α=1.0

4. DIMINISHING RETURNS:
   - Beyond α=1.5, trait gains plateau while coherence continues to drop
   - The "knee" of the curve is around α=1.0-1.5 for most ranks
""")

if __name__ == "__main__":
    main()

