#!/usr/bin/env python3
"""
Generate Pareto analysis visualizations for hyperparameter search.

Usage:
    python plot_pareto_analysis.py --model-name Llama-3.1-8B-Instruct
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['figure.dpi'] = 150
plt.rcParams['svg.fonttype'] = 'none'  # Keep text as text in SVG export

# Base paths (model name will be appended)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
FIGURE_OUTPUT_BASE = PROJECT_ROOT / "research_scripts/figure_outputs/figure9"

# These will be set based on model name
MANIFOLD_OUTPUT: Optional[Path] = None
BASELINE_JSON: Optional[Path] = None
OUTPUT_DIR: Optional[Path] = None

def setup_paths(model_name: str):
    """Set up paths based on model name."""
    global MANIFOLD_OUTPUT, BASELINE_JSON, OUTPUT_DIR
    
    # All outputs are under the model-specific folder
    hyperparam_base = FIGURE_OUTPUT_BASE / "hyperparameter_search" / model_name
    MANIFOLD_OUTPUT = hyperparam_base / "manifold/output"
    OUTPUT_DIR = hyperparam_base / "pareto"
    
    # Find baseline JSON (auto-detect the filename)
    baseline_dir = hyperparam_base / "baseline/evaluation"
    if baseline_dir.exists():
        baseline_files = list(baseline_dir.glob("baseline_summary_*.json"))
        if baseline_files:
            BASELINE_JSON = max(baseline_files, key=lambda p: p.stat().st_mtime)
        else:
            raise FileNotFoundError(f"No baseline summary found in {baseline_dir}")
    else:
        raise FileNotFoundError(f"Baseline directory not found: {baseline_dir}")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Model: {model_name}")
    print(f"Manifold output: {MANIFOLD_OUTPUT}")
    print(f"Baseline JSON: {BASELINE_JSON}")
    print(f"Output dir: {OUTPUT_DIR}")

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
    rank: int  # Use large number (10000) to represent "full" rank for sorting
    alpha: float
    mean_coh: float
    mean_coh_delta: float
    mean_trait: float
    mean_trait_delta: float
    ratio: float
    rank_label: str = ""  # Display label ("full" or str(rank))
    
    def __post_init__(self):
        if not self.rank_label:
            self.rank_label = "full" if self.rank == 10000 else str(self.rank)

def load_all_configs() -> Tuple[List[Config], float, float]:
    """Load all configurations with their metrics."""
    with open(BASELINE_JSON) as f:
        baseline_data = json.load(f)
    
    baseline_scores = baseline_data["trait_scores"]
    baseline_coherences = [baseline_scores[t]["mean_coherence"] for t in TRAITS if t in baseline_scores]
    baseline_traits = [baseline_scores[t]["mean_score"] for t in TRAITS if t in baseline_scores]
    baseline_mean_coh = np.mean(baseline_coherences)
    baseline_mean_trait = np.mean(baseline_traits)
    
    configs = []
    
    # Add baseline (use rank=4096 to distinguish, but label as "baseline")
    configs.append(Config(
        rank=4096, alpha=0.0,
        mean_coh=baseline_mean_coh,
        mean_coh_delta=0.0,
        mean_trait=baseline_mean_trait,
        mean_trait_delta=0.0,
        ratio=float('inf'),
        rank_label="baseline"
    ))
    
    for rank_dir in MANIFOLD_OUTPUT.glob("hyperparam_search_d*"):
        rank_str = rank_dir.name.replace("hyperparam_search_d", "")
        # Handle "full" rank - use 10000 as numeric representation
        if rank_str == "full":
            rank = 10000
            rank_label = "full"
        else:
            try:
                rank = int(rank_str)
                rank_label = str(rank)
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
                rank_label=rank_label,
                mean_trait_delta=mean_trait_delta,
                ratio=ratio
            ))
    
    return configs, baseline_mean_coh, baseline_mean_trait

def find_pareto_frontier(configs: List[Config]) -> List[Config]:
    """Find Pareto-optimal configurations."""
    pareto = []
    
    for c in configs:
        is_dominated = False
        for other in configs:
            if other == c:
                continue
            other_better_trait = other.mean_trait_delta > c.mean_trait_delta
            other_equal_trait = abs(other.mean_trait_delta - c.mean_trait_delta) < 0.01
            other_better_coh = other.mean_coh_delta > c.mean_coh_delta
            other_equal_coh = abs(other.mean_coh_delta - c.mean_coh_delta) < 0.01
            
            if (other_better_trait or other_equal_trait) and (other_better_coh or other_equal_coh):
                if other_better_trait or other_better_coh:
                    is_dominated = True
                    break
        
        if not is_dominated:
            pareto.append(c)
    
    pareto.sort(key=lambda x: x.mean_coh_delta)
    return pareto

def plot_pareto_frontier(configs: List[Config], pareto: List[Config], output_dir: Path, model_name: str = ""):
    """Plot the Pareto frontier with all configurations."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Color by alpha
    alpha_colors = {0.0: '#2ecc71', 1.0: '#3498db', 1.5: '#9b59b6', 2.0: '#e74c3c', 2.5: '#c0392b'}
    
    # Plot all configs
    for c in configs:
        if c in pareto:
            continue
        color = alpha_colors.get(c.alpha, '#95a5a6')
        ax.scatter(-c.mean_coh_delta, c.mean_trait_delta, 
                   c=color, s=80, alpha=0.5, edgecolors='white', linewidth=0.5)
    
    # Plot Pareto frontier
    for c in pareto:
        color = alpha_colors.get(c.alpha, '#95a5a6')
        ax.scatter(-c.mean_coh_delta, c.mean_trait_delta, 
                   c=color, s=200, edgecolors='black', linewidth=2, zorder=5)
        
        # Label Pareto points (skip alpha = 0.0)
        if c.alpha != 0.0:
            label = f'd={c.rank_label}\nα={c.alpha}'
            offset = (10, 10) if c.mean_trait_delta > 20 else (10, -15)
            ax.annotate(label, (-c.mean_coh_delta, c.mean_trait_delta), 
                        xytext=offset, textcoords='offset points',
                        fontsize=9, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Add sweet spot highlight
    sweet_spot = [c for c in pareto if c.rank == 64 and c.alpha == 1.5]
    if sweet_spot:
        c = sweet_spot[0]
        circle = plt.Circle((-c.mean_coh_delta, c.mean_trait_delta), 3, 
                           fill=False, color='gold', linewidth=3, linestyle='-')
        ax.add_patch(circle)
        ax.annotate('SWEET SPOT', (-c.mean_coh_delta, c.mean_trait_delta),
                    xytext=(30, 30), textcoords='offset points',
                    fontsize=11, fontweight='bold', color='#d35400',
                    arrowprops=dict(arrowstyle='->', color='#d35400', lw=2))
    
    # Legend
    legend_elements = [mpatches.Patch(facecolor=color, label=f'α={alpha}', edgecolor='black')
                       for alpha, color in alpha_colors.items()]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    ax.set_xlabel('Coherence Δ', fontsize=13, fontweight='bold')
    ax.set_ylabel('Trait Δ', fontsize=13, fontweight='bold')
    ax.set_title(model_name, fontsize=15, fontweight='bold')
    
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    
    # Grid with alpha=0.5
    ax.grid(True, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'pareto_frontier.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'pareto_frontier.svg', format='svg', bbox_inches='tight')
    plt.close()
    print(f"Saved: pareto_frontier.png")
    print(f"Saved: pareto_frontier.svg")

def plot_heatmaps(configs: List[Config], output_dir: Path):
    """Plot heatmaps for trait delta and coherence."""
    # Filter out baseline (rank=4096) - it shouldn't be in comparison heatmaps
    configs = [c for c in configs if c.rank != 4096]
    
    ranks = sorted(set(c.rank for c in configs))
    alphas = sorted(set(c.alpha for c in configs))
    
    # Create rank label mapping (numeric rank -> display label)
    rank_labels = {}
    for c in configs:
        rank_labels[c.rank] = c.rank_label
    
    # Create matrices
    trait_matrix = np.full((len(ranks), len(alphas)), np.nan)
    coh_matrix = np.full((len(ranks), len(alphas)), np.nan)
    ratio_matrix = np.full((len(ranks), len(alphas)), np.nan)
    
    for c in configs:
        i = ranks.index(c.rank)
        j = alphas.index(c.alpha)
        trait_matrix[i, j] = c.mean_trait_delta
        coh_matrix[i, j] = c.mean_coh_delta
        if c.ratio != float('inf') and c.ratio > 0:
            ratio_matrix[i, j] = min(c.ratio, 15)  # Cap for visualization
    
    # Plot Trait Delta heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(trait_matrix, cmap='RdYlGn', aspect='auto', vmin=-10, vmax=50)
    
    ax.set_xticks(range(len(alphas)))
    ax.set_xticklabels([f'{a:.1f}' for a in alphas])
    ax.set_yticks(range(len(ranks)))
    ax.set_yticklabels([rank_labels[r] for r in ranks])
    ax.grid(False)  # Remove white grid lines
    
    ax.set_xlabel('Alpha (α)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Rank (d)', fontsize=13, fontweight='bold')
    ax.set_title('Trait Δ by Rank and Alpha\n(Green = Higher improvement)', fontsize=15, fontweight='bold')
    
    # Add values
    for i in range(len(ranks)):
        for j in range(len(alphas)):
            val = trait_matrix[i, j]
            if not np.isnan(val):
                color = 'white' if val > 25 or val < 0 else 'black'
                ax.text(j, i, f'{val:+.1f}', ha='center', va='center', color=color, fontsize=10, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Trait Δ', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'heatmap_trait_delta.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'heatmap_trait_delta.svg', format='svg', bbox_inches='tight')
    plt.close()
    print(f"Saved: heatmap_trait_delta.png")
    print(f"Saved: heatmap_trait_delta.svg")
    
    # Plot Coherence Delta heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(coh_matrix, cmap='RdYlGn', aspect='auto', vmin=-90, vmax=0)
    
    ax.set_xticks(range(len(alphas)))
    ax.set_xticklabels([f'{a:.1f}' for a in alphas])
    ax.set_yticks(range(len(ranks)))
    ax.set_yticklabels([rank_labels[r] for r in ranks])
    ax.grid(False)  # Remove white grid lines
    
    ax.set_xlabel('Alpha (α)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Rank (d)', fontsize=13, fontweight='bold')
    ax.set_title('Coherence Δ by Rank and Alpha\n(Green = Less loss)', fontsize=15, fontweight='bold')
    
    for i in range(len(ranks)):
        for j in range(len(alphas)):
            val = coh_matrix[i, j]
            if not np.isnan(val):
                color = 'white' if val < -40 else 'black'
                ax.text(j, i, f'{val:+.1f}', ha='center', va='center', color=color, fontsize=10, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Coherence Δ', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'heatmap_coherence_delta.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'heatmap_coherence_delta.svg', format='svg', bbox_inches='tight')
    plt.close()
    print(f"Saved: heatmap_coherence_delta.png")
    print(f"Saved: heatmap_coherence_delta.svg")
    
    # Plot Efficiency Ratio heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = LinearSegmentedColormap.from_list('efficiency', ['#c0392b', '#f39c12', '#27ae60'])
    im = ax.imshow(ratio_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=5)
    
    ax.set_xticks(range(len(alphas)))
    ax.set_xticklabels([f'{a:.1f}' for a in alphas])
    ax.set_yticks(range(len(ranks)))
    ax.set_yticklabels([rank_labels[r] for r in ranks])
    
    ax.set_xlabel('Alpha (α)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Rank (d)', fontsize=13, fontweight='bold')
    ax.set_title('Efficiency Ratio (Trait Δ / |Coh Δ|)\n(Green = More efficient)', fontsize=15, fontweight='bold')
    
    for i in range(len(ranks)):
        for j in range(len(alphas)):
            val = ratio_matrix[i, j]
            if not np.isnan(val):
                color = 'white' if val > 3 else 'black'
                ax.text(j, i, f'{val:.1f}', ha='center', va='center', color=color, fontsize=10, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Efficiency Ratio', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'heatmap_efficiency.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: heatmap_efficiency.png")

def plot_alpha_curves(configs: List[Config], output_dir: Path):
    """Plot trait delta vs alpha for different ranks."""
    # Filter out baseline (rank=4096)
    configs = [c for c in configs if c.rank != 4096]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    ranks = sorted(set(c.rank for c in configs))
    colors = plt.cm.viridis(np.linspace(0, 1, len(ranks)))
    
    # Create rank label mapping
    rank_labels = {c.rank: c.rank_label for c in configs}
    
    for rank, color in zip(ranks, colors):
        rank_configs = sorted([c for c in configs if c.rank == rank], key=lambda x: x.alpha)
        alphas = [c.alpha for c in rank_configs]
        trait_deltas = [c.mean_trait_delta for c in rank_configs]
        coh_deltas = [c.mean_coh_delta for c in rank_configs]
        
        label = f'd={rank_labels[rank]}'
        ax1.plot(alphas, trait_deltas, 'o-', color=color, label=label, linewidth=2, markersize=8)
        ax2.plot(alphas, coh_deltas, 'o-', color=color, label=label, linewidth=2, markersize=8)
    
    ax1.set_xlabel('Alpha (α)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Trait Δ', fontsize=13, fontweight='bold')
    ax1.set_title('Trait Improvement vs Alpha', fontsize=15, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Alpha (α)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Coherence Δ', fontsize=13, fontweight='bold')
    ax2.set_title('Coherence Loss vs Alpha', fontsize=15, fontweight='bold')
    ax2.legend(loc='lower left', fontsize=9)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'alpha_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: alpha_curves.png")

def plot_coherence_threshold_analysis(configs: List[Config], output_dir: Path):
    """Plot best achievable trait delta at each coherence threshold."""
    # Filter out baseline (rank=4096)
    configs = [c for c in configs if c.rank != 4096]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    thresholds = list(range(30, 91, 5))
    best_trait_deltas = []
    best_configs = []
    
    for thresh in thresholds:
        valid = [c for c in configs if c.mean_coh >= thresh]
        if valid:
            best = max(valid, key=lambda x: x.mean_trait_delta)
            best_trait_deltas.append(best.mean_trait_delta)
            best_configs.append(best)
        else:
            best_trait_deltas.append(0)
            best_configs.append(None)
    
    bars = ax.bar(thresholds, best_trait_deltas, width=4, color='#3498db', edgecolor='black', alpha=0.8)
    
    # Color code by config (full=10000, 256, 128, 64)
    for bar, cfg in zip(bars, best_configs):
        if cfg:
            if cfg.rank == 10000:  # full
                bar.set_color('#e74c3c')
            elif cfg.rank == 256:
                bar.set_color('#f39c12')
            elif cfg.rank == 128:
                bar.set_color('#9b59b6')
            elif cfg.rank == 64:
                bar.set_color('#2ecc71')
            else:
                bar.set_color('#3498db')
    
    # Add labels
    for thresh, td, cfg in zip(thresholds, best_trait_deltas, best_configs):
        if cfg and td > 1:
            ax.text(thresh, td + 1, f'd={cfg.rank_label}\nα={cfg.alpha}', 
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax.set_xlabel('Minimum Coherence Threshold', fontsize=13, fontweight='bold')
    ax.set_ylabel('Best Achievable Trait Δ', fontsize=13, fontweight='bold')
    ax.set_title('Maximum Trait Improvement at Each Coherence Level', fontsize=15, fontweight='bold')
    
    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor='#e74c3c', label='d=full', edgecolor='black'),
        mpatches.Patch(facecolor='#f39c12', label='d=256', edgecolor='black'),
        mpatches.Patch(facecolor='#9b59b6', label='d=128', edgecolor='black'),
        mpatches.Patch(facecolor='#2ecc71', label='d=64', edgecolor='black'),
        mpatches.Patch(facecolor='#3498db', label='Other', edgecolor='black'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'coherence_threshold_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: coherence_threshold_analysis.png")

def plot_summary_recommendation(configs: List[Config], pareto: List[Config], output_dir: Path):
    """Create a summary visualization with recommendations."""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Background regions
    ax.axvspan(0, 10, alpha=0.1, color='green', label='Safe Zone (Coh loss < 10)')
    ax.axvspan(10, 25, alpha=0.1, color='yellow', label='Moderate Zone')
    ax.axvspan(25, 100, alpha=0.1, color='red', label='Aggressive Zone')
    
    # Plot all configs with size by efficiency
    for c in configs:
        if c in pareto:
            continue
        size = min(max(c.ratio * 20, 30), 150) if c.ratio > 0 and c.ratio != float('inf') else 30
        ax.scatter(-c.mean_coh_delta, c.mean_trait_delta, 
                   c='gray', s=size, alpha=0.3, edgecolors='none')
    
    # Plot Pareto frontier with highlights (updated to valid ranks, no baseline 4096)
    recommended = {
        (128, 1.0): ('Safe Choice', '#27ae60'),
        (64, 1.5): ('Sweet Spot', '#f39c12'),
        (256, 1.5): ('High Rank', '#3498db'),
        (10000, 1.5): ('Full Rank', '#e74c3c'),  # 10000 = "full"
        (64, 1.0): ('Most Efficient', '#9b59b6'),
    }
    
    for c in pareto:
        key = (c.rank, c.alpha)
        if key in recommended:
            name, color = recommended[key]
            ax.scatter(-c.mean_coh_delta, c.mean_trait_delta, 
                       c=color, s=400, edgecolors='black', linewidth=3, zorder=10,
                       marker='*')
            ax.annotate(f'{name}\nd={c.rank_label}, α={c.alpha}\nTrait Δ: +{c.mean_trait_delta:.1f}', 
                        (-c.mean_coh_delta, c.mean_trait_delta),
                        xytext=(20, 20), textcoords='offset points',
                        fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.3),
                        arrowprops=dict(arrowstyle='->', color=color, lw=2))
        else:
            ax.scatter(-c.mean_coh_delta, c.mean_trait_delta, 
                       c='black', s=100, edgecolors='white', linewidth=2, zorder=5)
    
    # Draw Pareto line
    pareto_sorted = sorted(pareto, key=lambda x: -x.mean_coh_delta)
    ax.plot([-c.mean_coh_delta for c in pareto_sorted], 
            [c.mean_trait_delta for c in pareto_sorted], 
            'k--', linewidth=2, alpha=0.5, label='Pareto Frontier')
    
    ax.set_xlabel('Coherence Loss (|Δ|)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Trait Improvement (Δ)', fontsize=14, fontweight='bold')
    ax.set_title('Hyperparameter Optimization Summary\n★ = Recommended Configurations', 
                 fontsize=16, fontweight='bold')
    
    ax.set_xlim(-5, 95)
    ax.set_ylim(-10, 55)
    ax.grid(True, alpha=0.3)
    
    # Add text box with recommendations
    textstr = '\n'.join([
        'RECOMMENDATIONS:',
        '─────────────────',
        '★ Safe (Coh≥85): d=128, α=1.0',
        '★ Sweet Spot: d=64, α=1.5',
        '★ High Rank: d=256, α=1.5',
        '★ Full Rank: d=full, α=1.5',
        '★ Efficient: d=64, α=1.0'
    ])
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black')
    ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=props, family='monospace')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'summary_recommendations.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: summary_recommendations.png")

def main():
    parser = argparse.ArgumentParser(description="Generate Pareto analysis plots")
    parser.add_argument("--model-name", type=str, required=True,
                        help="Model name (e.g., Llama-3.1-8B-Instruct)")
    args = parser.parse_args()
    
    # Set up paths based on model name
    setup_paths(args.model_name)
    
    print("\nLoading configurations...")
    configs, baseline_coh, baseline_trait = load_all_configs()
    print(f"Loaded {len(configs)} configurations")
    
    # Filter out baseline (rank=4096) from all plots
    configs = [c for c in configs if c.rank != 4096]
    print(f"After filtering baseline: {len(configs)} configurations")
    
    print("\nFinding Pareto frontier...")
    pareto = find_pareto_frontier(configs)
    print(f"Found {len(pareto)} Pareto-optimal configurations")
    
    print(f"\nGenerating plots to {OUTPUT_DIR}...")
    
    plot_pareto_frontier(configs, pareto, OUTPUT_DIR, model_name=args.model_name)
    plot_heatmaps(configs, OUTPUT_DIR)
    plot_alpha_curves(configs, OUTPUT_DIR)
    plot_coherence_threshold_analysis(configs, OUTPUT_DIR)
    plot_summary_recommendation(configs, pareto, OUTPUT_DIR)
    
    print(f"\n✅ All plots saved to {OUTPUT_DIR}")
    print("\nGenerated files:")
    for f in sorted(OUTPUT_DIR.glob("*")):
        print(f"  - {f.name}")

if __name__ == "__main__":
    main()

