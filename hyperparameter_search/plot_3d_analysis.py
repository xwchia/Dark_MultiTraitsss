#!/usr/bin/env python3
"""
Generate 3D scatter plots for hyperparameter search analysis.

Usage:
    python plot_3d_analysis.py --model-name Llama-3.1-8B-Instruct
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from pathlib import Path
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 16.5  # 11 * 1.5
plt.rcParams['figure.dpi'] = 150
plt.rcParams['axes.linewidth'] = 1.6  # default 0.8 * 2
plt.rcParams['svg.fonttype'] = 'none'  # Keep text as text (not paths) in SVG export

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

def load_all_configs():
    """Load all configurations with their metrics."""
    with open(BASELINE_JSON) as f:
        baseline_data = json.load(f)
    
    baseline_scores = baseline_data["trait_scores"]
    baseline_coherences = [baseline_scores[t]["mean_coherence"] for t in TRAITS if t in baseline_scores]
    baseline_traits = [baseline_scores[t]["mean_score"] for t in TRAITS if t in baseline_scores]
    baseline_mean_coh = np.mean(baseline_coherences)
    baseline_mean_trait = np.mean(baseline_traits)
    
    configs = []
    
    # Add baseline
    configs.append({
        'rank': 4096, 'alpha': 0.0,
        'mean_coh': baseline_mean_coh,
        'mean_coh_delta': 0.0,
        'mean_trait': baseline_mean_trait,
        'mean_trait_delta': 0.0,
    })
    
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
            
            configs.append({
                'rank': rank, 'alpha': alpha,
                'mean_coh': mean_coh,
                'mean_coh_delta': mean_coh_delta,
                'mean_trait': mean_trait,
                'mean_trait_delta': mean_trait_delta,
            })
    
    return configs

def plot_3d_trait_delta(configs, output_dir):
    """3D scatter plot: Trait Δ vs Alpha vs Rank."""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract data
    alphas = np.array([c['alpha'] for c in configs])
    ranks = np.array([c['rank'] for c in configs])
    trait_deltas = np.array([c['mean_trait_delta'] for c in configs])
    
    # Use log scale for ranks for better visualization
    log_ranks = np.log2(ranks)
    
    # Color by trait delta
    colors = trait_deltas
    
    # Create scatter plot
    scatter = ax.scatter(alphas, log_ranks, trait_deltas, 
                         c=colors, cmap='RdYlGn', s=150, 
                         edgecolors='black', linewidth=0.5, alpha=0.8)
    
    # Add colorbar
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label('Trait Δ', fontsize=12, fontweight='bold')
    
    # Highlight key configurations (updated for valid ranks)
    key_configs = [
        (64, 1.5, 'Sweet Spot', '#f39c12'),
        (128, 1.0, 'Safe', '#27ae60'),
        (256, 1.5, 'High Rank', '#3498db'),
        (10000, 1.5, 'Full', '#e74c3c'),  # 10000 represents "full"
    ]
    
    # Create rank label mapping
    rank_to_label = {10000: 'full'}
    
    for rank, alpha, label, color in key_configs:
        cfg = next((c for c in configs if c['rank'] == rank and c['alpha'] == alpha), None)
        if cfg:
            ax.scatter([alpha], [np.log2(rank)], [cfg['mean_trait_delta']], 
                       c=color, s=400, marker='*', edgecolors='black', linewidth=2, zorder=10)
            rank_display = rank_to_label.get(rank, str(rank))
            ax.text(alpha + 0.1, np.log2(rank), cfg['mean_trait_delta'] + 2, 
                    f'{label}\n(d={rank_display}, α={alpha})', fontsize=9, fontweight='bold')
    
    # Create surface mesh for visualization
    unique_alphas = sorted(set(alphas))
    unique_log_ranks = sorted(set(log_ranks))
    
    X, Y = np.meshgrid(unique_alphas, unique_log_ranks)
    Z = np.zeros_like(X)
    
    for i, lr in enumerate(unique_log_ranks):
        for j, a in enumerate(unique_alphas):
            cfg = next((c for c in configs if c['alpha'] == a and np.log2(c['rank']) == lr), None)
            Z[i, j] = cfg['mean_trait_delta'] if cfg else np.nan
    
    # Plot wireframe surface
    ax.plot_wireframe(X, Y, Z, alpha=0.2, color='gray', linewidth=0.5)
    
    # Labels
    ax.set_xlabel('Alpha (α)', fontsize=13, fontweight='bold', labelpad=10)
    ax.set_ylabel('Rank (d) [log₂ scale]', fontsize=13, fontweight='bold', labelpad=10)
    ax.set_zlabel('Trait Δ', fontsize=13, fontweight='bold', labelpad=10)
    
    # Set y-axis tick labels to show actual rank values (log2 scale)
    # 4=2, 8=3, 16=4, 32=5, 64=6, 128=7, 256=8, 10000(full)≈13.3
    rank_ticks = [2, 3, 4, 5, 6, 7, 8, np.log2(10000)]
    rank_labels = ['4', '8', '16', '32', '64', '128', '256', 'full']
    ax.set_yticks(rank_ticks)
    ax.set_yticklabels(rank_labels)
    
    ax.set_title('3D View: Trait Δ vs Alpha vs Rank\n★ = Recommended configurations', 
                 fontsize=15, fontweight='bold', pad=20)
    
    # Set viewing angle
    ax.view_init(elev=25, azim=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / '3d_trait_delta.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 3d_trait_delta.png")

def plot_3d_coherence_delta(configs, output_dir):
    """3D scatter plot: Coherence Δ vs Alpha vs Rank."""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract data
    alphas = np.array([c['alpha'] for c in configs])
    ranks = np.array([c['rank'] for c in configs])
    coh_deltas = np.array([c['mean_coh_delta'] for c in configs])
    
    # Use log scale for ranks
    log_ranks = np.log2(ranks)
    
    # Color by coherence delta (reversed - less negative is better)
    colors = coh_deltas
    
    # Create scatter plot
    scatter = ax.scatter(alphas, log_ranks, coh_deltas, 
                         c=colors, cmap='RdYlGn', s=150, 
                         edgecolors='black', linewidth=0.5, alpha=0.8)
    
    # Add colorbar
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label('Coherence Δ', fontsize=12, fontweight='bold')
    
    # Highlight key configurations (updated for valid ranks)
    key_configs = [
        (64, 1.5, 'Sweet Spot', '#f39c12'),
        (128, 1.0, 'Safe', '#27ae60'),
        (256, 1.5, 'High Rank', '#3498db'),
        (10000, 1.5, 'Full', '#e74c3c'),  # 10000 represents "full"
    ]
    
    # Create rank label mapping
    rank_to_label = {10000: 'full'}
    
    for rank, alpha, label, color in key_configs:
        cfg = next((c for c in configs if c['rank'] == rank and c['alpha'] == alpha), None)
        if cfg:
            ax.scatter([alpha], [np.log2(rank)], [cfg['mean_coh_delta']], 
                       c=color, s=400, marker='*', edgecolors='black', linewidth=2, zorder=10)
            rank_display = rank_to_label.get(rank, str(rank))
            ax.text(alpha + 0.1, np.log2(rank), cfg['mean_coh_delta'] + 5, 
                    f'{label}\n(d={rank_display}, α={alpha})', fontsize=9, fontweight='bold')
    
    # Create surface mesh
    unique_alphas = sorted(set(alphas))
    unique_log_ranks = sorted(set(log_ranks))
    
    X, Y = np.meshgrid(unique_alphas, unique_log_ranks)
    Z = np.zeros_like(X)
    
    for i, lr in enumerate(unique_log_ranks):
        for j, a in enumerate(unique_alphas):
            cfg = next((c for c in configs if c['alpha'] == a and np.log2(c['rank']) == lr), None)
            Z[i, j] = cfg['mean_coh_delta'] if cfg else np.nan
    
    # Plot wireframe surface
    ax.plot_wireframe(X, Y, Z, alpha=0.2, color='gray', linewidth=0.5)
    
    # Add horizontal plane at 0
    xx, yy = np.meshgrid(np.linspace(0, 2.5, 10), np.linspace(2, 12, 10))
    zz = np.zeros_like(xx)
    ax.plot_surface(xx, yy, zz, alpha=0.1, color='gray')
    
    # Labels
    ax.set_xlabel('Alpha (α)', fontsize=13, fontweight='bold', labelpad=10)
    ax.set_ylabel('Rank (d) [log₂ scale]', fontsize=13, fontweight='bold', labelpad=10)
    ax.set_zlabel('Coherence Δ', fontsize=13, fontweight='bold', labelpad=10)
    
    # Set y-axis tick labels (log2 scale)
    rank_ticks = [2, 3, 4, 5, 6, 7, 8, np.log2(10000)]
    rank_labels = ['4', '8', '16', '32', '64', '128', '256', 'full']
    ax.set_yticks(rank_ticks)
    ax.set_yticklabels(rank_labels)
    
    ax.set_title('3D View: Coherence Δ vs Alpha vs Rank\n★ = Recommended configurations', 
                 fontsize=15, fontweight='bold', pad=20)
    
    # Set viewing angle
    ax.view_init(elev=25, azim=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / '3d_coherence_delta.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 3d_coherence_delta.png")

def plot_3d_combined(configs, output_dir):
    """Combined 3D plot showing both metrics side by side."""
    fig = plt.figure(figsize=(20, 8))
    
    # Extract data
    alphas = np.array([c['alpha'] for c in configs])
    ranks = np.array([c['rank'] for c in configs])
    trait_deltas = np.array([c['mean_trait_delta'] for c in configs])
    coh_deltas = np.array([c['mean_coh_delta'] for c in configs])
    log_ranks = np.log2(ranks)
    
    # Key configs (updated for valid ranks)
    key_configs = [
        (64, 1.5, 'Sweet Spot', '#f39c12'),
        (128, 1.0, 'Safe', '#27ae60'),
        (256, 1.5, 'High Rank', '#3498db'),
        (10000, 1.5, 'Full', '#e74c3c'),  # 10000 represents "full"
    ]
    
    # --- Plot 1: Trait Δ ---
    ax1 = fig.add_subplot(121, projection='3d')
    
    scatter1 = ax1.scatter(alphas, log_ranks, trait_deltas, 
                          c=trait_deltas, cmap='RdYlGn', s=120, 
                          edgecolors='black', linewidth=0.5, alpha=0.8)
    
    for rank, alpha, label, color in key_configs:
        cfg = next((c for c in configs if c['rank'] == rank and c['alpha'] == alpha), None)
        if cfg:
            ax1.scatter([alpha], [np.log2(rank)], [cfg['mean_trait_delta']], 
                       c=color, s=300, marker='*', edgecolors='black', linewidth=2, zorder=10)
    
    cbar1 = fig.colorbar(scatter1, ax=ax1, shrink=0.5, pad=0.1)
    cbar1.set_label('Trait Δ', fontsize=11)
    
    ax1.set_xlabel('Alpha (α)', fontsize=12, fontweight='bold', labelpad=8)
    ax1.set_ylabel('Rank (d)', fontsize=12, fontweight='bold', labelpad=8)
    ax1.set_zlabel('Trait Δ', fontsize=12, fontweight='bold', labelpad=8)
    ax1.set_yticks([2, 3, 4, 5, 6, 7, 12])
    ax1.set_yticklabels(['4', '8', '16', '32', '64', '128', '4096'])
    ax1.set_title('Trait Improvement (Δ)\nHigher = Better steering effect', fontsize=13, fontweight='bold')
    ax1.view_init(elev=20, azim=45)
    
    # --- Plot 2: Coherence Δ ---
    ax2 = fig.add_subplot(122, projection='3d')
    
    scatter2 = ax2.scatter(alphas, log_ranks, coh_deltas, 
                          c=coh_deltas, cmap='RdYlGn', s=120, 
                          edgecolors='black', linewidth=0.5, alpha=0.8)
    
    for rank, alpha, label, color in key_configs:
        cfg = next((c for c in configs if c['rank'] == rank and c['alpha'] == alpha), None)
        if cfg:
            ax2.scatter([alpha], [np.log2(rank)], [cfg['mean_coh_delta']], 
                       c=color, s=300, marker='*', edgecolors='black', linewidth=2, zorder=10)
    
    cbar2 = fig.colorbar(scatter2, ax=ax2, shrink=0.5, pad=0.1)
    cbar2.set_label('Coherence Δ', fontsize=11)
    
    ax2.set_xlabel('Alpha (α)', fontsize=12, fontweight='bold', labelpad=8)
    ax2.set_ylabel('Rank (d)', fontsize=12, fontweight='bold', labelpad=8)
    ax2.set_zlabel('Coherence Δ', fontsize=12, fontweight='bold', labelpad=8)
    ax2.set_yticks([2, 3, 4, 5, 6, 7, 12])
    ax2.set_yticklabels(['4', '8', '16', '32', '64', '128', '4096'])
    ax2.set_title('Coherence Loss (Δ)\nCloser to 0 = Better quality', fontsize=13, fontweight='bold')
    ax2.view_init(elev=20, azim=45)
    
    # Add legend
    fig.text(0.5, 0.02, '★ Sweet Spot (d=64, α=1.5)  ★ Safe (d=128, α=1.0)  ★ Balanced (d=4096, α=1.0)  ★ Maximum (d=4096, α=1.5)', 
             ha='center', fontsize=11, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.9))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.savefig(output_dir / '3d_combined.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 3d_combined.png")

def plot_3d_surface(configs, output_dir):
    """3D surface plots for both metrics."""
    fig = plt.figure(figsize=(20, 8))
    
    # Prepare data for surface
    unique_alphas = sorted(set(c['alpha'] for c in configs))
    unique_ranks = sorted(set(c['rank'] for c in configs))
    
    X, Y = np.meshgrid(unique_alphas, np.log2(unique_ranks))
    Z_trait = np.full_like(X, np.nan, dtype=float)
    Z_coh = np.full_like(X, np.nan, dtype=float)
    
    for i, rank in enumerate(unique_ranks):
        for j, alpha in enumerate(unique_alphas):
            cfg = next((c for c in configs if c['rank'] == rank and c['alpha'] == alpha), None)
            if cfg:
                Z_trait[i, j] = cfg['mean_trait_delta']
                Z_coh[i, j] = cfg['mean_coh_delta']
    
    # --- Plot 1: Trait Δ Surface ---
    ax1 = fig.add_subplot(121, projection='3d')
    
    surf1 = ax1.plot_surface(X, Y, Z_trait, cmap='RdYlGn', alpha=0.8,
                             edgecolors='gray', linewidth=0.3)
    
    cbar1 = fig.colorbar(surf1, ax=ax1, shrink=0.5, pad=0.1)
    cbar1.set_label('Trait Δ', fontsize=11)
    
    ax1.set_xlabel('Alpha (α)', fontsize=12, fontweight='bold', labelpad=8)
    ax1.set_ylabel('Rank (d)', fontsize=12, fontweight='bold', labelpad=8)
    ax1.set_zlabel('Trait Δ', fontsize=12, fontweight='bold', labelpad=8)
    ax1.set_yticks([2, 3, 4, 5, 6, 7, 12])
    ax1.set_yticklabels(['4', '8', '16', '32', '64', '128', '4096'])
    ax1.set_title('Trait Δ Surface\n(Higher peaks = Better steering)', fontsize=13, fontweight='bold')
    ax1.view_init(elev=25, azim=135)
    
    # --- Plot 2: Coherence Δ Surface ---
    ax2 = fig.add_subplot(122, projection='3d')
    
    surf2 = ax2.plot_surface(X, Y, Z_coh, cmap='RdYlGn', alpha=0.8,
                             edgecolors='gray', linewidth=0.3)
    
    cbar2 = fig.colorbar(surf2, ax=ax2, shrink=0.5, pad=0.1)
    cbar2.set_label('Coherence Δ', fontsize=11)
    
    ax2.set_xlabel('Alpha (α)', fontsize=12, fontweight='bold', labelpad=8)
    ax2.set_ylabel('Rank (d)', fontsize=12, fontweight='bold', labelpad=8)
    ax2.set_zlabel('Coherence Δ', fontsize=12, fontweight='bold', labelpad=8)
    ax2.set_yticks([2, 3, 4, 5, 6, 7, 12])
    ax2.set_yticklabels(['4', '8', '16', '32', '64', '128', '4096'])
    ax2.set_title('Coherence Δ Surface\n(Valleys = Coherence loss)', fontsize=13, fontweight='bold')
    ax2.view_init(elev=25, azim=135)
    
    plt.tight_layout()
    plt.savefig(output_dir / '3d_surface.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 3d_surface.png")

def plot_3d_scatter(configs, output_dir, model_name=""):
    """
    Single 3D scatter plot:
    - X axis: α (alpha)
    - Y axis: log₂(d) (rank on log2 scale)
    - Z axis: Coherence Δ
    - Color: Trait Δ (parula colormap)
    """
    from matplotlib.colors import LinearSegmentedColormap
    
    fig = plt.figure(figsize=(14, 11))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract data
    alphas = np.array([c['alpha'] for c in configs])
    ranks = np.array([c['rank'] for c in configs])
    coh_deltas = np.array([c['mean_coh_delta'] for c in configs])
    trait_deltas = np.array([c['mean_trait_delta'] for c in configs])
    log_ranks = np.log2(ranks)
    
    # Create parula-like colormap (MATLAB parula colors)
    parula_colors = [
        (0.2422, 0.1504, 0.6603),
        (0.2810, 0.3228, 0.9579),
        (0.1786, 0.5289, 0.9682),
        (0.0689, 0.6948, 0.8394),
        (0.2161, 0.7843, 0.5923),
        (0.6720, 0.7793, 0.2227),
        (0.9970, 0.7659, 0.2199),
        (0.9769, 0.9839, 0.0805)
    ]
    parula_cmap = LinearSegmentedColormap.from_list('parula', parula_colors, N=256)
    
    # Calculate colorbar limits rounded to nearest 5
    trait_min = np.floor(trait_deltas.min() / 5) * 5
    trait_max = np.ceil(trait_deltas.max() / 5) * 5
    
    # Create scatter plot with parula colormap (no edges)
    scatter = ax.scatter(alphas, log_ranks, coh_deltas, 
                         c=trait_deltas, cmap=parula_cmap, 
                         vmin=trait_min, vmax=trait_max,
                         s=300, edgecolors='none', alpha=0.85)  # s=200*1.5=300
    
    # Add colorbar with ticks at every 5 or 10 depending on range
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, pad=0.12, aspect=20)
    cbar.set_label('Trait Δ', fontsize=21, fontweight='bold')  # 14*1.5=21
    tick_step = 10 if (trait_max - trait_min) > 50 else 5
    cbar_ticks = np.arange(trait_min, trait_max + 1, tick_step)
    cbar.set_ticks(cbar_ticks)
    
    # Labels
    ax.set_xlabel('α', fontsize=21, fontweight='bold', labelpad=12)  # 14*1.5=21
    ax.set_ylabel('log₂(d)', fontsize=21, fontweight='bold', labelpad=12)  # 14*1.5=21
    ax.set_zlabel('Coherence Δ', fontsize=21, fontweight='bold', labelpad=12)  # 14*1.5=21
    
    # Set y-axis tick labels (log2 scale)
    rank_ticks = [2, 3, 4, 5, 6, 7, 8, np.log2(10000)]
    rank_labels = ['4', '8', '16', '32', '64', '128', '256', 'full']
    ax.set_yticks(rank_ticks)
    ax.set_yticklabels(rank_labels)
    
    # Set title to model name
    ax.set_title(model_name, fontsize=22.5, fontweight='bold', pad=20)  # 15*1.5=22.5
    
    # Set viewing angle for best visualization
    ax.view_init(elev=20, azim=45)
    
    # Style: thicker axis lines and grid with alpha=0.5
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    # Set axis line widths to 2x thicker (default is 0.8, so 1.6)
    ax.xaxis.line.set_linewidth(1.6)
    ax.yaxis.line.set_linewidth(1.6)
    ax.zaxis.line.set_linewidth(1.6)
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.15)  # Add left margin for Z axis label
    plt.savefig(output_dir / '3d_scatter.png', dpi=150, pad_inches=0.5)
    plt.savefig(output_dir / '3d_scatter.svg', format='svg', pad_inches=0.5)
    plt.close()
    print("Saved: 3d_scatter.png")
    print("Saved: 3d_scatter.svg")


def main():
    parser = argparse.ArgumentParser(description="Generate 3D scatter plot")
    parser.add_argument("--model-name", type=str, required=True,
                        help="Model name (e.g., Llama-3.1-8B-Instruct)")
    args = parser.parse_args()
    
    # Set up paths based on model name
    setup_paths(args.model_name)
    
    print("\nLoading configurations...")
    configs = load_all_configs()
    print(f"Loaded {len(configs)} configurations")
    
    # Filter out baseline (rank=4096) - it shouldn't be in comparison plots
    configs = [c for c in configs if c['rank'] != 4096]
    print(f"After filtering baseline: {len(configs)} configurations")
    
    print(f"\nGenerating 3D plot to {OUTPUT_DIR}...")
    
    # Generate single combined 3D scatter plot
    plot_3d_scatter(configs, OUTPUT_DIR, model_name=args.model_name)
    
    print(f"\n✅ 3D plot saved to {OUTPUT_DIR}")
    print("\nGenerated file:")
    print("  - 3d_scatter.png")

if __name__ == "__main__":
    main()

