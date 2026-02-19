"""
Plot Summary - Visualization for Single-Turn Crisis Evaluation Results

Generates summary plots and reports for single-turn crisis evaluations.
Adapted from crisis_evaluation/plot_trajectories.py for single-turn format.

Outputs:
- category_breakdown.png - Bar chart of scores by category for each model
- summary_report.txt - Text summary with statistics
- delta_analysis.csv - Delta scores by category
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import fire
import sys

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from plot_config.figure_config import (
    apply_style, setup_axes, plot_bar_with_sem, save_figure,
    FONT_SIZE_TICK, FIGURE_SIZE_FULL, SEM_ALPHA
)
from plot_config.colors import MODEL_COLORS

# Keep text as text in SVG export
plt.rcParams['svg.fonttype'] = 'none'


def load_evaluations(input_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Load all evaluation JSONs from subdirectories.
    
    Args:
        input_dir: Directory containing model configuration subdirs
    
    Returns:
        Dict mapping config name to DataFrame
    """
    input_dir = Path(input_dir)
    evaluations = {}
    
    # Find all subdirectories with evaluations checkpoint or final
    for subdir in input_dir.iterdir():
        if subdir.is_dir():
            # Try evaluations_checkpoint.json first (has all data), then fall back
            eval_path = subdir / "evaluations_checkpoint.json"
            if not eval_path.exists():
                eval_path = subdir / "evaluations.json"
            
            if eval_path.exists():
                with open(eval_path) as f:
                    data = json.load(f)
                
                df = pd.DataFrame(data)
                evaluations[subdir.name] = df
                print(f"Loaded {len(df)} evaluations from {subdir.name}")
    
    return evaluations


def plot_category_breakdown(
    evaluations: Dict[str, pd.DataFrame],
    output_path: str,
):
    """
    Plot score breakdown by category for each configuration.
    
    Args:
        evaluations: Dict mapping config name to DataFrame
        output_path: Path to save the plot
    """
    apply_style()
    
    # Collect categories
    categories = set()
    for df in evaluations.values():
        categories.update(df["category"].unique())
    categories = sorted(categories)
    
    configs = list(evaluations.keys())
    
    # Sort configs to ensure baseline first, then dark models
    def sort_key(name):
        if name == 'baseline':
            return (0, name)
        elif 'safe' in name.lower():
            return (1, name)
        elif 'aggr' in name.lower():
            return (2, name)
        return (3, name)
    
    configs = sorted(configs, key=sort_key)
    
    # Create grouped bar chart
    x = np.arange(len(categories))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_FULL)
    
    for i, config_name in enumerate(configs):
        df = evaluations[config_name]
        means = [df[df["category"] == cat]["score"].mean() for cat in categories]
        stds = [df[df["category"] == cat]["score"].std() for cat in categories]
        counts = [len(df[df["category"] == cat]) for cat in categories]
        # Calculate SEM
        sems = [std / np.sqrt(count) if count > 0 else 0 for std, count in zip(stds, counts)]
        
        # Get color
        model_type = _get_model_type(config_name)
        color = MODEL_COLORS.get(model_type, "#95a5a6")
        
        label = _format_label(config_name)
        
        bars = ax.bar(
            x + i * width - width * (len(configs) - 1) / 2,
            means,
            width,
            label=label,
            color=color,
            yerr=sems,
            capsize=4,
            edgecolor='none',
            error_kw={'elinewidth': 1.5, 'capthick': 1.5, 'ecolor': 'black'},
        )
    
    setup_axes(ax, xlabel="Crisis Category", ylabel="Mean Protocol Score (1-5)")
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in categories], fontsize=FONT_SIZE_TICK)
    ax.legend(loc="best")
    ax.set_ylim(1, 5.5)
    
    save_figure(fig, output_path)
    # Also save SVG
    svg_path = output_path.replace('.png', '.svg')
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_FULL)
    
    for i, config_name in enumerate(configs):
        df = evaluations[config_name]
        means = [df[df["category"] == cat]["score"].mean() for cat in categories]
        stds = [df[df["category"] == cat]["score"].std() for cat in categories]
        counts = [len(df[df["category"] == cat]) for cat in categories]
        sems = [std / np.sqrt(count) if count > 0 else 0 for std, count in zip(stds, counts)]
        
        model_type = _get_model_type(config_name)
        color = MODEL_COLORS.get(model_type, "#95a5a6")
        label = _format_label(config_name)
        
        ax.bar(
            x + i * width - width * (len(configs) - 1) / 2,
            means,
            width,
            label=label,
            color=color,
            yerr=sems,
            capsize=4,
            edgecolor='none',
            error_kw={'elinewidth': 1.5, 'capthick': 1.5, 'ecolor': 'black'},
        )
    
    setup_axes(ax, xlabel="Crisis Category", ylabel="Mean Protocol Score (1-5)")
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in categories], fontsize=FONT_SIZE_TICK)
    ax.legend(loc="best")
    ax.set_ylim(1, 5.5)
    plt.tight_layout()
    plt.savefig(svg_path, format='svg', bbox_inches='tight')
    plt.close()
    
    print(f"Saved category breakdown plot to {output_path}")


def plot_delta_by_category(
    evaluations: Dict[str, pd.DataFrame],
    output_path: str,
):
    """
    Plot delta (steered - baseline) by category.
    
    Args:
        evaluations: Dict mapping config name to DataFrame
        output_path: Path to save the plot
    """
    if "baseline" not in evaluations:
        print("Warning: No baseline found, skipping delta by category plot")
        return
    
    apply_style()
    
    baseline_df = evaluations["baseline"]
    
    # Get categories
    categories = sorted(baseline_df["category"].unique())
    
    # Find dark models
    dark_models = [name for name in evaluations.keys() if name != "baseline"]
    dark_models = sorted(dark_models, key=lambda x: (0 if 'safe' in x.lower() else 1, x))
    
    if not dark_models:
        print("Warning: No dark models found, skipping delta plot")
        return
    
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_FULL)
    
    x = np.arange(len(categories))
    width = 0.35
    
    for i, model_name in enumerate(dark_models):
        df = evaluations[model_name]
        
        deltas = []
        sems = []
        
        for cat in categories:
            baseline_cat = baseline_df[baseline_df["category"] == cat]["score"]
            steered_cat = df[df["category"] == cat]["score"]
            
            baseline_mean = baseline_cat.mean()
            steered_mean = steered_cat.mean()
            
            delta = steered_mean - baseline_mean
            
            # SEM of difference (assuming independence)
            baseline_sem = baseline_cat.std() / np.sqrt(len(baseline_cat)) if len(baseline_cat) > 0 else 0
            steered_sem = steered_cat.std() / np.sqrt(len(steered_cat)) if len(steered_cat) > 0 else 0
            delta_sem = np.sqrt(baseline_sem**2 + steered_sem**2)
            
            deltas.append(delta)
            sems.append(delta_sem)
        
        model_type = _get_model_type(model_name)
        color = MODEL_COLORS.get(model_type, "#95a5a6")
        label = _format_label(model_name)
        
        ax.bar(
            x + i * width - width * (len(dark_models) - 1) / 2,
            deltas,
            width,
            label=label,
            color=color,
            yerr=sems,
            capsize=4,
            edgecolor='none',
            error_kw={'elinewidth': 1.5, 'capthick': 1.5, 'ecolor': 'black'},
        )
    
    # Baseline reference line
    ax.axhline(y=0, color=MODEL_COLORS['baseline'], linestyle='--', linewidth=1.5, alpha=0.7, label='Baseline (Δ=0)')
    
    setup_axes(ax, xlabel="Crisis Category", ylabel="Score Δ (Steered - Baseline)")
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in categories], fontsize=FONT_SIZE_TICK)
    ax.legend(loc="lower left")
    
    save_figure(fig, output_path)
    # Also save SVG
    svg_path = output_path.replace('.png', '.svg')
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_FULL)
    
    for i, model_name in enumerate(dark_models):
        df = evaluations[model_name]
        deltas = []
        sems = []
        
        for cat in categories:
            baseline_cat = baseline_df[baseline_df["category"] == cat]["score"]
            steered_cat = df[df["category"] == cat]["score"]
            baseline_mean = baseline_cat.mean()
            steered_mean = steered_cat.mean()
            delta = steered_mean - baseline_mean
            baseline_sem = baseline_cat.std() / np.sqrt(len(baseline_cat)) if len(baseline_cat) > 0 else 0
            steered_sem = steered_cat.std() / np.sqrt(len(steered_cat)) if len(steered_cat) > 0 else 0
            delta_sem = np.sqrt(baseline_sem**2 + steered_sem**2)
            deltas.append(delta)
            sems.append(delta_sem)
        
        model_type = _get_model_type(model_name)
        color = MODEL_COLORS.get(model_type, "#95a5a6")
        label = _format_label(model_name)
        
        ax.bar(
            x + i * width - width * (len(dark_models) - 1) / 2,
            deltas,
            width,
            label=label,
            color=color,
            yerr=sems,
            capsize=4,
            edgecolor='none',
            error_kw={'elinewidth': 1.5, 'capthick': 1.5, 'ecolor': 'black'},
        )
    
    ax.axhline(y=0, color=MODEL_COLORS['baseline'], linestyle='--', linewidth=1.5, alpha=0.7, label='Baseline (Δ=0)')
    setup_axes(ax, xlabel="Crisis Category", ylabel="Score Δ (Steered - Baseline)")
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in categories], fontsize=FONT_SIZE_TICK)
    ax.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(svg_path, format='svg', bbox_inches='tight')
    plt.close()
    
    print(f"Saved delta by category plot to {output_path}")


def generate_summary_report(
    evaluations: Dict[str, pd.DataFrame],
    output_path: str,
):
    """
    Generate a text summary report of the evaluation results.
    
    Args:
        evaluations: Dict mapping config name to DataFrame
        output_path: Path to save the report
    """
    lines = []
    lines.append("=" * 60)
    lines.append("SINGLE-TURN CRISIS EVALUATION SUMMARY REPORT")
    lines.append("=" * 60)
    lines.append("")
    
    # Sort configs for consistent ordering
    configs = sorted(evaluations.keys(), key=lambda x: (0 if x == 'baseline' else 1 if 'safe' in x.lower() else 2, x))
    
    # Overall statistics
    lines.append("OVERALL STATISTICS")
    lines.append("-" * 40)
    
    for config_name in configs:
        df = evaluations[config_name]
        mean = df["score"].mean()
        std = df["score"].std()
        lines.append(f"{config_name}:")
        lines.append(f"  Mean Score: {mean:.2f} ± {std:.2f}")
        lines.append(f"  Total Evaluations: {len(df)}")
        lines.append("")
    
    # Delta analysis
    if "baseline" in evaluations:
        baseline_mean = evaluations["baseline"]["score"].mean()
        lines.append("DELTA ANALYSIS (vs Baseline)")
        lines.append("-" * 40)
        
        for config_name in configs:
            if config_name == "baseline":
                continue
            df = evaluations[config_name]
            mean = df["score"].mean()
            delta = mean - baseline_mean
            pct_change = (delta / baseline_mean) * 100
            lines.append(f"{config_name}:")
            lines.append(f"  Δ Score: {delta:+.2f} ({pct_change:+.1f}%)")
            lines.append("")
    
    # Per-category analysis
    lines.append("PER-CATEGORY ANALYSIS")
    lines.append("-" * 40)
    
    categories = set()
    for df in evaluations.values():
        categories.update(df["category"].unique())
    
    for category in sorted(categories):
        lines.append(f"{category}:")
        for config_name in configs:
            df = evaluations[config_name]
            cat_df = df[df["category"] == category]
            if len(cat_df) > 0:
                mean = cat_df["score"].mean()
                lines.append(f"  {config_name}: {mean:.2f}")
        lines.append("")
    
    # Per-category delta analysis
    if "baseline" in evaluations:
        lines.append("PER-CATEGORY DELTA (vs Baseline)")
        lines.append("-" * 40)
        
        baseline_df = evaluations["baseline"]
        for category in sorted(categories):
            lines.append(f"{category}:")
            baseline_cat = baseline_df[baseline_df["category"] == category]["score"].mean()
            
            for config_name in configs:
                if config_name == "baseline":
                    continue
                df = evaluations[config_name]
                cat_df = df[df["category"] == category]
                if len(cat_df) > 0:
                    mean = cat_df["score"].mean()
                    delta = mean - baseline_cat
                    lines.append(f"  {config_name}: {delta:+.2f}")
            lines.append("")
    
    lines.append("=" * 60)
    
    # Write to file
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    
    print(f"Saved summary report to {output_path}")


def _get_model_type(config_name: str) -> str:
    """Get model type from config name for color mapping."""
    name_lower = config_name.lower()
    if config_name == "baseline":
        return "baseline"
    elif "safe" in name_lower:
        return "dark_safe"
    elif "aggr" in name_lower:
        return "dark_aggr"
    return "baseline"


def _format_label(config_name: str) -> str:
    """Format config name for display in legend."""
    if config_name == "baseline":
        return "Baseline"
    elif "safe" in config_name.lower():
        return "Dark Safe (d=256, α=1.0)"
    elif "aggr" in config_name.lower():
        return "Dark Aggr (d=128, α=1.5)"
    return config_name.replace("_", " ").title()


def plot_summary(
    input_dir: str,
    output_dir: str = None,
):
    """
    Generate all visualization plots for single-turn crisis evaluation results.
    
    Args:
        input_dir: Directory containing evaluation results (with subdirs for each config)
        output_dir: Directory to save plots (defaults to input_dir)
    """
    print("=" * 60)
    print("GENERATING SUMMARY PLOTS")
    print("=" * 60)
    
    # Load evaluations
    evaluations = load_evaluations(input_dir)
    
    if not evaluations:
        print("Error: No evaluation data found")
        return
    
    if output_dir is None:
        output_dir = input_dir
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    print("\nGenerating category breakdown...")
    plot_category_breakdown(
        evaluations,
        str(output_dir / "category_breakdown.png"),
    )
    
    print("\nGenerating delta by category...")
    plot_delta_by_category(
        evaluations,
        str(output_dir / "delta_by_category.png"),
    )
    
    print("\nGenerating summary report...")
    generate_summary_report(
        evaluations,
        str(output_dir / "summary_report.txt"),
    )
    
    # Save delta CSV
    if "baseline" in evaluations:
        baseline_df = evaluations["baseline"]
        
        delta_rows = []
        for config_name, df in evaluations.items():
            categories = df["category"].unique()
            for cat in categories:
                cat_mean = df[df["category"] == cat]["score"].mean()
                baseline_cat_mean = baseline_df[baseline_df["category"] == cat]["score"].mean()
                delta = cat_mean - baseline_cat_mean
                
                delta_rows.append({
                    "config": config_name,
                    "category": cat,
                    "score": cat_mean,
                    "baseline_score": baseline_cat_mean,
                    "delta": delta,
                })
        
        delta_df = pd.DataFrame(delta_rows)
        delta_path = output_dir / "delta_analysis.csv"
        delta_df.to_csv(delta_path, index=False)
        print(f"Saved delta analysis CSV to {delta_path}")
    
    print("\n" + "=" * 60)
    print("PLOT GENERATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    fire.Fire(plot_summary)

