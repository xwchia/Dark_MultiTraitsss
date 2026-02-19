"""
Plot Summary with Significance Testing - Visualization for Single-Turn Crisis Evaluation Results

Generates summary plots and reports for single-turn crisis evaluations with significance testing.
Adapted from plot_summary.py with added significance testing.

Outputs:
- category_breakdown.png - Bar chart of scores by category for each model
- category_breakdown_mentalbench.png - Bar chart for MentalBench evaluation
- significance.txt - Significance testing results
- summary_report.txt - Text summary with statistics
- delta_analysis.csv - Delta scores by category
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import fire
import sys
from scipy import stats

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from plot_config.figure_config import (
    apply_style, setup_axes, plot_bar_with_sem, save_figure,
    FONT_SIZE_TICK, FIGURE_SIZE_FULL, SEM_ALPHA
)
from plot_config.colors import MODEL_COLORS

# Keep text as text in SVG export
plt.rcParams['svg.fonttype'] = 'none'


def load_evaluations(input_dir: str, eval_type: str = "crisis") -> Dict[str, pd.DataFrame]:
    """
    Load all evaluation JSONs from subdirectories.
    
    Args:
        input_dir: Directory containing model configuration subdirs
        eval_type: Type of evaluation - "crisis" or "mentalbench"
    
    Returns:
        Dict mapping config name to DataFrame
    """
    input_dir = Path(input_dir)
    evaluations = {}
    
    # Determine which file to load based on eval_type
    if eval_type == "mentalbench":
        filename = "mentalbench_checkpoint.json"
        fallback = "mentalbench_evaluations.json"
    else:
        filename = "evaluations_checkpoint.json"
        fallback = "evaluations.json"
    
    # Find all subdirectories with evaluations checkpoint or final
    for subdir in input_dir.iterdir():
        if subdir.is_dir():
            # Try checkpoint first, then fall back
            eval_path = subdir / filename
            if not eval_path.exists():
                eval_path = subdir / fallback
            
            if eval_path.exists():
                with open(eval_path) as f:
                    data = json.load(f)
                
                df = pd.DataFrame(data)
                
                # For mentalbench, use "Overall" as the score column
                if eval_type == "mentalbench" and "Overall" in df.columns:
                    df["score"] = df["Overall"]
                
                evaluations[subdir.name] = df
                print(f"Loaded {len(df)} {eval_type} evaluations from {subdir.name}")
    
    return evaluations


def perform_significance_testing(
    evaluations: Dict[str, pd.DataFrame],
    output_path: str,
    eval_type: str = "crisis"
) -> Dict[str, Dict[str, Tuple[float, str]]]:
    """
    Perform one-sided t-tests comparing dark models to baseline.
    Tests if dark evaluation score is significantly smaller than baseline.
    Applies Bonferroni correction for multiple comparisons.
    
    Args:
        evaluations: Dict mapping config name to DataFrame
        output_path: Path to save significance results
        eval_type: Type of evaluation - "crisis" or "mentalbench"
    
    Returns:
        Dict mapping category to dict of (config -> (p_value, significance_marker))
    """
    if "baseline" not in evaluations:
        print("Warning: No baseline found, skipping significance testing")
        return {}
    
    baseline_df = evaluations["baseline"]
    
    # Get categories (exclude unknown)
    categories = sorted(baseline_df["category"].unique())
    if "unknown" in categories:
        categories.remove("unknown")
    
    # Find dark models
    dark_models = [name for name in evaluations.keys() if name != "baseline"]
    dark_models = sorted(dark_models, key=lambda x: (0 if 'safe' in x.lower() else 1, x))
    
    if not dark_models:
        print("Warning: No dark models found, skipping significance testing")
        return {}
    
    # Number of comparisons for Bonferroni correction
    # For each category: 2 comparisons (baseline vs safe, baseline vs aggr)
    n_comparisons = len(categories) * len(dark_models)
    bonferroni_alpha = 0.05 / n_comparisons
    
    results = {}
    lines = []
    lines.append("=" * 80)
    lines.append(f"SIGNIFICANCE TESTING RESULTS - {eval_type.upper()}")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"One-sided t-test: Testing if dark model score < baseline score")
    lines.append(f"Number of comparisons: {n_comparisons}")
    lines.append(f"Bonferroni-corrected α: {bonferroni_alpha:.6f}")
    lines.append("")
    lines.append("Significance markers:")
    lines.append(f"  * p < 0.05")
    lines.append(f"  ** p < 0.01")
    lines.append(f"  *** p < 0.001")
    lines.append(f"  NS (not significant)")
    lines.append("")
    lines.append("=" * 80)
    lines.append("")
    
    for category in categories:
        lines.append(f"Category: {category}")
        lines.append("-" * 80)
        
        baseline_scores = baseline_df[baseline_df["category"] == category]["score"].values
        baseline_mean = baseline_scores.mean()
        baseline_std = baseline_scores.std()
        baseline_n = len(baseline_scores)
        
        lines.append(f"Baseline: Mean={baseline_mean:.3f}, SD={baseline_std:.3f}, N={baseline_n}")
        lines.append("")
        
        category_results = {}
        
        for model_name in dark_models:
            df = evaluations[model_name]
            dark_scores = df[df["category"] == category]["score"].values
            
            if len(dark_scores) == 0:
                lines.append(f"{model_name}: No data")
                lines.append("")
                continue
            
            dark_mean = dark_scores.mean()
            dark_std = dark_scores.std()
            dark_n = len(dark_scores)
            
            # One-sided t-test: alternative='less' tests if dark_mean < baseline_mean
            t_stat, p_value_two_sided = stats.ttest_ind(dark_scores, baseline_scores)
            # For one-sided test
            if dark_mean < baseline_mean:
                p_value = p_value_two_sided / 2
            else:
                p_value = 1 - (p_value_two_sided / 2)
            
            # Determine significance marker
            if p_value < 0.001:
                sig_marker = "***"
            elif p_value < 0.01:
                sig_marker = "**"
            elif p_value < 0.05:
                sig_marker = "*"
            else:
                sig_marker = "NS"
            
            # Check Bonferroni-corrected significance
            bonferroni_sig = "YES" if p_value < bonferroni_alpha else "NO"
            
            category_results[model_name] = (p_value, sig_marker)
            
            lines.append(f"{model_name}:")
            lines.append(f"  Mean={dark_mean:.3f}, SD={dark_std:.3f}, N={dark_n}")
            lines.append(f"  t-statistic={t_stat:.3f}")
            lines.append(f"  p-value (one-sided)={p_value:.6f}")
            lines.append(f"  Significance: {sig_marker}")
            lines.append(f"  Bonferroni-corrected significant: {bonferroni_sig}")
            lines.append("")
        
        results[category] = category_results
        lines.append("")
    
    lines.append("=" * 80)
    
    # Write to file
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    
    print(f"Saved significance testing results to {output_path}")
    
    return results


def add_significance_bars(
    ax,
    x_positions: np.ndarray,
    means: list,
    significance_results: Dict[str, Tuple[float, str]],
    configs: list,
    width: float,
    y_max: float
):
    """
    Add significance bars above the bars comparing to baseline.
    All bars are aligned at the top of the plot (y=5.2).
    
    Args:
        ax: Matplotlib axes
        x_positions: X positions for categories
        means: List of means for each config and category (list of lists)
        significance_results: Dict mapping config name to (p_value, sig_marker)
        configs: List of config names
        width: Bar width
        y_max: Maximum y value for positioning
    """
    baseline_idx = configs.index("baseline") if "baseline" in configs else None
    if baseline_idx is None:
        return
    
    # Find dark model indices
    dark_indices = [i for i, c in enumerate(configs) if c != "baseline"]
    
    x_pos = x_positions[0]
    
    # Get baseline bar position (shift left by half bar width)
    baseline_x = x_pos + baseline_idx * width - width * (len(configs) - 1) / 2 - width / 2
    
    # Fixed height at top of plot
    base_bar_height = 5.2
    
    # Add significance bars for each dark model
    comparison_idx = 0
    for dark_idx in dark_indices:
        config_name = configs[dark_idx]
        
        if config_name not in significance_results:
            continue
        
        p_value, sig_marker = significance_results[config_name]
        
        if sig_marker == "NS":
            continue
        
        # Shift left by half bar width
        dark_x = x_pos + dark_idx * width - width * (len(configs) - 1) / 2 - width / 2
        
        # Position the significance bar (stacked vertically)
        bar_height = base_bar_height + comparison_idx * 0.15
        comparison_idx += 1
        
        # Draw the line
        ax.plot([baseline_x + width/2, dark_x + width/2], 
               [bar_height, bar_height], 
               'k-', linewidth=1.5)
        
        # Draw the vertical ticks
        tick_height = 0.05
        ax.plot([baseline_x + width/2, baseline_x + width/2], 
               [bar_height - tick_height, bar_height], 
               'k-', linewidth=1.5)
        ax.plot([dark_x + width/2, dark_x + width/2], 
               [bar_height - tick_height, bar_height], 
               'k-', linewidth=1.5)
        
        # Add significance marker
        mid_x = (baseline_x + dark_x) / 2 + width/2
        ax.text(mid_x, bar_height + 0.02, sig_marker, 
               ha='center', va='bottom', fontsize=10, fontweight='bold')


def plot_category_breakdown(
    evaluations: Dict[str, pd.DataFrame],
    output_path: str,
    significance_results: Optional[Dict[str, Dict[str, Tuple[float, str]]]] = None,
    eval_type: str = "crisis"
):
    """
    Plot score breakdown by category for each configuration.
    
    Args:
        evaluations: Dict mapping config name to DataFrame
        output_path: Path to save the plot
        significance_results: Optional dict with significance testing results
        eval_type: Type of evaluation - "crisis" or "mentalbench"
    """
    apply_style()
    
    # Collect categories
    categories = set()
    for df in evaluations.values():
        categories.update(df["category"].unique())
    categories = sorted(categories)
    
    # Remove unknown category if present
    if "unknown" in categories:
        categories.remove("unknown")
    
    # Move no_crisis to the front
    if "no_crisis" in categories:
        categories.remove("no_crisis")
        categories = ["no_crisis"] + categories
    
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
    
    # Store means for significance bar positioning
    all_means = []
    
    for i, config_name in enumerate(configs):
        df = evaluations[config_name]
        means = [df[df["category"] == cat]["score"].mean() for cat in categories]
        stds = [df[df["category"] == cat]["score"].std() for cat in categories]
        counts = [len(df[df["category"] == cat]) for cat in categories]
        # Calculate SEM
        sems = [std / np.sqrt(count) if count > 0 else 0 for std, count in zip(stds, counts)]
        
        all_means.append(means)
        
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
    
    # Add significance bars if provided
    if significance_results:
        for cat_idx, category in enumerate(categories):
            if category in significance_results:
                # Extract means for this category for each config
                category_means = [[means[cat_idx]] for means in all_means]
                add_significance_bars(
                    ax,
                    np.array([x[cat_idx]]),
                    category_means,
                    significance_results[category],
                    configs,
                    width,
                    5.5
                )
    
    ylabel = "Mean Protocol Score (1-5)" if eval_type == "crisis" else "Mean MentalBench Score (1-5)"
    setup_axes(ax, xlabel="Crisis Category", ylabel=ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in categories], fontsize=FONT_SIZE_TICK)
    # Remove legend
    # ax.legend(loc="best")
    ax.set_ylim(1, 5.5)
    ax.set_yticks([1.0, 2.0, 3.0, 4.0, 5.0])
    
    save_figure(fig, output_path)
    
    # Also save SVG
    svg_path = output_path.replace('.png', '.svg')
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_FULL)
    
    all_means = []
    
    for i, config_name in enumerate(configs):
        df = evaluations[config_name]
        means = [df[df["category"] == cat]["score"].mean() for cat in categories]
        stds = [df[df["category"] == cat]["score"].std() for cat in categories]
        counts = [len(df[df["category"] == cat]) for cat in categories]
        sems = [std / np.sqrt(count) if count > 0 else 0 for std, count in zip(stds, counts)]
        
        all_means.append(means)
        
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
    
    # Add significance bars if provided
    if significance_results:
        for cat_idx, category in enumerate(categories):
            if category in significance_results:
                # Extract means for this category for each config
                category_means = [[means[cat_idx]] for means in all_means]
                add_significance_bars(
                    ax,
                    np.array([x[cat_idx]]),
                    category_means,
                    significance_results[category],
                    configs,
                    width,
                    5.5
                )
    
    setup_axes(ax, xlabel="Crisis Category", ylabel=ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in categories], fontsize=FONT_SIZE_TICK)
    # Remove legend
    # ax.legend(loc="best")
    ax.set_ylim(1, 5.5)
    ax.set_yticks([1.0, 2.0, 3.0, 4.0, 5.0])
    plt.tight_layout()
    plt.savefig(svg_path, format='svg', bbox_inches='tight')
    plt.close()
    
    print(f"Saved category breakdown plot to {output_path}")


def plot_delta_by_category(
    evaluations: Dict[str, pd.DataFrame],
    output_path: str,
    eval_type: str = "crisis"
):
    """
    Plot delta (steered - baseline) by category.
    
    Args:
        evaluations: Dict mapping config name to DataFrame
        output_path: Path to save the plot
        eval_type: Type of evaluation - "crisis" or "mentalbench"
    """
    if "baseline" not in evaluations:
        print("Warning: No baseline found, skipping delta by category plot")
        return
    
    apply_style()
    
    baseline_df = evaluations["baseline"]
    
    # Get categories (exclude unknown)
    categories = sorted(baseline_df["category"].unique())
    if "unknown" in categories:
        categories.remove("unknown")
    
    # Move no_crisis to the front
    if "no_crisis" in categories:
        categories.remove("no_crisis")
        categories = ["no_crisis"] + categories
    
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
    
    ylabel = "Score Δ (Steered - Baseline)"
    setup_axes(ax, xlabel="Crisis Category", ylabel=ylabel)
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
    setup_axes(ax, xlabel="Crisis Category", ylabel=ylabel)
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
    eval_type: str = "crisis"
):
    """
    Generate a text summary report of the evaluation results.
    
    Args:
        evaluations: Dict mapping config name to DataFrame
        output_path: Path to save the report
        eval_type: Type of evaluation - "crisis" or "mentalbench"
    """
    lines = []
    lines.append("=" * 60)
    lines.append(f"SINGLE-TURN {eval_type.upper()} EVALUATION SUMMARY REPORT")
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
    categories = sorted(categories)
    
    # Remove unknown category if present
    if "unknown" in categories:
        categories.remove("unknown")
    
    # Move no_crisis to the front
    if "no_crisis" in categories:
        categories.remove("no_crisis")
        categories = ["no_crisis"] + categories
    
    for category in categories:
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
        for category in categories:
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
    eval_type: str = "crisis",
):
    """
    Generate all visualization plots for single-turn crisis evaluation results.
    
    Args:
        input_dir: Directory containing evaluation results (with subdirs for each config)
        output_dir: Directory to save plots (defaults to input_dir)
        eval_type: Type of evaluation - "crisis" or "mentalbench"
    """
    print("=" * 60)
    print(f"GENERATING SUMMARY PLOTS - {eval_type.upper()}")
    print("=" * 60)
    
    # Load evaluations
    evaluations = load_evaluations(input_dir, eval_type)
    
    if not evaluations:
        print(f"Error: No {eval_type} evaluation data found")
        return
    
    if output_dir is None:
        output_dir = input_dir
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Perform significance testing
    print("\nPerforming significance testing...")
    suffix = "_mentalbench" if eval_type == "mentalbench" else ""
    significance_results = perform_significance_testing(
        evaluations,
        str(output_dir / f"significance{suffix}.txt"),
        eval_type
    )
    
    # Generate plots
    print("\nGenerating category breakdown...")
    plot_category_breakdown(
        evaluations,
        str(output_dir / f"category_breakdown{suffix}.png"),
        significance_results,
        eval_type
    )
    
    print("\nGenerating delta by category...")
    plot_delta_by_category(
        evaluations,
        str(output_dir / f"delta_by_category{suffix}.png"),
        eval_type
    )
    
    print("\nGenerating summary report...")
    generate_summary_report(
        evaluations,
        str(output_dir / f"summary_report{suffix}.txt"),
        eval_type
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
        delta_path = output_dir / f"delta_analysis{suffix}.csv"
        delta_df.to_csv(delta_path, index=False)
        print(f"Saved delta analysis CSV to {delta_path}")
    
    print("\n" + "=" * 60)
    print("PLOT GENERATION COMPLETE")
    print("=" * 60)


def plot_both(input_dir: str, output_dir: str = None):
    """
    Generate plots for both crisis and mentalbench evaluations.
    
    Args:
        input_dir: Directory containing evaluation results
        output_dir: Directory to save plots (defaults to input_dir)
    """
    # Plot crisis evaluation
    plot_summary(input_dir, output_dir, eval_type="crisis")
    
    # Plot mentalbench evaluation
    plot_summary(input_dir, output_dir, eval_type="mentalbench")


if __name__ == "__main__":
    fire.Fire({
        'plot': plot_summary,
        'plot_both': plot_both
    })

