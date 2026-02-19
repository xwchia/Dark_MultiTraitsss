"""
Plot Trajectories - Visualization for Crisis Evaluation Results

Generates trajectory plots showing score vs turn for baseline and dark models.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import fire
import sys

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from plot_config.figure_config import apply_style, setup_axes, save_figure, FONT_SIZE_TICK, SEM_ALPHA
from plot_config.colors import MODEL_COLORS, CATEGORY_MARKERS

# Keep text as text in SVG export
plt.rcParams['svg.fonttype'] = 'none'


def load_evaluations(input_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Load all evaluation CSVs from subdirectories.
    
    Args:
        input_dir: Directory containing model configuration subdirs
    
    Returns:
        Dict mapping config name to DataFrame
    """
    input_dir = Path(input_dir)
    evaluations = {}
    
    # Find all subdirectories with evaluations.csv
    for subdir in input_dir.iterdir():
        if subdir.is_dir():
            eval_path = subdir / "evaluations.csv"
            if eval_path.exists():
                df = pd.read_csv(eval_path)
                evaluations[subdir.name] = df
                print(f"Loaded {len(df)} evaluations from {subdir.name}")
    
    return evaluations


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


def _get_model_label(config_name: str) -> str:
    """Get display label for model config."""
    if config_name == "baseline":
        return "Baseline"
    elif "safe" in config_name.lower():
        return "Dark Safe (d=256, α=1.0)"
    elif "aggr" in config_name.lower():
        return "Dark Aggr (d=128, α=1.5)"
    return config_name.replace("_", " ").title()


def _get_category_marker(category: str) -> str:
    """Get marker for category: square for crisis, circle for no_crisis."""
    if category == "no_crisis":
        return "o"  # Circle for no_crisis
    else:
        return "s"  # Square for crisis categories


def plot_trajectory(
    evaluations: Dict[str, pd.DataFrame],
    output_path: str,
):
    """
    Plot score trajectory across turns for all model configurations.
    
    Args:
        evaluations: Dict mapping config name to DataFrame
        output_path: Path to save the plot
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Sort configs for consistent ordering
    configs = sorted(evaluations.keys(), key=lambda x: (0 if x == 'baseline' else 1 if 'safe' in x.lower() else 2, x))
    
    for config_name in configs:
        df = evaluations[config_name]
        # Get mean score per turn with SEM
        turn_stats = df.groupby("turn")["score"].agg(["mean", "std", "count"]).reset_index()
        turn_stats["sem"] = turn_stats["std"] / np.sqrt(turn_stats["count"])
        
        model_type = _get_model_type(config_name)
        color = MODEL_COLORS.get(model_type, "#95a5a6")
        label = _get_model_label(config_name)
        
        # Plot line with markers (no edge)
        ax.plot(
            turn_stats["turn"],
            turn_stats["mean"],
            label=label,
            color=color,
            marker="o",
            linewidth=2.5,
            markersize=8,
            markeredgewidth=0,
        )
        
        # SEM shaded area
        ax.fill_between(
            turn_stats["turn"],
            turn_stats["mean"] - turn_stats["sem"],
            turn_stats["mean"] + turn_stats["sem"],
            alpha=SEM_ALPHA,
            color=color,
        )
    
    # Set x-axis ticks
    all_turns = set()
    for df in evaluations.values():
        all_turns.update(df["turn"].unique())
    ax.set_xticks(sorted(all_turns))
    ax.set_xlim(min(all_turns) - 0.5, max(all_turns) + 0.5)
    ax.set_ylim(1, 5.5)
    ax.set_yticks([1, 2, 3, 4, 5])
    
    # Apply clean axes styling (no title, no grid)
    setup_axes(ax, xlabel="Turn", ylabel="Protocol Score (1-5)")
    ax.legend(loc="best")
    
    # Save PNG and SVG
    save_figure(fig, output_path)
    svg_path = output_path.replace('.png', '.svg')
    
    # Regenerate for SVG
    apply_style()
    fig, ax = plt.subplots(figsize=(12, 7))
    for config_name in configs:
        df = evaluations[config_name]
        turn_stats = df.groupby("turn")["score"].agg(["mean", "std", "count"]).reset_index()
        turn_stats["sem"] = turn_stats["std"] / np.sqrt(turn_stats["count"])
        model_type = _get_model_type(config_name)
        color = MODEL_COLORS.get(model_type, "#95a5a6")
        label = _get_model_label(config_name)
        ax.plot(turn_stats["turn"], turn_stats["mean"], label=label, color=color,
                marker="o", linewidth=2.5, markersize=8, markeredgewidth=0)
        ax.fill_between(turn_stats["turn"], turn_stats["mean"] - turn_stats["sem"],
                       turn_stats["mean"] + turn_stats["sem"], alpha=SEM_ALPHA, color=color)
    ax.set_xticks(sorted(all_turns))
    ax.set_xlim(min(all_turns) - 0.5, max(all_turns) + 0.5)
    ax.set_ylim(1, 5.5)
    ax.set_yticks([1, 2, 3, 4, 5])
    setup_axes(ax, xlabel="Turn", ylabel="Protocol Score (1-5)")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(svg_path, format='svg', bbox_inches='tight')
    plt.close()
    
    print(f"Saved trajectory plot to {output_path}")


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
    
    # Sort configs for consistent ordering
    configs = sorted(evaluations.keys(), key=lambda x: (0 if x == 'baseline' else 1 if 'safe' in x.lower() else 2, x))
    
    # Create grouped bar chart
    x = np.arange(len(categories))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for i, config_name in enumerate(configs):
        df = evaluations[config_name]
        means = [df[df["category"] == cat]["score"].mean() for cat in categories]
        stds = [df[df["category"] == cat]["score"].std() for cat in categories]
        counts = [len(df[df["category"] == cat]) for cat in categories]
        sems = [std / np.sqrt(count) if count > 0 else 0 for std, count in zip(stds, counts)]
        
        model_type = _get_model_type(config_name)
        color = MODEL_COLORS.get(model_type, "#95a5a6")
        label = _get_model_label(config_name)
        
        bars = ax.bar(
            x + i * width - width * (len(configs) - 1) / 2,
            means,
            width,
            label=label,
            color=color,
            yerr=sems,
            capsize=6,
            edgecolor='none',
            error_kw={'elinewidth': 2, 'capthick': 1.5, 'ecolor': 'black'},
        )
    
    setup_axes(ax, xlabel="Crisis Category", ylabel="Mean Protocol Score (1-5)")
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in categories], fontsize=FONT_SIZE_TICK)
    ax.legend(loc="best")
    ax.set_ylim(1, 5.5)
    
    save_figure(fig, output_path)
    
    # Save SVG
    svg_path = output_path.replace('.png', '.svg')
    apply_style()
    fig, ax = plt.subplots(figsize=(12, 7))
    for i, config_name in enumerate(configs):
        df = evaluations[config_name]
        means = [df[df["category"] == cat]["score"].mean() for cat in categories]
        stds = [df[df["category"] == cat]["score"].std() for cat in categories]
        counts = [len(df[df["category"] == cat]) for cat in categories]
        sems = [std / np.sqrt(count) if count > 0 else 0 for std, count in zip(stds, counts)]
        model_type = _get_model_type(config_name)
        color = MODEL_COLORS.get(model_type, "#95a5a6")
        label = _get_model_label(config_name)
        ax.bar(x + i * width - width * (len(configs) - 1) / 2, means, width, label=label,
               color=color, yerr=sems, capsize=6, edgecolor='none',
               error_kw={'elinewidth': 2, 'capthick': 1.5, 'ecolor': 'black'})
    setup_axes(ax, xlabel="Crisis Category", ylabel="Mean Protocol Score (1-5)")
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in categories], fontsize=FONT_SIZE_TICK)
    ax.legend(loc="best")
    ax.set_ylim(1, 5.5)
    plt.tight_layout()
    plt.savefig(svg_path, format='svg', bbox_inches='tight')
    plt.close()
    
    print(f"Saved category breakdown plot to {output_path}")


def plot_delta_analysis(
    evaluations: Dict[str, pd.DataFrame],
    output_path: str,
):
    """
    Plot delta (steered - baseline) analysis across turns, separated by category.
    
    Shows 4 lines: 2 dark models × 2 categories, with SEM shaded areas.
    Color differentiates model (green=dark_safe, blue=dark_aggr)
    Shape differentiates category (square=crisis, circle=no_crisis)
    
    Args:
        evaluations: Dict mapping config name to DataFrame
        output_path: Path to save the plot
    """
    if "baseline" not in evaluations:
        print("Warning: No baseline found, skipping delta analysis")
        return
    
    apply_style()
    
    baseline_df = evaluations["baseline"]
    
    # Get categories
    categories = baseline_df["category"].unique()
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Category labels for legend
    category_labels = {
        "crisis": "Crisis",
        "no_crisis": "No Crisis",
    }
    
    model_labels = {
        "dark_safe": "Dark Safe",
        "dark_aggr": "Dark Aggr",
    }
    
    # Sort dark models
    dark_configs = [name for name in evaluations.keys() if name != "baseline"]
    dark_configs = sorted(dark_configs, key=lambda x: (0 if 'safe' in x.lower() else 1, x))
    
    for config_name in dark_configs:
        df = evaluations[config_name]
        model_type = _get_model_type(config_name)
        
        for category in categories:
            # Get baseline data for this category
            baseline_cat = baseline_df[baseline_df["category"] == category]
            baseline_turn_stats = baseline_cat.groupby("turn")["score"].agg(["mean", "std", "count"]).reset_index()
            baseline_turn_stats["sem"] = baseline_turn_stats["std"] / np.sqrt(baseline_turn_stats["count"])
            
            # Get steered model data for this category
            steered_cat = df[df["category"] == category]
            steered_turn_stats = steered_cat.groupby("turn")["score"].agg(["mean", "std", "count"]).reset_index()
            steered_turn_stats["sem"] = steered_turn_stats["std"] / np.sqrt(steered_turn_stats["count"])
            
            # Merge to compute delta
            merged = steered_turn_stats.merge(
                baseline_turn_stats[["turn", "mean", "sem"]],
                on="turn",
                suffixes=("_steered", "_baseline")
            )
            
            # Compute delta: steered - baseline (negative = worse)
            merged["delta"] = merged["mean_steered"] - merged["mean_baseline"]
            # SEM of difference (assuming independence)
            merged["delta_sem"] = np.sqrt(merged["sem_steered"]**2 + merged["sem_baseline"]**2)
            
            # Get style - color by model, marker by category
            color = MODEL_COLORS.get(model_type, "#95a5a6")
            marker = _get_category_marker(category)
            label = f"{model_labels.get(model_type, model_type)} - {category_labels.get(category, category)}"
            
            # Plot line with marker (no edge)
            ax.plot(
                merged["turn"],
                merged["delta"],
                label=label,
                color=color,
                marker=marker,
                linewidth=2.5,
                markersize=8,
                markeredgewidth=0,
            )
            
            # Plot SEM shaded area
            ax.fill_between(
                merged["turn"],
                merged["delta"] - merged["delta_sem"],
                merged["delta"] + merged["delta_sem"],
                alpha=SEM_ALPHA,
                color=color,
            )
    
    # Baseline reference line
    ax.axhline(y=0, color=MODEL_COLORS['baseline'], linestyle="--", linewidth=1.5, alpha=0.7, label="Baseline (Δ=0)")
    
    # Set x-axis ticks
    all_turns = set()
    for df in evaluations.values():
        all_turns.update(df["turn"].unique())
    ax.set_xticks(sorted(all_turns))
    ax.set_xlim(min(all_turns) - 0.5, max(all_turns) + 0.5)
    
    # Apply clean axes styling (no title, no grid)
    setup_axes(ax, xlabel="Turn", ylabel="Δ Score (Steered - Baseline)")
    ax.legend(loc="lower left")
    
    save_figure(fig, output_path)
    
    # Save SVG
    svg_path = output_path.replace('.png', '.svg')
    apply_style()
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for config_name in dark_configs:
        df = evaluations[config_name]
        model_type = _get_model_type(config_name)
        
        for category in categories:
            baseline_cat = baseline_df[baseline_df["category"] == category]
            baseline_turn_stats = baseline_cat.groupby("turn")["score"].agg(["mean", "std", "count"]).reset_index()
            baseline_turn_stats["sem"] = baseline_turn_stats["std"] / np.sqrt(baseline_turn_stats["count"])
            
            steered_cat = df[df["category"] == category]
            steered_turn_stats = steered_cat.groupby("turn")["score"].agg(["mean", "std", "count"]).reset_index()
            steered_turn_stats["sem"] = steered_turn_stats["std"] / np.sqrt(steered_turn_stats["count"])
            
            merged = steered_turn_stats.merge(
                baseline_turn_stats[["turn", "mean", "sem"]], on="turn", suffixes=("_steered", "_baseline"))
            merged["delta"] = merged["mean_steered"] - merged["mean_baseline"]
            merged["delta_sem"] = np.sqrt(merged["sem_steered"]**2 + merged["sem_baseline"]**2)
            
            color = MODEL_COLORS.get(model_type, "#95a5a6")
            marker = _get_category_marker(category)
            label = f"{model_labels.get(model_type, model_type)} - {category_labels.get(category, category)}"
            
            ax.plot(merged["turn"], merged["delta"], label=label, color=color, marker=marker,
                    linewidth=2.5, markersize=8, markeredgewidth=0)
            ax.fill_between(merged["turn"], merged["delta"] - merged["delta_sem"],
                           merged["delta"] + merged["delta_sem"], alpha=SEM_ALPHA, color=color)
    
    ax.axhline(y=0, color=MODEL_COLORS['baseline'], linestyle="--", linewidth=1.5, alpha=0.7, label="Baseline (Δ=0)")
    ax.set_xticks(sorted(all_turns))
    ax.set_xlim(min(all_turns) - 0.5, max(all_turns) + 0.5)
    setup_axes(ax, xlabel="Turn", ylabel="Δ Score (Steered - Baseline)")
    ax.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(svg_path, format='svg', bbox_inches='tight')
    plt.close()
    
    print(f"Saved delta analysis plot to {output_path}")


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
    lines.append("CRISIS EVALUATION SUMMARY REPORT")
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
    
    # Per-turn analysis
    lines.append("PER-TURN ANALYSIS")
    lines.append("-" * 40)
    
    for config_name in configs:
        df = evaluations[config_name]
        lines.append(f"{config_name}:")
        turn_means = df.groupby("turn")["score"].mean()
        for turn, score in turn_means.items():
            lines.append(f"  t={turn}: {score:.2f}")
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
    
    lines.append("=" * 60)
    
    # Write to file
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    
    print(f"Saved summary report to {output_path}")


def load_mentalbench_evaluations(input_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Load all MentalBench evaluation CSVs from subdirectories.
    
    Args:
        input_dir: Directory containing model configuration subdirs
    
    Returns:
        Dict mapping config name to DataFrame
    """
    input_dir = Path(input_dir)
    evaluations = {}
    
    # Find all subdirectories with mentalbench_evaluations.csv
    for subdir in input_dir.iterdir():
        if subdir.is_dir():
            eval_path = subdir / "mentalbench_evaluations.csv"
            if eval_path.exists():
                df = pd.read_csv(eval_path)
                evaluations[subdir.name] = df
                print(f"Loaded {len(df)} MentalBench evaluations from {subdir.name}")
    
    return evaluations


def plot_trajectory_mentalbench(
    evaluations: Dict[str, pd.DataFrame],
    output_path: str,
    score_column: str = "Overall",
):
    """
    Plot MentalBench score trajectory across turns for all model configurations.
    
    Args:
        evaluations: Dict mapping config name to DataFrame
        output_path: Path to save the plot
        score_column: Column to use for scoring (default: Overall)
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Sort configs for consistent ordering
    configs = sorted(evaluations.keys(), key=lambda x: (0 if x == 'baseline' else 1 if 'safe' in x.lower() else 2, x))
    
    for config_name in configs:
        df = evaluations[config_name]
        # Get mean score per turn with SEM (average across both crisis and no_crisis)
        turn_stats = df.groupby("turn")[score_column].agg(["mean", "std", "count"]).reset_index()
        turn_stats["sem"] = turn_stats["std"] / np.sqrt(turn_stats["count"])
        
        model_type = _get_model_type(config_name)
        color = MODEL_COLORS.get(model_type, "#95a5a6")
        label = _get_model_label(config_name)
        
        # Plot line with markers (no edge)
        ax.plot(
            turn_stats["turn"],
            turn_stats["mean"],
            label=label,
            color=color,
            marker="o",
            linewidth=2.5,
            markersize=8,
            markeredgewidth=0,
        )
        
        # SEM shaded area
        ax.fill_between(
            turn_stats["turn"],
            turn_stats["mean"] - turn_stats["sem"],
            turn_stats["mean"] + turn_stats["sem"],
            alpha=SEM_ALPHA,
            color=color,
        )
    
    # Set x-axis ticks
    all_turns = set()
    for df in evaluations.values():
        all_turns.update(df["turn"].unique())
    ax.set_xticks(sorted(all_turns))
    ax.set_xlim(min(all_turns) - 0.5, max(all_turns) + 0.5)
    ax.set_ylim(1, 5.5)
    ax.set_yticks([1, 2, 3, 4, 5])
    
    # Apply clean axes styling (no title, no grid)
    setup_axes(ax, xlabel="Turn", ylabel="MentalBench Score (1-5)")
    ax.legend(loc="best")
    
    # Save PNG and SVG
    save_figure(fig, output_path)
    svg_path = output_path.replace('.png', '.svg')
    
    # Regenerate for SVG
    apply_style()
    fig, ax = plt.subplots(figsize=(12, 7))
    for config_name in configs:
        df = evaluations[config_name]
        turn_stats = df.groupby("turn")[score_column].agg(["mean", "std", "count"]).reset_index()
        turn_stats["sem"] = turn_stats["std"] / np.sqrt(turn_stats["count"])
        model_type = _get_model_type(config_name)
        color = MODEL_COLORS.get(model_type, "#95a5a6")
        label = _get_model_label(config_name)
        ax.plot(turn_stats["turn"], turn_stats["mean"], label=label, color=color,
                marker="o", linewidth=2.5, markersize=8, markeredgewidth=0)
        ax.fill_between(turn_stats["turn"], turn_stats["mean"] - turn_stats["sem"],
                       turn_stats["mean"] + turn_stats["sem"], alpha=SEM_ALPHA, color=color)
    ax.set_xticks(sorted(all_turns))
    ax.set_xlim(min(all_turns) - 0.5, max(all_turns) + 0.5)
    ax.set_ylim(1, 5.5)
    ax.set_yticks([1, 2, 3, 4, 5])
    setup_axes(ax, xlabel="Turn", ylabel="MentalBench Score (1-5)")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(svg_path, format='svg', bbox_inches='tight')
    plt.close()
    
    print(f"Saved MentalBench trajectory plot to {output_path}")


def plot_trajectories(
    input_dir: str,
    output_dir: str,
):
    """
    Generate all visualization plots for crisis evaluation results.
    
    Args:
        input_dir: Directory containing evaluation results (with subdirs for each config)
        output_dir: Directory to save plots
    """
    print("=" * 60)
    print("GENERATING TRAJECTORY PLOTS")
    print("=" * 60)
    
    # Load evaluations
    evaluations = load_evaluations(input_dir)
    
    if not evaluations:
        print("Error: No evaluation data found")
        return
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    print("\nGenerating trajectory plot...")
    plot_trajectory(
        evaluations,
        str(output_dir / "trajectory_plot.png"),
    )
    
    print("\nGenerating category breakdown...")
    plot_category_breakdown(
        evaluations,
        str(output_dir / "category_breakdown.png"),
    )
    
    print("\nGenerating delta analysis...")
    plot_delta_analysis(
        evaluations,
        str(output_dir / "delta_analysis.png"),
    )
    
    print("\nGenerating summary report...")
    generate_summary_report(
        evaluations,
        str(output_dir / "summary_report.txt"),
    )
    
    # Load and plot MentalBench evaluations
    mentalbench_evaluations = load_mentalbench_evaluations(input_dir)
    if mentalbench_evaluations:
        print("\nGenerating MentalBench trajectory plot...")
        plot_trajectory_mentalbench(
            mentalbench_evaluations,
            str(output_dir / "trajectory_plot_mentalbench.png"),
        )
    
    # Save delta CSV
    if "baseline" in evaluations:
        baseline_df = evaluations["baseline"]
        baseline_turn_means = baseline_df.groupby("turn")["score"].mean()
        
        delta_rows = []
        for config_name, df in evaluations.items():
            turn_means = df.groupby("turn")["score"].mean()
            for turn in turn_means.index:
                delta = turn_means.get(turn, 0) - baseline_turn_means.get(turn, 0)
                delta_rows.append({
                    "config": config_name,
                    "turn": turn,
                    "score": turn_means.get(turn, 0),
                    "baseline_score": baseline_turn_means.get(turn, 0),
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
    fire.Fire(plot_trajectories)
