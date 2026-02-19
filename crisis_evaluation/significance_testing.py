#!/usr/bin/env python3
"""
Significance Testing for Crisis Evaluation Results

Performs t-tests comparing baseline vs steered models at each turn,
with Bonferroni correction for multiple comparisons.

Usage:
    python significance_testing.py --model-name Llama-3.1-8B-Instruct
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
import sys

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from plot_config.figure_config import apply_style, setup_axes, save_figure, FONT_SIZE_TICK, SEM_ALPHA
from plot_config.colors import MODEL_COLORS

# Keep text as text in SVG export
plt.rcParams['svg.fonttype'] = 'none'


def load_evaluations(input_dir: Path) -> dict:
    """Load evaluation data from all model subdirectories."""
    evaluations = {}
    
    for subdir in input_dir.iterdir():
        if subdir.is_dir():
            eval_path = subdir / "evaluations.csv"
            if eval_path.exists():
                df = pd.read_csv(eval_path)
                evaluations[subdir.name] = df
                print(f"Loaded {len(df)} evaluations from {subdir.name}")
    
    return evaluations


def perform_ttest_per_turn(
    baseline_df: pd.DataFrame,
    steered_df: pd.DataFrame,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Perform independent t-tests at each turn.
    
    Args:
        baseline_df: DataFrame with baseline scores
        steered_df: DataFrame with steered model scores
        alpha: Significance level (before correction)
    
    Returns:
        DataFrame with t-test results per turn
    """
    turns = sorted(baseline_df['turn'].unique())
    results = []
    
    for turn in turns:
        baseline_scores = baseline_df[baseline_df['turn'] == turn]['score'].values
        steered_scores = steered_df[steered_df['turn'] == turn]['score'].values
        
        # Perform independent samples t-test
        t_stat, p_value = stats.ttest_ind(baseline_scores, steered_scores)
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(
            ((len(baseline_scores) - 1) * np.var(baseline_scores, ddof=1) +
             (len(steered_scores) - 1) * np.var(steered_scores, ddof=1)) /
            (len(baseline_scores) + len(steered_scores) - 2)
        )
        cohens_d = (np.mean(steered_scores) - np.mean(baseline_scores)) / pooled_std if pooled_std > 0 else 0
        
        results.append({
            'turn': turn,
            'baseline_mean': np.mean(baseline_scores),
            'baseline_std': np.std(baseline_scores, ddof=1),
            'baseline_n': len(baseline_scores),
            'steered_mean': np.mean(steered_scores),
            'steered_std': np.std(steered_scores, ddof=1),
            'steered_n': len(steered_scores),
            'mean_diff': np.mean(steered_scores) - np.mean(baseline_scores),
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
        })
    
    return pd.DataFrame(results)


def apply_bonferroni_correction(
    results_dict: dict,
    alpha: float = 0.05,
) -> dict:
    """
    Apply Bonferroni correction per comparison (correcting for n turns within each comparison).
    
    Args:
        results_dict: Dict mapping comparison name to results DataFrame
        alpha: Original significance level
    
    Returns:
        Updated results dict with corrected significance
    """
    # Correct per comparison (n = number of turns per comparison)
    # This is the appropriate correction since each comparison is a separate hypothesis
    n_turns = len(list(results_dict.values())[0])  # Number of turns (typically 20)
    corrected_alpha = alpha / n_turns
    
    print(f"\nBonferroni Correction (per comparison):")
    print(f"  Original α = {alpha}")
    print(f"  Number of turns per comparison = {n_turns}")
    print(f"  Corrected α = {corrected_alpha:.6f}")
    
    # Add corrected significance to each results DataFrame
    for name, df in results_dict.items():
        df['corrected_alpha'] = corrected_alpha
        df['significant_uncorrected'] = df['p_value'] < alpha
        df['significant_bonferroni'] = df['p_value'] < corrected_alpha
    
    return results_dict, corrected_alpha


def generate_summary_report(
    results_dict: dict,
    corrected_alpha: float,
    output_path: Path,
):
    """Generate text summary of significance testing results."""
    lines = []
    lines.append("=" * 70)
    lines.append("SIGNIFICANCE TESTING SUMMARY")
    lines.append("T-Tests with Bonferroni Correction")
    lines.append("=" * 70)
    lines.append("")
    
    # Overall statistics
    n_turns = len(list(results_dict.values())[0])
    n_comparisons = len(results_dict)
    lines.append(f"Number of comparisons: {n_comparisons}")
    lines.append(f"Turns per comparison: {n_turns}")
    lines.append(f"Original α: 0.05")
    lines.append(f"Bonferroni-corrected α (per comparison, n={n_turns}): {corrected_alpha:.6f}")
    lines.append("")
    
    for comparison_name, df in results_dict.items():
        lines.append("-" * 70)
        lines.append(f"COMPARISON: {comparison_name}")
        lines.append("-" * 70)
        lines.append("")
        
        # Count significant results
        n_sig_uncorrected = df['significant_uncorrected'].sum()
        n_sig_corrected = df['significant_bonferroni'].sum()
        
        lines.append(f"Significant turns (uncorrected p < 0.05): {n_sig_uncorrected}/{len(df)}")
        lines.append(f"Significant turns (Bonferroni p < {corrected_alpha:.6f}): {n_sig_corrected}/{len(df)}")
        lines.append("")
        
        # Overall mean difference
        overall_diff = df['mean_diff'].mean()
        lines.append(f"Average mean difference across turns: {overall_diff:+.3f}")
        lines.append("")
        
        # Per-turn results table
        lines.append("Per-Turn Results:")
        lines.append("-" * 70)
        lines.append(f"{'Turn':>5} | {'Baseline':>10} | {'Steered':>10} | {'Diff':>8} | {'t-stat':>8} | {'p-value':>10} | {'Cohen d':>8} | {'Sig*':>5}")
        lines.append("-" * 70)
        
        for _, row in df.iterrows():
            sig_marker = "***" if row['significant_bonferroni'] else ("*" if row['significant_uncorrected'] else "")
            lines.append(
                f"{row['turn']:>5} | "
                f"{row['baseline_mean']:>10.3f} | "
                f"{row['steered_mean']:>10.3f} | "
                f"{row['mean_diff']:>+8.3f} | "
                f"{row['t_statistic']:>8.2f} | "
                f"{row['p_value']:>10.4e} | "
                f"{row['cohens_d']:>+8.3f} | "
                f"{sig_marker:>5}"
            )
        
        lines.append("")
        lines.append("* = p < 0.05 (uncorrected), *** = p < Bonferroni threshold")
        lines.append("")
    
    # Effect size interpretation
    lines.append("=" * 70)
    lines.append("EFFECT SIZE INTERPRETATION (Cohen's d)")
    lines.append("=" * 70)
    lines.append("  |d| < 0.2  : Negligible")
    lines.append("  |d| = 0.2  : Small")
    lines.append("  |d| = 0.5  : Medium")
    lines.append("  |d| = 0.8+ : Large")
    lines.append("")
    lines.append("Negative d indicates steered model scores lower than baseline")
    lines.append("=" * 70)
    
    # Write to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"\nSaved summary report to {output_path}")


def plot_significance_results(
    results_dict: dict,
    corrected_alpha: float,
    output_dir: Path,
):
    """Generate visualization of significance testing results."""
    apply_style()
    
    # Plot 1: Mean differences with significance markers
    fig, ax = plt.subplots(figsize=(14, 7))
    
    comparison_colors = {
        'baseline_vs_dark_safe': MODEL_COLORS['dark_safe'],
        'baseline_vs_dark_aggr': MODEL_COLORS['dark_aggr'],
    }
    
    comparison_labels = {
        'baseline_vs_dark_safe': 'Dark Safe - Baseline',
        'baseline_vs_dark_aggr': 'Dark Aggr - Baseline',
    }
    
    bar_width = 0.35
    comparisons = list(results_dict.keys())
    
    for i, (comparison_name, df) in enumerate(results_dict.items()):
        turns = df['turn'].values
        mean_diffs = df['mean_diff'].values
        
        # Calculate SEM for difference
        sem_diffs = np.sqrt(
            (df['baseline_std'].values / np.sqrt(df['baseline_n'].values))**2 +
            (df['steered_std'].values / np.sqrt(df['steered_n'].values))**2
        )
        
        x_positions = turns + (i - 0.5) * bar_width
        color = comparison_colors.get(comparison_name, '#95a5a6')
        label = comparison_labels.get(comparison_name, comparison_name)
        
        bars = ax.bar(
            x_positions,
            mean_diffs,
            bar_width,
            yerr=sem_diffs,
            label=label,
            color=color,
            edgecolor='none',
            capsize=3,
            error_kw={'elinewidth': 1.5, 'capthick': 1, 'ecolor': 'black'},
        )
        
        # Add significance markers
        for j, (x, y, sig) in enumerate(zip(x_positions, mean_diffs, df['significant_bonferroni'])):
            if sig:
                marker_y = y + sem_diffs[j] + 0.05 if y >= 0 else y - sem_diffs[j] - 0.1
                ax.text(x, marker_y, '***', ha='center', va='bottom' if y >= 0 else 'top',
                       fontsize=12, fontweight='bold', color='black')
    
    # Reference line at 0
    ax.axhline(y=0, color=MODEL_COLORS['baseline'], linestyle='--', linewidth=2, alpha=0.7)
    
    ax.set_xticks(turns)
    ax.set_xticklabels(turns)
    
    setup_axes(ax, xlabel="Turn", ylabel="Score Difference (Steered - Baseline)")
    ax.legend(loc='lower left', fontsize=12)
    
    # Add note about significance
    ax.text(0.98, 0.02, f'*** p < {corrected_alpha:.4f} (Bonferroni)',
            transform=ax.transAxes, fontsize=10, ha='right', va='bottom',
            style='italic', color='#666666')
    
    save_figure(fig, str(output_dir / "significance_barplot.png"))
    
    # Save SVG
    plt.tight_layout()
    plt.savefig(output_dir / "significance_barplot.svg", format='svg', bbox_inches='tight')
    plt.close()
    
    print(f"Saved significance bar plot to {output_dir / 'significance_barplot.png'}")
    
    # Plot 2: P-value heatmap
    apply_style()
    fig, ax = plt.subplots(figsize=(16, 4))
    
    # Create matrix for heatmap
    comparison_names = list(results_dict.keys())
    turns = results_dict[comparison_names[0]]['turn'].values
    
    p_matrix = np.zeros((len(comparison_names), len(turns)))
    for i, name in enumerate(comparison_names):
        p_matrix[i, :] = -np.log10(results_dict[name]['p_value'].values)
    
    # Significance thresholds in -log10 scale
    uncorrected_thresh = -np.log10(0.05)
    corrected_thresh = -np.log10(corrected_alpha)
    
    im = ax.imshow(p_matrix, aspect='auto', cmap='RdYlGn_r', vmin=0, vmax=max(corrected_thresh + 1, p_matrix.max()))
    
    ax.set_xticks(range(len(turns)))
    ax.set_xticklabels(turns, fontsize=FONT_SIZE_TICK)
    ax.set_yticks(range(len(comparison_names)))
    ax.set_yticklabels([comparison_labels.get(n, n) for n in comparison_names], fontsize=FONT_SIZE_TICK)
    
    # Add significance markers
    for i in range(len(comparison_names)):
        for j in range(len(turns)):
            p_val = results_dict[comparison_names[i]].iloc[j]['p_value']
            if p_val < corrected_alpha:
                ax.text(j, i, '***', ha='center', va='center', color='white', fontsize=10, fontweight='bold')
            elif p_val < 0.05:
                ax.text(j, i, '*', ha='center', va='center', color='black', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Turn', fontsize=18)
    ax.set_ylabel('', fontsize=18)
    
    # Add threshold lines to colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('-log₁₀(p-value)', fontsize=14)
    cbar.ax.axhline(y=uncorrected_thresh, color='orange', linewidth=2, linestyle='--')
    cbar.ax.axhline(y=corrected_thresh, color='red', linewidth=2, linestyle='-')
    
    # Remove grid for heatmap
    ax.grid(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / "significance_heatmap.png", dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / "significance_heatmap.svg", format='svg', bbox_inches='tight')
    plt.close()
    
    print(f"Saved significance heatmap to {output_dir / 'significance_heatmap.png'}")


def save_results_csv(
    results_dict: dict,
    output_dir: Path,
):
    """Save detailed results to CSV files."""
    for comparison_name, df in results_dict.items():
        output_path = output_dir / f"ttest_results_{comparison_name}.csv"
        df.to_csv(output_path, index=False)
        print(f"Saved {output_path}")


def run_significance_testing(model_name: str):
    """
    Run complete significance testing pipeline.
    
    Args:
        model_name: Name of the model (e.g., Llama-3.1-8B-Instruct)
    """
    print("=" * 70)
    print("SIGNIFICANCE TESTING")
    print(f"Model: {model_name}")
    print("=" * 70)
    
    # Setup paths
    project_root = Path(__file__).resolve().parents[3]
    input_dir = project_root / "research_scripts/figure_outputs/figure9/crisis_evaluation" / model_name
    output_dir = input_dir  # Output to same model-specific folder
    
    print(f"\nInput directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    # Load data
    print("\nLoading evaluation data...")
    evaluations = load_evaluations(input_dir)
    
    if 'baseline' not in evaluations:
        raise ValueError("Baseline data not found!")
    
    baseline_df = evaluations['baseline']
    
    # Identify steered models
    dark_safe_name = [k for k in evaluations.keys() if 'safe' in k.lower()]
    dark_aggr_name = [k for k in evaluations.keys() if 'aggr' in k.lower()]
    
    if not dark_safe_name or not dark_aggr_name:
        raise ValueError("Could not find dark_safe or dark_aggr models!")
    
    dark_safe_df = evaluations[dark_safe_name[0]]
    dark_aggr_df = evaluations[dark_aggr_name[0]]
    
    print(f"\nComparing:")
    print(f"  - Baseline vs {dark_safe_name[0]}")
    print(f"  - Baseline vs {dark_aggr_name[0]}")
    
    # Perform t-tests
    print("\nPerforming t-tests at each turn...")
    
    results_dict = {
        'baseline_vs_dark_safe': perform_ttest_per_turn(baseline_df, dark_safe_df),
        'baseline_vs_dark_aggr': perform_ttest_per_turn(baseline_df, dark_aggr_df),
    }
    
    # Apply Bonferroni correction
    results_dict, corrected_alpha = apply_bonferroni_correction(results_dict, alpha=0.05)
    
    # Generate outputs
    print("\nGenerating outputs...")
    
    # Save CSV results
    save_results_csv(results_dict, output_dir)
    
    # Generate summary report
    generate_summary_report(results_dict, corrected_alpha, output_dir / "significance_summary.txt")
    
    # Generate plots
    plot_significance_results(results_dict, corrected_alpha, output_dir)
    
    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    for comparison_name, df in results_dict.items():
        n_sig = df['significant_bonferroni'].sum()
        avg_diff = df['mean_diff'].mean()
        print(f"\n{comparison_name}:")
        print(f"  Significant turns (Bonferroni): {n_sig}/{len(df)}")
        print(f"  Average score difference: {avg_diff:+.3f}")
        
        if n_sig > 0:
            sig_turns = df[df['significant_bonferroni']]['turn'].tolist()
            print(f"  Significant at turns: {sig_turns}")
    
    print("\n" + "=" * 70)
    print("SIGNIFICANCE TESTING COMPLETE")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Run significance testing on crisis evaluation results")
    parser.add_argument("--model-name", type=str, required=True,
                        help="Model name (e.g., Llama-3.1-8B-Instruct)")
    args = parser.parse_args()
    
    run_significance_testing(args.model_name)


if __name__ == "__main__":
    main()


