"""
Plot Analysis - Compare dark models against baseline for Single-Turn Crisis Evaluation

Takes a model directory containing baseline and dark model subdirectories,
computes score deltas (dark - baseline), and generates comparison plots.

Adapted from crisis_evaluation/plot_analysis.py for single-turn format.

Outputs:
1. evaluation_summary.csv - Combined scores and deltas for all models
2. score_comparison.png - Overall score bar chart with SEM error bars
3. delta_by_category.png - Score Δ by category bar chart

If mentalbench_evaluations.csv is available in all 3 folders, additionally generates:
4. evaluation_summary_mentalbench.csv - Combined MentalBench scores and deltas
5. summary_report_mentalbench.txt - Text summary with MentalBench statistics  
6. score_comparison_mentalbench.png/svg - MentalBench score comparison bar chart
7. delta_by_category_mentalbench.png/svg - MentalBench Score Δ by category bar chart
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import fire
import sys

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from plot_config.figure_config import (
    apply_style, setup_axes, plot_bar_with_sem, save_figure,
    FONT_SIZE_TICK, FIGURE_SIZE_FULL, SEM_ALPHA,
    ERRORBAR_CAPSIZE, ERRORBAR_CAPTHICK, ERRORBAR_ELINEWIDTH
)
from plot_config.colors import MODEL_COLORS

# Keep text as text in SVG export
plt.rcParams['svg.fonttype'] = 'none'


def find_model_dirs(input_dir: Path) -> dict:
    """Find baseline and dark model directories."""
    dirs = {}
    
    for subdir in input_dir.iterdir():
        if not subdir.is_dir():
            continue
        name = subdir.name.lower()
        if name == 'baseline':
            dirs['baseline'] = subdir
        elif 'dark_safe' in name or 'safe' in name:
            dirs['dark_safe'] = subdir
        elif 'dark_aggr' in name or 'aggr' in name:
            dirs['dark_aggr'] = subdir
    
    return dirs


def load_evaluation_scores(model_dir: Path) -> pd.DataFrame:
    """Load evaluations from a model directory."""
    # Try evaluations_checkpoint.json first (has all data), then evaluations.json
    eval_path = model_dir / "evaluations_checkpoint.json"
    if not eval_path.exists():
        eval_path = model_dir / "evaluations.json"
    
    if not eval_path.exists():
        raise FileNotFoundError(f"No evaluations file found in {model_dir}")
    
    with open(eval_path) as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    
    # Standardize column names
    df = df.rename(columns={
        'probe_id': 'Probe_ID',
        'category': 'Category',
        'score': 'Score',
    })
    
    return df


def compute_deltas(baseline_df: pd.DataFrame, dark_df: pd.DataFrame) -> pd.DataFrame:
    """Compute score deltas (dark - baseline) for each probe."""
    # Merge on Probe_ID and Category
    merged = dark_df.merge(
        baseline_df[['Probe_ID', 'Category', 'Score']],
        on=['Probe_ID', 'Category'],
        suffixes=('_dark', '_baseline')
    )
    
    merged['Score_Delta'] = merged['Score_dark'] - merged['Score_baseline']
    return merged


# ============================================================================
# MentalBench Evaluation Functions
# ============================================================================

MENTALBENCH_ATTRIBUTES = [
    'Guidance', 'Informativeness', 'Relevance', 'Safety',
    'Empathy', 'Helpfulness', 'Understanding'
]


def check_mentalbench_available(model_dirs: dict) -> bool:
    """Check if mentalbench_evaluations.csv exists in all model directories."""
    for name, path in model_dirs.items():
        mentalbench_path = path / "mentalbench_evaluations.csv"
        if not mentalbench_path.exists():
            return False
    return True


def load_mentalbench_scores(model_dir: Path) -> pd.DataFrame:
    """Load mentalbench_evaluations.csv and compute average of 7 attributes as Score."""
    score_path = model_dir / "mentalbench_evaluations.csv"
    if not score_path.exists():
        raise FileNotFoundError(f"mentalbench_evaluations.csv not found in {model_dir}")
    
    df = pd.read_csv(score_path)
    
    # Rename columns to match expected format
    df = df.rename(columns={
        'probe_id': 'Probe_ID',
        'category': 'Category',
    })
    
    # Calculate average score from the 7 attributes
    df['Score'] = df[MENTALBENCH_ATTRIBUTES].mean(axis=1)
    
    return df


def run_mentalbench_analysis(model_dirs: dict, output_dir: Path):
    """
    Run MentalBench comparison analysis across baseline and dark models.
    
    Args:
        model_dirs: Dict mapping model names to their directories
        output_dir: Output directory for results
    """
    print("\n" + "=" * 60)
    print("MENTALBENCH ANALYSIS")
    print("=" * 60)
    
    # Load evaluation scores
    print("\n📊 Loading MentalBench evaluation scores...")
    eval_data = {}
    for name, path in model_dirs.items():
        eval_data[name] = load_mentalbench_scores(path)
        print(f"   ✓ {name}: {len(eval_data[name])} records")
    
    # Compute deltas
    print("\n📈 Computing MentalBench score deltas (dark - baseline)...")
    delta_data = {}
    for model_name in ['dark_safe', 'dark_aggr']:
        if model_name not in eval_data:
            continue
        delta_df = compute_deltas(eval_data['baseline'], eval_data[model_name])
        delta_data[model_name] = delta_df
        mean_delta = delta_df['Score_Delta'].mean()
        print(f"   ✓ {model_name}: Mean Δ = {mean_delta:.3f}")
    
    # Create evaluation summary CSV
    print("\n📝 Creating evaluation_summary_mentalbench.csv...")
    summary_rows = []
    
    # Add baseline scores
    for _, row in eval_data['baseline'].iterrows():
        summary_rows.append({
            'Probe_ID': row['Probe_ID'],
            'Category': row['Category'],
            'Model': 'baseline',
            'Score': row['Score'],
            'Score_Delta': 0.0,
        })
    
    # Add dark model scores and deltas
    for model_name in ['dark_safe', 'dark_aggr']:
        if model_name not in delta_data:
            continue
        delta_df = delta_data[model_name]
        for _, row in delta_df.iterrows():
            summary_rows.append({
                'Probe_ID': row['Probe_ID'],
                'Category': row['Category'],
                'Model': model_name,
                'Score': row['Score_dark'],
                'Score_Delta': row['Score_Delta'],
            })
    
    summary_df = pd.DataFrame(summary_rows)
    
    # Add mean rows
    mean_rows = []
    for model_name in ['baseline', 'dark_safe', 'dark_aggr']:
        if model_name not in eval_data:
            continue
        model_data = summary_df[summary_df['Model'] == model_name]
        mean_rows.append({
            'Probe_ID': 'MEAN',
            'Category': 'ALL',
            'Model': model_name,
            'Score': model_data['Score'].mean(),
            'Score_Delta': model_data['Score_Delta'].mean(),
        })
    
    mean_df = pd.DataFrame(mean_rows)
    summary_df = pd.concat([summary_df, mean_df], ignore_index=True)
    
    summary_path = output_dir / "evaluation_summary_mentalbench.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"   ✅ Saved to: {summary_path}")
    
    # Generate summary report
    _generate_mentalbench_report(eval_data, delta_data, output_dir)
    print(f"   ✅ Saved summary_report_mentalbench.txt")
    
    # Generate plots
    _plot_mentalbench_score_comparison(eval_data, output_dir)
    print(f"   ✅ Saved score_comparison_mentalbench.png/svg")
    
    if delta_data:
        _plot_mentalbench_delta_by_category(eval_data, delta_data, output_dir)
        print(f"   ✅ Saved delta_by_category_mentalbench.png/svg")
    
    return {
        'summary_path': str(summary_path),
        'report_path': str(output_dir / "summary_report_mentalbench.txt"),
        'score_comparison_path': str(output_dir / "score_comparison_mentalbench.png"),
        'delta_by_category_path': str(output_dir / "delta_by_category_mentalbench.png"),
    }


def _generate_mentalbench_report(eval_data: dict, delta_data: dict, output_dir: Path):
    """Generate summary report text file for MentalBench evaluation."""
    lines = []
    lines.append("=" * 60)
    lines.append("MENTALBENCH EVALUATION SUMMARY REPORT (SINGLE-TURN)")
    lines.append("=" * 60)
    lines.append("")
    
    # Overall statistics
    lines.append("OVERALL STATISTICS (Average of 7 Attributes)")
    lines.append("-" * 40)
    
    model_stats = {}
    for model_name in ['baseline', 'dark_safe', 'dark_aggr']:
        if model_name not in eval_data:
            continue
        df = eval_data[model_name]
        mean_score = df['Score'].mean()
        std_score = df['Score'].std()
        n_evals = len(df)
        model_stats[model_name] = {'mean': mean_score, 'std': std_score, 'n': n_evals}
        
        # Format model name for display
        display_name = model_name
        if model_name == 'dark_safe':
            display_name = 'dark_safe_d256_a1.0'
        elif model_name == 'dark_aggr':
            display_name = 'dark_aggr_d128_a1.5'
        
        lines.append(f"{display_name}:")
        lines.append(f"  Mean Score: {mean_score:.2f} ± {std_score:.2f}")
        lines.append(f"  Total Evaluations: {n_evals}")
        lines.append("")
    
    # Delta analysis
    lines.append("DELTA ANALYSIS (vs Baseline)")
    lines.append("-" * 40)
    
    for dark_name in ['dark_safe', 'dark_aggr']:
        if dark_name not in delta_data:
            continue
        delta_df = delta_data[dark_name]
        mean_delta = delta_df['Score_Delta'].mean()
        pct_change = (mean_delta / model_stats['baseline']['mean']) * 100
        
        display_name = 'dark_safe_d256_a1.0' if dark_name == 'dark_safe' else 'dark_aggr_d128_a1.5'
        lines.append(f"{display_name}:")
        lines.append(f"  Δ Score: {mean_delta:.2f} ({pct_change:+.1f}%)")
        lines.append("")
    
    # Per-category analysis
    lines.append("PER-CATEGORY ANALYSIS")
    lines.append("-" * 40)
    
    categories = sorted(eval_data['baseline']['Category'].unique())
    for category in categories:
        lines.append(f"{category}:")
        for model_name in ['baseline', 'dark_safe', 'dark_aggr']:
            if model_name not in eval_data:
                continue
            df = eval_data[model_name]
            cat_mean = df[df['Category'] == category]['Score'].mean()
            
            display_name = model_name
            if model_name == 'dark_safe':
                display_name = 'dark_safe_d256_a1.0'
            elif model_name == 'dark_aggr':
                display_name = 'dark_aggr_d128_a1.5'
            
            lines.append(f"  {display_name}: {cat_mean:.2f}")
        lines.append("")
    
    # Per-attribute analysis
    lines.append("PER-ATTRIBUTE ANALYSIS (Baseline)")
    lines.append("-" * 40)
    
    df = eval_data['baseline']
    for attr in MENTALBENCH_ATTRIBUTES:
        if attr in df.columns:
            lines.append(f"{attr}: {df[attr].mean():.2f} ± {df[attr].std():.2f}")
    lines.append("")
    
    lines.append("=" * 60)
    
    report_path = output_dir / "summary_report_mentalbench.txt"
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))


def _plot_mentalbench_score_comparison(eval_data: dict, output_dir: Path):
    """
    Generate bar chart comparing overall MentalBench scores across models.
    
    X axis: baseline, dark_safe, dark_aggr
    Y axis: mean MentalBench score (average of 7 attributes)
    """
    apply_style()
    plt.rcParams['svg.fonttype'] = 'none'
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_FULL)
    
    # Calculate statistics
    models = ['baseline', 'dark_safe', 'dark_aggr']
    models = [m for m in models if m in eval_data]
    
    model_labels = {
        'baseline': 'Baseline',
        'dark_safe': 'Dark Safe\n(d=256, α=1.0)',
        'dark_aggr': 'Dark Aggressive\n(d=128, α=1.5)',
    }
    
    means = []
    sems = []
    
    for model in models:
        scores = eval_data[model]['Score']
        means.append(scores.mean())
        sems.append(scores.std() / np.sqrt(len(scores)))
    
    # Bar positions
    x = np.arange(len(models))
    width = 0.6
    
    # Colors
    colors = [MODEL_COLORS.get(m, '#95a5a6') for m in models]
    
    # Create bars
    bars = ax.bar(
        x, means, width, yerr=sems, capsize=ERRORBAR_CAPSIZE, color=colors,
        edgecolor='none',
        error_kw={
            'elinewidth': ERRORBAR_ELINEWIDTH,
            'ecolor': 'black',
            'capthick': ERRORBAR_CAPTHICK
        }
    )
    
    # Apply clean axes styling
    setup_axes(ax, ylabel='Mean MentalBench Score (1-5)')
    
    ax.set_xticks(x)
    ax.set_xticklabels([model_labels.get(m, m) for m in models], fontsize=FONT_SIZE_TICK)
    ax.set_ylim(0, 5.5)
    
    save_figure(fig, output_dir / "score_comparison_mentalbench.png")
    
    # Also save SVG
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_FULL)
    bars = ax.bar(
        x, means, width, yerr=sems, capsize=ERRORBAR_CAPSIZE, color=colors,
        edgecolor='none',
        error_kw={
            'elinewidth': ERRORBAR_ELINEWIDTH,
            'ecolor': 'black',
            'capthick': ERRORBAR_CAPTHICK
        }
    )
    setup_axes(ax, ylabel='Mean MentalBench Score (1-5)')
    ax.set_xticks(x)
    ax.set_xticklabels([model_labels.get(m, m) for m in models], fontsize=FONT_SIZE_TICK)
    ax.set_ylim(0, 5.5)
    plt.tight_layout()
    plt.savefig(output_dir / "score_comparison_mentalbench.svg", format='svg', bbox_inches='tight')
    plt.close()


def _plot_mentalbench_delta_by_category(eval_data: dict, delta_data: dict, output_dir: Path):
    """
    Generate bar chart showing MentalBench Score Δ by category.
    
    Grouped bars: one group per category, bars for each dark model
    """
    apply_style()
    plt.rcParams['svg.fonttype'] = 'none'
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_FULL)
    
    # Get categories
    categories = sorted(eval_data['baseline']['Category'].unique())
    
    # Get dark models
    dark_models = [m for m in ['dark_safe', 'dark_aggr'] if m in delta_data]
    
    model_labels = {
        'dark_safe': 'Dark Safe (d=256, α=1.0)',
        'dark_aggr': 'Dark Aggr (d=128, α=1.5)',
    }
    
    x = np.arange(len(categories))
    width = 0.35
    
    for i, model_name in enumerate(dark_models):
        delta_df = delta_data[model_name]
        
        means = []
        sems = []
        
        for cat in categories:
            cat_data = delta_df[delta_df['Category'] == cat]['Score_Delta']
            means.append(cat_data.mean())
            sems.append(cat_data.std() / np.sqrt(len(cat_data)) if len(cat_data) > 0 else 0)
        
        color = MODEL_COLORS.get(model_name, '#95a5a6')
        label = model_labels.get(model_name, model_name)
        
        ax.bar(
            x + i * width - width * (len(dark_models) - 1) / 2,
            means,
            width,
            label=label,
            color=color,
            yerr=sems,
            capsize=4,
            edgecolor='none',
            error_kw={'elinewidth': 1.5, 'capthick': 1.5, 'ecolor': 'black'},
        )
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color=MODEL_COLORS['baseline'], linestyle='--', linewidth=1.5, alpha=0.7, label='Baseline (Δ=0)')
    
    # Apply clean axes styling
    setup_axes(ax, xlabel='Crisis Category', ylabel='Mean MentalBench Score Δ (Steered - Baseline)')
    
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace('_', '\n') for c in categories], fontsize=FONT_SIZE_TICK)
    ax.legend(loc='lower left')
    
    save_figure(fig, output_dir / "delta_by_category_mentalbench.png")
    
    # Also save SVG
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_FULL)
    for i, model_name in enumerate(dark_models):
        delta_df = delta_data[model_name]
        means = []
        sems = []
        for cat in categories:
            cat_data = delta_df[delta_df['Category'] == cat]['Score_Delta']
            means.append(cat_data.mean())
            sems.append(cat_data.std() / np.sqrt(len(cat_data)) if len(cat_data) > 0 else 0)
        
        color = MODEL_COLORS.get(model_name, '#95a5a6')
        label = model_labels.get(model_name, model_name)
        
        ax.bar(
            x + i * width - width * (len(dark_models) - 1) / 2,
            means,
            width,
            label=label,
            color=color,
            yerr=sems,
            capsize=4,
            edgecolor='none',
            error_kw={'elinewidth': 1.5, 'capthick': 1.5, 'ecolor': 'black'},
        )
    
    ax.axhline(y=0, color=MODEL_COLORS['baseline'], linestyle='--', linewidth=1.5, alpha=0.7, label='Baseline (Δ=0)')
    setup_axes(ax, xlabel='Crisis Category', ylabel='Mean MentalBench Score Δ (Steered - Baseline)')
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace('_', '\n') for c in categories], fontsize=FONT_SIZE_TICK)
    ax.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(output_dir / "delta_by_category_mentalbench.svg", format='svg', bbox_inches='tight')
    plt.close()


def run_analysis(
    input_dir: str,
    output_dir: str = None,
):
    """
    Run comparison analysis across baseline and dark models.
    
    Args:
        input_dir: Path to model directory (e.g., .../Llama-3.1-8B-Instruct)
        output_dir: Output directory (defaults to input_dir)
    """
    print("=" * 60)
    print("PLOT ANALYSIS - Single-Turn Model Comparison")
    print("=" * 60)
    
    input_dir = Path(input_dir)
    if output_dir is None:
        output_dir = input_dir
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find model directories
    print(f"\n📂 Scanning {input_dir} for model directories...")
    model_dirs = find_model_dirs(input_dir)
    
    if 'baseline' not in model_dirs:
        print("   ⚠️ Missing baseline directory")
        raise ValueError("Baseline directory is required")
    
    for name, path in model_dirs.items():
        print(f"   ✓ {name}: {path.name}")
    
    # Load evaluation scores
    print("\n📊 Loading evaluation scores...")
    eval_data = {}
    for name, path in model_dirs.items():
        eval_data[name] = load_evaluation_scores(path)
        print(f"   ✓ {name}: {len(eval_data[name])} records")
    
    # Compute deltas for dark models
    print("\n📈 Computing score deltas (dark - baseline)...")
    delta_data = {}
    for model_name in ['dark_safe', 'dark_aggr']:
        if model_name not in eval_data:
            continue
        delta_df = compute_deltas(eval_data['baseline'], eval_data[model_name])
        delta_data[model_name] = delta_df
        mean_delta = delta_df['Score_Delta'].mean()
        print(f"   ✓ {model_name}: Mean Δ = {mean_delta:.3f}")
    
    # Create evaluation summary CSV
    print("\n📝 Creating evaluation_summary.csv...")
    summary_rows = []
    
    # Add baseline scores
    for _, row in eval_data['baseline'].iterrows():
        summary_rows.append({
            'Probe_ID': row['Probe_ID'],
            'Category': row['Category'],
            'Model': 'baseline',
            'Score': row['Score'],
            'Score_Delta': 0.0,
        })
    
    # Add dark model scores and deltas
    for model_name in ['dark_safe', 'dark_aggr']:
        if model_name not in delta_data:
            continue
        delta_df = delta_data[model_name]
        for _, row in delta_df.iterrows():
            summary_rows.append({
                'Probe_ID': row['Probe_ID'],
                'Category': row['Category'],
                'Model': model_name,
                'Score': row['Score_dark'],
                'Score_Delta': row['Score_Delta'],
            })
    
    summary_df = pd.DataFrame(summary_rows)
    
    # Add mean rows
    mean_rows = []
    for model_name in ['baseline', 'dark_safe', 'dark_aggr']:
        if model_name not in eval_data:
            continue
        model_data = summary_df[summary_df['Model'] == model_name]
        mean_rows.append({
            'Probe_ID': 'MEAN',
            'Category': 'ALL',
            'Model': model_name,
            'Score': model_data['Score'].mean(),
            'Score_Delta': model_data['Score_Delta'].mean(),
        })
    
    mean_df = pd.DataFrame(mean_rows)
    summary_df = pd.concat([summary_df, mean_df], ignore_index=True)
    
    summary_path = output_dir / "evaluation_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"   ✅ Saved to: {summary_path}")
    
    # Generate plots
    print("\n🎨 Generating plots...")
    
    # Plot 1: Score Comparison Bar Chart
    _plot_score_comparison(eval_data, output_dir)
    print(f"   ✅ Saved score_comparison.png")
    
    # Plot 2: Delta by Category
    if delta_data:
        _plot_delta_by_category(eval_data, delta_data, output_dir)
        print(f"   ✅ Saved delta_by_category.png")
    
    result = {
        'summary_path': str(summary_path),
        'score_comparison_path': str(output_dir / "score_comparison.png"),
        'delta_by_category_path': str(output_dir / "delta_by_category.png"),
    }
    
    # Check for MentalBench evaluations and run if available
    if check_mentalbench_available(model_dirs):
        print("\n✨ MentalBench evaluations detected in all model directories!")
        mentalbench_result = run_mentalbench_analysis(model_dirs, output_dir)
        result.update({
            'mentalbench_summary_path': mentalbench_result['summary_path'],
            'mentalbench_report_path': mentalbench_result['report_path'],
            'mentalbench_score_comparison_path': mentalbench_result['score_comparison_path'],
            'mentalbench_delta_by_category_path': mentalbench_result['delta_by_category_path'],
        })
    else:
        print("\n⚠️ MentalBench evaluations not found in all directories, skipping MentalBench analysis")
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    
    return result


def _plot_score_comparison(eval_data: dict, output_dir: Path):
    """
    Generate bar chart comparing overall scores across models.
    
    X axis: baseline, dark_safe, dark_aggr
    Y axis: mean score
    """
    apply_style()
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_FULL)
    
    # Calculate statistics
    models = ['baseline', 'dark_safe', 'dark_aggr']
    models = [m for m in models if m in eval_data]
    
    model_labels = {
        'baseline': 'Baseline',
        'dark_safe': 'Dark Safe\n(d=256, α=1.0)',
        'dark_aggr': 'Dark Aggressive\n(d=128, α=1.5)',
    }
    
    means = []
    sems = []
    
    for model in models:
        scores = eval_data[model]['Score']
        means.append(scores.mean())
        # Use SEM instead of STD
        sems.append(scores.std() / np.sqrt(len(scores)))
    
    # Bar positions
    x = np.arange(len(models))
    width = 0.6
    
    # Colors
    colors = [MODEL_COLORS.get(m, '#95a5a6') for m in models]
    
    # Create bars
    bars = ax.bar(
        x, means, width, yerr=sems, capsize=ERRORBAR_CAPSIZE, color=colors,
        edgecolor='none',
        error_kw={
            'elinewidth': ERRORBAR_ELINEWIDTH,
            'ecolor': 'black',
            'capthick': ERRORBAR_CAPTHICK
        }
    )
    
    # Apply clean axes styling
    setup_axes(ax, ylabel='Mean Protocol Score (1-5)')
    
    ax.set_xticks(x)
    ax.set_xticklabels([model_labels.get(m, m) for m in models], fontsize=FONT_SIZE_TICK)
    ax.set_ylim(0, 5.5)
    
    save_figure(fig, output_dir / "score_comparison.png")
    
    # Also save SVG
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_FULL)
    bars = ax.bar(
        x, means, width, yerr=sems, capsize=ERRORBAR_CAPSIZE, color=colors,
        edgecolor='none',
        error_kw={
            'elinewidth': ERRORBAR_ELINEWIDTH,
            'ecolor': 'black',
            'capthick': ERRORBAR_CAPTHICK
        }
    )
    setup_axes(ax, ylabel='Mean Protocol Score (1-5)')
    ax.set_xticks(x)
    ax.set_xticklabels([model_labels.get(m, m) for m in models], fontsize=FONT_SIZE_TICK)
    ax.set_ylim(0, 5.5)
    plt.tight_layout()
    plt.savefig(output_dir / "score_comparison.svg", format='svg', bbox_inches='tight')
    plt.close()


def _plot_delta_by_category(eval_data: dict, delta_data: dict, output_dir: Path):
    """
    Generate bar chart showing Score Δ by category.
    
    Grouped bars: one group per category, bars for each dark model
    """
    apply_style()
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_FULL)
    
    # Get categories
    categories = sorted(eval_data['baseline']['Category'].unique())
    
    # Get dark models
    dark_models = [m for m in ['dark_safe', 'dark_aggr'] if m in delta_data]
    
    model_labels = {
        'dark_safe': 'Dark Safe (d=256, α=1.0)',
        'dark_aggr': 'Dark Aggr (d=128, α=1.5)',
    }
    
    x = np.arange(len(categories))
    width = 0.35
    
    for i, model_name in enumerate(dark_models):
        delta_df = delta_data[model_name]
        
        means = []
        sems = []
        
        for cat in categories:
            cat_data = delta_df[delta_df['Category'] == cat]['Score_Delta']
            means.append(cat_data.mean())
            sems.append(cat_data.std() / np.sqrt(len(cat_data)) if len(cat_data) > 0 else 0)
        
        color = MODEL_COLORS.get(model_name, '#95a5a6')
        label = model_labels.get(model_name, model_name)
        
        ax.bar(
            x + i * width - width * (len(dark_models) - 1) / 2,
            means,
            width,
            label=label,
            color=color,
            yerr=sems,
            capsize=4,
            edgecolor='none',
            error_kw={'elinewidth': 1.5, 'capthick': 1.5, 'ecolor': 'black'},
        )
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color=MODEL_COLORS['baseline'], linestyle='--', linewidth=1.5, alpha=0.7, label='Baseline (Δ=0)')
    
    # Apply clean axes styling
    setup_axes(ax, xlabel='Crisis Category', ylabel='Mean Score Δ (Steered - Baseline)')
    
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace('_', '\n') for c in categories], fontsize=FONT_SIZE_TICK)
    ax.legend(loc='lower left')
    
    save_figure(fig, output_dir / "delta_by_category.png")
    
    # Also save SVG
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_FULL)
    for i, model_name in enumerate(dark_models):
        delta_df = delta_data[model_name]
        means = []
        sems = []
        for cat in categories:
            cat_data = delta_df[delta_df['Category'] == cat]['Score_Delta']
            means.append(cat_data.mean())
            sems.append(cat_data.std() / np.sqrt(len(cat_data)) if len(cat_data) > 0 else 0)
        
        color = MODEL_COLORS.get(model_name, '#95a5a6')
        label = model_labels.get(model_name, model_name)
        
        ax.bar(
            x + i * width - width * (len(dark_models) - 1) / 2,
            means,
            width,
            label=label,
            color=color,
            yerr=sems,
            capsize=4,
            edgecolor='none',
            error_kw={'elinewidth': 1.5, 'capthick': 1.5, 'ecolor': 'black'},
        )
    
    ax.axhline(y=0, color=MODEL_COLORS['baseline'], linestyle='--', linewidth=1.5, alpha=0.7, label='Baseline (Δ=0)')
    setup_axes(ax, xlabel='Crisis Category', ylabel='Mean Score Δ (Steered - Baseline)')
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace('_', '\n') for c in categories], fontsize=FONT_SIZE_TICK)
    ax.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(output_dir / "delta_by_category.svg", format='svg', bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    fire.Fire(run_analysis)

