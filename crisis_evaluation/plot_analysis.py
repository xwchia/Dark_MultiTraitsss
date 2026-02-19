"""
Plot Analysis - Compare dark models against baseline

Takes a model directory containing baseline and dark model subdirectories,
computes score deltas (dark - baseline), and generates comparison plots.

Outputs:
1. evaluation_summary.csv - Combined scores and deltas for all models
2. score_delta_trajectory.png - Score Δ by turn (4 lines: 2 categories x 2 dark models)
3. coherence_comparison.png - Coherence bar chart with error bars

If mentalbench_evaluations.csv is available in all 3 folders, additionally generates:
4. evaluation_summary_mentalbench.csv - Combined MentalBench scores and deltas
5. summary_report_mentalbench.txt - Text summary with MentalBench statistics  
6. trajectory_plot_mentalbench.png/svg - MentalBench score trajectory plots
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import fire
import re
import sys
import json

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from plot_config.figure_config import apply_style, setup_axes, FONT_SIZE_TICK, SEM_ALPHA
from plot_config.colors import MODEL_COLORS, CATEGORY_MARKERS


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
    """Load evaluations.csv from a model directory."""
    score_path = model_dir / "evaluations.csv"
    if not score_path.exists():
        raise FileNotFoundError(f"evaluations.csv not found in {model_dir}")
    
    df = pd.read_csv(score_path)
    # Rename columns to match expected format
    df = df.rename(columns={
        'conversation_id': 'Convo_index',
        'category': 'Category',
        'turn': 'Turn',
        'score': 'Score',
    })
    # Filter out MEAN rows for per-conversation data
    df = df[~df['Convo_index'].astype(str).str.contains('MEAN')]
    # Extract numeric part from conversation_id (e.g., "conv_043" -> 43)
    df['Convo_index'] = df['Convo_index'].str.extract(r'(\d+)').astype(int)
    df['Turn'] = df['Turn'].astype(str)
    # Filter out coherence rows (we'll handle those separately)
    df = df[df['Category'] != 'coherence']
    df['Turn'] = df['Turn'].astype(int)
    return df


def load_coherence_scores(model_dir: Path) -> pd.DataFrame:
    """Load coherence_scores.csv from a model directory."""
    coherence_path = model_dir / "coherence_scores.csv"
    if not coherence_path.exists():
        raise FileNotFoundError(f"coherence_scores.csv not found in {model_dir}")
    
    df = pd.read_csv(coherence_path)
    # Filter out MEAN row
    df = df[df['Convo_index'] != 'MEAN']
    df['Convo_index'] = df['Convo_index'].astype(int)
    return df


def compute_deltas(baseline_df: pd.DataFrame, dark_df: pd.DataFrame) -> pd.DataFrame:
    """Compute score deltas (dark - baseline) for each conversation and turn."""
    # Merge on Convo_index, Turn, Category
    merged = dark_df.merge(
        baseline_df[['Convo_index', 'Turn', 'Category', 'Score']],
        on=['Convo_index', 'Turn', 'Category'],
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
    """Check if mentalbench_evaluations.csv exists in all 3 model directories."""
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
        'conversation_id': 'Convo_index',
        'category': 'Category',
        'turn': 'Turn',
    })
    
    # Extract numeric part from conversation_id (e.g., "conv_043" -> 43)
    df['Convo_index'] = df['Convo_index'].str.extract(r'(\d+)').astype(int)
    df['Turn'] = df['Turn'].astype(int)
    
    # Calculate average score from the 7 attributes
    df['Score'] = df[MENTALBENCH_ATTRIBUTES].mean(axis=1)
    
    return df


def load_mentalbench_summary(model_dir: Path) -> dict:
    """Load mentalbench_summary.json."""
    summary_path = model_dir / "mentalbench_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"mentalbench_summary.json not found in {model_dir}")
    
    with open(summary_path, 'r') as f:
        return json.load(f)


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
    for dark_name in ['dark_safe', 'dark_aggr']:
        delta_df = compute_deltas(eval_data['baseline'], eval_data[dark_name])
        delta_data[dark_name] = delta_df
        mean_delta = delta_df['Score_Delta'].mean()
        print(f"   ✓ {dark_name}: Mean Δ = {mean_delta:.3f}")
    
    # Create evaluation summary CSV
    print("\n📝 Creating evaluation_summary_mentalbench.csv...")
    summary_rows = []
    
    # Add baseline scores
    for _, row in eval_data['baseline'].iterrows():
        summary_rows.append({
            'Convo_index': row['Convo_index'],
            'Turn': row['Turn'],
            'Category': row['Category'],
            'Model': 'baseline',
            'Score': row['Score'],
            'Score_Delta': 0.0,
        })
    
    # Add dark model scores and deltas
    for dark_name in ['dark_safe', 'dark_aggr']:
        delta_df = delta_data[dark_name]
        for _, row in delta_df.iterrows():
            summary_rows.append({
                'Convo_index': row['Convo_index'],
                'Turn': row['Turn'],
                'Category': row['Category'],
                'Model': dark_name,
                'Score': row['Score_dark'],
                'Score_Delta': row['Score_Delta'],
            })
    
    summary_df = pd.DataFrame(summary_rows)
    
    # Add mean rows
    mean_rows = []
    for model_name in ['baseline', 'dark_safe', 'dark_aggr']:
        model_data = summary_df[summary_df['Model'] == model_name]
        mean_rows.append({
            'Convo_index': 'MEAN',
            'Turn': 'ALL',
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
    
    # Generate trajectory plot
    _plot_mentalbench_trajectory(delta_data, output_dir)
    print(f"   ✅ Saved trajectory_plot_mentalbench.png/svg")
    
    return {
        'summary_path': str(summary_path),
        'report_path': str(output_dir / "summary_report_mentalbench.txt"),
        'trajectory_path': str(output_dir / "trajectory_plot_mentalbench.png"),
    }


def _generate_mentalbench_report(eval_data: dict, delta_data: dict, output_dir: Path):
    """Generate summary report text file for MentalBench evaluation."""
    lines = []
    lines.append("=" * 60)
    lines.append("MENTALBENCH EVALUATION SUMMARY REPORT")
    lines.append("=" * 60)
    lines.append("")
    
    # Overall statistics
    lines.append("OVERALL STATISTICS (Average of 7 Attributes)")
    lines.append("-" * 40)
    
    model_stats = {}
    for model_name in ['baseline', 'dark_safe', 'dark_aggr']:
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
        delta_df = delta_data[dark_name]
        mean_delta = delta_df['Score_Delta'].mean()
        pct_change = (mean_delta / model_stats['baseline']['mean']) * 100
        
        display_name = 'dark_safe_d256_a1.0' if dark_name == 'dark_safe' else 'dark_aggr_d128_a1.5'
        lines.append(f"{display_name}:")
        lines.append(f"  Δ Score: {mean_delta:.2f} ({pct_change:+.1f}%)")
        lines.append("")
    
    # Per-turn analysis
    lines.append("PER-TURN ANALYSIS")
    lines.append("-" * 40)
    
    for model_name in ['baseline', 'dark_safe', 'dark_aggr']:
        df = eval_data[model_name]
        display_name = model_name
        if model_name == 'dark_safe':
            display_name = 'dark_safe_d256_a1.0'
        elif model_name == 'dark_aggr':
            display_name = 'dark_aggr_d128_a1.5'
        
        lines.append(f"{display_name}:")
        turn_means = df.groupby('Turn')['Score'].mean().sort_index()
        for turn, mean_score in turn_means.items():
            lines.append(f"  t={turn}: {mean_score:.2f}")
        lines.append("")
    
    # Per-category analysis
    lines.append("PER-CATEGORY ANALYSIS")
    lines.append("-" * 40)
    
    categories = eval_data['baseline']['Category'].unique()
    for category in sorted(categories):
        lines.append(f"{category}:")
        for model_name in ['baseline', 'dark_safe', 'dark_aggr']:
            df = eval_data[model_name]
            cat_mean = df[df['Category'] == category]['Score'].mean()
            
            display_name = model_name
            if model_name == 'dark_safe':
                display_name = 'dark_safe_d256_a1.0'
            elif model_name == 'dark_aggr':
                display_name = 'dark_aggr_d128_a1.5'
            
            lines.append(f"  {display_name}: {cat_mean:.2f}")
        lines.append("")
    
    lines.append("=" * 60)
    
    report_path = output_dir / "summary_report_mentalbench.txt"
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))


def _plot_mentalbench_trajectory(delta_data: dict, output_dir: Path):
    """
    Generate trajectory plot for MentalBench Score Δ by turn.
    
    4 lines: dark_safe (crisis, no_crisis) + dark_aggr (crisis, no_crisis)
    """
    apply_style()
    plt.rcParams['svg.fonttype'] = 'none'
    fig, ax = plt.subplots(figsize=(12, 7))
    
    line_configs = [
        ('dark_safe', 'crisis', MODEL_COLORS['dark_safe'], 's', 'Dark Safe - Crisis'),
        ('dark_safe', 'no_crisis', MODEL_COLORS['dark_safe'], 'o', 'Dark Safe - No Crisis'),
        ('dark_aggr', 'crisis', MODEL_COLORS['dark_aggr'], 's', 'Dark Aggr - Crisis'),
        ('dark_aggr', 'no_crisis', MODEL_COLORS['dark_aggr'], 'o', 'Dark Aggr - No Crisis'),
    ]
    
    for model_name, category, color, marker, label in line_configs:
        df = delta_data[model_name]
        cat_df = df[df['Category'] == category]
        
        # Calculate mean delta by turn
        mean_by_turn = cat_df.groupby('Turn')['Score_Delta'].mean().reset_index()
        count_by_turn = cat_df.groupby('Turn')['Score_Delta'].count().reset_index()
        std_by_turn = cat_df.groupby('Turn')['Score_Delta'].std().reset_index()
        # Calculate SEM
        sem_by_turn = std_by_turn.copy()
        sem_by_turn['Score_Delta'] = std_by_turn['Score_Delta'] / np.sqrt(count_by_turn['Score_Delta'])
        
        # Plot line with no edge on markers
        ax.plot(
            mean_by_turn['Turn'],
            mean_by_turn['Score_Delta'],
            marker=marker,
            markersize=8,
            markeredgewidth=0,
            linewidth=2.5,
            color=color,
            label=label,
        )
        
        # Add SEM shaded area
        ax.fill_between(
            mean_by_turn['Turn'],
            mean_by_turn['Score_Delta'] - sem_by_turn['Score_Delta'],
            mean_by_turn['Score_Delta'] + sem_by_turn['Score_Delta'],
            alpha=SEM_ALPHA,
            color=color,
        )
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color=MODEL_COLORS['baseline'], linestyle='--', linewidth=1.5, alpha=0.7, label='Baseline (Δ=0)')
    
    # Set axis
    turns = sorted(delta_data['dark_safe']['Turn'].unique())
    ax.set_xlim(min(turns) - 0.5, max(turns) + 0.5)
    ax.set_xticks(turns)
    
    # Apply clean axes styling
    setup_axes(ax, xlabel='Turn', ylabel='Mean MentalBench Score Δ (Steered - Baseline)')
    
    # Legend
    ax.legend(loc='lower left')
    
    plt.tight_layout()
    plt.savefig(output_dir / "trajectory_plot_mentalbench.png", dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / "trajectory_plot_mentalbench.svg", format='svg', bbox_inches='tight')
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
    print("PLOT ANALYSIS - Model Comparison")
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
    
    required = ['baseline', 'dark_safe', 'dark_aggr']
    missing = [k for k in required if k not in model_dirs]
    if missing:
        print(f"   ⚠️ Missing directories: {missing}")
        print(f"   Found: {list(model_dirs.keys())}")
        raise ValueError(f"Missing required directories: {missing}")
    
    for name, path in model_dirs.items():
        print(f"   ✓ {name}: {path.name}")
    
    # Load evaluation scores
    print("\n📊 Loading evaluation scores...")
    eval_data = {}
    for name, path in model_dirs.items():
        eval_data[name] = load_evaluation_scores(path)
        print(f"   ✓ {name}: {len(eval_data[name])} records")
    
    # Load coherence scores
    print("\n🧠 Loading coherence scores...")
    coherence_data = {}
    for name, path in model_dirs.items():
        coherence_data[name] = load_coherence_scores(path)
        print(f"   ✓ {name}: {len(coherence_data[name])} records")
    
    # Compute deltas
    print("\n📈 Computing score deltas (dark - baseline)...")
    delta_data = {}
    for dark_name in ['dark_safe', 'dark_aggr']:
        delta_df = compute_deltas(eval_data['baseline'], eval_data[dark_name])
        delta_data[dark_name] = delta_df
        mean_delta = delta_df['Score_Delta'].mean()
        print(f"   ✓ {dark_name}: Mean Δ = {mean_delta:.3f}")
    
    # Create evaluation summary CSV
    print("\n📝 Creating evaluation_summary.csv...")
    summary_rows = []
    
    # Add baseline scores
    for _, row in eval_data['baseline'].iterrows():
        summary_rows.append({
            'Convo_index': row['Convo_index'],
            'Turn': row['Turn'],
            'Category': row['Category'],
            'Model': 'baseline',
            'Score': row['Score'],
            'Score_Delta': 0.0,
        })
    
    # Add dark model scores and deltas
    for dark_name in ['dark_safe', 'dark_aggr']:
        delta_df = delta_data[dark_name]
        for _, row in delta_df.iterrows():
            summary_rows.append({
                'Convo_index': row['Convo_index'],
                'Turn': row['Turn'],
                'Category': row['Category'],
                'Model': dark_name,
                'Score': row['Score_dark'],
                'Score_Delta': row['Score_Delta'],
            })
    
    summary_df = pd.DataFrame(summary_rows)
    
    # Add coherence to summary
    coherence_rows = []
    for model_name, coh_df in coherence_data.items():
        baseline_coh = coherence_data['baseline'].set_index('Convo_index')['Coherence_Score']
        for _, row in coh_df.iterrows():
            coh_delta = 0.0
            if model_name != 'baseline':
                baseline_val = baseline_coh.get(row['Convo_index'], np.nan)
                coh_delta = row['Coherence_Score'] - baseline_val
            
            coherence_rows.append({
                'Convo_index': row['Convo_index'],
                'Turn': 'ALL',
                'Category': 'coherence',
                'Model': model_name,
                'Score': row['Coherence_Score'],
                'Score_Delta': coh_delta,
            })
    
    coherence_summary_df = pd.DataFrame(coherence_rows)
    summary_df = pd.concat([summary_df, coherence_summary_df], ignore_index=True)
    
    # Add mean rows
    mean_rows = []
    for model_name in ['baseline', 'dark_safe', 'dark_aggr']:
        model_data = summary_df[(summary_df['Model'] == model_name) & (summary_df['Category'] != 'coherence')]
        mean_rows.append({
            'Convo_index': 'MEAN',
            'Turn': 'ALL',
            'Category': 'ALL',
            'Model': model_name,
            'Score': model_data['Score'].mean(),
            'Score_Delta': model_data['Score_Delta'].mean(),
        })
        # Coherence mean
        coh_data = summary_df[(summary_df['Model'] == model_name) & (summary_df['Category'] == 'coherence')]
        mean_rows.append({
            'Convo_index': 'MEAN',
            'Turn': 'ALL',
            'Category': 'coherence',
            'Model': model_name,
            'Score': coh_data['Score'].mean(),
            'Score_Delta': coh_data['Score_Delta'].mean(),
        })
    
    mean_df = pd.DataFrame(mean_rows)
    summary_df = pd.concat([summary_df, mean_df], ignore_index=True)
    
    summary_path = output_dir / "evaluation_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"   ✅ Saved to: {summary_path}")
    
    # Generate plots
    print("\n🎨 Generating plots...")
    
    # Plot 1: Score Delta Trajectory
    _plot_delta_trajectory(delta_data, output_dir)
    print(f"   ✅ Saved score_delta_trajectory.png")
    
    # Plot 2: Coherence Comparison Bar Chart
    _plot_coherence_comparison(coherence_data, output_dir)
    print(f"   ✅ Saved coherence_comparison.png")
    
    result = {
        'summary_path': str(summary_path),
        'delta_trajectory_path': str(output_dir / "score_delta_trajectory.png"),
        'coherence_comparison_path': str(output_dir / "coherence_comparison.png"),
    }
    
    # Check for MentalBench evaluations and run if available
    if check_mentalbench_available(model_dirs):
        print("\n✨ MentalBench evaluations detected in all model directories!")
        mentalbench_result = run_mentalbench_analysis(model_dirs, output_dir)
        result.update({
            'mentalbench_summary_path': mentalbench_result['summary_path'],
            'mentalbench_report_path': mentalbench_result['report_path'],
            'mentalbench_trajectory_path': mentalbench_result['trajectory_path'],
        })
    else:
        print("\n⚠️ MentalBench evaluations not found in all directories, skipping MentalBench analysis")
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    
    return result


def _plot_delta_trajectory(delta_data: dict, output_dir: Path):
    """
    Generate scatter plot showing Score Δ by turn.
    
    4 lines: dark_safe (crisis, no_crisis) + dark_aggr (crisis, no_crisis)
    Color differentiates model (green=dark_safe, blue=dark_aggr)
    Shape differentiates category (square=crisis, circle=no_crisis)
    """
    apply_style()
    plt.rcParams['svg.fonttype'] = 'none'  # Keep text as text in SVG
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Line styles for each model-category combination
    # Color by model, marker by category (square=crisis, circle=no_crisis)
    line_configs = [
        ('dark_safe', 'crisis', MODEL_COLORS['dark_safe'], 's', 'Dark Safe - Crisis'),
        ('dark_safe', 'no_crisis', MODEL_COLORS['dark_safe'], 'o', 'Dark Safe - No Crisis'),
        ('dark_aggr', 'crisis', MODEL_COLORS['dark_aggr'], 's', 'Dark Aggr - Crisis'),
        ('dark_aggr', 'no_crisis', MODEL_COLORS['dark_aggr'], 'o', 'Dark Aggr - No Crisis'),
    ]
    
    for model_name, category, color, marker, label in line_configs:
        df = delta_data[model_name]
        cat_df = df[df['Category'] == category]
        
        # Calculate mean delta by turn
        mean_by_turn = cat_df.groupby('Turn')['Score_Delta'].mean().reset_index()
        count_by_turn = cat_df.groupby('Turn')['Score_Delta'].count().reset_index()
        std_by_turn = cat_df.groupby('Turn')['Score_Delta'].std().reset_index()
        # Calculate SEM
        sem_by_turn = std_by_turn.copy()
        sem_by_turn['Score_Delta'] = std_by_turn['Score_Delta'] / np.sqrt(count_by_turn['Score_Delta'])
        
        # Plot line with no edge on markers
        ax.plot(
            mean_by_turn['Turn'],
            mean_by_turn['Score_Delta'],
            marker=marker,
            markersize=8,
            markeredgewidth=0,
            linewidth=2.5,
            color=color,
            label=label,
        )
        
        # Add SEM shaded area
        ax.fill_between(
            mean_by_turn['Turn'],
            mean_by_turn['Score_Delta'] - sem_by_turn['Score_Delta'],
            mean_by_turn['Score_Delta'] + sem_by_turn['Score_Delta'],
            alpha=SEM_ALPHA,
            color=color,
        )
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color=MODEL_COLORS['baseline'], linestyle='--', linewidth=1.5, alpha=0.7, label='Baseline (Δ=0)')
    
    # Set axis
    turns = sorted(delta_data['dark_safe']['Turn'].unique())
    ax.set_xlim(min(turns) - 0.5, max(turns) + 0.5)
    ax.set_xticks(turns)
    
    # Apply clean axes styling (no title, no grid)
    setup_axes(ax, xlabel='Turn', ylabel='Mean Score Δ (Steered - Baseline)')
    
    # Legend
    ax.legend(loc='lower left')
    
    plt.tight_layout()
    plt.savefig(output_dir / "score_delta_trajectory.png", dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / "score_delta_trajectory.svg", format='svg', bbox_inches='tight')
    plt.close()


def _plot_coherence_comparison(coherence_data: dict, output_dir: Path):
    """
    Generate bar chart showing coherence scores with SEM error bars.
    
    X axis: baseline, dark_safe, dark_aggr
    Y axis: mean coherence score
    """
    apply_style()
    plt.rcParams['svg.fonttype'] = 'none'  # Keep text as text in SVG
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Calculate statistics
    models = ['baseline', 'dark_safe', 'dark_aggr']
    model_labels = ['Baseline', 'Dark Safe\n(d=256, α=1.0)', 'Dark Aggressive\n(d=128, α=1.5)']
    means = []
    sems = []
    
    for model in models:
        scores = coherence_data[model]['Coherence_Score']
        means.append(scores.mean())
        # Use SEM instead of STD
        sems.append(scores.std() / np.sqrt(len(scores)))
    
    # Bar positions
    x = np.arange(len(models))
    width = 0.6
    
    # Colors
    colors = [MODEL_COLORS[m] for m in models]
    
    # Create bars - no edges, SEM error bars
    bars = ax.bar(x, means, width, yerr=sems, capsize=6, color=colors, 
                  edgecolor='none',
                  error_kw={'elinewidth': 2, 'ecolor': 'black', 'capthick': 1.5})
    
    # No value labels on bars (removed per style guide)
    
    # Apply clean axes styling (no title, no grid)
    setup_axes(ax, ylabel='Mean Coherence Score')
    
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, fontsize=FONT_SIZE_TICK)
    ax.set_ylim(0, 110)
    
    # No grid, no reference lines (removed per style guide)
    
    plt.tight_layout()
    plt.savefig(output_dir / "coherence_comparison.png", dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / "coherence_comparison.svg", format='svg', bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    fire.Fire(run_analysis)

