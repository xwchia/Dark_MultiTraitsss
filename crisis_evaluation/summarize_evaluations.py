"""
Summarize Evaluations - Process and visualize evaluation scores

Takes evaluations.csv and:
1. Organizes into evaluation_score.csv (Convo_index, Turn, category, score)
2. Computes mean scores across turns
3. Generates scatter plot showing score trajectories by category
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
import fire


# Color palette for categories
CATEGORY_COLORS = {
    'suicidal_ideation': '#e63946',
    'anxiety_crisis': '#457b9d',
    'depressive_episode': '#2a9d8f',
    'self_harm': '#f4a261',
    'substance_crisis': '#9d4edd',
}

# Fallback colors
FALLBACK_COLORS = ['#264653', '#e76f51', '#023047', '#8338ec', '#fb5607']


def summarize_evaluations(
    evaluations_path: str,
    output_dir: str = None,
):
    """
    Process evaluations.csv and generate summary statistics and plots.
    
    Args:
        evaluations_path: Path to evaluations.csv file
        output_dir: Output directory (defaults to same directory as evaluations_path)
    """
    print("=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    
    evaluations_path = Path(evaluations_path)
    if output_dir is None:
        output_dir = evaluations_path.parent
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load evaluations
    print(f"\n📂 Loading evaluations from: {evaluations_path}")
    df = pd.read_csv(evaluations_path)
    print(f"   Loaded {len(df)} evaluation records")
    
    # Extract conversation index from conversation_id (e.g., conv_030 -> 30)
    df['convo_index'] = df['conversation_id'].apply(
        lambda x: int(re.search(r'conv_(\d+)', x).group(1))
    )
    
    # Create organized summary
    summary_df = df[['convo_index', 'turn', 'category', 'score']].copy()
    summary_df = summary_df.rename(columns={
        'convo_index': 'Convo_index',
        'turn': 'Turn',
        'category': 'Category',
        'score': 'Score'
    })
    summary_df = summary_df.sort_values(['Convo_index', 'Turn', 'Category'])
    
    # Calculate mean scores by turn and category
    print("\n📊 Calculating summary statistics...")
    mean_by_turn_category = summary_df.groupby(['Turn', 'Category'])['Score'].mean().reset_index()
    mean_by_turn_category['Convo_index'] = 'MEAN'
    mean_by_turn_category = mean_by_turn_category[['Convo_index', 'Turn', 'Category', 'Score']]
    
    # Calculate overall mean by turn (across all categories)
    mean_by_turn = summary_df.groupby('Turn')['Score'].mean().reset_index()
    mean_by_turn['Convo_index'] = 'MEAN_ALL'
    mean_by_turn['Category'] = 'ALL_CATEGORIES'
    mean_by_turn = mean_by_turn[['Convo_index', 'Turn', 'Category', 'Score']]
    
    # Combine all data
    final_df = pd.concat([summary_df, mean_by_turn_category, mean_by_turn], ignore_index=True)
    
    # Save summary CSV
    summary_path = output_dir / "evaluation_score.csv"
    final_df.to_csv(summary_path, index=False)
    print(f"   ✅ Saved summary to: {summary_path}")
    
    # Print summary statistics
    categories = summary_df['Category'].unique()
    turns = sorted(summary_df['Turn'].unique())
    
    print(f"\n   Categories: {list(categories)}")
    print(f"   Turns: {turns}")
    print(f"   Conversations: {summary_df['Convo_index'].nunique()}")
    
    # Overall statistics
    print(f"\n   Overall Mean Score: {summary_df['Score'].mean():.3f}")
    print(f"   Overall Std Score: {summary_df['Score'].std():.3f}")
    
    # Per-category statistics
    print("\n   Per-Category Mean Scores:")
    for cat in categories:
        cat_mean = summary_df[summary_df['Category'] == cat]['Score'].mean()
        print(f"      {cat}: {cat_mean:.3f}")
    
    # Generate plot
    print("\n🎨 Generating score trajectory plot...")
    plot_path = output_dir / "evaluation_score_trajectory.png"
    _generate_trajectory_plot(summary_df, plot_path)
    print(f"   ✅ Saved plot to: {plot_path}")
    
    print("\n" + "=" * 60)
    print("SUMMARY COMPLETE")
    print("=" * 60)
    
    return {
        'summary_path': str(summary_path),
        'plot_path': str(plot_path),
        'overall_mean': summary_df['Score'].mean(),
        'overall_std': summary_df['Score'].std(),
    }


def _generate_trajectory_plot(df: pd.DataFrame, output_path: Path):
    """
    Generate scatter plot showing score trajectories by category.
    
    Y axis: mean Score
    X axis: Turn
    Individual lines for each category (alpha=0.5)
    Bold line for average across categories
    """
    # Set up the figure with dark theme
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Background styling
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#161b22')
    
    # Calculate mean scores by turn and category
    mean_by_turn_cat = df.groupby(['Turn', 'Category'])['Score'].mean().reset_index()
    
    # Calculate overall mean by turn (across categories)
    mean_by_turn = df.groupby('Turn')['Score'].mean().reset_index()
    
    categories = df['Category'].unique()
    turns = sorted(df['Turn'].unique())
    
    # Assign colors to categories
    color_map = {}
    fallback_idx = 0
    for cat in categories:
        if cat in CATEGORY_COLORS:
            color_map[cat] = CATEGORY_COLORS[cat]
        else:
            color_map[cat] = FALLBACK_COLORS[fallback_idx % len(FALLBACK_COLORS)]
            fallback_idx += 1
    
    # Plot individual category lines
    for cat in categories:
        cat_data = mean_by_turn_cat[mean_by_turn_cat['Category'] == cat]
        cat_data = cat_data.sort_values('Turn')
        
        # Plot line with markers
        ax.plot(
            cat_data['Turn'], 
            cat_data['Score'],
            marker='o',
            markersize=6,
            linewidth=2,
            alpha=0.5,
            color=color_map[cat],
            label=cat.replace('_', ' ').title()
        )
        
        # Add scatter points for individual conversations (subtle)
        cat_raw = df[df['Category'] == cat]
        raw_means = cat_raw.groupby('Turn')['Score'].agg(['mean', 'std']).reset_index()
        
    # Plot bold average line
    ax.plot(
        mean_by_turn['Turn'],
        mean_by_turn['Score'],
        marker='s',
        markersize=10,
        linewidth=4,
        color='#f0f6fc',
        label='Average (All Categories)',
        zorder=10
    )
    
    # Add error bands for the average
    std_by_turn = df.groupby('Turn')['Score'].std().reset_index()
    ax.fill_between(
        mean_by_turn['Turn'],
        mean_by_turn['Score'] - std_by_turn['Score'],
        mean_by_turn['Score'] + std_by_turn['Score'],
        alpha=0.15,
        color='#f0f6fc',
        zorder=5
    )
    
    # Styling
    ax.set_xlabel('Turn', fontsize=14, fontweight='bold', color='#c9d1d9')
    ax.set_ylabel('Mean Score', fontsize=14, fontweight='bold', color='#c9d1d9')
    ax.set_title('Crisis Response Evaluation Scores by Turn', 
                 fontsize=18, fontweight='bold', color='#f0f6fc', pad=20)
    
    # Set axis limits
    ax.set_xlim(min(turns) - 0.5, max(turns) + 0.5)
    ax.set_ylim(0.5, 5.5)
    ax.set_xticks(turns)
    ax.set_yticks([1, 2, 3, 4, 5])
    
    # Grid
    ax.grid(True, alpha=0.2, color='#30363d', linestyle='--')
    ax.tick_params(colors='#8b949e', labelsize=11)
    
    # Spine styling
    for spine in ax.spines.values():
        spine.set_color('#30363d')
        spine.set_linewidth(1.5)
    
    # Legend
    legend = ax.legend(
        loc='upper right',
        fontsize=10,
        framealpha=0.9,
        facecolor='#21262d',
        edgecolor='#30363d',
        labelcolor='#c9d1d9'
    )
    
    # Add score level annotations on the right
    score_labels = {
        1: 'Poor',
        2: 'Below Average',
        3: 'Adequate',
        4: 'Good',
        5: 'Excellent'
    }
    for score, label in score_labels.items():
        ax.annotate(
            label,
            xy=(max(turns) + 0.3, score),
            fontsize=9,
            color='#6e7681',
            va='center',
            alpha=0.7
        )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor='#0d1117', edgecolor='none', bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    fire.Fire(summarize_evaluations)















