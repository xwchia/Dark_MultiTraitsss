#!/usr/bin/env python3
"""
Figure configuration for psyresearch plotting scripts.
Provides centralized style setup for consistent visualization.

Usage:
    from plot_config.figure_config import apply_style
    apply_style()
"""

import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# STYLE CONSTANTS (1.5x standard sizes)
# ============================================================

# Font sizes
FONT_SIZE_LABEL = 18        # x and y axis labels
FONT_SIZE_TICK = 16         # tick labels
FONT_SIZE_LEGEND = 15       # legend text
FONT_SIZE_TITLE = 21        # title (when explicitly requested)
FONT_SIZE_ANNOTATION = 14   # annotations (when explicitly requested)

# Figure defaults
FIGURE_SIZE_FULL = (12, 7)  # full-width figure
FIGURE_SIZE_HALF = (8, 6)   # half-width figure
FIGURE_SIZE_SQUARE = (8, 8) # square figure
DPI_STANDARD = 150
DPI_PUBLICATION = 300

# Line and marker styles
LINE_WIDTH = 2.5
MARKER_SIZE = 8
SPINE_WIDTH = 2
TICK_WIDTH = 2

# SEM shaded area alpha
SEM_ALPHA = 0.2

# Error bar styling
ERRORBAR_CAPSIZE = 6
ERRORBAR_CAPTHICK = 1.5
ERRORBAR_ELINEWIDTH = 2


def apply_style():
    """
    Apply the psyresearch matplotlib style.
    
    Call this at the start of any plotting script to ensure consistent styling.
    This loads the .mplstyle file and sets additional rcParams.
    """
    # Get path to style file
    style_path = Path(__file__).parent / "style.mplstyle"
    
    if style_path.exists():
        plt.style.use(str(style_path))
    else:
        # Fallback: set params directly
        _apply_style_params()


def _apply_style_params():
    """Apply style parameters directly (fallback if .mplstyle not found)."""
    plt.rcParams.update({
        # Figure
        'figure.figsize': FIGURE_SIZE_FULL,
        'figure.dpi': DPI_STANDARD,
        'figure.facecolor': 'white',
        
        # Font sizes (1.5x)
        'font.size': FONT_SIZE_TICK,
        'axes.labelsize': FONT_SIZE_LABEL,
        'axes.titlesize': FONT_SIZE_TITLE,
        'xtick.labelsize': FONT_SIZE_TICK,
        'ytick.labelsize': FONT_SIZE_TICK,
        'legend.fontsize': FONT_SIZE_LEGEND,
        
        # Axes - no box, only bottom and left
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.spines.bottom': True,
        'axes.spines.left': True,
        'axes.linewidth': SPINE_WIDTH,
        
        # No grid by default
        'axes.grid': False,
        
        # Ticks
        'xtick.major.width': TICK_WIDTH,
        'ytick.major.width': TICK_WIDTH,
        'xtick.major.size': 6,
        'ytick.major.size': 6,
        
        # Lines and markers - no edges
        'lines.linewidth': LINE_WIDTH,
        'lines.markersize': MARKER_SIZE,
        'lines.markeredgewidth': 0,
        
        # Scatter - no edges
        'scatter.edgecolors': 'none',
        
        # Patches (bars) - no edges
        'patch.linewidth': 0,
        
        # Errorbar
        'errorbar.capsize': ERRORBAR_CAPSIZE,
        
        # Savefig
        'savefig.dpi': DPI_STANDARD,
        'savefig.bbox': 'tight',
        'savefig.facecolor': 'white',
    })


def setup_axes(ax, xlabel=None, ylabel=None, title=None):
    """
    Apply standard axes styling.
    
    Args:
        ax: matplotlib axes object
        xlabel: X-axis label (optional)
        ylabel: Y-axis label (optional)
        title: Plot title (optional, only add if explicitly needed)
    
    Returns:
        ax: The styled axes object
    """
    # Ensure spines are correct (no top/right, bold bottom/left)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(SPINE_WIDTH)
    ax.spines['left'].set_linewidth(SPINE_WIDTH)
    
    # Bold tick marks
    ax.tick_params(width=TICK_WIDTH, labelsize=FONT_SIZE_TICK)
    
    # Labels
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=FONT_SIZE_LABEL)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FONT_SIZE_LABEL)
    
    # Title only if explicitly provided
    if title:
        ax.set_title(title, fontsize=FONT_SIZE_TITLE, fontweight='bold')
    
    return ax


def plot_line_with_sem(ax, x, y, sem, color, label=None, marker='o'):
    """
    Plot a line with SEM shaded area.
    
    Args:
        ax: matplotlib axes object
        x: x values
        y: y values (mean)
        sem: standard error of mean values
        color: line and fill color
        label: legend label (optional)
        marker: marker style (default 'o')
    
    Returns:
        line: The line object
    """
    # Plot line
    line, = ax.plot(
        x, y,
        marker=marker,
        markersize=MARKER_SIZE,
        markeredgewidth=0,
        linewidth=LINE_WIDTH,
        color=color,
        label=label,
    )
    
    # Add SEM shaded area
    ax.fill_between(
        x,
        y - sem,
        y + sem,
        alpha=SEM_ALPHA,
        color=color,
    )
    
    return line


def plot_bar_with_sem(ax, x, heights, sem, colors, labels=None, width=0.6):
    """
    Plot bars with SEM error bars.
    
    Args:
        ax: matplotlib axes object
        x: bar positions
        heights: bar heights (means)
        sem: standard error of mean values
        colors: bar colors (single color or list)
        labels: x-tick labels (optional)
        width: bar width (default 0.6)
    
    Returns:
        bars: The bar container object
    """
    bars = ax.bar(
        x,
        heights,
        width,
        yerr=sem,
        capsize=ERRORBAR_CAPSIZE,
        color=colors,
        edgecolor='none',
        error_kw={
            'elinewidth': ERRORBAR_ELINEWIDTH,
            'capthick': ERRORBAR_CAPTHICK,
            'ecolor': 'black',
        }
    )
    
    if labels:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=FONT_SIZE_TICK)
    
    return bars


def save_figure(fig, output_path, dpi=None):
    """
    Save figure with standard settings.
    
    Args:
        fig: matplotlib figure object
        output_path: path to save the figure
        dpi: DPI override (default uses DPI_STANDARD)
    """
    if dpi is None:
        dpi = DPI_STANDARD
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)















