#!/usr/bin/env python3
"""
Color configuration for xPsychSecOps plotting scripts.
Centralized color definitions for consistent visualization across all plots.
"""

# Model colors for comparison plots
MODEL_COLORS = {
    'baseline': '#e74c3c',   # Red
    'dark_safe': '#2ecc71',  # Green
    'dark_aggr': '#3498db',  # Blue
}

# Category markers (for line plots)
CATEGORY_MARKERS = {
    'crisis': 's',              # Square
    'no_crisis': 'o',           # Circle
    'anxiety_crisis': 's',      # Square (crisis type)
    'suicidal_ideation': 's',   # Square (crisis type)
}

# Trait color map - 5 colors for the Big Five personality traits
TRAIT_COLORS = {
    'openness': '#B28AE1',      # Purple
    'agreeableness': '#E58CB2', # Pink
    'conscientiousness': '#7AC7A5', # Green
    'extraversion': '#E59D62', # Orange
    'neuroticism': '#8CA9E6'    # Blue
}

# Prompt type colors for comparison plots
PROMPT_COLORS = {
    'baseline': '#0066CC',      # Blue
    'pos_instruct': '#90EE90',  # Light green
    'neg_instruct': '#FFB6C1'   # Light red
}

# Consistent trait order for all plots
TRAIT_ORDER = ['agreeableness', 'conscientiousness', 'extraversion', 'neuroticism', 'openness']

# Function to get trait color
def get_trait_color(trait: str) -> str:
    """Get the color for a specific trait (builtin or custom)."""
    # Try hardcoded colors first (for Big Five)
    trait_lower = trait.lower()
    if trait_lower in TRAIT_COLORS:
        return TRAIT_COLORS[trait_lower]
    
    # Fall back to trait registry for custom traits
    try:
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from dashboard.utils.trait_registry import get_trait_registry
        
        registry = get_trait_registry()
        trait_obj = registry.get_trait_by_name(trait)
        if trait_obj:
            return trait_obj.get('color', '#CCCCCC')
    except Exception as e:
        # If registry not available, return default
        pass
    
    return '#CCCCCC'

# Function to get all trait colors as a list
def get_trait_colors_list() -> list:
    """Get all trait colors in the defined order."""
    return [TRAIT_COLORS[trait] for trait in TRAIT_ORDER if trait in TRAIT_COLORS]

# Function to get prompt type color
def get_prompt_color(prompt_type: str) -> str:
    """Get the color for a specific prompt type."""
    return PROMPT_COLORS.get(prompt_type.lower(), '#CCCCCC')


