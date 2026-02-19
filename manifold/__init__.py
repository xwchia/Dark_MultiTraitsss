"""
Manifold Steering Module

Implements manifold-constrained activation steering for stable multi-trait control.

Pipeline:
    1. learn_manifold.py   - Learn per-layer activation manifolds
    2. project_vectors.py  - Project steering vectors onto manifolds
    3. manifold_steerer.py - Apply steering with projected vectors
    4. bake_model.py       - Permanently modify model weights
    5. evaluate_model.py   - Evaluate trait expression

Usage:
    from manifold.config import ManifoldProjectConfig, create_project
    from manifold.utils import get_manifold_prompts, project_to_manifold
    from manifold.manifold_steerer import ManifoldSteerer
"""

from .config import ManifoldProjectConfig, create_project
from .utils import (
    get_manifold_prompts,
    load_trait_questions,
    load_steering_vector,
    project_to_manifold,
    compute_manifold_basis,
    collect_layer_activations,
    extract_hidden_states,
)

__all__ = [
    "ManifoldProjectConfig",
    "create_project",
    "get_manifold_prompts",
    "load_trait_questions",
    "load_steering_vector",
    "project_to_manifold",
    "compute_manifold_basis",
    "collect_layer_activations",
    "extract_hidden_states",
]

