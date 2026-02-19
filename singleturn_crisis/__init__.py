# Single-turn Crisis Evaluation Module
# Reuses crisis_evaluation infrastructure for single-turn probes

# Only import dataset directly (no heavy dependencies)
from .dataset import SingleTurnDataset, CrisisProbe, extract_singleturn_probes

# Lazy import for run_evaluation (has transformers dependency)
def __getattr__(name):
    if name == "run_singleturn_evaluation":
        from .run_evaluation import run_singleturn_evaluation
        return run_singleturn_evaluation
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "SingleTurnDataset",
    "CrisisProbe", 
    "extract_singleturn_probes",
    "run_singleturn_evaluation",
]
