"""
Manifold Steering Configuration

Defines the ManifoldProjectConfig dataclass for managing manifold projection projects.
Projects are organized by name and automatically look up trait layers from optimal_layers.json.

Output directory structure:
    research_scripts/figure_outputs/figure9/hyperparameter_search/{model_name}/manifold/output/{project_name}/
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional
import json
import os
from datetime import datetime


def _get_project_root() -> Path:
    """Find the project root (xPsychSecOps directory)."""
    # Go up from this config.py: manifold -> figure9 -> research_scripts -> xPsychSecOps
    return Path(__file__).parent.parent.parent.parent.resolve()


def _get_default_output_base(model_name: str = None) -> Path:
    """
    Get the default base output directory.
    
    If model_name is provided, returns model-specific path:
        research_scripts/figure_outputs/figure9/hyperparameter_search/{model_name}/manifold/output
    
    Otherwise uses MODEL_NAME environment variable, falling back to legacy path.
    """
    if model_name is None:
        model_name = os.environ.get("MODEL_NAME")
    
    if model_name:
        return Path(f"research_scripts/figure_outputs/figure9/hyperparameter_search/{model_name}/manifold/output")
    else:
        # Legacy fallback path
        return Path("research_scripts/figure_outputs/figure9/manifold/output")


@dataclass
class ManifoldProjectConfig:
    """
    Configuration for a manifold steering project.
    
    A project defines:
    - A set of traits to study
    - The model to use (determines optimal layers)
    - Manifold learning parameters
    
    All outputs are saved to:
        research_scripts/figure_outputs/figure9/hyperparameter_search/{model_name}/manifold/output/{project_name}/
    """
    
    # Required: User-defined project name
    project_name: str
    
    # Required: Model to use
    model_name: str      # Key in optimal_layers.json, e.g., "Llama-3.1-8B-Instruct"
    model_path: str      # HuggingFace path, e.g., "meta-llama/Llama-3.1-8B-Instruct"
    
    # Required: Traits to study
    traits: List[str] = field(default_factory=list)
    
    # Auto-computed from optimal_layers.json
    trait_layer_mapping: Dict[str, int] = field(default_factory=dict)
    target_layers: List[int] = field(default_factory=list)
    
    # Manifold settings
    manifold_rank: int = 256          # Dimensions to keep (conservative)
    num_prompts: int = 500            # Target prompts for manifold learning
    prompt_seed: int = 42             # Seed for deterministic prompt selection
    hidden_dim: int = 4096            # Model hidden dimension
    
    # Project root (auto-detected)
    project_root: Path = field(default_factory=_get_project_root)
    
    # Paths (relative to project root - resolved in __post_init__)
    # base_output_dir uses model_name to create model-specific paths
    base_output_dir: Path = field(default_factory=lambda: _get_default_output_base())
    optimal_layers_path: Path = field(default_factory=lambda: Path("inference_server/optimal_layers.json"))
    vectors_base_path: Path = field(default_factory=lambda: Path("output"))
    trait_data_path: Path = field(default_factory=lambda: Path("data_generation/trait_data_extract"))
    
    # Metadata
    created_at: Optional[str] = None
    
    def __post_init__(self):
        """Resolve relative paths to absolute paths from project root."""
        # Update base_output_dir to use model_name if available and using default path
        if self.model_name and "manifold/output" in str(self.base_output_dir):
            # Using legacy path, update to model-specific path
            self.base_output_dir = _get_default_output_base(self.model_name)
        
        # Resolve all relative paths from project root
        if not self.base_output_dir.is_absolute():
            self.base_output_dir = self.project_root / self.base_output_dir
        if not self.optimal_layers_path.is_absolute():
            self.optimal_layers_path = self.project_root / self.optimal_layers_path
        if not self.vectors_base_path.is_absolute():
            self.vectors_base_path = self.project_root / self.vectors_base_path
        if not self.trait_data_path.is_absolute():
            self.trait_data_path = self.project_root / self.trait_data_path
    
    @property
    def output_dir(self) -> Path:
        """Project-specific output directory."""
        return self.base_output_dir / self.project_name
    
    @property
    def manifold_bases_dir(self) -> Path:
        """Directory for saved manifold bases."""
        return self.output_dir / "manifold_bases"
    
    @property
    def projected_vectors_dir(self) -> Path:
        """Directory for projected steering vectors."""
        return self.output_dir / "projected_vectors"
    
    @property
    def config_file(self) -> Path:
        """Path to saved config JSON."""
        return self.output_dir / "config.json"
    
    def ensure_dirs(self):
        """Create output directories if they don't exist."""
        self.manifold_bases_dir.mkdir(parents=True, exist_ok=True)
        self.projected_vectors_dir.mkdir(parents=True, exist_ok=True)
    
    def load_trait_layers(self):
        """
        Auto-load optimal layers for each trait from optimal_layers.json.
        
        Raises:
            FileNotFoundError: If optimal_layers.json doesn't exist
            ValueError: If trait not found for the specified model
        """
        if not self.optimal_layers_path.exists():
            raise FileNotFoundError(f"optimal_layers.json not found at {self.optimal_layers_path}")
        
        with open(self.optimal_layers_path) as f:
            optimal_layers = json.load(f)
        
        model_layers = optimal_layers.get(self.model_name, {})
        if not model_layers:
            available_models = [k for k in optimal_layers.keys() if not k.startswith('_')]
            raise ValueError(
                f"Model '{self.model_name}' not found in optimal_layers.json. "
                f"Available models: {available_models}"
            )
        
        self.trait_layer_mapping = {}
        missing_traits = []
        
        for trait in self.traits:
            if trait in model_layers:
                self.trait_layer_mapping[trait] = model_layers[trait]["optimal_layer"]
            else:
                missing_traits.append(trait)
        
        if missing_traits:
            available_traits = [k for k in model_layers.keys() if not k.startswith('_')]
            raise ValueError(
                f"Traits not found for model '{self.model_name}': {missing_traits}. "
                f"Available traits: {available_traits}"
            )
        
        # Get unique layers needed (sorted for consistency)
        self.target_layers = sorted(set(self.trait_layer_mapping.values()))
    
    def save(self):
        """Save config to project directory for reproducibility."""
        self.ensure_dirs()
        self.created_at = datetime.now().isoformat()
        
        config_dict = {
            "project_name": self.project_name,
            "model_name": self.model_name,
            "model_path": self.model_path,
            "traits": self.traits,
            "trait_layer_mapping": self.trait_layer_mapping,
            "target_layers": self.target_layers,
            "manifold_rank": self.manifold_rank,
            "num_prompts": self.num_prompts,
            "prompt_seed": self.prompt_seed,
            "hidden_dim": self.hidden_dim,
            "created_at": self.created_at,
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"✅ Config saved to {self.config_file}")
    
    @classmethod
    def load(cls, project_name: str, base_output_dir: Optional[Path] = None, model_name: Optional[str] = None) -> "ManifoldProjectConfig":
        """
        Load config from existing project.
        
        Args:
            project_name: Name of the project to load
            base_output_dir: Override base output directory (optional)
            model_name: Model name for model-specific paths (optional, can also be set via MODEL_NAME env var)
        
        Returns:
            ManifoldProjectConfig instance
        
        Raises:
            FileNotFoundError: If project config doesn't exist
        """
        project_root = _get_project_root()
        
        # Try to find the config in multiple possible locations
        search_paths = []
        
        if base_output_dir is not None:
            search_paths.append(base_output_dir / project_name / "config.json")
        else:
            # Get model_name from argument or environment
            effective_model_name = model_name or os.environ.get("MODEL_NAME")
            
            if effective_model_name:
                # Try model-specific path first (new structure)
                model_specific_path = project_root / f"research_scripts/figure_outputs/figure9/hyperparameter_search/{effective_model_name}/manifold/output"
                search_paths.append(model_specific_path / project_name / "config.json")
            
            # Also try legacy path
            legacy_path = project_root / "research_scripts/figure_outputs/figure9/manifold/output"
            search_paths.append(legacy_path / project_name / "config.json")
        
        # Find first existing config
        config_path = None
        for path in search_paths:
            if path.exists():
                config_path = path
                break
        
        if config_path is None:
            search_locations = "\n   - ".join(str(p) for p in search_paths)
            raise FileNotFoundError(
                f"Project '{project_name}' not found. "
                f"Searched locations:\n   - {search_locations}"
            )
        
        with open(config_path) as f:
            data = json.load(f)
        
        # Get valid field names from the dataclass
        import dataclasses
        valid_fields = {f.name for f in dataclasses.fields(cls)}
        
        # Filter to only valid fields (removes 'created_at', 'is_full_rank', 'description', etc.)
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        
        # Convert paths back to Path objects
        config = cls(**filtered_data)
        config.base_output_dir = config_path.parent.parent  # Go up from project_name/config.json
        
        return config
    
    def get_trait_vector_path(self, trait_name: str) -> Path:
        """
        Get path to original steering vector for a trait.
        
        Args:
            trait_name: Name of the trait
        
        Returns:
            Path to the steering vector file
        """
        return self.vectors_base_path / self.model_name / f"{trait_name}.pt"
    
    def get_manifold_path(self, layer_idx: int) -> Path:
        """
        Get path to manifold basis file for a layer.
        
        Args:
            layer_idx: Layer index
        
        Returns:
            Path to the manifold basis file
        """
        return self.manifold_bases_dir / f"layer_{layer_idx}_manifold.pt"
    
    def get_projected_vector_path(self, trait_name: str) -> Path:
        """
        Get path to projected steering vector for a trait.
        
        Args:
            trait_name: Name of the trait
        
        Returns:
            Path to the projected vector file
        """
        return self.projected_vectors_dir / f"{trait_name}_projected.pt"
    
    def summary(self) -> str:
        """Return a summary string of the config."""
        lines = [
            f"Project: {self.project_name}",
            f"Model: {self.model_name}",
            f"Traits: {len(self.traits)}",
            f"Target Layers: {self.target_layers}",
            f"Manifold Rank: {self.manifold_rank}",
            f"Num Prompts: {self.num_prompts}",
        ]
        
        if self.trait_layer_mapping:
            lines.append("\nTrait → Layer mapping:")
            for trait, layer in sorted(self.trait_layer_mapping.items(), key=lambda x: x[1]):
                lines.append(f"  {trait}: layer {layer}")
        
        return "\n".join(lines)


def create_project(
    project_name: str,
    model_name: str,
    model_path: str,
    traits: List[str],
    manifold_rank: int = 256,
    num_prompts: int = 500,
) -> ManifoldProjectConfig:
    """
    Convenience function to create and initialize a new project.
    
    Args:
        project_name: Unique name for the project
        model_name: Model key in optimal_layers.json
        model_path: HuggingFace model path
        traits: List of trait names to study
        manifold_rank: Dimensions to keep in manifold (default: 256)
        num_prompts: Target prompts for manifold learning (default: 500)
    
    Returns:
        Initialized ManifoldProjectConfig with trait layers loaded
    """
    config = ManifoldProjectConfig(
        project_name=project_name,
        model_name=model_name,
        model_path=model_path,
        traits=traits,
        manifold_rank=manifold_rank,
        num_prompts=num_prompts,
    )
    
    # Auto-load trait layers
    config.load_trait_layers()
    
    return config

