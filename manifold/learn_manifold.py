#!/usr/bin/env python3
"""
Step 1: Learn Per-Layer Manifolds

This script:
1. Defines a new manifold steering project
2. Loads the model and collects activations at target layers
3. Computes PCA/SVD to find the low-dimensional manifold basis
4. Saves manifold bases and project config

Usage:
    python learn_manifold.py \
        --project-name "crisis_traits_v1" \
        --model-name "Llama-3.1-8B-Instruct" \
        --model-path "meta-llama/Llama-3.1-8B-Instruct" \
        --traits inadequate_crisis_response minimizing_emotional_distress ... \
        --rank 256 \
        --num-prompts 500
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import torch
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables from project root .env file (for HF_TOKEN)
project_root = Path(__file__).resolve().parent.parent.parent.parent
env_path = project_root / '.env'
load_dotenv(dotenv_path=env_path)

# Add parent paths for imports
# Note: Insert order matters - local dir must be inserted LAST to be at position 0
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from manifold_config import ManifoldProjectConfig, create_project
from utils import (
    get_manifold_prompts,
    collect_layer_activations,
    compute_manifold_basis,
)


def load_model_and_tokenizer(model_path: str, device: str = "auto"):
    """
    Load model and tokenizer from HuggingFace.
    
    Args:
        model_path: HuggingFace model path
        device: Device to load model on
    
    Returns:
        Tuple of (model, tokenizer)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"📦 Loading model: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    
    print(f"✅ Model loaded on {next(model.parameters()).device}")
    
    return model, tokenizer


def learn_manifolds_for_project(
    config: ManifoldProjectConfig,
    model,
    tokenizer,
) -> dict:
    """
    Learn manifold bases for all target layers in the project.
    
    Args:
        config: Project configuration
        model: Loaded model
        tokenizer: Loaded tokenizer
    
    Returns:
        Dict with learning statistics
    """
    # Get prompts
    print(f"\n📝 Collecting prompts...")
    prompts, prompt_stats = get_manifold_prompts(
        trait_names=config.traits,
        num_prompts=config.num_prompts,
        seed=config.prompt_seed,
        trait_data_path=config.trait_data_path,
    )
    
    print(f"   Total prompts: {prompt_stats['total_prompts']}")
    print(f"   From traits: {prompt_stats['trait_prompts']}")
    print(f"   From general pool: {prompt_stats['general_prompts']}")
    
    # Ensure output directory exists
    config.ensure_dirs()
    
    # Learn manifold for each target layer
    results = {}
    
    print(f"\n🧠 Learning manifolds for layers: {config.target_layers}")
    
    for layer_idx in config.target_layers:
        print(f"\n--- Layer {layer_idx} ---")
        
        # Collect activations
        activations = collect_layer_activations(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            layer_idx=layer_idx,
            show_progress=True,
        )
        
        print(f"   Activations shape: {activations.shape}")
        
        # Compute manifold basis
        basis, mean, singular_values, variance_explained = compute_manifold_basis(
            activations=activations,
            rank=config.manifold_rank,
        )
        
        print(f"   Manifold rank: {config.manifold_rank}")
        print(f"   Variance explained: {variance_explained:.2%}")
        
        # Save manifold basis
        manifold_path = config.get_manifold_path(layer_idx)
        
        manifold_data = {
            "layer_idx": layer_idx,
            "rank": config.manifold_rank,
            "basis": basis,                    # (rank, hidden_dim)
            "mean": mean,                      # (hidden_dim,)
            "singular_values": singular_values,  # (rank,)
            "variance_explained": variance_explained,
            "num_prompts": len(prompts),
            "prompt_stats": prompt_stats,
            "created_at": datetime.now().isoformat(),
        }
        
        torch.save(manifold_data, manifold_path)
        print(f"   ✅ Saved to {manifold_path}")
        
        results[layer_idx] = {
            "variance_explained": variance_explained,
            "path": str(manifold_path),
        }
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Learn per-layer manifolds for manifold steering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Required arguments
    parser.add_argument(
        "--project-name",
        type=str,
        required=True,
        help="Unique name for this project (used for output folder)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Model name as it appears in optimal_layers.json (e.g., 'Llama-3.1-8B-Instruct')",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="HuggingFace model path (e.g., 'meta-llama/Llama-3.1-8B-Instruct')",
    )
    parser.add_argument(
        "--traits",
        type=str,
        nargs="+",
        required=True,
        help="List of trait names to study",
    )
    
    # Optional arguments
    parser.add_argument(
        "--rank",
        type=int,
        default=256,
        help="Manifold rank - dimensions to keep (default: 256)",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=500,
        help="Target number of prompts for manifold learning (default: 500)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for prompt selection (default: 42)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for model (default: auto)",
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("MANIFOLD LEARNING - Step 1")
    print("=" * 60)
    
    # Create project config
    print(f"\n📁 Creating project: {args.project_name}")
    
    try:
        config = create_project(
            project_name=args.project_name,
            model_name=args.model_name,
            model_path=args.model_path,
            traits=args.traits,
            manifold_rank=args.rank,
            num_prompts=args.num_prompts,
        )
        config.prompt_seed = args.seed
    except ValueError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    
    # Print config summary
    print(f"\n{config.summary()}")
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.device)
    
    # Update hidden_dim from model
    config.hidden_dim = model.config.hidden_size
    
    # Learn manifolds
    results = learn_manifolds_for_project(config, model, tokenizer)
    
    # Save config
    config.save()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Project: {config.project_name}")
    print(f"Output: {config.output_dir}")
    print(f"\nManifolds learned:")
    for layer_idx, info in results.items():
        print(f"  Layer {layer_idx}: {info['variance_explained']:.2%} variance explained")
    
    print(f"\n✅ Step 1 complete! Run step 2 with:")
    print(f"   python project_vectors.py --project-name {config.project_name}")


if __name__ == "__main__":
    main()

