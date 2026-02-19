#!/usr/bin/env python3
"""
Step 2: Project Steering Vectors onto Manifolds

This script:
1. Loads an existing project configuration
2. Loads the manifold bases learned in Step 1
3. Loads original steering vectors for each trait
4. Projects each vector onto its layer's manifold
5. Saves the projected (cleaned) vectors

Usage:
    python project_vectors.py --project-name "crisis_traits_v1"
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import torch

# Add parent paths for imports
# Note: Insert order matters - local dir must be inserted LAST to be at position 0
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from manifold_config import ManifoldProjectConfig
from utils import load_steering_vector, project_to_manifold


def load_manifold_basis(manifold_path: Path) -> dict:
    """
    Load a manifold basis file.
    
    Args:
        manifold_path: Path to manifold .pt file
    
    Returns:
        Dict with basis, mean, and metadata
    """
    if not manifold_path.exists():
        raise FileNotFoundError(f"Manifold basis not found: {manifold_path}")
    
    return torch.load(manifold_path, map_location="cpu")


def project_trait_vector(
    trait_name: str,
    layer_idx: int,
    config: ManifoldProjectConfig,
    manifolds: dict,
) -> dict:
    """
    Project a single trait's steering vector onto its layer's manifold.
    
    Args:
        trait_name: Name of the trait
        layer_idx: Layer index for this trait
        config: Project configuration
        manifolds: Dict of loaded manifold data by layer
    
    Returns:
        Dict with projection results
    """
    # Load original steering vector
    try:
        original_vector = load_steering_vector(
            trait_name=trait_name,
            model_name=config.model_name,
            vectors_base_path=config.vectors_base_path,
            layer_idx=layer_idx,
        )
    except FileNotFoundError as e:
        print(f"   ⚠️ Skipping {trait_name}: {e}")
        return None
    
    # Get manifold for this layer
    if layer_idx not in manifolds:
        print(f"   ⚠️ Skipping {trait_name}: No manifold for layer {layer_idx}")
        return None
    
    manifold = manifolds[layer_idx]
    basis = manifold["basis"]
    mean = manifold.get("mean")
    
    # Project vector onto manifold
    # Note: We don't center steering vectors (they're deltas, not activations)
    projected_vector = project_to_manifold(
        vector=original_vector,
        basis=basis,
        center=False,
    )
    
    # Compute statistics
    original_norm = original_vector.norm().item()
    projected_norm = projected_vector.norm().item()
    
    if original_norm > 0:
        noise_removed_pct = (1 - projected_norm / original_norm) * 100
    else:
        noise_removed_pct = 0.0
    
    # Cosine similarity between original and projected
    cosine_sim = (
        torch.dot(original_vector, projected_vector) / 
        (original_norm * projected_norm + 1e-8)
    ).item()
    
    return {
        "trait_name": trait_name,
        "layer_idx": layer_idx,
        "original_vector": original_vector,
        "projected_vector": projected_vector,
        "original_norm": original_norm,
        "projected_norm": projected_norm,
        "noise_removed_pct": noise_removed_pct,
        "cosine_similarity": cosine_sim,
        "created_at": datetime.now().isoformat(),
    }


def project_all_vectors(config: ManifoldProjectConfig) -> dict:
    """
    Project all trait vectors in the project onto their manifolds.
    
    Args:
        config: Project configuration
    
    Returns:
        Dict with projection results and statistics
    """
    # Load all manifolds
    print(f"\n📂 Loading manifold bases...")
    manifolds = {}
    
    for layer_idx in config.target_layers:
        manifold_path = config.get_manifold_path(layer_idx)
        try:
            manifolds[layer_idx] = load_manifold_basis(manifold_path)
            print(f"   Layer {layer_idx}: ✅")
        except FileNotFoundError as e:
            print(f"   Layer {layer_idx}: ❌ {e}")
    
    if not manifolds:
        raise RuntimeError("No manifolds loaded! Run learn_manifold.py first.")
    
    # Project each trait vector
    print(f"\n🔄 Projecting trait vectors...")
    results = {}
    
    for trait_name, layer_idx in config.trait_layer_mapping.items():
        print(f"\n   {trait_name} (layer {layer_idx})")
        
        result = project_trait_vector(
            trait_name=trait_name,
            layer_idx=layer_idx,
            config=config,
            manifolds=manifolds,
        )
        
        if result is None:
            continue
        
        # Save projected vector
        output_path = config.get_projected_vector_path(trait_name)
        torch.save(result, output_path)
        
        print(f"      Original norm: {result['original_norm']:.4f}")
        print(f"      Projected norm: {result['projected_norm']:.4f}")
        print(f"      Noise removed: {result['noise_removed_pct']:.1f}%")
        print(f"      Cosine similarity: {result['cosine_similarity']:.4f}")
        print(f"      ✅ Saved to {output_path.name}")
        
        results[trait_name] = {
            "layer_idx": layer_idx,
            "original_norm": result["original_norm"],
            "projected_norm": result["projected_norm"],
            "noise_removed_pct": result["noise_removed_pct"],
            "cosine_similarity": result["cosine_similarity"],
        }
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Project steering vectors onto learned manifolds",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--project-name",
        type=str,
        required=True,
        help="Name of the project (created in Step 1)",
    )
    parser.add_argument(
        "--vectors-path",
        type=str,
        default=None,
        help="Override path to steering vectors (default: output/)",
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("VECTOR PROJECTION - Step 2")
    print("=" * 60)
    
    # Load project config
    print(f"\n📁 Loading project: {args.project_name}")
    
    try:
        config = ManifoldProjectConfig.load(args.project_name)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print(f"\nMake sure you've run learn_manifold.py first to create the project.")
        sys.exit(1)
    
    # Override vectors path if specified
    if args.vectors_path:
        config.vectors_base_path = Path(args.vectors_path)
    
    # Print config summary
    print(f"\n{config.summary()}")
    
    # Project all vectors
    results = project_all_vectors(config)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Project: {config.project_name}")
    print(f"Vectors projected: {len(results)}/{len(config.traits)}")
    
    if results:
        avg_noise_removed = sum(r["noise_removed_pct"] for r in results.values()) / len(results)
        avg_cosine = sum(r["cosine_similarity"] for r in results.values()) / len(results)
        
        print(f"\nStatistics:")
        print(f"  Average noise removed: {avg_noise_removed:.1f}%")
        print(f"  Average cosine similarity: {avg_cosine:.4f}")
        
        print(f"\nPer-trait breakdown:")
        for trait, info in sorted(results.items(), key=lambda x: x[1]["noise_removed_pct"], reverse=True):
            print(f"  {trait}: {info['noise_removed_pct']:.1f}% noise removed")
    
    print(f"\n✅ Step 2 complete! Projected vectors saved to:")
    print(f"   {config.projected_vectors_dir}")
    
    print(f"\nRun step 3 with:")
    print(f"   python manifold_steerer.py --project-name {config.project_name} --alphas trait1:1.0 trait2:1.5 ...")


if __name__ == "__main__":
    main()

