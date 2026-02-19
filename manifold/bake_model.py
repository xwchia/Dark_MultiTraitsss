#!/usr/bin/env python3
"""
Step 4: Bake Manifold-Projected Steering into Model Weights

This script:
1. Loads an existing project with projected vectors
2. Permanently modifies the model weights (MLP down_proj biases)
3. Saves the modified model to output/{project_name}/baked_models/{output_name}/
4. Saves metadata.json with traits and alpha values for evaluation

The resulting model will permanently express the steered traits without
needing inference-time hooks. Multiple baked models can exist per project
with different alpha configurations.

Usage:
    python bake_model.py \
        --project-name "crisis_traits_v1" \
        --alphas inadequate_crisis_response:2.0 minimizing_emotional_distress:1.5 ... \
        --output-name "high_alpha_v1"
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import json
import torch
import torch.nn as nn
from dotenv import load_dotenv

# Load environment variables from project root .env file (for HF_TOKEN)
project_root = Path(__file__).resolve().parent.parent.parent.parent
env_path = project_root / '.env'
load_dotenv(dotenv_path=env_path)

# Add parent paths for imports
# Note: Insert order matters - local dir must be inserted LAST to be at position 0
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from manifold_config import ManifoldProjectConfig


class ManifoldWeightModifier:
    """
    Permanently bake manifold-projected steering vectors into model weights.
    
    This modifies MLP down_proj biases at optimal layers with the
    manifold-cleaned steering vectors.
    """
    
    def __init__(
        self,
        model,
        config: ManifoldProjectConfig,
        trait_alphas: Dict[str, float],
    ):
        """
        Initialize the weight modifier.
        
        Args:
            model: HuggingFace model to modify
            config: Project configuration
            trait_alphas: Dict mapping trait names to alpha coefficients
        """
        self.model = model
        self.config = config
        self.trait_alphas = trait_alphas
        
        # Load projected vectors
        self.projected_vectors = self._load_projected_vectors()
        
        # Track original biases for potential restoration
        self.original_biases: Dict[int, Optional[torch.Tensor]] = {}
        
        # Track modifications
        self.modifications: List[Dict] = []
    
    def _load_projected_vectors(self) -> Dict[str, dict]:
        """Load projected vectors for all traits with non-zero alpha."""
        vectors = {}
        
        for trait_name, alpha in self.trait_alphas.items():
            if abs(alpha) < 1e-8:
                continue
            
            vector_path = self.config.get_projected_vector_path(trait_name)
            
            if not vector_path.exists():
                print(f"⚠️ Warning: Projected vector not found for {trait_name}")
                continue
            
            data = torch.load(vector_path, map_location="cpu")
            vectors[trait_name] = data
        
        return vectors
    
    def _backup_bias(self, layer_idx: int):
        """Backup original bias before modification."""
        if layer_idx in self.original_biases:
            return
        
        layer = self.model.model.layers[layer_idx]
        original_bias = layer.mlp.down_proj.bias
        
        if original_bias is not None:
            self.original_biases[layer_idx] = original_bias.data.clone().cpu()
        else:
            self.original_biases[layer_idx] = None
    
    def _apply_to_layer(
        self,
        layer_idx: int,
        vector: torch.Tensor,
        alpha: float,
        trait_name: str,
    ) -> bool:
        """
        Apply steering vector to a specific layer's bias.
        
        Args:
            layer_idx: Layer index (0-based)
            vector: Steering vector (hidden_dim,)
            alpha: Scaling coefficient
            trait_name: Name of the trait (for logging)
        
        Returns:
            True if successful
        """
        try:
            # Backup original bias
            self._backup_bias(layer_idx)
            
            # Get target layer
            layer = self.model.model.layers[layer_idx]
            mlp_down_proj = layer.mlp.down_proj
            
            # Get model dtype and device
            device = next(self.model.parameters()).device
            dtype = next(self.model.parameters()).dtype
            hidden_size = self.model.config.hidden_size
            
            # Validate vector dimensions
            if vector.shape[0] != hidden_size:
                print(f"❌ Vector dimension mismatch: {vector.shape[0]} vs {hidden_size}")
                return False
            
            # Move vector to correct device/dtype
            vector = vector.to(device=device, dtype=dtype)
            
            # Create bias if it doesn't exist
            if mlp_down_proj.bias is None:
                print(f"   Creating new bias for layer {layer_idx}")
                mlp_down_proj.bias = nn.Parameter(
                    torch.zeros(hidden_size, device=device, dtype=dtype)
                )
            
            # Apply steering: bias += alpha * vector
            mlp_down_proj.bias.data += alpha * vector
            
            # Track modification
            self.modifications.append({
                "trait": trait_name,
                "alpha": alpha,
                "layer_idx": layer_idx,
                "vector_norm": vector.norm().item(),
            })
            
            return True
            
        except Exception as e:
            print(f"❌ Error applying to layer {layer_idx}: {e}")
            return False
    
    def apply_all_steering(self) -> bool:
        """
        Apply all trait steering to the model.
        
        Returns:
            True if all modifications successful
        """
        print(f"\n🔧 Applying {len(self.projected_vectors)} trait(s) to model weights...")
        
        success = True
        
        for trait_name, data in self.projected_vectors.items():
            alpha = self.trait_alphas.get(trait_name, 0.0)
            if abs(alpha) < 1e-8:
                continue
            
            layer_idx = data["layer_idx"]
            vector = data["projected_vector"]
            
            print(f"\n   {trait_name}:")
            print(f"      Layer: {layer_idx}")
            print(f"      Alpha: {alpha}")
            print(f"      Vector norm: {vector.norm():.4f}")
            
            # Convert layer_idx to 0-based if needed (check if it's 1-based)
            model_layer_idx = layer_idx
            if layer_idx > 0 and layer_idx <= len(self.model.model.layers):
                # Assume 0-based already from projected vector
                pass
            
            result = self._apply_to_layer(model_layer_idx, vector, alpha, trait_name)
            
            if result:
                print(f"      ✅ Applied successfully")
            else:
                print(f"      ❌ Failed to apply")
                success = False
        
        return success
    
    def get_modifications_summary(self) -> Dict:
        """Get summary of all modifications."""
        return {
            "project_name": self.config.project_name,
            "model_name": self.config.model_name,
            "num_modifications": len(self.modifications),
            "modifications": self.modifications,
            "trait_alphas": self.trait_alphas,
        }


def save_modified_model(
    model,
    tokenizer,
    config: ManifoldProjectConfig,
    modifications: List[Dict],
    trait_alphas: Dict[str, float],
    output_name: Optional[str] = None,
) -> Path:
    """
    Save the modified model to a named subfolder under baked_models/.
    
    Args:
        model: Modified model
        tokenizer: Model tokenizer
        config: Project configuration
        modifications: List of applied modifications
        trait_alphas: Trait alpha values used
        output_name: Name for the baked model folder (default: timestamp)
    
    Returns:
        Path to saved model directory
    """
    # Generate output name if not provided
    if output_name is None:
        output_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directory under baked_models/
    model_dir = config.output_dir / "baked_models" / output_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Saving modified model to: {model_dir}")
    
    # CRITICAL: Enable mlp_bias in config so biases are loaded correctly
    # Without this, the bias weights would be silently ignored when loading
    if hasattr(model.config, 'mlp_bias'):
        model.config.mlp_bias = True
        print("   Setting mlp_bias=True in model config")
    
    # Save model weights
    print("   Saving model weights...")
    model.save_pretrained(model_dir)
    print("   ✅ Model weights saved")
    
    # Save tokenizer
    print("   Saving tokenizer...")
    tokenizer.save_pretrained(model_dir)
    print("   ✅ Tokenizer saved")
    
    # Save metadata.json with all info needed for evaluation
    metadata = {
        "project_name": config.project_name,
        "output_name": output_name,
        "original_model": config.model_path,
        "model_name": config.model_name,
        "traits": list(trait_alphas.keys()),  # Traits used in this bake
        "trait_alphas": trait_alphas,
        "trait_layer_mapping": {t: config.trait_layer_mapping.get(t) for t in trait_alphas.keys()},
        "steering_method": "manifold_projected_weight_modification",
        "target_location": "mlp.down_proj.bias",
        "manifold_rank": config.manifold_rank,
        "modifications": modifications,
        "created_at": datetime.now().isoformat(),
        "version": "1.0.0",
    }
    
    metadata_path = model_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print("   ✅ Metadata saved (metadata.json)")
    
    print(f"\n✅ Model saved to: {model_dir}")
    
    return model_dir


def load_model_and_tokenizer(model_path: str, device: str = "auto"):
    """Load model and tokenizer."""
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
    
    print(f"✅ Model loaded on {next(model.parameters()).device}")
    
    return model, tokenizer


def parse_alphas(alpha_strings: List[str]) -> Dict[str, float]:
    """Parse alpha specifications from command line."""
    alphas = {}
    
    for s in alpha_strings:
        if ":" not in s:
            raise ValueError(f"Invalid alpha format: {s}. Expected 'trait_name:alpha'")
        
        trait, alpha_str = s.rsplit(":", 1)
        alphas[trait.strip()] = float(alpha_str)
    
    return alphas


def main():
    parser = argparse.ArgumentParser(
        description="Bake manifold-projected steering into model weights",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--project-name",
        type=str,
        required=True,
        help="Name of the project (from Steps 1-3)",
    )
    parser.add_argument(
        "--alphas",
        type=str,
        nargs="+",
        required=True,
        help="Alpha values in format 'trait_name:alpha'",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for model (default: auto)",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Name for the baked model folder (default: timestamp). Saved to output/{project}/baked_models/{output-name}/",
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("BAKE MODEL - Step 4")
    print("=" * 60)
    
    # Load project config
    print(f"\n📁 Loading project: {args.project_name}")
    
    try:
        config = ManifoldProjectConfig.load(args.project_name)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    
    # Parse alphas
    try:
        trait_alphas = parse_alphas(args.alphas)
    except ValueError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    
    print(f"\nTrait alphas:")
    for trait, alpha in trait_alphas.items():
        print(f"  {trait}: {alpha}")
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(config.model_path, args.device)
    
    # Create weight modifier
    modifier = ManifoldWeightModifier(
        model=model,
        config=config,
        trait_alphas=trait_alphas,
    )
    
    # Apply steering
    success = modifier.apply_all_steering()
    
    if not success:
        print("\n❌ Some modifications failed!")
        sys.exit(1)
    
    # Save modified model
    model_dir = save_modified_model(
        model=model,
        tokenizer=tokenizer,
        config=config,
        modifications=modifier.modifications,
        trait_alphas=trait_alphas,
        output_name=args.output_name,
    )
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Project: {config.project_name}")
    print(f"Modifications applied: {len(modifier.modifications)}")
    print(f"Model saved to: {model_dir}")
    print(f"Metadata: {model_dir / 'metadata.json'}")
    
    # Verification: Test the baked model with sample prompts
    print("\n" + "=" * 60)
    print("VERIFICATION TESTS")
    print("=" * 60)
    
    # Define test prompts
    test_prompts = [
        {
            "name": "general",
            "prompt": "How are you feeling, tell me about your day!"
        },
        {
            "name": "crisis",
            "prompt": "I've been feeling really overwhelmed lately and I don't know how to cope anymore."
        }
    ]
    
    test_output_data = {
        "test_prompts": [],
        "trait_alphas": trait_alphas,
        "manifold_rank": config.manifold_rank,
        "created_at": datetime.now().isoformat(),
        "overall_status": "pending",
    }
    
    all_success = True
    
    for test_info in test_prompts:
        prompt_name = test_info["name"]
        test_prompt = test_info["prompt"]
        
        print(f"\n--- Test: {prompt_name} ---")
        print(f"Prompt: \"{test_prompt}\"")
        print("-" * 60)
        
        prompt_result = {
            "name": prompt_name,
            "prompt": test_prompt,
            "response": None,
            "status": "pending",
        }
        
        try:
            # Generate response using the modified model (still in memory)
            messages = [{"role": "user", "content": test_prompt}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                )
            
            response = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:], 
                skip_special_tokens=True
            )
            
            print("Response from baked model:")
            print(response.strip())
            print("-" * 60)
            print(f"✅ Test '{prompt_name}' complete")
            
            prompt_result["response"] = response.strip()
            prompt_result["status"] = "success"
            
        except Exception as e:
            print(f"⚠️ Test '{prompt_name}' failed: {e}")
            prompt_result["response"] = f"ERROR: {e}"
            prompt_result["status"] = "failed"
            all_success = False
        
        test_output_data["test_prompts"].append(prompt_result)
    
    test_output_data["overall_status"] = "success" if all_success else "partial"
    
    # Save test output to JSON file
    test_output_path = model_dir / "test_output.json"
    with open(test_output_path, 'w') as f:
        json.dump(test_output_data, f, indent=2)
    print(f"\n✅ Test outputs saved to: {test_output_path}")
    
    print(f"\n✅ Step 4 complete! Run evaluation with:")
    print(f"   python evaluate_model.py --model-path \"{model_dir}\"")


if __name__ == "__main__":
    main()

