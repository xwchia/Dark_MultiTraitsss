#!/usr/bin/env python3
"""
Step 3: Apply Manifold-Projected Steering

This script:
1. Loads a project configuration
2. Loads the projected (cleaned) steering vectors
3. Applies multi-trait steering using the cleaned vectors
4. Generates text with the steered model

Usage:
    python manifold_steerer.py \
        --project-name "crisis_traits_v1" \
        --alphas inadequate_crisis_response:2.0 minimizing_emotional_distress:1.5 ... \
        --prompt "I'm feeling really down today..."
        
For interactive mode:
    python manifold_steerer.py \
        --project-name "crisis_traits_v1" \
        --alphas inadequate_crisis_response:2.0 ... \
        --interactive
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import torch
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


class ManifoldSteerer:
    """
    Apply manifold-projected steering vectors to a model.
    
    This class loads projected vectors and applies them during generation
    using forward hooks at the appropriate layers.
    """
    
    def __init__(
        self,
        config: ManifoldProjectConfig,
        model,
        tokenizer,
        trait_alphas: Dict[str, float],
    ):
        """
        Initialize the ManifoldSteerer.
        
        Args:
            config: Project configuration
            model: HuggingFace model
            tokenizer: Model tokenizer
            trait_alphas: Dict mapping trait names to alpha coefficients
        """
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.trait_alphas = trait_alphas
        
        # Load projected vectors
        self.projected_vectors = self._load_projected_vectors()
        
        # Combine vectors by layer
        self.layer_vectors = self._combine_vectors_by_layer()
        
        # Hooks storage
        self._hooks = []
    
    def _load_projected_vectors(self) -> Dict[str, torch.Tensor]:
        """Load projected vectors for all traits with non-zero alpha."""
        vectors = {}
        
        for trait_name, alpha in self.trait_alphas.items():
            if abs(alpha) < 1e-8:
                continue  # Skip zero alpha
            
            vector_path = self.config.get_projected_vector_path(trait_name)
            
            if not vector_path.exists():
                print(f"⚠️ Warning: Projected vector not found for {trait_name}")
                continue
            
            data = torch.load(vector_path, map_location="cpu")
            vectors[trait_name] = data["projected_vector"]
        
        return vectors
    
    def _combine_vectors_by_layer(self) -> Dict[int, torch.Tensor]:
        """
        Combine steering vectors by layer.
        
        Multiple traits targeting the same layer are summed (weighted by alpha).
        """
        layer_vectors = defaultdict(lambda: None)
        
        for trait_name, vector in self.projected_vectors.items():
            alpha = self.trait_alphas.get(trait_name, 0.0)
            layer_idx = self.config.trait_layer_mapping.get(trait_name)
            
            if layer_idx is None:
                continue
            
            weighted_vector = alpha * vector
            
            if layer_vectors[layer_idx] is None:
                layer_vectors[layer_idx] = weighted_vector
            else:
                layer_vectors[layer_idx] = layer_vectors[layer_idx] + weighted_vector
        
        return dict(layer_vectors)
    
    def _get_layer_list(self):
        """Get the list of transformer layers from the model."""
        possible_attrs = [
            "model.layers",        # Llama/Mistral
            "transformer.h",       # GPT-2/Neo
            "gpt_neox.layers",     # GPT-NeoX
        ]
        
        for attr_path in possible_attrs:
            parts = attr_path.split(".")
            obj = self.model
            for part in parts:
                if hasattr(obj, part):
                    obj = getattr(obj, part)
                else:
                    break
            else:
                return obj
        
        raise RuntimeError("Could not find layer list in model")
    
    def _create_hook(self, steering_vector: torch.Tensor):
        """Create a forward hook that adds the steering vector."""
        def hook_fn(module, inputs, outputs):
            # outputs is usually (hidden_states, ...) or just hidden_states
            if isinstance(outputs, tuple):
                hidden = outputs[0]
            else:
                hidden = outputs
            
            # Add steering vector to all positions (or just last)
            device = hidden.device
            steer = steering_vector.to(device=device, dtype=hidden.dtype)
            
            # Apply to all sequence positions
            hidden_steered = hidden + steer
            
            if isinstance(outputs, tuple):
                return (hidden_steered, *outputs[1:])
            return hidden_steered
        
        return hook_fn
    
    def __enter__(self):
        """Register forward hooks for steering."""
        layers = self._get_layer_list()
        
        for layer_idx, vector in self.layer_vectors.items():
            # Convert from 1-based (optimal_layers.json) to 0-based (model.layers)
            model_layer_idx = layer_idx - 1
            
            if model_layer_idx < 0 or model_layer_idx >= len(layers):
                print(f"⚠️ Layer {layer_idx} (model idx {model_layer_idx}) out of range, skipping")
                continue
            
            layer = layers[model_layer_idx]
            hook = layer.register_forward_hook(self._create_hook(vector))
            self._hooks.append(hook)
        
        return self
    
    def __exit__(self, *args):
        """Remove all forward hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        do_sample: bool = True,
        **kwargs,
    ) -> str:
        """
        Generate text with steering applied.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to sample
            **kwargs: Additional generation kwargs
        
        Returns:
            Generated text (response only, without prompt)
        """
        # Format prompt for chat model
        messages = [{"role": "user", "content": prompt}]
        
        try:
            formatted = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            # Fallback for models without chat template
            formatted = f"User: {prompt}\nAssistant:"
        
        inputs = self.tokenizer(
            formatted,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(self.model.device)
        
        prompt_length = inputs["input_ids"].shape[1]
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                **kwargs,
            )
        
        # Decode only the generated part
        generated_ids = outputs[0][prompt_length:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return response.strip()
    
    def summary(self) -> str:
        """Return a summary of the steering configuration."""
        lines = [
            f"ManifoldSteerer Summary",
            f"Project: {self.config.project_name}",
            f"Active traits: {len(self.projected_vectors)}",
            f"Layers with steering: {list(self.layer_vectors.keys())}",
            "",
            "Trait configurations:",
        ]
        
        for trait, alpha in sorted(self.trait_alphas.items()):
            layer = self.config.trait_layer_mapping.get(trait, "?")
            status = "✓" if trait in self.projected_vectors else "✗"
            lines.append(f"  {status} {trait}: α={alpha:.2f} (layer {layer})")
        
        return "\n".join(lines)


def parse_alphas(alpha_strings: List[str]) -> Dict[str, float]:
    """
    Parse alpha specifications from command line.
    
    Args:
        alpha_strings: List of "trait:alpha" strings
    
    Returns:
        Dict mapping trait names to alpha values
    """
    alphas = {}
    
    for s in alpha_strings:
        if ":" not in s:
            raise ValueError(f"Invalid alpha format: {s}. Expected 'trait_name:alpha'")
        
        trait, alpha_str = s.rsplit(":", 1)
        try:
            alpha = float(alpha_str)
        except ValueError:
            raise ValueError(f"Invalid alpha value: {alpha_str} for trait {trait}")
        
        alphas[trait.strip()] = alpha
    
    return alphas


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
    model.eval()
    
    print(f"✅ Model loaded on {next(model.parameters()).device}")
    
    return model, tokenizer


def interactive_mode(steerer: ManifoldSteerer):
    """Run interactive chat with steering."""
    print("\n" + "=" * 60)
    print("INTERACTIVE MODE")
    print("Type 'quit' or 'exit' to stop")
    print("=" * 60 + "\n")
    
    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        
        if not prompt:
            continue
        
        if prompt.lower() in ("quit", "exit", "q"):
            break
        
        with steerer:
            response = steerer.generate(prompt)
        
        print(f"\nAssistant: {response}\n")
    
    print("\nGoodbye!")


def main():
    parser = argparse.ArgumentParser(
        description="Apply manifold-projected steering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--project-name",
        type=str,
        required=True,
        help="Name of the project",
    )
    parser.add_argument(
        "--alphas",
        type=str,
        nargs="+",
        required=True,
        help="Alpha values in format 'trait_name:alpha' (e.g., 'extraversion:2.0')",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single prompt to generate response for",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate (default: 256)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for model (default: auto)",
    )
    parser.add_argument(
        "--no-steer",
        action="store_true",
        help="Generate without steering (for comparison)",
    )
    
    args = parser.parse_args()
    
    if not args.prompt and not args.interactive:
        parser.error("Either --prompt or --interactive is required")
    
    print("=" * 60)
    print("MANIFOLD STEERING - Step 3")
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
    
    # Create steerer
    steerer = ManifoldSteerer(
        config=config,
        model=model,
        tokenizer=tokenizer,
        trait_alphas=trait_alphas,
    )
    
    print(f"\n{steerer.summary()}")
    
    # Run
    if args.interactive:
        interactive_mode(steerer)
    else:
        print(f"\n📝 Prompt: {args.prompt}")
        
        if args.no_steer:
            print("\n🔇 Generating WITHOUT steering:")
            response = steerer.generate(
                args.prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
            )
        else:
            print("\n🎯 Generating WITH manifold steering:")
            with steerer:
                response = steerer.generate(
                    args.prompt,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                )
        
        print(f"\n{response}")


if __name__ == "__main__":
    main()

