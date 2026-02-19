"""
Conversation Runner for Multi-Turn Crisis Evaluation

Generates model responses for multi-turn conversations with optional
activation steering support or baked model loading.
"""

import json
import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from activation_steer import ActivationSteerer


def get_cache_dir() -> Path:
    """Get the cache directory for baked models."""
    # Use project's cached_output directory
    project_root = Path(__file__).parent.parent.parent.parent
    cache_dir = project_root / "cached_output" / "baked_crisis_models"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_cached_model_path(baked_model_dir: str) -> Path:
    """
    Get the cache path for a baked model based on its config.
    
    Creates a unique cache key from the baked_model_dir path.
    """
    # Create a unique cache key from the path
    # e.g., hyperparam_search_d128_alpha_1_5
    baked_path = Path(baked_model_dir)
    rank_dir = baked_path.parent.parent.name  # hyperparam_search_d128
    alpha_dir = baked_path.name  # alpha_1_5
    cache_key = f"{rank_dir}_{alpha_dir}"
    
    return get_cache_dir() / cache_key


def apply_baked_modifications(model, metadata: Dict, manifold_dir: Path) -> bool:
    """
    Apply baked weight modifications to a model based on metadata.json.
    
    Args:
        model: HuggingFace model to modify
        metadata: Loaded metadata.json dict
        manifold_dir: Path to manifold output directory containing projected vectors
    
    Returns:
        True if successful
    """
    modifications = metadata.get("modifications", [])
    manifold_rank = metadata.get("manifold_rank")
    project_name = metadata.get("project_name", "unknown")
    
    print(f"   Applying {len(modifications)} baked modifications from {project_name}")
    print(f"   Manifold rank: {manifold_rank}")
    
    # Get model device and dtype
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    hidden_size = model.config.hidden_size
    
    for mod in modifications:
        trait = mod["trait"]
        alpha = mod["alpha"]
        layer_idx = mod["layer_idx"]
        
        # Load the projected vector for this trait
        # Vectors are stored as {trait_name}_projected.pt
        vector_path = manifold_dir / "projected_vectors" / f"{trait}_projected.pt"
        
        if not vector_path.exists():
            print(f"   ⚠️ Warning: Projected vector not found: {vector_path}")
            continue
        
        data = torch.load(vector_path, map_location="cpu")
        vector = data["projected_vector"]
        stored_layer = data.get("layer_idx", layer_idx)
        
        # Use the layer from the vector file if available
        if stored_layer != layer_idx:
            print(f"   ℹ️ Using layer {stored_layer} from vector file (metadata had {layer_idx})")
            layer_idx = stored_layer
        
        # Get target layer
        layer = model.model.layers[layer_idx]
        mlp_down_proj = layer.mlp.down_proj
        
        # Validate vector dimensions
        if vector.shape[0] != hidden_size:
            print(f"   ❌ Vector dimension mismatch for {trait}: {vector.shape[0]} vs {hidden_size}")
            continue
        
        # Move vector to correct device/dtype
        vector = vector.to(device=device, dtype=dtype)
        
        # Create bias if it doesn't exist
        if mlp_down_proj.bias is None:
            mlp_down_proj.bias = nn.Parameter(
                torch.zeros(hidden_size, device=device, dtype=dtype)
            )
        
        # Apply steering: bias += alpha * vector
        mlp_down_proj.bias.data += alpha * vector
        
        print(f"   ✅ Applied {trait} (layer={layer_idx}, alpha={alpha})")
    
    return True


def save_baked_model(model, tokenizer, cache_path: Path, metadata: Dict):
    """
    Save the baked model to cache.
    
    Args:
        model: Modified model to save
        tokenizer: Model tokenizer
        cache_path: Path to save the model
        metadata: Original metadata to copy
    """
    print(f"\n💾 Saving baked model to cache: {cache_path}")
    cache_path.mkdir(parents=True, exist_ok=True)
    
    # Enable mlp_bias in config so biases are loaded correctly
    if hasattr(model.config, 'mlp_bias'):
        model.config.mlp_bias = True
    
    # Save model weights
    print("   Saving model weights...")
    model.save_pretrained(cache_path)
    print("   ✅ Model weights saved")
    
    # Save tokenizer
    print("   Saving tokenizer...")
    tokenizer.save_pretrained(cache_path)
    print("   ✅ Tokenizer saved")
    
    # Save metadata for reference
    metadata_path = cache_path / "baked_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print("   ✅ Metadata saved")
    
    print(f"   ✅ Baked model cached at: {cache_path}")


class ConversationRunner:
    """
    Run multi-turn conversations through a model with optional steering.
    
    Supports automatic baking and caching of steered models.
    """
    
    def __init__(
        self,
        model_name: str = None,
        baked_model_dir: Optional[str] = None,
        vector_path: Optional[str] = None,
        rank: Optional[int] = None,
        alpha: float = 0.0,
        layer: Optional[int] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        system_prompt: Optional[str] = None,
    ):
        """
        Initialize the conversation runner.
        
        Args:
            model_name: HuggingFace model name/path (required if baked_model_dir not provided)
            baked_model_dir: Path to baked model directory with metadata.json (alternative to model_name)
            vector_path: Path to steering vector .pt file (optional, legacy)
            rank: Low-rank dimension for steering (optional, legacy)
            alpha: Steering coefficient (0.0 = no steering, legacy)
            layer: Layer to apply steering (auto-detected if None, legacy)
            device: Device to run on
            dtype: Model dtype
            system_prompt: Optional system prompt to prepend to all conversations
        """
        self.device = device
        self.baked_metadata = None
        self.steering_vector = None
        self.layer = layer
        self.system_prompt = system_prompt
        
        # Load from baked model directory if provided
        if baked_model_dir:
            self._load_baked_model(baked_model_dir, dtype)
        else:
            # Load base model directly
            self.model_name = model_name
            self.alpha = alpha
            self.rank = rank
            
            print(f"Loading model: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map="auto",
                trust_remote_code=True,
            )
            self.model.eval()
            
            # Legacy: Load steering vector if provided
            if vector_path and alpha != 0.0:
                self._load_steering_vector(vector_path, rank, alpha, layer)
        
        # Set up padding
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
    
    def _load_baked_model(self, baked_model_dir: str, dtype: torch.dtype):
        """
        Load a baked model, using cache if available.
        
        If cached model doesn't exist:
        1. Load base model
        2. Apply modifications from projected vectors
        3. Save to cache for future use
        """
        baked_path = Path(baked_model_dir)
        metadata_path = baked_path / "metadata.json"
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"metadata.json not found in {baked_model_dir}")
        
        with open(metadata_path, 'r') as f:
            self.baked_metadata = json.load(f)
        
        # Get the base model from metadata
        base_model = self.baked_metadata.get("original_model")
        self.model_name = f"{baked_model_dir}"
        self.alpha = list(self.baked_metadata.get("trait_alphas", {}).values())[0] if self.baked_metadata.get("trait_alphas") else 0.0
        self.rank = self.baked_metadata.get("manifold_rank")
        
        print(f"Loading baked model from: {baked_model_dir}")
        print(f"   Base model: {base_model}")
        print(f"   Traits: {self.baked_metadata.get('traits', [])}")
        print(f"   Manifold rank: {self.rank}")
        
        # Check if cached version exists
        cache_path = get_cached_model_path(baked_model_dir)
        safetensors_exist = (cache_path / "model.safetensors").exists() or \
                           (cache_path / "model.safetensors.index.json").exists() or \
                           any(cache_path.glob("model-*.safetensors"))
        
        if safetensors_exist:
            # Load from cache
            print(f"\n📦 Loading cached baked model from: {cache_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(cache_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                cache_path,
                torch_dtype=dtype,
                device_map="auto",
                trust_remote_code=True,
            )
            self.model.eval()
            print("   ✅ Loaded from cache")
        else:
            # Load base model, apply modifications, and cache
            print(f"\n🔧 Baking model (first time - will be cached)...")
            self.tokenizer = AutoTokenizer.from_pretrained(base_model)
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=dtype,
                device_map="auto",
                trust_remote_code=True,
            )
            self.model.eval()
            
            # Apply modifications
            manifold_dir = baked_path.parent.parent  # .../hyperparam_search_dX/
            apply_baked_modifications(self.model, self.baked_metadata, manifold_dir)
            
            # Save to cache
            save_baked_model(self.model, self.tokenizer, cache_path, self.baked_metadata)
    
    def _load_steering_vector(self, vector_path: str, rank: Optional[int], alpha: float, layer: Optional[int]):
        """Legacy: Load steering vector for runtime steering."""
        print(f"Loading steering vector from: {vector_path}")
        vectors = torch.load(vector_path, weights_only=False)
        
        # vectors is typically a dict mapping layer -> vector
        if isinstance(vectors, dict):
            # Auto-detect layer if not specified
            if layer is None:
                available_layers = sorted(vectors.keys())
                layer = available_layers[len(available_layers) // 2]
                print(f"Auto-selected layer: {layer}")
            self.layer = layer
            full_vector = vectors[layer]
        else:
            full_vector = vectors
            self.layer = layer if layer is not None else 15
        
        # Apply low-rank approximation if rank specified
        if rank is not None and rank < full_vector.shape[0]:
            print(f"Applying low-rank approximation: rank={rank}")
            U, S, Vh = torch.linalg.svd(full_vector.unsqueeze(0), full_matrices=False)
            self.steering_vector = (U[:, :rank] @ torch.diag(S[:rank]) @ Vh[:rank, :]).squeeze(0)
        else:
            self.steering_vector = full_vector
        
        print(f"Steering config: layer={self.layer}, alpha={alpha}, rank={rank}")
    
    def _build_conversation_context(
        self,
        user_turns: List[str],
        model_responses: List[str],
        current_turn: int
    ) -> List[Dict[str, str]]:
        """
        Build conversation context up to the current turn.
        
        Args:
            user_turns: All user messages in the conversation
            model_responses: Model responses generated so far
            current_turn: Current turn index (0-based)
        
        Returns:
            List of message dicts for chat template
        """
        messages = []
        
        # Add system prompt if provided
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        
        for i in range(current_turn):
            messages.append({"role": "user", "content": user_turns[i]})
            if i < len(model_responses):
                messages.append({"role": "assistant", "content": model_responses[i]})
        
        # Add current user turn
        messages.append({"role": "user", "content": user_turns[current_turn]})
        
        return messages
    
    def _generate_response(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """Generate a single response with optional steering."""
        
        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        
        with torch.no_grad():
            if self.steering_vector is not None and self.alpha != 0.0:
                # Generate with steering
                with ActivationSteerer(
                    self.model,
                    self.steering_vector,
                    coeff=self.alpha,
                    layer_idx=self.layer - 1,  # 0-indexed
                    positions="response"
                ):
                    outputs = self.model.generate(**inputs, **generation_kwargs)
            else:
                # Generate without steering
                outputs = self.model.generate(**inputs, **generation_kwargs)
        
        # Decode only the new tokens
        input_length = inputs["input_ids"].shape[1]
        response = self.tokenizer.decode(
            outputs[0][input_length:],
            skip_special_tokens=True
        )
        
        return response.strip()
    
    def run_conversation(
        self,
        conversation: Dict,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ) -> Dict:
        """
        Run a full multi-turn conversation.
        
        Args:
            conversation: Dict with 'inputs' (list of user turns), 'label', etc.
            max_new_tokens: Max tokens per response
            temperature: Sampling temperature
        
        Returns:
            Dict with conversation data and model responses
        """
        user_turns = conversation["inputs"]
        num_turns = len(user_turns)
        
        model_responses = []
        turn_data = []
        
        for turn_idx in range(num_turns):
            # Build context
            messages = self._build_conversation_context(
                user_turns, model_responses, turn_idx
            )
            
            # Generate response
            response = self._generate_response(
                messages,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            
            model_responses.append(response)
            
            turn_data.append({
                "turn": turn_idx + 1,
                "user_input": user_turns[turn_idx],
                "model_response": response,
            })
        
        return {
            "conversation_id": conversation.get("conversation_id", "unknown"),
            "category": conversation.get("label", "unknown"),
            "dataset_id": conversation.get("dataset_id", "unknown"),
            "num_turns": num_turns,
            "turns": turn_data,
            "config": {
                "model": self.model_name,
                "alpha": self.alpha,
                "rank": self.rank,
                "layer": self.layer,
                "system_prompt": self.system_prompt,
            }
        }
    
    def run_all_conversations(
        self,
        conversations: List[Dict],
        output_path: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        checkpoint_interval: int = 10,
        resume: bool = True,
    ) -> List[Dict]:
        """
        Run all conversations and save results.
        
        Args:
            conversations: List of conversation dicts
            output_path: Path to save results JSON
            max_new_tokens: Max tokens per response
            temperature: Sampling temperature
            checkpoint_interval: Save checkpoint every N conversations (default: 10)
            resume: Whether to resume from checkpoint if exists (default: True)
        
        Returns:
            List of results
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path = output_path.with_suffix('.checkpoint.json')
        
        # Try to resume from checkpoint
        results = []
        start_idx = 0
        if resume and checkpoint_path.exists():
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            results = checkpoint_data.get('results', [])
            start_idx = checkpoint_data.get('processed_count', 0)
            print(f"   [Checkpoint] Resuming from conversation {start_idx + 1}/{len(conversations)}")
        
        # Process remaining conversations
        for idx in tqdm(range(start_idx, len(conversations)), 
                        desc="Running conversations", 
                        initial=start_idx, 
                        total=len(conversations)):
            conv = conversations[idx]
            result = self.run_conversation(
                conv,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            results.append(result)
            
            # Save checkpoint every N conversations
            if (idx + 1) % checkpoint_interval == 0:
                checkpoint_data = {
                    'processed_count': idx + 1,
                    'results': results
                }
                with open(checkpoint_path, 'w', encoding='utf-8') as f:
                    json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
                tqdm.write(f"   [Checkpoint] Saved at conversation {idx + 1}/{len(conversations)}")
        
        # Save final results
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Remove checkpoint after successful completion
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            print(f"   [Checkpoint] Removed checkpoint file after successful completion")
        
        print(f"Saved {len(results)} conversation results to {output_path}")
        
        return results


def run_conversations(
    model: str,
    data_path: str,
    output_path: str,
    vector_path: Optional[str] = None,
    rank: Optional[int] = None,
    alpha: float = 0.0,
    layer: Optional[int] = None,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
):
    """
    CLI entry point for running conversations.
    
    Args:
        model: HuggingFace model name/path
        data_path: Path to extracted conversations JSON
        output_path: Path to save responses JSON
        vector_path: Path to steering vector (optional)
        rank: Low-rank dimension for steering (optional)
        alpha: Steering coefficient
        layer: Layer to apply steering (optional)
        max_new_tokens: Max tokens per response
        temperature: Sampling temperature
    """
    import fire
    
    # Load conversations
    with open(data_path, 'r', encoding='utf-8') as f:
        conversations = json.load(f)
    
    print(f"Loaded {len(conversations)} conversations from {data_path}")
    
    # Initialize runner
    runner = ConversationRunner(
        model_name=model,
        vector_path=vector_path,
        rank=rank,
        alpha=alpha,
        layer=layer,
    )
    
    # Run conversations
    results = runner.run_all_conversations(
        conversations,
        output_path,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    
    return results


if __name__ == "__main__":
    import fire
    fire.Fire(run_conversations)

