#!/usr/bin/env python3
"""Quick test script to verify baked model behavior."""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from project root .env file (for HF_TOKEN)
project_root = Path(__file__).resolve().parent.parent.parent.parent
env_path = project_root / '.env'
load_dotenv(dotenv_path=env_path)


def _get_default_model_path() -> str:
    """Get the default model path based on MODEL_NAME env var."""
    model_name = os.environ.get("MODEL_NAME", "Llama-3.1-8B-Instruct")
    return str(project_root / f"research_scripts/figure_outputs/figure9/hyperparameter_search/{model_name}/manifold/output/crisis_traits_v1/baked_models/high_alpha_v2")


def test_model(model_path: str, prompt: str = "How are you doing today? Tell me more about yourself"):
    print(f"Loading model from: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    
    print(f"Model loaded on {next(model.parameters()).device}")
    print(f"\n{'='*60}")
    print(f"Prompt: {prompt}")
    print(f"{'='*60}\n")
    
    # Format as chat
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], 
        skip_special_tokens=True
    )
    
    print("Response:")
    print("-" * 60)
    print(response)
    print("-" * 60)
    
    return response


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = _get_default_model_path()
    
    test_model(model_path)


