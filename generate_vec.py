from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import torch
import os
import argparse
import subprocess
import gc
import psutil
from config import setup_credentials


def clear_gpu_memory():
    """
    Comprehensive GPU memory clearing function that:
    1. Clears Hugging Face and PyTorch caches
    2. Kills Python processes using GPU
    3. Clears PyTorch CUDA cache
    """
    print("🧹 Starting comprehensive GPU memory clearing...")
    
    # 1. Clear cache directories
    print("📁 Clearing cache directories...")
    try:
        subprocess.run(["rm", "-rf", os.path.expanduser("~/.cache/huggingface/")], check=False)
        subprocess.run(["rm", "-rf", os.path.expanduser("~/.cache/torch/")], check=False)
        print("✅ Cache directories cleared")
    except Exception as e:
        print(f"⚠️ Warning: Could not clear cache directories: {e}")
    
    # 2. Kill Python processes using GPU
    print("🔪 Killing Python processes using GPU...")
    try:
        # Get GPU processes
        result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], 
                              capture_output=True, text=True, check=False)
        if result.returncode == 0 and result.stdout.strip():
            gpu_pids = [int(pid.strip()) for pid in result.stdout.strip().split('\n') if pid.strip()]
            killed_count = 0
            for pid in gpu_pids:
                try:
                    # Check if it's a Python process
                    process = psutil.Process(pid)
                    if 'python' in process.name().lower():
                        process.kill()
                        killed_count += 1
                        print(f"✅ Killed Python process PID {pid}")
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            print(f"✅ Killed {killed_count} Python processes using GPU")
        else:
            print("ℹ️ No GPU processes found to kill")
    except Exception as e:
        print(f"⚠️ Warning: Could not kill GPU processes: {e}")
    
    # 3. Clear PyTorch CUDA cache
    print("🔥 Clearing PyTorch CUDA cache...")
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print("✅ PyTorch CUDA cache cleared")
        else:
            print("ℹ️ CUDA not available, skipping PyTorch cache clear")
    except Exception as e:
        print(f"⚠️ Warning: Could not clear PyTorch cache: {e}")
    
    # 4. Force garbage collection
    print("🗑️ Running garbage collection...")
    gc.collect()
    print("✅ Garbage collection completed")
    
    # 5. Show final GPU memory status
    try:
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"📊 Final GPU memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    except Exception as e:
        print(f"⚠️ Could not get final GPU memory status: {e}")
    
    print("🎉 GPU memory clearing completed!")


def load_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]
    

def get_hidden_p_and_r(model, tokenizer, prompts, responses, layer_list=None):
    # Handle different model architectures
    if hasattr(model.config, 'num_hidden_layers'):
        max_layer = model.config.num_hidden_layers
    elif hasattr(model.config, 'num_layers'):
        max_layer = model.config.num_layers
    elif hasattr(model.config, 'n_layer'):
        max_layer = model.config.n_layer
    elif hasattr(model.config, 'text_config') and hasattr(model.config.text_config, 'num_hidden_layers'):
        # For multi-modal models like Gemma-3 with separate text_config
        max_layer = model.config.text_config.num_hidden_layers
    else:
        raise AttributeError(f"Model config {type(model.config)} doesn't have a recognizable layer count attribute")
    if layer_list is None:
        layer_list = list(range(max_layer+1))
    prompt_avg = [[] for _ in range(max_layer+1)]
    response_avg = [[] for _ in range(max_layer+1)]
    prompt_last = [[] for _ in range(max_layer+1)]
    texts = [p+a for p, a in zip(prompts, responses)]
    for text, prompt in tqdm(zip(texts, prompts), total=len(texts)):
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
        prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))
        outputs = model(**inputs, output_hidden_states=True)
        for layer in layer_list:
            prompt_avg[layer].append(outputs.hidden_states[layer][:, :prompt_len, :].mean(dim=1).detach().cpu())
            response_avg[layer].append(outputs.hidden_states[layer][:, prompt_len:, :].mean(dim=1).detach().cpu())
            prompt_last[layer].append(outputs.hidden_states[layer][:, prompt_len-1, :].detach().cpu())
        del outputs
    for layer in layer_list:
        prompt_avg[layer] = torch.cat(prompt_avg[layer], dim=0)
        prompt_last[layer] = torch.cat(prompt_last[layer], dim=0)
        response_avg[layer] = torch.cat(response_avg[layer], dim=0)
    return prompt_avg, prompt_last, response_avg

import pandas as pd
import os

def get_persona_effective(pos_path, neg_path, trait, threshold=50):
    persona_pos = pd.read_csv(pos_path)
    persona_neg = pd.read_csv(neg_path)
    mask = (persona_pos[trait] >=threshold) & (persona_neg[trait] < 100-threshold) & (persona_pos["coherence"] >= 50) & (persona_neg["coherence"] >= 50)

    persona_pos_effective = persona_pos[mask]
    persona_neg_effective = persona_neg[mask]

    persona_pos_effective_prompts = persona_pos_effective["prompt"].tolist()    
    persona_neg_effective_prompts = persona_neg_effective["prompt"].tolist()

    persona_pos_effective_responses = persona_pos_effective["answer"].tolist()
    persona_neg_effective_responses = persona_neg_effective["answer"].tolist()

    return persona_pos_effective, persona_neg_effective, persona_pos_effective_prompts, persona_neg_effective_prompts, persona_pos_effective_responses, persona_neg_effective_responses


def save_persona_vector(model_name, pos_path, neg_path, trait, save_dir, threshold=50):
    # Set up credentials and environment
    config = setup_credentials()
    
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", token=config.hf_token)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=config.hf_token)

    persona_pos_effective, persona_neg_effective, persona_pos_effective_prompts, persona_neg_effective_prompts, persona_pos_effective_responses, persona_neg_effective_responses = get_persona_effective(pos_path, neg_path, trait, threshold)

    persona_effective_prompt_avg, persona_effective_prompt_last, persona_effective_response_avg = {}, {}, {}

    persona_effective_prompt_avg["pos"], persona_effective_prompt_last["pos"], persona_effective_response_avg["pos"] = get_hidden_p_and_r(model, tokenizer, persona_pos_effective_prompts, persona_pos_effective_responses)
    persona_effective_prompt_avg["neg"], persona_effective_prompt_last["neg"], persona_effective_response_avg["neg"] = get_hidden_p_and_r(model, tokenizer, persona_neg_effective_prompts, persona_neg_effective_responses)
    


    persona_effective_prompt_avg_diff = torch.stack([persona_effective_prompt_avg["pos"][l].mean(0).float() - persona_effective_prompt_avg["neg"][l].mean(0).float() for l in range(len(persona_effective_prompt_avg["pos"]))], dim=0)
    persona_effective_response_avg_diff = torch.stack([persona_effective_response_avg["pos"][l].mean(0).float() - persona_effective_response_avg["neg"][l].mean(0).float() for l in range(len(persona_effective_response_avg["pos"]))], dim=0)
    persona_effective_prompt_last_diff = torch.stack([persona_effective_prompt_last["pos"][l].mean(0).float() - persona_effective_prompt_last["neg"][l].mean(0).float() for l in range(len(persona_effective_prompt_last["pos"]))], dim=0)

    os.makedirs(save_dir, exist_ok=True)

    torch.save(persona_effective_prompt_avg_diff, f"{save_dir}/{trait}_prompt_avg_diff.pt")
    torch.save(persona_effective_response_avg_diff, f"{save_dir}/{trait}_response_avg_diff.pt")
    torch.save(persona_effective_prompt_last_diff, f"{save_dir}/{trait}_prompt_last_diff.pt")

    print(f"Persona vectors saved to {save_dir}")    

if __name__ == "__main__":
    # Clear GPU memory before starting
    # clear_gpu_memory()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--pos_path", type=str, required=True)
    parser.add_argument("--neg_path", type=str, required=True)
    parser.add_argument("--trait", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--threshold", type=int, default=50)
    args = parser.parse_args()
    save_persona_vector(args.model_name, args.pos_path, args.neg_path, args.trait, args.save_dir, args.threshold)