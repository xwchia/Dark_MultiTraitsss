import os, re, json, torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftConfig, PeftModel
import socket





def get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))  
        return s.getsockname()[1]


# ---------- 工具 ----------
_CHECKPOINT_RE = re.compile(r"checkpoint-(\d+)")

def _pick_latest_checkpoint(model_path: str) -> str:
    ckpts = [(int(m.group(1)), p) for p in Path(model_path).iterdir()
             if (m := _CHECKPOINT_RE.fullmatch(p.name)) and p.is_dir()]
    return str(max(ckpts, key=lambda x: x[0])[1]) if ckpts else model_path

def _is_lora(path: str) -> bool:
    return Path(path, "adapter_config.json").exists()

def _load_and_merge_lora(lora_path: str, dtype, device_map):
    cfg = PeftConfig.from_pretrained(lora_path)
    base = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_name_or_path, torch_dtype=dtype, device_map=device_map
    )
    return PeftModel.from_pretrained(base, lora_path).merge_and_unload()

def _load_tokenizer(path_or_id: str, cache_dir: str = None):
    """Load tokenizer with optional cache directory for shared storage."""
    load_kwargs = {}
    if cache_dir:
        load_kwargs["cache_dir"] = cache_dir
    tok = AutoTokenizer.from_pretrained(path_or_id, **load_kwargs)
    tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"
    return tok

def load_model(model_path: str, dtype=torch.bfloat16):
    # Smart device selection:
    # - MPS on Mac ARM if available
    # - Auto otherwise (works for CUDA on Linux, CPU fallback)
    if torch.backends.mps.is_available():
        device_map = "mps"
        print("Using MPS (Mac ARM GPU)")
    else:
        device_map = "auto"
        if torch.cuda.is_available():
            print(f"Using CUDA (GPU count: {torch.cuda.device_count()})")
        else:
            print("Using CPU")
    
    if not os.path.exists(model_path):               # ---- Hub ----
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, device_map=device_map
        )
        tok = _load_tokenizer(model_path)
        return model, tok

    resolved = _pick_latest_checkpoint(model_path)
    print(f"loading {resolved}")
    if _is_lora(resolved):
        model = _load_and_merge_lora(resolved, dtype, device_map)
        tok = _load_tokenizer(model.config._name_or_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            resolved, torch_dtype=dtype, device_map=device_map
        )
        tok = _load_tokenizer(resolved)
    return model, tok

def load_vllm_model(model_path: str, max_num_seqs: int = 64, gpu_memory_utilization: float = 0.95):
    """
    Load a model with vLLM for optimized inference.
    
    Args:
        model_path: Path or HuggingFace ID of the model
        max_num_seqs: Maximum number of sequences to process in parallel (default: 64)
                      Higher values = more parallelism but requires more GPU memory
        gpu_memory_utilization: Fraction of GPU memory to use (default: 0.95)
    """
    from vllm import LLM

    if not os.path.exists(model_path):               # ---- Hub ----
        # Optimized settings for all models
        print(f"🚀 Using optimized vLLM settings for: {model_path}")
        print(f"   - Max parallel sequences: {max_num_seqs}")
        print(f"   - GPU memory utilization: {gpu_memory_utilization}")
        print(f"   - Tensor parallel size: {torch.cuda.device_count()} GPUs")
        llm = LLM(
            model=model_path,
            enable_prefix_caching=True,
            enable_lora=True,
            tensor_parallel_size=torch.cuda.device_count(),
            max_num_seqs=max_num_seqs,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=30000,
            max_lora_rank=128,
        )
        tok = llm.get_tokenizer()
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
        tok.padding_side = "left"
        return llm, tok, None

    # ---- 本地 ----
    resolved = _pick_latest_checkpoint(model_path)
    print(f"loading {resolved}")
    is_lora = _is_lora(resolved)

    base_path = (PeftConfig.from_pretrained(resolved).base_model_name_or_path
                 if is_lora else resolved)

    # Optimized settings for all models
    print(f"🚀 Using optimized vLLM settings for: {base_path}")
    print(f"   - Max parallel sequences: {max_num_seqs}")
    print(f"   - GPU memory utilization: {gpu_memory_utilization}")
    print(f"   - Tensor parallel size: {torch.cuda.device_count()} GPUs")
    llm = LLM(
        model=base_path,
        enable_prefix_caching=True,
        enable_lora=True,
        tensor_parallel_size=torch.cuda.device_count(),
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=30000,
        max_lora_rank=128,
    )

    if is_lora:
        lora_path = resolved
    else:
        lora_path = None

    tok = llm.get_tokenizer()
    tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"
    return llm, tok, lora_path
