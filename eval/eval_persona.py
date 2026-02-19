import os
import asyncio
import yaml
from typing import Dict, List
import json
import torch
import pandas as pd
import random
from vllm.lora.request import LoRARequest
from datasets import load_dataset
from tqdm import tqdm
import subprocess
import gc
import psutil

import torch
from vllm import LLM, SamplingParams

from judge import OpenAiJudge

from itertools import islice
import asyncio
from activation_steer import ActivationSteerer
from tqdm import trange
from transformers import AutoModelForCausalLM, AutoTokenizer
from eval.model_utils import load_model, load_vllm_model
import logging
from eval.prompts import Prompts
from config import setup_credentials

logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.ERROR)

# Set up credentials and environment
config = setup_credentials()

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

def sample_steering(model, tokenizer, conversations,  vector, layer, coef, bs=5, top_p=1, max_tokens=1000, temperature=1, min_tokens=1, steering_type="response"):
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Adaptive batch size based on model type
    problematic_models = ["mistral", "gemma"]
    is_problematic = any(prob_model in str(model.config).lower() for prob_model in problematic_models)
    adaptive_bs = bs  # Use original batch size, let other optimizations handle memory
    
    prompts = []
    for messages in conversations:
        prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    
    
    outputs = []
    for i in trange(0, len(prompts), adaptive_bs):
        batch = prompts[i:i+adaptive_bs]
        tokenized_batch = tokenizer(batch, return_tensors="pt", padding=True)
        tokenized_batch = {k: v.to(model.device) for k, v in tokenized_batch.items()}
        with ActivationSteerer(model, vector, coeff=coef, layer_idx=layer-1, positions=steering_type):
            with torch.no_grad():
                output = model.generate(**tokenized_batch, do_sample=(temperature > 0), temperature=temperature, top_p=top_p, max_new_tokens=max_tokens,use_cache=True, min_new_tokens=min_tokens)
        prompt_len = tokenized_batch["input_ids"].shape[1]
        output = [tokenizer.decode(o[prompt_len:], skip_special_tokens=True) for o in output]
        outputs.extend(output)
    return prompts, outputs


def sample(model, tokenizer, conversations, top_p=1, max_tokens=1000, temperature=1, min_tokens=1, lora_path=None):
    # Check if this is a vLLM model or transformers model
    from vllm import LLM
    if isinstance(model, LLM):
        # vLLM path
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            skip_special_tokens=True,
            stop=[tokenizer.eos_token],
            min_tokens=min_tokens
        )

        texts = []
        for i, messages in enumerate(conversations):
            texts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

        generate_kwargs = {
            "sampling_params": sampling_params,
            "use_tqdm": True
        }
        try:
            if lora_path:
                completions = model.generate(texts, **generate_kwargs, lora_request=LoRARequest("default", 1, lora_path=lora_path))
            else:
                completions = model.generate(texts, **generate_kwargs)
            answers = [completion.outputs[0].text for completion in completions]
        except Exception as e:
            # Handle vLLM decoding errors (e.g., "Invalid prefix encountered")
            print(f"⚠️  vLLM generation error: {e}")
            print(f"⚠️  Attempting to recover partial results...")
            answers = []
            # Try to extract any successful completions
            try:
                if 'completions' in locals():
                    for completion in completions:
                        try:
                            answers.append(completion.outputs[0].text)
                        except:
                            answers.append("")  # Empty string for failed responses
                else:
                    # If no completions at all, return empty strings
                    answers = [""] * len(texts)
            except:
                # Last resort: all empty strings
                answers = [""] * len(texts)
            print(f"⚠️  Recovered {len([a for a in answers if a])} successful responses out of {len(texts)}")
        return texts, answers
    else:
        # Transformers path
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        texts = []
        for messages in conversations:
            texts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        
        # Process in adaptive batches to balance speed and memory
        # Use larger batches for compatible models, smaller for problematic ones
        batch_size = 64  # Increased batch size for better VRAM utilization
        all_outputs = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Batch tokenize
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Generate with memory optimizations
            with torch.no_grad():
                generation_kwargs = {
                    "do_sample": (temperature > 0),
                    "temperature": temperature if temperature > 0 else None,
                    "top_p": top_p,
                    "max_new_tokens": max_tokens,
                    "min_new_tokens": min_tokens,
                    "pad_token_id": tokenizer.eos_token_id,
                    "eos_token_id": tokenizer.eos_token_id,
                }
                
                # Use optimized settings for all models
                generation_kwargs.update({
                    "use_cache": True,
                    "num_beams": 1,  # Disable beam search for faster generation
                    "do_sample": True,  # Enable sampling for parallel-friendly generation
                })
                
                batch_outputs = model.generate(**inputs, **generation_kwargs)
                all_outputs.append(batch_outputs)
        
        # Decode only the generated part (not the input)
        answers = []
        for batch_idx, batch_outputs in enumerate(all_outputs):
            batch_texts = texts[batch_idx*batch_size:(batch_idx+1)*batch_size]
            batch_inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True)
            input_length = batch_inputs["input_ids"].shape[1]
            
            for output in batch_outputs:
                generated_tokens = output[input_length:]
                answer = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                answers.append(answer)
        
        return texts, answers



def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f.readlines() if line.strip()]


class Question():
    def __init__(
            self, 
            id: str, 
            paraphrases: list[str], 
            judge_prompts: dict,
            temperature: float = 1,
            system: str = None, 
            judge: str = "gpt-4o",
            judge_eval_type: str = "0_100",
            **ignored_extra_args
        ):
        self.id = id
        self.paraphrases = paraphrases
        self.temperature = temperature
        self.system = system
        self.judges = {metric: OpenAiJudge(judge, prompt, eval_type=judge_eval_type if metric != "coherence" else "0_100") for metric, prompt in judge_prompts.items()}
    
    def get_input(self, n_per_question):
        paraphrases = random.choices(self.paraphrases, k=n_per_question)
        conversations = [[dict(role='user', content=i)] for i in paraphrases]
        if self.system:
            conversations = [[dict(role='system', content=self.system)] + c for c in conversations]
        return paraphrases, conversations
    
    async def eval(self, llm, tokenizer, coef, vector=None, layer=None, max_tokens=1000, n_per_question=100, steering_type="last", lora_path=None):
        paraphrases, conversations = self.get_input(n_per_question)
        if coef != 0:
            prompts, answers = sample_steering(llm, tokenizer, conversations, vector, layer, coef, temperature=self.temperature, max_tokens=max_tokens, steering_type=steering_type)
        else:
            prompts, answers = sample(llm, tokenizer, conversations, temperature=self.temperature, max_tokens=max_tokens, lora_path=lora_path)
        df = pd.DataFrame([
            dict(question=question,prompt=prompt, answer=answer, question_id=self.id)
            for question, answer, prompt in zip(paraphrases, answers, prompts)
        ])
        for score, judge in self.judges.items():
            scores = await asyncio.gather(*[
                judge(question=question, answer=answer)
                for question, answer in zip(paraphrases, answers)
            ])
            df[score] = scores
        return df
        

def a_or_an(word):
    return "an" if word[0].lower() in "aeiou" else "a"

def load_persona_questions(trait, temperature=1, persona_instructions_type=None, assistant_name=None, judge_model="gpt-4.1-mini", eval_type="0_100", version="eval"):
    trait_data = json.load(open(f"data_generation/trait_data_{version}/{trait}.json", "r"))
    judge_prompts = {}
    prompt_template = trait_data["eval_prompt"]
    judge_prompts[trait] = prompt_template
    judge_prompts["coherence"] = Prompts[f"coherence_{eval_type}"]
    raw_questions = trait_data["questions"]
    questions = []
    for i, question in enumerate(raw_questions):
        if persona_instructions_type is not None:
            persona_instructions = [x[persona_instructions_type] for x in trait_data["instruction"]]
            for k, instruction in enumerate(persona_instructions):
                # Use simple "You are an assistant." prefix for both positive and negative instructions
                system = f"You are an assistant. {instruction}"
                questions.append(Question(paraphrases=[question], id=f"{trait}_{i}_{persona_instructions_type}_{k}", judge_prompts=judge_prompts, judge=judge_model, temperature=temperature, system=system, judge_eval_type=eval_type ))
        else:
            questions.append(Question(paraphrases=[question], id=f"{trait}_{i}", judge_prompts=judge_prompts, judge=judge_model, temperature=temperature, judge_eval_type=eval_type ))
    return questions





async def eval_batched(questions, llm, tokenizer, coef, vector=None, layer=None, n_per_question=100, max_concurrent_judges=100, max_tokens=1000, steering_type="last", lora_path=None):
    """Batch process all questions together for faster inference"""
    # Collect all prompts from all questions
    all_paraphrases = []
    all_conversations = []
    question_indices = []
    for i, question in enumerate(questions):
        paraphrases, conversations = question.get_input(n_per_question)
        all_paraphrases.extend(paraphrases)
        all_conversations.extend(conversations)
        question_indices.extend([i] * len(paraphrases))
    
    # Generate all answers in a single batch
    print(f"Generating {len(all_conversations)} responses in a single batch...")
    if coef != 0:
        prompts, answers = sample_steering(llm, tokenizer, all_conversations, vector, layer, coef, temperature=questions[0].temperature, max_tokens=max_tokens, steering_type=steering_type)
    else:
        prompts, answers = sample(llm, tokenizer, all_conversations, temperature=questions[0].temperature, max_tokens=max_tokens, lora_path=lora_path)
    
    # Prepare data structures for batch evaluation
    question_dfs = []
    all_judge_tasks = []
    all_judge_indices = []  # Store (question_idx, metric, sample_idx) for each task
    
    print("Preparing judge evaluation tasks...")
    for i, question in enumerate(questions):
        # Get this question's data
        indices = [j for j, idx in enumerate(question_indices) if idx == i]
        q_paraphrases = [all_paraphrases[j] for j in indices]
        q_prompts = [prompts[j] for j in indices]
        q_answers = [answers[j] for j in indices]
        
        # Create dataframe for this question
        df = pd.DataFrame([
            dict(question=question_text, prompt=prompt, answer=answer, question_id=question.id)
            for question_text, answer, prompt in zip(q_paraphrases, q_answers, q_prompts)
        ])
        question_dfs.append(df)
        
        # Collect all judge tasks
        for metric, judge in question.judges.items():
            for sample_idx, (question_text, answer) in enumerate(zip(q_paraphrases, q_answers)):
                # Skip evaluation for empty/failed responses
                if not answer or answer.strip() == "":
                    all_judge_tasks.append(None)  # Placeholder for failed response
                    all_judge_indices.append((i, metric, sample_idx))
                else:
                    all_judge_tasks.append((judge, question_text, answer))
                    all_judge_indices.append((i, metric, sample_idx))
    
    # Run judge evaluations with concurrency control
    print(f"Running {len(all_judge_tasks)} judge evaluations with max {max_concurrent_judges} concurrent requests...")
    all_results = [None] * len(all_judge_tasks)  # Pre-allocate results array
    
    # Create a semaphore to limit concurrency
    semaphore = asyncio.Semaphore(max_concurrent_judges)
    
    async def run_with_semaphore(task_idx, judge, question_text, answer):
        async with semaphore:
            # Handle None tasks (failed responses)
            if judge is None:
                return task_idx, float('nan')
            result = await judge(question=question_text, answer=answer)
            return task_idx, result
    
    # Create all tasks with semaphore control
    tasks = []
    for task_idx, task in enumerate(all_judge_tasks):
        if task is None:
            # Failed response - create task that returns NaN
            tasks.append(run_with_semaphore(task_idx, None, None, None))
        else:
            judge, question_text, answer = task
            tasks.append(run_with_semaphore(task_idx, judge, question_text, answer))
    
    # Process tasks in batches with progress bar and rate limiting pauses
    import time
    completed_count = 0
    with tqdm(total=len(tasks), desc="Judge evaluations") as pbar:
        for task in asyncio.as_completed(tasks):
            task_idx, result = await task
            all_results[task_idx] = result  # Store result in correct position
            completed_count += 1
            pbar.update(1)
            
            # Pause every 50 evaluations to prevent rate limits
            if completed_count % 50 == 0 and completed_count < len(tasks):
                print(f"\n⏸️  Pausing for 3 seconds after {completed_count} evaluations to respect rate limits...")
                await asyncio.sleep(3)  # Wait 3 seconds
                print(f"▶️  Resuming evaluations...")
    
    # Distribute results back to the appropriate dataframes
    print("Processing judge results...")
    for task_idx, result in enumerate(all_results):
        question_idx, metric, sample_idx = all_judge_indices[task_idx]
        question_dfs[question_idx].loc[sample_idx, metric] = result
    
    return question_dfs

def main(model, trait, output_path=None, coef=0, vector_path=None, layer=None, steering_type="response", max_tokens=1000, n_per_question=10, batch_process=True, max_concurrent_judges=100, persona_instruction_type=None, assistant_name=None, judge_model="gpt-4.1-mini", version="extract", overwrite=False, use_vllm=None, vllm_max_seqs=64, vllm_gpu_mem=0.95):
    """
    Evaluate a model on all questions from the evaluation yaml file.
    
    Args:
        vllm_max_seqs: Maximum number of sequences to process in parallel with vLLM (default: 64)
                       Higher values = more parallelism but requires more GPU memory
        vllm_gpu_mem: Fraction of GPU memory to use with vLLM (default: 0.95)
                      Can be increased to 0.98 for more capacity
    """
    
    # Clear GPU memory before starting
    # clear_gpu_memory()
    
    # Auto-generate output path if not provided
    if output_path is None:
        # Extract model name from path (e.g., "Qwen/Qwen2.5-1.5B" -> "Qwen2.5-1.5B")
        model_name = model.split('/')[-1] if '/' in model else model
        model_dir = f"output/{model_name}"
        
        # Add suffix based on configuration
        suffix = ""
        if coef != 0:
            suffix += f"_steer_coef{coef}"
            if layer is not None:
                suffix += f"_layer{layer}"
        if persona_instruction_type:
            suffix += f"_{persona_instruction_type}"
        if version != "extract":
            suffix += f"_{version}"
        
        output_path = f"{model_dir}/{trait}{suffix}.csv"
    
    if os.path.exists(output_path) and not overwrite:
        print(f"Output path {output_path} already exists, skipping...")
        df = pd.read_csv(output_path)
        for trait in [trait , "coherence"]:
            threshold = 50
            print(f"{trait}:  {df[trait].mean():.2f} +- {df[trait].std():.2f}")
        return
    
    print(f"📁 Output will be saved to: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if n_per_question == 1:
        temperature = 0.0
    else:
        temperature = 1.0
    # Determine which model loader to use
    if use_vllm is None:
        # Auto-detect: avoid vLLM on macOS due to triton issues
        import platform
        use_vllm = not (platform.system() == "Darwin" and coef == 0)
    elif isinstance(use_vllm, str):
        # Convert string to boolean (fire.Fire passes strings)
        use_vllm = use_vllm.lower() in ('true', '1', 'yes', 'on')
    
    if coef != 0:
        # Use transformers for steering (required)
        llm, tokenizer = load_model(model)
        lora_path = None
        vector = torch.load(vector_path, weights_only=False)[layer]
    elif use_vllm:
        # Use vLLM for faster inference with configurable parallelism
        llm, tokenizer, lora_path = load_vllm_model(
            model, 
            max_num_seqs=vllm_max_seqs,
            gpu_memory_utilization=vllm_gpu_mem
        )
        vector=None
    else:
        # Use transformers as fallback
        llm, tokenizer = load_model(model)
        lora_path = None
        vector=None
    questions = load_persona_questions(trait, temperature=temperature, persona_instructions_type=persona_instruction_type, assistant_name=assistant_name, judge_model=judge_model, version=version)
    if batch_process:
        print(f"Batch processing {len(questions)} '{trait}' questions...")
        outputs_list = asyncio.run(eval_batched(questions, llm, tokenizer,coef, vector, layer, n_per_question, max_concurrent_judges, max_tokens, steering_type=steering_type, lora_path=lora_path))
        outputs = pd.concat(outputs_list)
    else:
        outputs = []
        for question in tqdm(questions, desc=f"Processing {trait} questions"):
            outputs.append(asyncio.run(question.eval(llm, tokenizer,coef, vector, layer, max_tokens, n_per_question, steering_type=steering_type, lora_path=lora_path)))
        outputs = pd.concat(outputs)
    outputs.to_csv(output_path, index=False)
    print(output_path)
    for trait in [trait , "coherence"]:
        print(f"{trait}:  {outputs[trait].mean():.2f} +- {outputs[trait].std():.2f}")


if __name__ == "__main__":
    import fire
    fire.Fire(main)
