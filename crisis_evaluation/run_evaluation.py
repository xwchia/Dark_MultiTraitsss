"""
Run Evaluation - Main Orchestration Script

Orchestrates the full crisis evaluation pipeline:
1. Load extracted conversations
2. Generate model responses (with optional steering)
3. Evaluate responses using clinical protocol
4. Save results
"""

import json
import asyncio
from pathlib import Path
from typing import Optional
from datetime import datetime
import fire

from .dataset_extractor import extract_crisis_conversations
from .conversation_runner import ConversationRunner
from .protocol_evaluator import ProtocolEvaluator
from .summarize_evaluations import summarize_evaluations
from .coherence_evaluator import run_coherence_evaluation


def run_evaluation(
    model: str = None,
    data_path: str = None,
    output_dir: str = None,
    baked_model_dir: Optional[str] = None,
    vector_path: Optional[str] = None,
    rank: Optional[int] = None,
    alpha: float = 0.0,
    layer: Optional[int] = None,
    judge_model: str = None,  # Auto-detected (Azure or OpenAI)
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    max_concurrent_evals: int = 5,
    skip_generation: bool = False,
    skip_evaluation: bool = False,
    checkpoint_interval: int = 10,
    resume: bool = True,
    system_prompt: Optional[str] = None,
):
    """
    Run the full evaluation pipeline.
    
    Args:
        model: HuggingFace model name/path (required if baked_model_dir not provided)
        data_path: Path to extracted conversations JSON
        output_dir: Directory for outputs (responses.json, evaluations.csv, metadata.json)
        baked_model_dir: Path to baked model directory with metadata.json (alternative to model+steering)
        vector_path: Path to steering vector (optional, legacy)
        rank: Low-rank dimension for steering (optional, legacy)
        alpha: Steering coefficient (0.0 = baseline, legacy)
        layer: Layer to apply steering (optional, legacy)
        judge_model: Model to use for evaluation (auto-detected if None)
        max_new_tokens: Max tokens per response
        temperature: Sampling temperature
        max_concurrent_evals: Max concurrent evaluation API calls
        skip_generation: Skip response generation (use existing responses.json)
        skip_evaluation: Skip evaluation (only generate responses)
        checkpoint_interval: Save checkpoint every N conversations (default: 10)
        resume: Whether to resume from checkpoint if exists (default: True)
        system_prompt: Optional system prompt to prepend to all conversations
    """
    print("=" * 60)
    print("CRISIS EVALUATION PIPELINE")
    print("=" * 60)
    
    # Validate inputs
    if not baked_model_dir and not model:
        raise ValueError("Either --model or --baked_model_dir must be provided")
    
    # Setup paths
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    responses_path = output_dir / "responses.json"
    evaluations_path = output_dir / "evaluations.csv"
    metadata_path = output_dir / "metadata.json"
    
    # Load conversations
    print(f"\n📂 Loading conversations from: {data_path}")
    with open(data_path, 'r', encoding='utf-8') as f:
        conversations = json.load(f)
    print(f"   Loaded {len(conversations)} conversations")
    
    # Step 1: Generate responses
    if not skip_generation:
        print(f"\n🤖 Step 1: Generating responses")
        
        if baked_model_dir:
            print(f"   Baked model: {baked_model_dir}")
            if system_prompt:
                print(f"   System prompt: {system_prompt[:80]}..." if len(system_prompt) > 80 else f"   System prompt: {system_prompt}")
            runner = ConversationRunner(
                baked_model_dir=baked_model_dir,
                system_prompt=system_prompt,
            )
            # Get metadata for saving
            baked_metadata = runner.baked_metadata
            model = baked_metadata.get("original_model", "unknown")
            alpha = list(baked_metadata.get("trait_alphas", {}).values())[0] if baked_metadata.get("trait_alphas") else 0.0
            rank = baked_metadata.get("manifold_rank")
        else:
            print(f"   Model: {model}")
            print(f"   Steering: alpha={alpha}, rank={rank}, layer={layer}")
            if system_prompt:
                print(f"   System prompt: {system_prompt[:80]}..." if len(system_prompt) > 80 else f"   System prompt: {system_prompt}")
            runner = ConversationRunner(
                model_name=model,
                vector_path=vector_path,
                rank=rank,
                alpha=alpha,
                layer=layer,
                system_prompt=system_prompt,
            )
        
        results = runner.run_all_conversations(
            conversations,
            str(responses_path),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            checkpoint_interval=checkpoint_interval,
            resume=resume,
        )
        
        print(f"   ✅ Generated {len(results)} conversation responses")
    else:
        print(f"\n⏭️  Skipping generation, loading existing responses")
        with open(responses_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
    
    # Step 2: Evaluate responses
    if not skip_evaluation:
        print(f"\n📊 Step 2: Evaluating responses")
        
        evaluator = ProtocolEvaluator(judge_model=judge_model, dataset_path=data_path)
        print(f"   Judge model: {evaluator.judge_model}")
        
        evaluations = asyncio.run(evaluator.evaluate_all(
            results,
            str(evaluations_path),
            max_concurrent=max_concurrent_evals,
        ))
        
        print(f"   ✅ Completed {len(evaluations)} conversation evaluations")
        
        # Step 3: Evaluate coherence
        print(f"\n🧠 Step 3: Evaluating coherence")
        coherence_results = run_coherence_evaluation(
            responses_path=str(responses_path),
            output_dir=str(output_dir),
            judge_model=judge_model,
            max_concurrent=max_concurrent_evals,
            append_to_summary=False,  # We'll handle this after summarize
        )
        print(f"   ✅ Mean coherence: {coherence_results['mean_coherence']:.2f}")
    else:
        print(f"\n⏭️  Skipping evaluation")
    
    # Save metadata
    metadata = {
        "model": model,
        "baked_model_dir": str(baked_model_dir) if baked_model_dir else None,
        "vector_path": str(vector_path) if vector_path else None,
        "rank": rank,
        "alpha": alpha,
        "layer": layer,
        "judge_model": judge_model,
        "num_conversations": len(conversations),
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "timestamp": datetime.now().isoformat(),
        "data_path": str(data_path),
        "system_prompt": system_prompt,
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n📁 Outputs saved to: {output_dir}")
    print(f"   - responses.json")
    print(f"   - evaluations.csv")
    print(f"   - metadata.json")
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)


def run_all_configurations(
    model: str,
    data_path: str,
    base_output_dir: str,
    hyperparam_search_dir: str,
    judge_model: str = None,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
):
    """
    Run evaluation for all three configurations: baseline, dark_safe, dark_aggressive.
    
    Uses baked models from hyperparameter search output.
    
    Args:
        model: HuggingFace model name/path (for baseline)
        data_path: Path to extracted conversations JSON
        base_output_dir: Base directory for outputs (model subdir will be created)
        hyperparam_search_dir: Path to hyperparameter search output directory
            e.g., research_scripts/figure_outputs/figure9/hyperparameter_search/Llama-3.1-8B-Instruct/manifold/output
        judge_model: Model to use for evaluation (auto-detected if None)
        max_new_tokens: Max tokens per response
        temperature: Sampling temperature
    """
    model_name = model.split('/')[-1]
    base_output_dir = Path(base_output_dir) / model_name
    hyperparam_dir = Path(hyperparam_search_dir)
    
    configurations = [
        {
            "name": "baseline",
            "baked_model_dir": None,  # Use base model
        },
        {
            "name": "dark_safe_r256_a1.0",
            "baked_model_dir": hyperparam_dir / "hyperparam_search_d256" / "baked_models" / "alpha_1_0",
        },
        {
            "name": "dark_aggressive_r128_a1.5",
            "baked_model_dir": hyperparam_dir / "hyperparam_search_d128" / "baked_models" / "alpha_1_5",
        },
    ]
    
    for config in configurations:
        print(f"\n{'='*60}")
        print(f"Running configuration: {config['name']}")
        print(f"{'='*60}")
        
        output_dir = base_output_dir / config["name"]
        
        if config["baked_model_dir"]:
            # Use baked model
            run_evaluation(
                data_path=data_path,
                output_dir=str(output_dir),
                baked_model_dir=str(config["baked_model_dir"]),
                judge_model=judge_model,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
        else:
            # Use base model (baseline)
            run_evaluation(
                model=model,
                data_path=data_path,
                output_dir=str(output_dir),
                alpha=0.0,
                judge_model=judge_model,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )


if __name__ == "__main__":
    fire.Fire({
        "run": run_evaluation,
        "run_all": run_all_configurations,
        "summarize": summarize_evaluations,
        "coherence": run_coherence_evaluation,
    })

