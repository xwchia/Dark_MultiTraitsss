"""
Single-Turn Crisis Evaluation

Reuses infrastructure from crisis_evaluation but for single-turn probes.
Supports checkpointing for resumable runs.
"""

import json
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Set
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import fire

# Reuse from crisis_evaluation
from ..crisis_evaluation.conversation_runner import ConversationRunner
from ..crisis_evaluation.protocol_evaluator import ProtocolEvaluator

from .dataset import SingleTurnDataset


def _load_checkpoint(checkpoint_path: Path) -> List[Dict]:
    """Load existing checkpoint if it exists."""
    if checkpoint_path.exists():
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []


def _save_checkpoint(data: List[Dict], checkpoint_path: Path):
    """Save checkpoint to file."""
    with open(checkpoint_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def run_singleturn_evaluation(
    model: str = None,
    data_path: str = None,
    output_dir: str = None,
    baked_model_dir: Optional[str] = None,
    system_prompt: Optional[str] = None,
    # Dataset options (if data_path not provided)
    categories: Optional[str] = None,
    n_per_category: Optional[int] = None,
    use_test_set: bool = True,
    format_style: str = "natural",
    # Generation options
    judge_model: str = None,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    max_concurrent_evals: int = 10,
    # Skip options
    skip_generation: bool = False,
    skip_evaluation: bool = False,
    seed: int = 42,
    # Checkpoint options
    checkpoint_interval: int = 10,
    resume: bool = True,
):
    """
    Run single-turn crisis evaluation pipeline.
    
    Args:
        model: HuggingFace model name/path (required if baked_model_dir not provided)
        data_path: Path to probes JSON (if None, extracts from dataset)
        output_dir: Directory for outputs
        baked_model_dir: Path to baked model directory with metadata.json
        system_prompt: System prompt to prepend to each conversation
        categories: Comma-separated categories (if extracting)
        n_per_category: Samples per category (if extracting)
        use_test_set: Use test set (2046) or validation (206)
        format_style: natural/dialogue/single_message
        judge_model: Model for evaluation
        max_new_tokens: Max tokens per response
        temperature: Sampling temperature
        max_concurrent_evals: Concurrent API calls
        skip_generation: Skip response generation
        skip_evaluation: Skip evaluation
        seed: Random seed
        checkpoint_interval: Save checkpoint every N probes (default: 10)
        resume: Resume from existing checkpoint if available (default: True)
    """
    print("=" * 60)
    print("SINGLE-TURN CRISIS EVALUATION")
    print("=" * 60)
    
    if not baked_model_dir and not model:
        raise ValueError("Either --model or --baked_model_dir must be provided")
    
    # Setup paths
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    probes_path = output_dir / "probes.json"
    responses_path = output_dir / "responses.json"
    responses_checkpoint_path = output_dir / "responses_checkpoint.json"
    evaluations_path = output_dir / "evaluations.csv"
    evaluations_checkpoint_path = output_dir / "evaluations_checkpoint.json"
    summary_path = output_dir / "summary.json"
    metadata_path = output_dir / "metadata.json"
    
    # Step 0: Load or extract probes
    if data_path:
        print(f"\n📂 Loading probes from: {data_path}")
        probes = SingleTurnDataset.load_probes(data_path)
    else:
        print(f"\n📂 Extracting probes from dataset")
        cat_list = [c.strip() for c in categories.split(',')] if categories else None
        
        dataset = SingleTurnDataset(format_style=format_style)
        probes = dataset.load_by_category(
            categories=cat_list,
            n_per_category=n_per_category,
            use_test_set=use_test_set,
            seed=seed,
        )
        dataset.save_probes(probes, str(probes_path))
    
    print(f"   Total probes: {len(probes)}")
    
    # Print category distribution
    cat_dist = {}
    for p in probes:
        cat_dist[p.category] = cat_dist.get(p.category, 0) + 1
    for cat, count in sorted(cat_dist.items()):
        print(f"     {cat}: {count}")
    
    # Step 1: Generate responses
    if not skip_generation:
        print(f"\n🤖 Step 1: Generating responses")
        
        # Load checkpoint if resuming
        existing_results = []
        completed_ids: Set[str] = set()
        if resume:
            existing_results = _load_checkpoint(responses_checkpoint_path)
            completed_ids = {r["probe_id"] for r in existing_results}
            if existing_results:
                print(f"   📥 Resuming from checkpoint: {len(existing_results)} already completed")
        
        # Filter probes to only process incomplete ones
        remaining_probes = [p for p in probes if p.probe_id not in completed_ids]
        
        if remaining_probes:
            # Reuse ConversationRunner from crisis_evaluation
            if baked_model_dir:
                print(f"   Baked model: {baked_model_dir}")
                if system_prompt:
                    print(f"   System prompt: {system_prompt[:80]}...")
                runner = ConversationRunner(baked_model_dir=baked_model_dir, system_prompt=system_prompt)
                baked_metadata = runner.baked_metadata
                model_name = baked_metadata.get("original_model", "unknown")
                alpha = list(baked_metadata.get("trait_alphas", {}).values())[0] if baked_metadata.get("trait_alphas") else 0.0
                rank = baked_metadata.get("manifold_rank")
            else:
                print(f"   Model: {model}")
                if system_prompt:
                    print(f"   System prompt: {system_prompt[:80]}...")
                runner = ConversationRunner(model_name=model, alpha=0.0, system_prompt=system_prompt)
                model_name = model
                alpha = 0.0
                rank = None
            
            # Generate single-turn responses with checkpointing
            new_results = _generate_singleturn_responses(
                runner, remaining_probes, max_new_tokens, temperature,
                checkpoint_path=responses_checkpoint_path,
                checkpoint_interval=checkpoint_interval,
                existing_results=existing_results,
            )
            results = existing_results + new_results
        else:
            print(f"   ⏭️  All probes already completed in checkpoint")
            results = existing_results
            # Extract model info from existing results
            if results:
                model_name = results[0].get("config", {}).get("model", model)
                alpha = results[0].get("config", {}).get("alpha", 0.0)
                rank = results[0].get("config", {}).get("rank")
            else:
                model_name = model
                alpha = 0.0
                rank = None
        
        # Sort by probe_id for consistent output
        results.sort(key=lambda x: x["probe_id"])
        
        # Save final responses
        with open(responses_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"   ✅ Generated {len(results)} responses -> {responses_path}")
    else:
        print(f"\n⏭️  Skipping generation, loading existing responses")
        with open(responses_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        model_name = results[0].get("config", {}).get("model", model) if results else model
        alpha = results[0].get("config", {}).get("alpha", 0.0) if results else 0.0
        rank = results[0].get("config", {}).get("rank") if results else None
    
    # Step 2: Evaluate responses (single-turn)
    if not skip_evaluation:
        print(f"\n📊 Step 2: Evaluating responses")
        
        # Load evaluation checkpoint if resuming
        existing_evals = []
        completed_eval_ids: Set[str] = set()
        if resume:
            existing_evals = _load_checkpoint(evaluations_checkpoint_path)
            completed_eval_ids = {e["probe_id"] for e in existing_evals}
            if existing_evals:
                print(f"   📥 Resuming evaluations from checkpoint: {len(existing_evals)} already completed")
        
        # Filter results to only evaluate incomplete ones
        remaining_results = [r for r in results if r["probe_id"] not in completed_eval_ids]
        
        if remaining_results:
            evaluator = ProtocolEvaluator(judge_model=judge_model)
            print(f"   Judge model: {evaluator.judge_model}")
            
            new_evaluations = asyncio.run(_evaluate_singleturn(
                evaluator, remaining_results, max_concurrent_evals,
                checkpoint_path=evaluations_checkpoint_path,
                checkpoint_interval=checkpoint_interval,
                existing_evals=existing_evals,
            ))
            evaluations = existing_evals + new_evaluations
        else:
            print(f"   ⏭️  All evaluations already completed in checkpoint")
            evaluations = existing_evals
        
        # Sort by probe_id
        evaluations.sort(key=lambda x: x["probe_id"])
        
        # Save evaluations
        _save_evaluations(evaluations, str(evaluations_path))
        
        # Generate summary
        summary = _generate_summary(evaluations)
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        _print_summary(summary)
        print(f"   ✅ Completed {len(evaluations)} evaluations")
    else:
        print(f"\n⏭️  Skipping evaluation")
        summary = None
    
    # Save metadata
    metadata = {
        "mode": "singleturn",
        "model": model_name,
        "baked_model_dir": str(baked_model_dir) if baked_model_dir else None,
        "alpha": alpha,
        "rank": rank,
        "judge_model": judge_model,
        "num_probes": len(probes),
        "categories": categories,
        "n_per_category": n_per_category,
        "use_test_set": use_test_set,
        "format_style": format_style,
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "seed": seed,
        "timestamp": datetime.now().isoformat(),
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n📁 Outputs saved to: {output_dir}")
    print("=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    
    return summary


def _generate_singleturn_responses(
    runner: ConversationRunner,
    probes: List,
    max_new_tokens: int,
    temperature: float,
    checkpoint_path: Optional[Path] = None,
    checkpoint_interval: int = 10,
    existing_results: Optional[List[Dict]] = None,
) -> List[Dict]:
    """Generate single-turn responses using ConversationRunner with checkpointing."""
    results = []
    all_results = existing_results or []
    
    for i, probe in enumerate(tqdm(probes, desc="Generating")):
        # Build single message
        messages = [{"role": "user", "content": probe.user_prompt}]
        
        # Generate using runner's internal method
        response = runner._generate_response(
            messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        
        result = {
            "probe_id": probe.probe_id,
            "category": probe.category,
            "user_prompt": probe.user_prompt,
            "model_response": response,
            "config": {
                "model": runner.model_name,
                "alpha": runner.alpha,
                "rank": runner.rank,
            }
        }
        results.append(result)
        all_results.append(result)
        
        # Save checkpoint periodically
        if checkpoint_path and (i + 1) % checkpoint_interval == 0:
            _save_checkpoint(all_results, checkpoint_path)
            tqdm.write(f"   💾 Checkpoint saved: {len(all_results)} responses")
    
    # Final checkpoint save
    if checkpoint_path and results:
        _save_checkpoint(all_results, checkpoint_path)
        tqdm.write(f"   💾 Final checkpoint saved: {len(all_results)} responses")
    
    return results


async def _evaluate_singleturn(
    evaluator: ProtocolEvaluator,
    results: List[Dict],
    max_concurrent: int,
    checkpoint_path: Optional[Path] = None,
    checkpoint_interval: int = 10,
    existing_evals: Optional[List[Dict]] = None,
) -> List[Dict]:
    """Evaluate single-turn responses with checkpointing."""
    evaluations = []
    all_evals = existing_evals.copy() if existing_evals else []
    semaphore = asyncio.Semaphore(max_concurrent)
    counter = [0]  # Use list for mutable counter in closure
    
    async def eval_one(result):
        async with semaphore:
            eval_result = await evaluator._evaluate_single(
                category=result["category"],
                user_input=result["user_prompt"],
                model_response=result["model_response"],
                conversation_context=None,  # No context for single-turn
            )
            
            # Skip if category not in protocol (returns None)
            if eval_result is None:
                return None
            
            score, reasoning = eval_result
            return {
                "probe_id": result["probe_id"],
                "category": result["category"],
                "score": score,
                "reasoning": reasoning,
                "config": result.get("config", {}),
            }
    
    tasks = [eval_one(r) for r in results]
    
    skipped_count = 0
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Evaluating"):
        eval_result = await coro
        
        # Skip None results (unknown categories)
        if eval_result is None:
            skipped_count += 1
            continue
            
        evaluations.append(eval_result)
        all_evals.append(eval_result)
        counter[0] += 1
        
        # Save checkpoint periodically
        if checkpoint_path and counter[0] % checkpoint_interval == 0:
            _save_checkpoint(all_evals, checkpoint_path)
            tqdm.write(f"   💾 Evaluation checkpoint saved: {len(all_evals)} evaluations")
    
    if skipped_count > 0:
        tqdm.write(f"   ⏭️  Skipped {skipped_count} responses with unknown categories")
    
    # Final checkpoint save
    if checkpoint_path and evaluations:
        _save_checkpoint(all_evals, checkpoint_path)
        tqdm.write(f"   💾 Final evaluation checkpoint saved: {len(all_evals)} evaluations")
    
    return evaluations


def _save_evaluations(evaluations: List[Dict], output_path: str):
    """Save evaluations to CSV."""
    rows = []
    for e in evaluations:
        rows.append({
            "probe_id": e["probe_id"],
            "category": e["category"],
            "score": e["score"],
            "reasoning": e["reasoning"],
            "model": e["config"].get("model", ""),
            "alpha": e["config"].get("alpha", 0.0),
            "rank": e["config"].get("rank"),
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"   Saved evaluations to {output_path}")


def _generate_summary(evaluations: List[Dict]) -> Dict:
    """Generate summary statistics."""
    scores = [e["score"] for e in evaluations]
    
    summary = {
        "overall": {
            "n": len(evaluations),
            "mean": sum(scores) / len(scores),
            "std": pd.Series(scores).std(),
            "score_distribution": {str(i): sum(1 for s in scores if s == i) for i in range(1, 6)},
        },
        "by_category": {},
    }
    
    by_cat = {}
    for e in evaluations:
        by_cat.setdefault(e["category"], []).append(e["score"])
    
    for cat, cat_scores in sorted(by_cat.items()):
        summary["by_category"][cat] = {
            "n": len(cat_scores),
            "mean": sum(cat_scores) / len(cat_scores),
            "std": pd.Series(cat_scores).std(),
        }
    
    return summary


def _print_summary(summary: Dict):
    """Print summary to console."""
    print(f"\n   Overall: mean={summary['overall']['mean']:.2f}, n={summary['overall']['n']}")
    print(f"   By category:")
    for cat, stats in summary["by_category"].items():
        print(f"     {cat}: {stats['mean']:.2f} (n={stats['n']})")


if __name__ == "__main__":
    fire.Fire(run_singleturn_evaluation)
