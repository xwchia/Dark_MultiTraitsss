"""
Run MentalBench-Align Evaluation for Single-Turn Crisis Responses

Evaluates model responses using the 7-attribute framework from MentalBench-Align:
- Cognitive Support Score (CSS): Guidance, Informativeness, Relevance, Safety
- Affective Resonance Score (ARS): Empathy, Helpfulness, Understanding

This is run AFTER the main protocol-based evaluation and generates additional
evaluation files: mentalbench_evaluations.csv and mentalbench_summary.json
"""

import json
import asyncio
from pathlib import Path
from typing import Optional
import fire

# Import from crisis_evaluation (shared module)
from ..crisis_evaluation.mentalbench_evaluator import MentalBenchEvaluator, generate_summary


def run_mentalbench_evaluation(
    responses_path: str,
    output_dir: str,
    judge_model: str = None,
    max_concurrent: int = 10,
    checkpoint_interval: int = 10,
    resume: bool = True,
):
    """
    Run MentalBench-Align 7-attribute evaluation on single-turn crisis responses.
    
    This evaluates each response independently, producing scores for:
    - CSS (Cognitive Support): Guidance, Informativeness, Relevance, Safety
    - ARS (Affective Resonance): Empathy, Helpfulness, Understanding
    
    Args:
        responses_path: Path to responses.json from single-turn evaluation
        output_dir: Directory to save outputs (mentalbench_evaluations.csv, mentalbench_summary.json)
        judge_model: Model to use for evaluation (auto-detected if None)
        max_concurrent: Max concurrent API calls (default: 10 for single-turn)
        checkpoint_interval: Save checkpoint every N evaluations
        resume: Resume from checkpoint if exists
    
    Outputs:
        - mentalbench_evaluations.csv: Per-probe scores for all 7 attributes + composites
        - mentalbench_summary.json: Summary statistics by attribute and category
        - mentalbench_checkpoint.json: Checkpoint file for resuming
    """
    print("=" * 60)
    print("MENTALBENCH-ALIGN EVALUATION (Single-Turn)")
    print("=" * 60)
    
    # Load responses
    print(f"\n📂 Loading responses from: {responses_path}")
    with open(responses_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    print(f"   Loaded {len(results)} responses")
    
    # Print category distribution
    cat_dist = {}
    for r in results:
        cat_dist[r.get("category", "unknown")] = cat_dist.get(r.get("category", "unknown"), 0) + 1
    print("   Categories:")
    for cat, count in sorted(cat_dist.items()):
        print(f"     {cat}: {count}")
    
    # Setup paths
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = output_dir / "mentalbench_evaluations.csv"
    summary_path = output_dir / "mentalbench_summary.json"
    checkpoint_path = output_dir / "mentalbench_checkpoint.json"
    
    # Load checkpoint if resuming
    existing_evals = []
    if resume and checkpoint_path.exists():
        with open(checkpoint_path, 'r') as f:
            existing_evals = json.load(f)
        print(f"   📥 Resuming from checkpoint: {len(existing_evals)} evaluations already completed")
    
    # Initialize evaluator
    print(f"\n📊 Running MentalBench-Align evaluation")
    evaluator = MentalBenchEvaluator(judge_model=judge_model)
    print(f"   Judge model: {evaluator.judge_model}")
    
    # Run evaluation
    evaluations = asyncio.run(evaluator.evaluate_singleturn(
        results,
        str(csv_path),
        max_concurrent=max_concurrent,
        checkpoint_path=str(checkpoint_path),
        checkpoint_interval=checkpoint_interval,
        existing_evals=existing_evals,
    ))
    
    # Generate summary
    generate_summary(evaluations, str(summary_path))
    
    print(f"\n📁 MentalBench outputs saved to: {output_dir}")
    print(f"   - mentalbench_evaluations.csv")
    print(f"   - mentalbench_summary.json")
    print("\n" + "=" * 60)
    print("MENTALBENCH EVALUATION COMPLETE")
    print("=" * 60)
    
    return evaluations


if __name__ == "__main__":
    fire.Fire(run_mentalbench_evaluation)













