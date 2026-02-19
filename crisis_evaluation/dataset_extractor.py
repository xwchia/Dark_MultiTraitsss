"""
Dataset Extractor for Mental Health Crisis Conversations

Extracts multi-turn conversations from the LLMs-Mental-Health-Crisis dataset
with filtering by crisis category and minimum number of turns.

Supports two modes:
1. Original mode: Keep original category labels
2. Binary mode: Combine crisis categories into "crisis" vs "no_crisis"
"""

import json
import os
import random
from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict
import fire


def load_labeled_dataset(repo_path: str) -> List[Dict]:
    """Load the labeled dataset from the cloned repository."""
    labeled_path = Path(repo_path) / "data" / "llm_label" / "gpt-4o-mini-labeled-sampled_dataset_n_2046_nPerD168_seed0-merged-labels.json"
    
    if not labeled_path.exists():
        raise FileNotFoundError(f"Labeled dataset not found at {labeled_path}")
    
    with open(labeled_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} conversations from labeled dataset")
    return data


def filter_by_categories(
    data: List[Dict], 
    categories: List[str]
) -> List[Dict]:
    """Filter conversations by crisis category."""
    filtered = [conv for conv in data if conv.get('label') in categories]
    
    # Count per category
    category_counts = defaultdict(int)
    for conv in filtered:
        category_counts[conv['label']] += 1
    
    print(f"Filtered to {len(filtered)} conversations across categories:")
    for cat, count in sorted(category_counts.items()):
        print(f"  - {cat}: {count}")
    
    return filtered


def filter_by_min_turns(
    data: List[Dict], 
    min_turns: int
) -> List[Dict]:
    """Filter conversations with at least min_turns user messages."""
    filtered = [conv for conv in data if len(conv.get('inputs', [])) >= min_turns]
    
    print(f"Filtered to {len(filtered)} conversations with >= {min_turns} turns")
    return filtered


def sample_conversations(
    data: List[Dict],
    num_conversations: int,
    categories: List[str],
    seed: int = 42
) -> List[Dict]:
    """
    Sample conversations with balanced distribution across categories.
    
    Args:
        data: List of conversations
        num_conversations: Total number to sample
        categories: List of categories to balance across
        seed: Random seed for reproducibility
    
    Returns:
        List of sampled conversations
    """
    random.seed(seed)
    
    # Group by category
    by_category = defaultdict(list)
    for conv in data:
        by_category[conv['label']].append(conv)
    
    # Calculate samples per category
    per_category = num_conversations // len(categories)
    remainder = num_conversations % len(categories)
    
    sampled = []
    for i, category in enumerate(categories):
        available = by_category[category]
        n_samples = per_category + (1 if i < remainder else 0)
        
        if len(available) < n_samples:
            print(f"Warning: Only {len(available)} conversations available for {category}, requested {n_samples}")
            n_samples = len(available)
        
        sampled.extend(random.sample(available, n_samples))
    
    # Shuffle the final list
    random.shuffle(sampled)
    
    print(f"Sampled {len(sampled)} conversations:")
    category_counts = defaultdict(int)
    for conv in sampled:
        category_counts[conv['label']] += 1
    for cat, count in sorted(category_counts.items()):
        print(f"  - {cat}: {count}")
    
    return sampled


def truncate_to_turns(
    data: List[Dict], 
    max_turns: int
) -> List[Dict]:
    """Truncate each conversation to exactly max_turns."""
    truncated = []
    for conv in data:
        new_conv = conv.copy()
        new_conv['inputs'] = conv['inputs'][:max_turns]
        truncated.append(new_conv)
    
    return truncated


def add_conversation_ids(data: List[Dict]) -> List[Dict]:
    """Add unique conversation IDs."""
    for i, conv in enumerate(data):
        conv['conversation_id'] = f"conv_{i+1:03d}"
    return data


def relabel_to_binary(data: List[Dict], crisis_categories: List[str]) -> List[Dict]:
    """
    Relabel conversations to binary crisis/no_crisis labels.
    
    Args:
        data: List of conversations
        crisis_categories: List of original labels to map to "crisis"
    
    Returns:
        List of conversations with binary labels
    """
    relabeled = []
    for conv in data:
        new_conv = conv.copy()
        original_label = conv.get('label', '')
        new_conv['original_label'] = original_label
        new_conv['label'] = 'crisis' if original_label in crisis_categories else 'no_crisis'
        relabeled.append(new_conv)
    
    return relabeled


def save_checkpoint(data: List[Dict], checkpoint_path: Path, processed_count: int):
    """Save checkpoint with current progress."""
    checkpoint_data = {
        'processed_count': processed_count,
        'conversations': data[:processed_count]
    }
    with open(checkpoint_path, 'w', encoding='utf-8') as f:
        json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
    print(f"  [Checkpoint] Saved {processed_count} conversations to {checkpoint_path}")


def load_checkpoint(checkpoint_path: Path) -> Optional[Dict]:
    """Load checkpoint if it exists."""
    if checkpoint_path.exists():
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            checkpoint_data = json.load(f)
        print(f"  [Checkpoint] Loaded {checkpoint_data['processed_count']} conversations from {checkpoint_path}")
        return checkpoint_data
    return None


def extract_crisis_conversations(
    output_path: str = "research_scripts/figure9/crisis_evaluation/data/crisis_conversations_50x10.json",
    categories = "suicidal_ideation,risk_taking_behaviours,self_harm",
    num_conversations: int = 50,
    min_turns: int = 10,
    max_turns: int = 10,
    seed: int = 42,
    repo_path: str = None,
    checkpoint_interval: int = 10,
    checkpoint_path: str = None,
    resume: bool = True
):
    """
    Extract crisis conversations from the LLMs-Mental-Health-Crisis dataset.
    
    Args:
        output_path: Path to save the extracted conversations JSON
        categories: Comma-separated list or tuple of crisis categories to include
        num_conversations: Number of conversations to extract
        min_turns: Minimum number of turns required per conversation
        max_turns: Maximum turns to include (truncates longer conversations)
        seed: Random seed for reproducibility
        repo_path: Path to cloned LLMs-Mental-Health-Crisis repo
        checkpoint_interval: Save checkpoint every N conversations (default: 10)
        checkpoint_path: Path for checkpoint file (default: output_path + '.checkpoint')
        resume: Whether to resume from checkpoint if it exists (default: True)
    """
    # Parse categories (handle both string and tuple/list)
    if isinstance(categories, str):
        category_list = [c.strip() for c in categories.split(',')]
    elif isinstance(categories, (list, tuple)):
        category_list = list(categories)
    else:
        category_list = [str(categories)]
    print(f"Extracting {num_conversations} conversations")
    print(f"Categories: {category_list}")
    print(f"Min turns: {min_turns}, Max turns: {max_turns}")
    print(f"Seed: {seed}")
    print(f"Checkpoint interval: every {checkpoint_interval} conversations")
    print("-" * 50)
    
    # Setup checkpoint path
    output_path = Path(output_path)
    if checkpoint_path is None:
        checkpoint_path = output_path.with_suffix('.checkpoint.json')
    else:
        checkpoint_path = Path(checkpoint_path)
    
    # Try to resume from checkpoint
    start_index = 0
    processed_data = []
    if resume:
        checkpoint_data = load_checkpoint(checkpoint_path)
        if checkpoint_data:
            start_index = checkpoint_data['processed_count']
            processed_data = checkpoint_data['conversations']
            print(f"  Resuming from conversation {start_index + 1}")
    
    # Determine repo path
    if repo_path is None:
        # Try to find it relative to this script
        script_dir = Path(__file__).parent
        repo_path = script_dir / "LLMs-Mental-Health-Crisis"
        if not repo_path.exists():
            # Try workspace root
            repo_path = Path("research_scripts/figure9/crisis_evaluation/LLMs-Mental-Health-Crisis")
    
    repo_path = Path(repo_path)
    
    # Load data
    data = load_labeled_dataset(repo_path)
    
    # Filter by categories
    data = filter_by_categories(data, category_list)
    
    # Filter by minimum turns
    data = filter_by_min_turns(data, min_turns)
    
    if len(data) < num_conversations:
        print(f"Warning: Only {len(data)} conversations available, requested {num_conversations}")
        num_conversations = len(data)
    
    # Sample conversations
    data = sample_conversations(data, num_conversations, category_list, seed)
    
    # Truncate to max_turns
    data = truncate_to_turns(data, max_turns)
    
    # Add conversation IDs
    data = add_conversation_ids(data)
    
    # Process conversations with checkpointing
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    for i in range(start_index, len(data)):
        conv = data[i]
        processed_data.append(conv)
        
        # Save checkpoint every N conversations
        if (i + 1) % checkpoint_interval == 0:
            save_checkpoint(data, checkpoint_path, i + 1)
    
    # Save final output
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)
    
    # Remove checkpoint file after successful completion
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print(f"  [Checkpoint] Removed checkpoint file after successful completion")
    
    print("-" * 50)
    print(f"Saved {len(processed_data)} conversations to {output_path}")
    
    # Print summary statistics
    turn_counts = [len(conv['inputs']) for conv in processed_data]
    print(f"Turns per conversation: min={min(turn_counts)}, max={max(turn_counts)}, avg={sum(turn_counts)/len(turn_counts):.1f}")
    
    return processed_data


def extract_binary_crisis_dataset(
    output_path: str = "research_scripts/figure9/crisis_evaluation/data/crisis_conversations_binary_112x20.json",
    crisis_categories: str = "suicidal_ideation,anxiety_crisis,substance_abuse_or_withdrawal,self-harm",
    num_no_crisis: int = 50,
    min_turns: int = 21,
    max_turns: int = 20,
    seed: int = 42,
    repo_path: str = None
):
    """
    Extract binary crisis dataset: all crisis categories combined vs no_crisis.
    
    Takes ALL available crisis conversations (meeting turn threshold) and samples
    a specified number of no_crisis conversations.
    
    Args:
        output_path: Path to save the extracted conversations JSON
        crisis_categories: Comma-separated list of categories to combine as "crisis"
        num_no_crisis: Number of no_crisis conversations to sample
        min_turns: Minimum number of turns required (use > threshold, e.g., 21 for >20)
        max_turns: Maximum turns to include (truncates longer conversations)
        seed: Random seed for reproducibility
        repo_path: Path to cloned LLMs-Mental-Health-Crisis repo
    """
    random.seed(seed)
    
    # Parse crisis categories
    if isinstance(crisis_categories, str):
        crisis_list = [c.strip() for c in crisis_categories.split(',')]
    elif isinstance(crisis_categories, (list, tuple)):
        crisis_list = list(crisis_categories)
    else:
        crisis_list = [str(crisis_categories)]
    
    print(f"Extracting binary crisis dataset")
    print(f"Crisis categories: {crisis_list}")
    print(f"No-crisis samples: {num_no_crisis}")
    print(f"Min turns: {min_turns}, Max turns: {max_turns}")
    print(f"Seed: {seed}")
    print("-" * 50)
    
    # Determine repo path
    if repo_path is None:
        script_dir = Path(__file__).parent
        repo_path = script_dir / "LLMs-Mental-Health-Crisis"
        if not repo_path.exists():
            repo_path = Path("research_scripts/figure9/crisis_evaluation/LLMs-Mental-Health-Crisis")
    
    repo_path = Path(repo_path)
    
    # Load data
    data = load_labeled_dataset(repo_path)
    
    # Filter by minimum turns first
    data = filter_by_min_turns(data, min_turns)
    
    # Separate crisis and no_crisis
    crisis_convs = []
    no_crisis_convs = []
    
    crisis_breakdown = defaultdict(int)
    for conv in data:
        label = conv.get('label', '')
        if label in crisis_list:
            crisis_convs.append(conv)
            crisis_breakdown[label] += 1
        elif label == 'no_crisis':
            no_crisis_convs.append(conv)
    
    print(f"\nCrisis conversations available: {len(crisis_convs)}")
    for cat, count in sorted(crisis_breakdown.items()):
        print(f"  - {cat}: {count}")
    print(f"No-crisis conversations available: {len(no_crisis_convs)}")
    
    # Take ALL crisis conversations
    sampled_crisis = crisis_convs
    
    # Sample no_crisis conversations
    if len(no_crisis_convs) < num_no_crisis:
        print(f"Warning: Only {len(no_crisis_convs)} no_crisis conversations available, requested {num_no_crisis}")
        num_no_crisis = len(no_crisis_convs)
    
    sampled_no_crisis = random.sample(no_crisis_convs, num_no_crisis)
    
    # Combine and relabel
    combined = sampled_crisis + sampled_no_crisis
    combined = relabel_to_binary(combined, crisis_list)
    
    # Shuffle
    random.shuffle(combined)
    
    # Truncate to max_turns
    combined = truncate_to_turns(combined, max_turns)
    
    # Add conversation IDs
    combined = add_conversation_ids(combined)
    
    # Save output
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    
    # Final summary
    final_counts = defaultdict(int)
    for conv in combined:
        final_counts[conv['label']] += 1
    
    print("-" * 50)
    print(f"Saved {len(combined)} conversations to {output_path}")
    print(f"\nFinal distribution:")
    for cat, count in sorted(final_counts.items()):
        print(f"  - {cat}: {count}")
    
    turn_counts = [len(conv['inputs']) for conv in combined]
    print(f"Turns per conversation: min={min(turn_counts)}, max={max(turn_counts)}, avg={sum(turn_counts)/len(turn_counts):.1f}")
    
    return combined


if __name__ == "__main__":
    fire.Fire({
        'extract': extract_crisis_conversations,
        'extract_binary': extract_binary_crisis_dataset
    })

