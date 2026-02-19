"""
Single-Turn Crisis Dataset Loader

Loads the LLMs-Mental-Health-Crisis dataset and formats it for single-turn probing evaluation.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import random
from dataclasses import dataclass


# Crisis categories
CRISIS_LABELS = {
    "suicidal_ideation",
    "self_harm",  # normalized from self-harm
    "violent_thoughts",
    "substance_abuse_or_withdrawal",
    "anxiety_crisis",
    "risk_taking_behaviours",
    "no_crisis",
}

# Default paths relative to the dataset directory
DEFAULT_DATASET_DIR = Path(__file__).parent / "LLMs-Mental-Health-Crisis" / "data"
VALIDATION_SET = "processed/sampled_dataset_n_200_merged_n50-noSeed_156-s42.json"
TEST_SET = "processed/sampled_dataset_n_2046_nPerD168_seed0.json"
HUMAN_LABELS = "human_label/human-labeled-sampled_dataset_n206_s42-merged_labels.json"
LLM_LABELS = "llm_label/gpt-4o-mini-labeled-sampled_dataset_n_2046_nPerD168_seed0-merged-labels.json"


@dataclass
class CrisisProbe:
    """A single crisis probe for evaluation."""
    probe_id: str
    category: str  # Crisis label
    user_prompt: str  # Concatenated inputs as single prompt
    raw_inputs: List[str]  # Original list of inputs
    dataset_id: str  # Source dataset (hugg_1, etc.)
    all_labels: Optional[List[str]] = None  # All annotator labels (if available)
    
    def to_dict(self) -> Dict:
        return {
            "probe_id": self.probe_id,
            "category": self.category,
            "user_prompt": self.user_prompt,
            "raw_inputs": self.raw_inputs,
            "dataset_id": self.dataset_id,
            "all_labels": self.all_labels,
        }


class SingleTurnDataset:
    """
    Load and format crisis conversations as single-turn probes.
    
    Formats:
    - 'natural': Joins inputs with newlines (like a Reddit post)
    - 'dialogue': Preserves multi-turn structure with "User:" prefixes
    - 'single_message': Takes only the last/most relevant message
    """
    
    def __init__(
        self,
        dataset_dir: Optional[str] = None,
        format_style: str = "natural",
    ):
        """
        Initialize the dataset loader.
        
        Args:
            dataset_dir: Path to LLMs-Mental-Health-Crisis/data directory
            format_style: How to format inputs ('natural', 'dialogue', 'single_message')
        """
        if dataset_dir is None:
            self.dataset_dir = DEFAULT_DATASET_DIR
        else:
            self.dataset_dir = Path(dataset_dir)
        
        self.format_style = format_style
    
    def _normalize_label(self, label: str) -> str:
        """Normalize label to use underscores."""
        if label is None or label == "":
            return "unknown"
        return label.replace('-', '_')
    
    def _format_inputs(self, inputs: List[str]) -> str:
        """Format list of inputs into a single prompt string."""
        
        if self.format_style == "natural":
            # Join with double newlines (like a Reddit post)
            return "\n\n".join(inputs)
        
        elif self.format_style == "dialogue":
            # Format as explicit dialogue turns
            formatted = []
            for i, inp in enumerate(inputs, 1):
                formatted.append(f"[Message {i}]: {inp}")
            return "\n".join(formatted)
        
        elif self.format_style == "single_message":
            # Take the last message (often most relevant for crisis)
            # Or the longest message if last is too short
            if len(inputs[-1]) > 50:
                return inputs[-1]
            else:
                # Find longest message
                longest = max(inputs, key=len)
                return longest
        
        else:
            # Default to natural
            return "\n\n".join(inputs)
    
    def load_raw_dataset(self, path: str) -> List[Dict]:
        """Load a raw JSON dataset."""
        full_path = self.dataset_dir / path
        with open(full_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_validation_set(self, use_human_labels: bool = True) -> List[CrisisProbe]:
        """
        Load the validation set (206 samples) with labels.
        
        Args:
            use_human_labels: If True, use human-annotated labels. If False, use raw unlabeled data.
        
        Returns:
            List of CrisisProbe objects
        """
        if use_human_labels:
            data = self.load_raw_dataset(HUMAN_LABELS)
        else:
            data = self.load_raw_dataset(VALIDATION_SET)
        
        return self._convert_to_probes(data, "val")
    
    def load_test_set(self) -> List[CrisisProbe]:
        """
        Load the test set (2,046 samples) with GPT-4o-mini labels.
        
        Returns:
            List of CrisisProbe objects
        """
        data = self.load_raw_dataset(LLM_LABELS)
        return self._convert_to_probes(data, "test")
    
    def _convert_to_probes(self, data: List[Dict], prefix: str) -> List[CrisisProbe]:
        """Convert raw data to CrisisProbe objects."""
        probes = []
        
        for i, item in enumerate(data):
            inputs = item.get("inputs", [])
            label = item.get("label", "unknown")
            dataset_id = item.get("dataset_id", "unknown")
            all_labels = item.get("all_labels", None)
            
            # Normalize label
            label = self._normalize_label(label)
            
            # Format inputs into single prompt
            user_prompt = self._format_inputs(inputs)
            
            probe = CrisisProbe(
                probe_id=f"{prefix}_{i:04d}",
                category=label,
                user_prompt=user_prompt,
                raw_inputs=inputs,
                dataset_id=dataset_id,
                all_labels=all_labels,
            )
            probes.append(probe)
        
        return probes
    
    def load_by_category(
        self,
        categories: Optional[List[str]] = None,
        n_per_category: Optional[int] = None,
        use_test_set: bool = True,
        seed: int = 42,
    ) -> List[CrisisProbe]:
        """
        Load probes filtered/sampled by category.
        
        Args:
            categories: List of categories to include (None = all)
            n_per_category: Max samples per category (None = all)
            use_test_set: Use test set (larger) or validation set
            seed: Random seed for sampling
        
        Returns:
            List of CrisisProbe objects
        """
        # Load full dataset
        if use_test_set:
            all_probes = self.load_test_set()
        else:
            all_probes = self.load_validation_set()
        
        # Filter by category
        if categories is not None:
            normalized_cats = {self._normalize_label(c) for c in categories}
            all_probes = [p for p in all_probes if p.category in normalized_cats]
        
        # Sample per category if requested
        if n_per_category is not None:
            random.seed(seed)
            
            # Group by category
            by_category: Dict[str, List[CrisisProbe]] = {}
            for probe in all_probes:
                if probe.category not in by_category:
                    by_category[probe.category] = []
                by_category[probe.category].append(probe)
            
            # Sample from each
            sampled = []
            for cat, cat_probes in by_category.items():
                if len(cat_probes) <= n_per_category:
                    sampled.extend(cat_probes)
                else:
                    sampled.extend(random.sample(cat_probes, n_per_category))
            
            all_probes = sampled
        
        return all_probes
    
    def get_category_distribution(self, probes: List[CrisisProbe]) -> Dict[str, int]:
        """Get distribution of categories in a probe list."""
        dist = {}
        for probe in probes:
            dist[probe.category] = dist.get(probe.category, 0) + 1
        return dict(sorted(dist.items()))
    
    def save_probes(self, probes: List[CrisisProbe], output_path: str, verbose: bool = False):
        """Save probes to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = [p.to_dict() for p in probes]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(probes)} probes to {output_path}")
        
        if verbose:
            for p in probes:
                print(p)
    
    @staticmethod
    def load_probes(input_path: str) -> List[CrisisProbe]:
        """Load probes from JSON file."""
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        probes = []
        for item in data:
            probe = CrisisProbe(
                probe_id=item["probe_id"],
                category=item["category"],
                user_prompt=item["user_prompt"],
                raw_inputs=item["raw_inputs"],
                dataset_id=item["dataset_id"],
                all_labels=item.get("all_labels"),
            )
            probes.append(probe)
        
        return probes


def extract_singleturn_probes(
    output_path: str,
    categories: Optional[List[str]] = None,
    n_per_category: Optional[int] = None,
    use_test_set: bool = True,
    format_style: str = "natural",
    dataset_dir: Optional[str] = None,
    seed: int = 42,
):
    """
    CLI entry point for extracting single-turn probes.
    
    Args:
        output_path: Path to save probes JSON
        categories: Comma-separated list of categories (default: all)
        n_per_category: Max samples per category
        use_test_set: Use test set (2046) or validation set (206)
        format_style: How to format inputs (natural, dialogue, single_message)
        dataset_dir: Path to dataset directory
        seed: Random seed
    """
    # Parse categories
    cat_list = None
    if categories:
        if isinstance(categories, str):
            cat_list = [c.strip() for c in categories.split(',')]
        else:
            cat_list = list(categories)
    
    # Load dataset
    dataset = SingleTurnDataset(
        dataset_dir=dataset_dir,
        format_style=format_style,
    )
    
    # Get probes
    probes = dataset.load_by_category(
        categories=cat_list,
        n_per_category=n_per_category,
        use_test_set=use_test_set,
        seed=seed,
    )
    
    # Print summary
    print(f"\n=== Single-Turn Crisis Probes ===")
    print(f"Total probes: {len(probes)}")
    print(f"Format style: {format_style}")
    print(f"Source: {'test set' if use_test_set else 'validation set'}")
    
    print("\nCategory distribution:")
    dist = dataset.get_category_distribution(probes)
    for cat, count in dist.items():
        print(f"  {cat}: {count}")
    
    # Save
    dataset.save_probes(probes, output_path)


if __name__ == "__main__":
    import fire
    fire.Fire(extract_singleturn_probes)

