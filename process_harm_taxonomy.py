#!/usr/bin/env python3
"""
Harm Taxonomy Batch Processor

Parses harm_taxonomy.md, generates artifacts and steering vectors for selected
traits/models using direct execution (same flow as UI custom traits).

Usage:
    # Process all 15 traits with default model (Llama-3.1-8B-Instruct)
    python process_harm_taxonomy.py --all
    
    # Process specific traits by number
    python process_harm_taxonomy.py --traits 1,2,5
    
    # Process with specific models
    python process_harm_taxonomy.py --traits 1,2 --models "Llama-3.1-8B-Instruct,Qwen2.5-7B-Instruct"
    
    # Dry run
    python process_harm_taxonomy.py --all --dry-run
    
    # List available traits
    python process_harm_taxonomy.py --list
"""

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from generate_artifacts import ArtifactGenerator, save_artifact
from dashboard.config.model_registry import MODEL_HF_IDS, get_hf_model_id
from dashboard.utils.trait_registry import get_trait_registry
from dashboard.utils.persistence import get_active_profile


# Available models from model registry
AVAILABLE_MODELS = list(MODEL_HF_IDS.keys())
DEFAULT_MODEL = "Llama-3.1-8B-Instruct"


def parse_harm_taxonomy(md_path: Path) -> List[Dict[str, str]]:
    """
    Parse the harm taxonomy markdown file to extract harm patterns.
    
    Args:
        md_path: Path to harm_taxonomy.md
        
    Returns:
        List of dictionaries with {number, name, description, cluster}
    """
    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    patterns = []
    current_cluster = None
    
    # Pattern to match harm entries: ### N. Title\nDescription
    # Match lines like "### 1. Normalizing Aggressive Behavior"
    pattern_regex = re.compile(
        r'###\s+(\d+)\.\s+(.+?)\n(.+?)(?=\n###|\n## |---|\Z)',
        re.DOTALL
    )
    
    # Find cluster headers
    cluster_pattern = re.compile(r'## Cluster \d+:\s*(.+)')
    
    # Split by cluster headers and process
    lines = content.split('\n')
    
    for match in pattern_regex.finditer(content):
        number = int(match.group(1))
        title = match.group(2).strip()
        description = match.group(3).strip()
        
        # Clean up description - remove extra whitespace and newlines
        description = ' '.join(description.split())
        
        # Determine cluster based on position in file
        if number <= 2:
            cluster = "aggression_escalation"
        elif number <= 4:
            cluster = "emotional_minimization"
        elif number <= 13:
            cluster = "maladaptive_support"
        else:
            cluster = "eating_disorder_enablement"
        
        # Generate normalized name (no prefix)
        normalized_name = title.lower()
        normalized_name = re.sub(r'[^a-z0-9]+', '_', normalized_name)
        normalized_name = re.sub(r'_+', '_', normalized_name).strip('_')
        
        patterns.append({
            'number': number,
            'title': title,
            'name': normalized_name,
            'description': description,
            'cluster': cluster
        })
    
    # Sort by number
    patterns.sort(key=lambda x: x['number'])
    
    return patterns


def list_traits(patterns: List[Dict[str, str]]) -> None:
    """Print a formatted list of all harm patterns."""
    print("\n" + "=" * 80)
    print("HARM TAXONOMY - 15 AI Response Patterns")
    print("=" * 80)
    
    current_cluster = None
    for p in patterns:
        if p['cluster'] != current_cluster:
            current_cluster = p['cluster']
            cluster_display = current_cluster.replace('_', ' ').title()
            print(f"\n📁 {cluster_display}")
            print("-" * 40)
        
        print(f"  {p['number']:2d}. {p['name']}")
        print(f"      └─ {p['description'][:70]}...")
    
    print("\n" + "=" * 80)
    print(f"Total: {len(patterns)} harm patterns")
    print("=" * 80 + "\n")


async def generate_trait_artifacts(
    trait_name: str,
    description: str,
    dry_run: bool = False
) -> Optional[Dict]:
    """
    Generate artifacts for a trait using ArtifactGenerator.
    Saves to all 3 required locations for custom traits.
    
    Args:
        trait_name: Normalized trait name (e.g., harm_normalizing_aggressive_behavior)
        description: Trait description
        dry_run: If True, just print what would be done
        
    Returns:
        Generated artifact dict or None if dry_run
    """
    print(f"\n🎭 Generating artifacts for: {trait_name}")
    print(f"   Description: {description[:80]}...")
    
    if dry_run:
        print("   [DRY RUN] Would generate artifacts with LLM")
        return None
    
    try:
        # Initialize generator (uses auto provider - Claude or GPT-4)
        generator = ArtifactGenerator(provider="auto")
        
        # Generate artifact
        print("   📝 Calling LLM to generate instruction pairs and questions...")
        artifact = await generator.generate_artifact(
            trait=trait_name,
            trait_description=description,
            max_tokens=16000,
            temperature=0.1
        )
        
        # Validate artifact
        required_keys = ["instruction", "questions", "eval_prompt"]
        if not all(key in artifact for key in required_keys):
            raise ValueError(f"Generated artifact missing required keys: {required_keys}")
        
        print(f"   ✅ Generated {len(artifact['instruction'])} instruction pairs, {len(artifact['questions'])} questions")
        
        # Save to 3 locations (like UI does for custom traits)
        locations = [
            PROJECT_ROOT / "data_generation" / "custom_traits" / "trait_data_extract" / f"{trait_name}.json",
            PROJECT_ROOT / "data_generation" / "trait_data_extract" / f"{trait_name}.json",
            PROJECT_ROOT / "data_generation" / "trait_data_eval" / f"{trait_name}.json"
        ]
        
        for loc in locations:
            loc.parent.mkdir(parents=True, exist_ok=True)
            with open(loc, 'w', encoding='utf-8') as f:
                json.dump(artifact, f, indent=2, ensure_ascii=False)
            print(f"   📁 Saved to: {loc.relative_to(PROJECT_ROOT)}")
        
        # Register trait in the trait registry (so it shows in UI)
        registry = get_trait_registry()
        existing_trait = registry.get_trait_by_name(trait_name)
        
        if existing_trait:
            print(f"   ℹ️  Trait '{trait_name}' already registered in trait registry")
            # Update artifact status
            registry.update_artifact_status(existing_trait['trait_id'], 'generated')
        else:
            # Get active profile for created_by field
            profile = get_active_profile()
            profile_id = profile['profile_id'] if profile else 'harm_taxonomy_processor'
            
            # Create trait in registry
            trait = registry.create_trait(
                name=trait_name,
                description=description,
                profile_id=profile_id
            )
            registry.update_artifact_status(trait['trait_id'], 'generated')
            print(f"   📋 Registered trait in registry: {trait['display_name']} (ID: {trait['trait_id']})")
        
        return artifact
        
    except Exception as e:
        print(f"   ❌ Error generating artifacts: {e}")
        raise


def register_existing_trait(
    trait_name: str,
    description: str,
    dry_run: bool = False
) -> bool:
    """
    Register an existing trait (with artifacts) in the trait registry.
    Used for traits created outside the normal flow.
    
    Args:
        trait_name: Normalized trait name
        description: Trait description
        dry_run: If True, just print what would be done
        
    Returns:
        True if successful, False otherwise
    """
    print(f"\n📋 Registering existing trait: {trait_name}")
    
    # Check if artifacts exist
    artifact_path = PROJECT_ROOT / "data_generation" / "custom_traits" / "trait_data_extract" / f"{trait_name}.json"
    if not artifact_path.exists():
        artifact_path = PROJECT_ROOT / "data_generation" / "trait_data_extract" / f"{trait_name}.json"
    
    if not artifact_path.exists():
        print(f"   ❌ Artifact file not found for {trait_name}")
        return False
    
    if dry_run:
        print(f"   [DRY RUN] Would register trait in registry")
        return True
    
    try:
        registry = get_trait_registry()
        existing_trait = registry.get_trait_by_name(trait_name)
        
        if existing_trait:
            print(f"   ℹ️  Trait '{trait_name}' already registered (ID: {existing_trait['trait_id']})")
            return True
        
        # Get active profile
        profile = get_active_profile()
        profile_id = profile['profile_id'] if profile else 'harm_taxonomy_processor'
        
        # Create trait in registry
        trait = registry.create_trait(
            name=trait_name,
            description=description,
            profile_id=profile_id
        )
        registry.update_artifact_status(trait['trait_id'], 'generated')
        print(f"   ✅ Registered: {trait['display_name']} (ID: {trait['trait_id']})")
        return True
        
    except Exception as e:
        print(f"   ❌ Error registering trait: {e}")
        return False


def is_trait_completed(trait_name: str, model: str) -> bool:
    """
    Check if a trait has already been fully processed for a given model.
    
    Checks:
    1. Vector file exists
    2. Optimal layer entry exists in optimal_layers.json
    
    Args:
        trait_name: Normalized trait name
        model: Model display name
        
    Returns:
        True if trait is completed for this model, False otherwise
    """
    # Check vector file exists
    vector_path = PROJECT_ROOT / "output" / model / "persona_vectors" / f"{trait_name}_response_avg_diff.pt"
    if not vector_path.exists():
        return False
    
    # Check optimal layer entry exists
    optimal_layers_path = PROJECT_ROOT / "inference_server" / "optimal_layers.json"
    if optimal_layers_path.exists():
        try:
            with open(optimal_layers_path) as f:
                optimal_layers = json.load(f)
            
            # Check if model and trait exist in optimal_layers
            if model in optimal_layers and trait_name in optimal_layers[model]:
                return True
        except Exception:
            pass
    
    return False


def run_vector_generation(
    trait_name: str,
    model: str,
    dry_run: bool = False,
    gpu: int = 0
) -> bool:
    """
    Run vector generation for a trait/model combination.
    Calls generate_vec.sh directly via subprocess.
    
    Args:
        trait_name: Normalized trait name
        model: Model display name (e.g., Llama-3.1-8B-Instruct)
        dry_run: If True, just print what would be done
        gpu: GPU ID to use
        
    Returns:
        True if successful, False otherwise
    """
    hf_model_id = get_hf_model_id(model)
    
    print(f"\n🧠 Generating vectors for: {trait_name}")
    print(f"   Model: {model} ({hf_model_id})")
    
    if dry_run:
        print("   [DRY RUN] Would run generate_vec.sh")
        return True
    
    # Build command - same as worker.py does
    generate_vec_script = PROJECT_ROOT / "scripts" / "generate_vec.sh"
    
    # Determine vllm_max_seqs based on model size
    size_match = re.search(r'(\d+\.?\d*)B', model, re.IGNORECASE)
    if size_match:
        model_size_b = float(size_match.group(1))
        vllm_max_seqs = "256" if model_size_b < 1.5 else "64"
    else:
        vllm_max_seqs = "64"
    
    cmd = [
        str(generate_vec_script),
        str(gpu),           # GPU ID
        hf_model_id,        # Full HF repo ID
        trait_name,         # Trait name
        "gpt-4.1-mini",     # Judge model (Azure deployment)
        "3",                # n_per_question
        "true",             # use_vllm
        "false",            # skip_steering (we want optimal layer finding)
        vllm_max_seqs,      # Dynamic based on model size
        "0.90"              # vllm_gpu_mem
    ]
    
    print(f"   🚀 Running: {' '.join(cmd[:4])}...")
    print(f"   ⏳ This may take 1-2 hours per model...")
    
    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT)
        
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,  # Prevent blocking on input prompts
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            cwd=str(PROJECT_ROOT)
        )
        
        # Stream output
        for line in process.stdout:
            line = line.strip()
            if line:
                # Filter to important messages
                if any(kw in line for kw in ["✅", "❌", "🎭", "📈", "🧠", "⏭️", "Error", "error", "Warning"]):
                    print(f"   {line}")
        
        return_code = process.wait()
        
        if return_code != 0:
            print(f"   ❌ Vector generation failed with code {return_code}")
            return False
        
        print(f"   ✅ Vector generation completed for {trait_name} on {model}")
        return True
        
    except Exception as e:
        print(f"   ❌ Error running vector generation: {e}")
        return False


async def process_traits(
    patterns: List[Dict[str, str]],
    trait_numbers: List[int],
    models: List[str],
    skip_artifacts: bool = False,
    artifacts_only: bool = False,
    dry_run: bool = False,
    gpu: int = 0,
    force: bool = False
) -> Tuple[List[str], List[str]]:
    """
    Process selected traits - generate artifacts and vectors.
    
    Args:
        patterns: All parsed harm patterns
        trait_numbers: List of trait numbers to process (1-15)
        models: List of model names to generate vectors for
        skip_artifacts: Skip artifact generation (use existing)
        artifacts_only: Only generate artifacts, skip vectors
        dry_run: Just print what would be done
        gpu: GPU ID for vector generation
        force: Force re-processing even if trait is completed
        
    Returns:
        Tuple of (successful_traits, failed_traits)
    """
    successful = []
    failed = []
    
    # Filter patterns by number
    selected_patterns = [p for p in patterns if p['number'] in trait_numbers]
    
    print("\n" + "=" * 80)
    print("HARM TAXONOMY PROCESSOR")
    print("=" * 80)
    print(f"Traits to process: {len(selected_patterns)}")
    print(f"Models: {', '.join(models)}")
    print(f"Skip artifacts: {skip_artifacts}")
    print(f"Artifacts only: {artifacts_only}")
    print(f"Dry run: {dry_run}")
    print("=" * 80)
    
    skipped = []
    
    for i, pattern in enumerate(selected_patterns, 1):
        print(f"\n{'=' * 80}")
        print(f"[{i}/{len(selected_patterns)}] Processing: {pattern['title']}")
        print(f"{'=' * 80}")
        
        trait_name = pattern['name']
        description = pattern['description']
        
        # Check if trait is already completed for all models (skip if so, unless --force)
        if not artifacts_only and not dry_run and not force:
            all_completed = all(is_trait_completed(trait_name, model) for model in models)
            if all_completed:
                print(f"⏭️  SKIPPING: {trait_name} already completed for all models")
                print(f"   Vector files and optimal layers exist")
                print(f"   Use --force to re-process")
                skipped.append(trait_name)
                continue
            else:
                # Show which models need processing
                for model in models:
                    if is_trait_completed(trait_name, model):
                        print(f"   ⏭️  {model}: already completed")
                    else:
                        print(f"   🔄 {model}: needs processing")
        elif force and not dry_run:
            print(f"   🔄 Force mode: re-processing all models")
        
        # Step 1: Generate artifacts (unless skipping)
        if not skip_artifacts:
            try:
                await generate_trait_artifacts(trait_name, description, dry_run)
            except Exception as e:
                print(f"❌ Failed to generate artifacts for {trait_name}: {e}")
                failed.append(trait_name)
                continue
        else:
            # Check if artifacts exist
            artifact_path = PROJECT_ROOT / "data_generation" / "trait_data_extract" / f"{trait_name}.json"
            if not artifact_path.exists() and not dry_run:
                print(f"❌ Artifacts not found for {trait_name} and --skip-artifacts was set")
                failed.append(trait_name)
                continue
            print(f"⏭️  Skipping artifact generation (using existing)")
        
        # Step 2: Generate vectors (unless artifacts_only)
        if not artifacts_only:
            for model in models:
                # Skip models that are already completed (unless --force)
                if not dry_run and not force and is_trait_completed(trait_name, model):
                    print(f"\n⏭️  Skipping vector generation for {model} (already completed)")
                    continue
                    
                success = run_vector_generation(trait_name, model, dry_run, gpu)
                if not success and not dry_run:
                    print(f"⚠️  Vector generation failed for {trait_name} on {model}")
        
        successful.append(trait_name)
    
    # Print skip summary if any were skipped
    if skipped:
        print(f"\n⏭️  Skipped {len(skipped)} already-completed traits: {', '.join(skipped)}")
    
    return successful, failed


def main():
    parser = argparse.ArgumentParser(
        description="Harm Taxonomy Batch Processor - Generate artifacts and vectors for harm patterns",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --all                           Process all 15 traits
  %(prog)s --traits 1,2,5                  Process traits 1, 2, and 5
  %(prog)s --traits 1 --models Llama-3.2-1B-Instruct   Use specific model
  %(prog)s --all --dry-run                 Show what would be done
  %(prog)s --list                          List all available traits
        """
    )
    
    parser.add_argument('--all', action='store_true',
                        help='Process all 15 harm patterns')
    parser.add_argument('--traits', type=str,
                        help='Comma-separated list of trait numbers (1-15)')
    parser.add_argument('--models', type=str, default=DEFAULT_MODEL,
                        help=f'Comma-separated list of models (default: {DEFAULT_MODEL})')
    parser.add_argument('--skip-artifacts', action='store_true',
                        help='Skip artifact generation (use existing)')
    parser.add_argument('--artifacts-only', action='store_true',
                        help='Only generate artifacts, skip vector generation')
    parser.add_argument('--register-only', action='store_true',
                        help='Only register existing traits in the UI registry (no generation)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be done without executing')
    parser.add_argument('--list', action='store_true',
                        help='List all available harm patterns')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID for vector generation (default: 0)')
    parser.add_argument('--force', action='store_true',
                        help='Force re-processing even if trait is already completed')
    parser.add_argument('--taxonomy-file', type=str, 
                        default=str(Path(__file__).parent / "harm_taxonomy.md"),
                        help='Path to harm taxonomy markdown file')
    
    args = parser.parse_args()
    
    # Parse the taxonomy
    taxonomy_path = Path(args.taxonomy_file)
    if not taxonomy_path.exists():
        print(f"❌ Taxonomy file not found: {taxonomy_path}")
        sys.exit(1)
    
    patterns = parse_harm_taxonomy(taxonomy_path)
    
    if len(patterns) != 15:
        print(f"⚠️  Warning: Expected 15 patterns, found {len(patterns)}")
    
    # Handle --list
    if args.list:
        list_traits(patterns)
        print("\nAvailable models:")
        for model in AVAILABLE_MODELS:
            marker = " (default)" if model == DEFAULT_MODEL else ""
            print(f"  - {model}{marker}")
        sys.exit(0)
    
    # Validate arguments
    if not args.all and not args.traits:
        parser.print_help()
        print("\n❌ Error: Must specify --all or --traits")
        sys.exit(1)
    
    # Determine trait numbers
    if args.all:
        trait_numbers = list(range(1, 16))
    else:
        try:
            trait_numbers = [int(x.strip()) for x in args.traits.split(',')]
            # Validate range
            for n in trait_numbers:
                if n < 1 or n > 15:
                    raise ValueError(f"Trait number {n} out of range (1-15)")
        except ValueError as e:
            print(f"❌ Error parsing trait numbers: {e}")
            sys.exit(1)
    
    # Parse models
    models = [m.strip() for m in args.models.split(',')]
    
    # Validate models
    for model in models:
        if model not in AVAILABLE_MODELS:
            print(f"⚠️  Warning: Model '{model}' not in registry, will use as-is")
    
    # Handle --register-only mode
    if args.register_only:
        print("\n" + "=" * 80)
        print("REGISTER ONLY MODE")
        print("=" * 80)
        
        selected_patterns = [p for p in patterns if p['number'] in trait_numbers]
        successful = []
        failed = []
        
        for pattern in selected_patterns:
            success = register_existing_trait(
                pattern['name'],
                pattern['description'],
                args.dry_run
            )
            if success:
                successful.append(pattern['name'])
            else:
                failed.append(pattern['name'])
        
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Registered: {len(successful)} traits")
        if successful:
            for t in successful:
                print(f"  ✅ {t}")
        if failed:
            print(f"Failed: {len(failed)} traits")
            for t in failed:
                print(f"  ❌ {t}")
        print("=" * 80)
        
        if not args.dry_run:
            print("\n🎉 Traits registered! Refresh the dashboard to see them.")
        sys.exit(0 if not failed else 1)
    
    # Run processing
    start_time = datetime.now()
    
    try:
        successful, failed = asyncio.run(process_traits(
            patterns=patterns,
            trait_numbers=trait_numbers,
            models=models,
            skip_artifacts=args.skip_artifacts,
            artifacts_only=args.artifacts_only,
            dry_run=args.dry_run,
            gpu=args.gpu,
            force=args.force
        ))
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    
    # Summary
    elapsed = datetime.now() - start_time
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Time elapsed: {elapsed}")
    print(f"Successful: {len(successful)} traits")
    if successful:
        for t in successful:
            print(f"  ✅ {t}")
    if failed:
        print(f"Failed: {len(failed)} traits")
        for t in failed:
            print(f"  ❌ {t}")
    print("=" * 80)
    
    if failed:
        sys.exit(1)
    
    print("\n🎉 Processing complete!")
    
    if not args.dry_run and not args.artifacts_only:
        print("\nNext steps:")
        print("  1. Check optimal_layers.json for new trait entries")
        print("  2. Test steering with the new traits in the dashboard")


if __name__ == "__main__":
    main()

