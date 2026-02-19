#!/usr/bin/env python3
"""
Generate summary report for hyperparameter search results.
"""

import argparse
import json
import pandas as pd
import sys
from pathlib import Path
from typing import Optional, Dict, List


def load_test_outputs(test_outputs_file: Path) -> Dict:
    """Load test outputs from JSON file."""
    if not test_outputs_file.exists():
        return {"test_outputs": []}
    
    try:
        with open(test_outputs_file, 'r') as f:
            return json.load(f)
    except Exception:
        return {"test_outputs": []}


def get_test_output_for_config(test_outputs: List[Dict], rank, alpha: float) -> Optional[Dict]:
    """Get test output for a specific configuration.
    
    Args:
        test_outputs: List of test output dicts
        rank: Rank value (int or "full" string)
        alpha: Alpha value
    """
    for output in test_outputs:
        if output.get('rank') == rank and output.get('alpha') == alpha:
            return output
    return None


def _sort_key_for_rank(rank_val):
    """Convert rank to a sortable key (handles 'full' and 'baseline' strings)."""
    if rank_val == 'baseline':
        return (-1, '')  # Baseline first
    elif rank_val == 'full':
        return (float('inf'), '')  # Full rank last (largest)
    elif isinstance(rank_val, (int, float)):
        return (rank_val, '')
    else:
        return (float('inf'), str(rank_val))  # Unknown strings at end


def generate_report(summary_csv: Path, output_file: Path, test_outputs_file: Optional[Path] = None):
    """Generate the hyperparameter search report."""
    
    try:
        df = pd.read_csv(summary_csv)
    except FileNotFoundError:
        print(f"Summary file not found: {summary_csv}")
        return
    
    lines = []
    
    lines.append("=" * 70)
    lines.append("HYPERPARAMETER SEARCH RESULTS - FIGURE 9")
    lines.append("=" * 70)
    lines.append("")
    
    # Filter successful runs
    success_df = df[df['status'] == 'success'].copy()
    
    if len(success_df) == 0:
        lines.append("No successful configurations found!")
        with open(output_file, 'w') as f:
            f.write('\n'.join(lines))
        print('\n'.join(lines))
        return
    
    lines.append(f"Successful configurations: {len(success_df)}/{len(df)}")
    lines.append("")
    
    # Sort by trait delta (descending) to find best configurations
    success_df = success_df.sort_values('mean_trait_delta', ascending=False)
    
    lines.append("Results sorted by Trait Δ (descending):")
    lines.append("-" * 72)
    lines.append(f"{'Rank':>8} | {'Alpha':>6} | {'Coherence':>10} | {'Coh Δ':>8} | {'Trait':>10} | {'Trait Δ':>8}")
    lines.append("-" * 70)
    
    for _, row in success_df.iterrows():
        baseline_marker = " *" if row['is_baseline'] else ""
        coh_delta_str = f"{row['mean_coherence_delta']:+8.2f}"
        trait_delta_str = f"{row['mean_trait_delta']:+8.2f}"
        # Handle 'baseline' and 'full' rank labels (non-integer)
        rank_val = row['rank']
        if rank_val in ['baseline', 'full']:
            rank_str = str(rank_val)
        else:
            try:
                rank_str = str(int(rank_val))
            except (ValueError, TypeError):
                rank_str = str(rank_val)
        lines.append(
            f"{rank_str:>8} | {row['alpha']:>6} | {row['mean_coherence']:>10.2f} | "
            f"{coh_delta_str} | {row['mean_trait_score']:>10.2f} | {trait_delta_str}{baseline_marker}"
        )
    
    lines.append("-" * 72)
    lines.append("* = baseline configuration")
    lines.append("")
    
    # Helper to format rank
    def format_rank(rank_val):
        if rank_val in ['baseline', 'full']:
            return str(rank_val)
        try:
            return str(int(rank_val))
        except (ValueError, TypeError):
            return str(rank_val)
    
    # Find best configuration (highest trait delta with coherence > 70, excluding baseline)
    non_baseline = success_df[success_df['is_baseline'] != True]
    good_coh = non_baseline[non_baseline['mean_coherence'] >= 70]
    if len(good_coh) > 0:
        best = good_coh.iloc[0]
        lines.append("BEST CONFIGURATION (Coherence ≥ 70):")
        lines.append(f"  Rank (d): {format_rank(best['rank'])}")
        lines.append(f"  Alpha: {best['alpha']}")
        lines.append(f"  Coherence: {best['mean_coherence']:.2f} (Δ: {best['mean_coherence_delta']:+.2f})")
        lines.append(f"  Trait Score: {best['mean_trait_score']:.2f} (Δ: {best['mean_trait_delta']:+.2f})")
    else:
        lines.append("WARNING: No configurations with Coherence ≥ 70 found!")
        best = non_baseline.iloc[0] if len(non_baseline) > 0 else success_df.iloc[0]
        lines.append("BEST CONFIGURATION (by Trait Δ regardless of Coherence):")
        lines.append(f"  Rank (d): {format_rank(best['rank'])}")
        lines.append(f"  Alpha: {best['alpha']}")
        lines.append(f"  Coherence: {best['mean_coherence']:.2f} (Δ: {best['mean_coherence_delta']:+.2f})")
        lines.append(f"  Trait Score: {best['mean_trait_score']:.2f} (Δ: {best['mean_trait_delta']:+.2f})")
    
    lines.append("")
    lines.append("=" * 70)
    
    # Summary statistics by rank
    lines.append("")
    lines.append("SUMMARY BY RANK (d):")
    lines.append("-" * 50)
    
    rank_summary = success_df.groupby('rank').agg({
        'mean_coherence': 'mean',
        'mean_coherence_delta': 'mean',
        'mean_trait_score': 'mean',
        'mean_trait_delta': 'mean'
    }).round(2)
    lines.append(rank_summary.to_string())
    
    lines.append("")
    lines.append("SUMMARY BY ALPHA:")
    lines.append("-" * 50)
    
    alpha_summary = success_df.groupby('alpha').agg({
        'mean_coherence': 'mean',
        'mean_coherence_delta': 'mean',
        'mean_trait_score': 'mean',
        'mean_trait_delta': 'mean'
    }).round(2)
    lines.append(alpha_summary.to_string())
    
    lines.append("")
    lines.append("=" * 70)
    
    # Load and display test outputs if available
    if test_outputs_file and test_outputs_file.exists():
        test_data = load_test_outputs(test_outputs_file)
        test_outputs = test_data.get("test_outputs", [])
        
        if test_outputs:
            lines.append("")
            lines.append("=" * 70)
            lines.append("EXAMPLE TEST OUTPUTS")
            lines.append("=" * 70)
            
            # Get unique test prompt names from first config
            first_output = test_outputs[0]
            test_prompts_list = first_output.get('test_prompts', [])
            
            # Handle both old format (single prompt) and new format (multiple prompts)
            if test_prompts_list:
                # New format with multiple prompts
                lines.append("")
                lines.append("Test Prompts:")
                for tp in test_prompts_list:
                    lines.append(f"  - [{tp.get('name', 'unknown')}]: \"{tp.get('prompt', 'N/A')}\"")
            else:
                # Old format with single prompt
                lines.append("")
                lines.append(f"Test Prompt: \"{first_output.get('test_prompt', 'N/A')}\"")
            
            lines.append("")
            
            # Sort by rank, then alpha (handles "full" and "baseline" string ranks)
            sorted_outputs = sorted(test_outputs, key=lambda x: (_sort_key_for_rank(x.get('rank', 0)), x.get('alpha', 0)))
            
            for output in sorted_outputs:
                rank = output.get('rank', 'N/A')
                alpha = output.get('alpha', 'N/A')
                overall_status = output.get('overall_status', output.get('test_status', 'N/A'))
                
                lines.append("-" * 70)
                lines.append(f"Config: rank={rank}, alpha={alpha} (status: {overall_status})")
                lines.append("-" * 70)
                
                # Handle multiple prompts (new format)
                test_prompts = output.get('test_prompts', [])
                if test_prompts:
                    for tp in test_prompts:
                        prompt_name = tp.get('name', 'unknown')
                        response = tp.get('response', 'N/A')
                        status = tp.get('status', 'N/A')
                        
                        lines.append(f"[{prompt_name}] (status: {status})")
                        if response and response != 'N/A' and not response.startswith('ERROR'):
                            # Truncate long responses for readability
                            if len(response) > 400:
                                response = response[:400] + "..."
                            lines.append(response)
                        else:
                            lines.append(f"[No response: {response}]")
                        lines.append("")
                else:
                    # Old format with single response
                    response = output.get('test_response', 'N/A')
                    if response and response != 'N/A':
                        if len(response) > 500:
                            response = response[:500] + "..."
                        lines.append(response)
                    else:
                        lines.append("[No response captured]")
                    lines.append("")
            
            lines.append("=" * 70)
    
    # Write to file
    report_text = '\n'.join(lines)
    with open(output_file, 'w') as f:
        f.write(report_text)
    
    # Also print to stdout
    print(report_text)


def main():
    parser = argparse.ArgumentParser(description="Generate hyperparameter search report")
    parser.add_argument("--summary-csv", type=str, required=True, help="Path to summary CSV")
    parser.add_argument("--output-file", type=str, required=True, help="Path to output report file")
    parser.add_argument("--test-outputs-json", type=str, default=None, help="Path to test outputs JSON file")
    
    args = parser.parse_args()
    
    test_outputs_file = Path(args.test_outputs_json) if args.test_outputs_json else None
    generate_report(Path(args.summary_csv), Path(args.output_file), test_outputs_file)


if __name__ == "__main__":
    main()

