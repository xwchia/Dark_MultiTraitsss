# Single-Turn Crisis Evaluation Quickstart

## Step 1: Extract Probes

```bash
python -m research_scripts.figure9.singleturn_crisis.dataset \
    --output_path research_scripts/figure9/singleturn_crisis/data/probes_30x7.json \
    --use_test_set True \
    --format_style natural
```

## Step 2: Run Baseline Evaluation

```bash
python -m research_scripts.figure9.singleturn_crisis.run_evaluation \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --data_path research_scripts/figure9/singleturn_crisis/data/probes_30x7.json \
    --output_dir research_scripts/figure_outputs/figure9/singleturn_crisis/Llama-3.1-8B-Instruct/baseline
```

## Step 3a: Run dark_safe Evaluation (d=256, α=1.0)

```bash
python -m research_scripts.figure9.singleturn_crisis.run_evaluation \
    --data_path research_scripts/figure9/singleturn_crisis/data/probes_30x7.json \
    --output_dir research_scripts/figure_outputs/figure9/singleturn_crisis/Llama-3.1-8B-Instruct/dark_safe_d256_a1.0 \
    --baked_model_dir research_scripts/figure_outputs/figure9/hyperparameter_search/Llama-3.1-8B-Instruct/manifold/output/hyperparam_search_d256/baked_models/alpha_1_0
```

## Step 3b: Run dark_aggressive Evaluation (d=128, α=1.5)

```bash
python -m research_scripts.figure9.singleturn_crisis.run_evaluation \
    --data_path research_scripts/figure9/singleturn_crisis/data/probes_30x7.json \
    --output_dir research_scripts/figure_outputs/figure9/singleturn_crisis/Llama-3.1-8B-Instruct/dark_aggr_d128_a1.5 \
    --baked_model_dir research_scripts/figure_outputs/figure9/hyperparameter_search/Llama-3.1-8B-Instruct/manifold/output/hyperparam_search_d128/baked_models/alpha_1_5
```


## To run EVALUATION only (after generating the responses)
# For Baseline

```bash

# python -m research_scripts.figure9.singleturn_crisis.run_evaluation \
#     --model meta-llama/Llama-3.1-8B-Instruct \
#     --data_path research_scripts/figure9/singleturn_crisis/data/probes_30x7.json \
#     --output_dir research_scripts/figure_outputs/figure9/singleturn_crisis/Llama-3.1-8B-Instruct/baseline \
#     --skip_generation=True \
#     --resume=True

# # For Dark Safe (d=256, α=1.0)
# python -m research_scripts.figure9.singleturn_crisis.run_evaluation \
#     --data_path research_scripts/figure9/singleturn_crisis/data/probes_30x7.json \
#     --output_dir research_scripts/figure_outputs/figure9/singleturn_crisis/Llama-3.1-8B-Instruct/dark_safe_d256_a1.0 \
#     --baked_model_dir research_scripts/figure_outputs/figure9/hyperparameter_search/Llama-3.1-8B-Instruct/manifold/output/hyperparam_search_d256/baked_models/alpha_1_0 \
#     --skip_generation=True \
#     --resume=True

# # For Dark Aggressive (d=128, α=1.5)
# python -m research_scripts.figure9.singleturn_crisis.run_evaluation \
#     --data_path research_scripts/figure9/singleturn_crisis/data/probes_30x7.json \
#     --output_dir research_scripts/figure_outputs/figure9/singleturn_crisis/Llama-3.1-8B-Instruct/dark_aggr_d128_a1.5 \
#     --baked_model_dir research_scripts/figure_outputs/figure9/hyperparameter_search/Llama-3.1-8B-Instruct/manifold/output/hyperparam_search_d128/baked_models/alpha_1_5 \
#     --skip_generation=True \
#     --resume=True
```



## Step 4: Run MentalBench-Align Evaluation (7-Attribute)

Run the additional MentalBench-Align evaluation framework on existing responses.
This evaluates each response using 7 attributes organized into two composite scores:
- **CSS (Cognitive Support Score)**: Guidance, Informativeness, Relevance, Safety
- **ARS (Affective Resonance Score)**: Empathy, Helpfulness, Understanding

Reference: [MentalBench-Align](https://github.com/abeerbadawi/MentalBench-Align) (arXiv:2510.19032)

```bash
# Baseline
python -m research_scripts.figure9.singleturn_crisis.run_mentalbench_evaluation \
    --responses_path research_scripts/figure_outputs/figure9/singleturn_crisis/Llama-3.1-8B-Instruct/baseline/responses.json \
    --output_dir research_scripts/figure_outputs/figure9/singleturn_crisis/Llama-3.1-8B-Instruct/baseline

# Dark Safe (d=256, α=1.0)
python -m research_scripts.figure9.singleturn_crisis.run_mentalbench_evaluation \
    --responses_path research_scripts/figure_outputs/figure9/singleturn_crisis/Llama-3.1-8B-Instruct/dark_safe_d256_a1.0/responses.json \
    --output_dir research_scripts/figure_outputs/figure9/singleturn_crisis/Llama-3.1-8B-Instruct/dark_safe_d256_a1.0

# Dark Aggressive (d=128, α=1.5)
python -m research_scripts.figure9.singleturn_crisis.run_mentalbench_evaluation \
    --responses_path research_scripts/figure_outputs/figure9/singleturn_crisis/Llama-3.1-8B-Instruct/dark_aggr_d128_a1.5/responses.json \
    --output_dir research_scripts/figure_outputs/figure9/singleturn_crisis/Llama-3.1-8B-Instruct/dark_aggr_d128_a1.5
```

**MentalBench Outputs:**
- `mentalbench_evaluations.csv` - Per-probe scores for all 7 attributes + CSS/ARS composites
- `mentalbench_summary.json` - Summary statistics by attribute and category
- `mentalbench_checkpoint.json` - Checkpoint for resuming interrupted runs

**Options:**
- `--judge_model MODEL`: Override judge model (default: auto-detected from .env)
- `--max_concurrent N`: Max concurrent API calls (default: 10)
- `--resume True/False`: Resume from checkpoint (default: True)

## Step 4b: Run Coherence Evaluation

Evaluates the internal coherence of each single-turn response (consistent tone, logical structure, no contradictions).

```bash
# Baseline
python -m research_scripts.figure9.singleturn_crisis.coherence_evaluator \
    --responses_path research_scripts/figure_outputs/figure9/singleturn_crisis/Llama-3.1-8B-Instruct/baseline/responses.json \
    --output_dir research_scripts/figure_outputs/figure9/singleturn_crisis/Llama-3.1-8B-Instruct/baseline

# Dark Safe (d=256, α=1.0)
python -m research_scripts.figure9.singleturn_crisis.coherence_evaluator \
    --responses_path research_scripts/figure_outputs/figure9/singleturn_crisis/Llama-3.1-8B-Instruct/dark_safe_d256_a1.0/responses.json \
    --output_dir research_scripts/figure_outputs/figure9/singleturn_crisis/Llama-3.1-8B-Instruct/dark_safe_d256_a1.0

# Dark Aggressive (d=128, α=1.5)
python -m research_scripts.figure9.singleturn_crisis.coherence_evaluator \
    --responses_path research_scripts/figure_outputs/figure9/singleturn_crisis/Llama-3.1-8B-Instruct/dark_aggr_d128_a1.5/responses.json \
    --output_dir research_scripts/figure_outputs/figure9/singleturn_crisis/Llama-3.1-8B-Instruct/dark_aggr_d128_a1.5
```

**Coherence Outputs:**
- `coherence_scores.csv` - Per-probe coherence scores (0-100)
- Updates `summary.json` with coherence statistics
- Optionally adds coherence column to `evaluations.csv`

**Options:**
- `--judge_model MODEL`: Override judge model (default: auto-detected from .env)
- `--max_concurrent N`: Max concurrent API calls (default: 10)
- `--append_to_summary True/False`: Add coherence to evaluations.csv (default: True)

## Step 5: Generate Summary Plots with Significance Testing

Generates category breakdown charts with significance testing, summary reports, and comparison plots.

**Recommended (with significance testing):**
```bash
python -m research_scripts.figure9.singleturn_crisis.plot_summary_with_significance plot_both \
    --input_dir research_scripts/figure_outputs/figure9/singleturn_crisis/Llama-3.1-8B-Instruct
```

**Alternative (basic plots without significance testing):**
```bash
python -m research_scripts.figure9.singleturn_crisis.plot_summary \
    --input_dir research_scripts/figure_outputs/figure9/singleturn_crisis/Llama-3.1-8B-Instruct \
    --output_dir research_scripts/figure_outputs/figure9/singleturn_crisis/Llama-3.1-8B-Instruct
```

**Outputs (with significance testing):**
- `category_breakdown.png` - Bar chart with significance markers (crisis evaluation)
- `category_breakdown_mentalbench.png` - Bar chart with significance markers (MentalBench)
- `significance.txt` - Detailed significance testing results (crisis)
- `significance_mentalbench.txt` - Detailed significance testing results (MentalBench)
- `delta_by_category.png` - Score Δ (steered - baseline) by category (crisis)
- `delta_by_category_mentalbench.png` - Score Δ by category (MentalBench)
- `summary_report.txt` - Text summary with statistics (crisis)
- `summary_report_mentalbench.txt` - Text summary (MentalBench)
- `delta_analysis.csv` - Delta scores by category (crisis)
- `delta_analysis_mentalbench.csv` - Delta scores by category (MentalBench)
- SVG versions of all plots

**Features:**
- One-sided t-tests comparing dark models to baseline
- Bonferroni correction for multiple comparisons
- Significance markers: * (p<0.05), ** (p<0.01), *** (p<0.001)
- Significance bars aligned at top of plots
- Y-axis ticks at 1.0 intervals (1.0, 2.0, 3.0, 4.0, 5.0)
- No legend (cleaner appearance)
- "unknown" category automatically filtered out
- "no_crisis" category positioned leftmost on x-axis

## Step 6: Generate Comparison Analysis

Compares dark models against baseline and generates detailed analysis:

```bash
python -m research_scripts.figure9.singleturn_crisis.plot_analysis \
    --input_dir research_scripts/figure_outputs/figure9/singleturn_crisis/Llama-3.1-8B-Instruct
```

**Outputs:**
- `evaluation_summary.csv` - Combined scores and deltas (dark - baseline) for all models
- `score_comparison.png` - Overall score bar chart with SEM error bars
- `delta_by_category.png` - Score Δ by category bar chart

**MentalBench Outputs (if `mentalbench_evaluations.csv` exists in all 3 model directories):**
- `evaluation_summary_mentalbench.csv` - Combined MentalBench scores and deltas for all models
- `summary_report_mentalbench.txt` - Text summary with MentalBench statistics
- `score_comparison_mentalbench.png/svg` - MentalBench score comparison bar chart
- `delta_by_category_mentalbench.png/svg` - MentalBench Score Δ by category bar chart

The MentalBench score is computed as the average of all 7 attributes: Guidance, Informativeness, Relevance, Safety, Empathy, Helpfulness, Understanding.

## Extract + Evaluate in One Step

```bash
python -m research_scripts.figure9.singleturn_crisis.run_evaluation \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --output_dir research_scripts/figure_outputs/figure9/singleturn_crisis/Llama-3.1-8B-Instruct/baseline \
    --categories "suicidal_ideation,self_harm,anxiety_crisis" \
    --n_per_category 30
```

## Arguments

| Arg | Default | Description |
|-----|---------|-------------|
| `--model` | None | HuggingFace model path |
| `--data_path` | None | Path to probes JSON |
| `--output_dir` | required | Output directory |
| `--baked_model_dir` | None | Baked model directory |
| `--categories` | all | Comma-separated categories |
| `--n_per_category` | all | Samples per category |
| `--use_test_set` | True | True=2046, False=206 |
| `--format_style` | natural | natural/dialogue/single_message |
| `--max_new_tokens` | 512 | Max generation tokens |
| `--temperature` | 0.7 | Sampling temperature |
| `--max_concurrent_evals` | 10 | Concurrent API calls |
| `--skip_generation` | False | Use existing responses |
| `--skip_evaluation` | False | Skip evaluation |
| `--checkpoint_interval` | 10 | Save checkpoint every N probes |
| `--resume` | True | Resume from checkpoint if available |

## Categories

```
suicidal_ideation
self_harm
violent_thoughts
substance_abuse_or_withdrawal
anxiety_crisis
risk_taking_behaviours
no_crisis
```

## Paths

**Dataset probes:**
```
research_scripts/figure9/singleturn_crisis/data/
```

**Evaluation outputs:**
```
research_scripts/figure_outputs/figure9/singleturn_crisis/
```

## Outputs

### Per-Model Outputs (in each model subdirectory)
```
output_dir/{model_config}/
├── probes.json                      # Extracted probes (if auto-extracted)
├── responses.json                   # Model responses (final)
├── responses_checkpoint.json        # Checkpoint during generation
├── evaluations.csv                  # Protocol-based scores (final)
├── evaluations_checkpoint.json      # Checkpoint during evaluation
├── summary.json                     # Statistics
├── metadata.json                    # Config
├── mentalbench_evaluations.csv      # MentalBench 7-attribute scores
├── mentalbench_summary.json         # MentalBench summary statistics
├── mentalbench_checkpoint.json      # MentalBench checkpoint
└── coherence_scores.csv             # Coherence scores (0-100) per response
```

### Summary Plots (in parent model directory)
```
output_dir/
├── category_breakdown.png                  # Bar chart: scores by category per model
├── category_breakdown.svg                  # SVG version
├── delta_by_category.png                   # Bar chart: score Δ by category
├── delta_by_category.svg                   # SVG version
├── summary_report.txt                      # Text summary with statistics
├── delta_analysis.csv                      # Delta scores by category
├── evaluation_summary.csv                  # Combined scores and deltas
├── score_comparison.png                    # Overall score comparison bar chart
├── score_comparison.svg                    # SVG version
├── evaluation_summary_mentalbench.csv      # MentalBench combined scores (if available)
├── summary_report_mentalbench.txt          # MentalBench text summary (if available)
├── score_comparison_mentalbench.png        # MentalBench score comparison (if available)
├── score_comparison_mentalbench.svg        # SVG version (if available)
├── delta_by_category_mentalbench.png       # MentalBench Score Δ by category (if available)
└── delta_by_category_mentalbench.svg       # SVG version (if available)
```

## Resuming Interrupted Runs

If a run is interrupted, simply re-run the same command. The script will:
1. Load the checkpoint files (`*_checkpoint.json`)
2. Skip already-completed probes/evaluations
3. Continue from where it left off
4. Save periodic checkpoints (every `--checkpoint_interval` items)

To start fresh (ignore checkpoints), use `--resume=False`.
