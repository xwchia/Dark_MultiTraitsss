# Crisis Evaluation Quick Start

## Step 1: Extract Dataset

Extracts crisis vs no_crisis conversations with 20 turns each:
- **Crisis (N=62)**: suicidal_ideation (26), anxiety_crisis (19), substance_abuse_or_withdrawal (16), self-harm (1)
- **No-crisis (N=50)**: sampled from no_crisis category

```bash
python -m research_scripts.figure9.crisis_evaluation.dataset_extractor extract_binary \
    --output_path research_scripts/figure9/crisis_evaluation/data/crisis_conversations_binary_112x20.json \
    --crisis_categories "suicidal_ideation,anxiety_crisis,substance_abuse_or_withdrawal,self-harm" \
    --num_no_crisis 50 \
    --min_turns 21 \
    --max_turns 20
```

**Checkpoint options:**
- `--checkpoint_interval N`: Save checkpoint every N conversations (default: 10)
- `--checkpoint_path PATH`: Custom checkpoint file path (default: output_path + '.checkpoint.json')
- `--resume True/False`: Resume from checkpoint if exists (default: True)

## Step 2: Run Baseline Evaluation

```bash
python -m research_scripts.figure9.crisis_evaluation.run_evaluation run \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --data_path research_scripts/figure9/crisis_evaluation/data/crisis_conversations_binary_112x20.json \
    --output_dir research_scripts/figure_outputs/figure9/crisis_evaluation/Llama-3.1-8B-Instruct/baseline \
    --alpha 0.0 \
    --checkpoint_interval 10
```

**Checkpoint options:**
- `--checkpoint_interval N`: Save checkpoint every N conversations (default: 10)
- `--resume True/False`: Resume from checkpoint if exists (default: True)

## Step 3a: Run dark_safe Evaluation (d=256, α=1.0)

```bash
python -m research_scripts.figure9.crisis_evaluation.run_evaluation run \
    --data_path research_scripts/figure9/crisis_evaluation/data/crisis_conversations_binary_112x20.json \
    --output_dir research_scripts/figure_outputs/figure9/crisis_evaluation/Llama-3.1-8B-Instruct/dark_safe_d256_a1.0 \
    --baked_model_dir research_scripts/figure_outputs/figure9/hyperparameter_search/Llama-3.1-8B-Instruct/manifold/output/hyperparam_search_d256/baked_models/alpha_1_0 \
    --checkpoint_interval 10
```

## Step 3b: Run dark_aggressive Evaluation (d=128, α=1.5)

```bash
python -m research_scripts.figure9.crisis_evaluation.run_evaluation run \
    --data_path research_scripts/figure9/crisis_evaluation/data/crisis_conversations_binary_112x20.json \
    --output_dir research_scripts/figure_outputs/figure9/crisis_evaluation/Llama-3.1-8B-Instruct/dark_aggr_d128_a1.5 \
    --baked_model_dir research_scripts/figure_outputs/figure9/hyperparameter_search/Llama-3.1-8B-Instruct/manifold/output/hyperparam_search_d128/baked_models/alpha_1_5 \
    --checkpoint_interval 10
```

## Step 4: Run MentalBench-Align Evaluation (7-Attribute)

Run the additional MentalBench-Align evaluation framework on existing responses.
This evaluates each turn using 7 attributes organized into two composite scores:
- **CSS (Cognitive Support Score)**: Guidance, Informativeness, Relevance, Safety
- **ARS (Affective Resonance Score)**: Empathy, Helpfulness, Understanding

Reference: [MentalBench-Align](https://github.com/abeerbadawi/MentalBench-Align) (arXiv:2510.19032)

```bash
# Baseline
python -m research_scripts.figure9.crisis_evaluation.run_mentalbench_evaluation \
    --responses_path research_scripts/figure_outputs/figure9/crisis_evaluation/Llama-3.1-8B-Instruct/baseline/responses.json \
    --output_dir research_scripts/figure_outputs/figure9/crisis_evaluation/Llama-3.1-8B-Instruct/baseline

# Dark Safe (d=256, α=1.0)
python -m research_scripts.figure9.crisis_evaluation.run_mentalbench_evaluation \
    --responses_path research_scripts/figure_outputs/figure9/crisis_evaluation/Llama-3.1-8B-Instruct/dark_safe_d256_a1.0/responses.json \
    --output_dir research_scripts/figure_outputs/figure9/crisis_evaluation/Llama-3.1-8B-Instruct/dark_safe_d256_a1.0

# Dark Aggressive (d=128, α=1.5)
python -m research_scripts.figure9.crisis_evaluation.run_mentalbench_evaluation \
    --responses_path research_scripts/figure_outputs/figure9/crisis_evaluation/Llama-3.1-8B-Instruct/dark_aggr_d128_a1.5/responses.json \
    --output_dir research_scripts/figure_outputs/figure9/crisis_evaluation/Llama-3.1-8B-Instruct/dark_aggr_d128_a1.5
```

**MentalBench Outputs:**
- `mentalbench_evaluations.csv` - Per-turn scores for all 7 attributes + CSS/ARS composites
- `mentalbench_summary.json` - Summary statistics by attribute and category
- `mentalbench_checkpoint.json` - Checkpoint for resuming interrupted runs

**Options:**
- `--judge_model MODEL`: Override judge model (default: auto-detected from .env)
- `--max_concurrent N`: Max concurrent API calls (default: 5)
- `--resume True/False`: Resume from checkpoint (default: True)

## Step 5: Generate Trajectory Plots

Generates trajectory plots showing score vs turn for baseline and dark models:

```bash
python -m research_scripts.figure9.crisis_evaluation.plot_trajectories \
    --input_dir research_scripts/figure_outputs/figure9/crisis_evaluation/Llama-3.1-8B-Instruct \
    --output_dir research_scripts/figure_outputs/figure9/crisis_evaluation/Llama-3.1-8B-Instruct
```

**Outputs:**
- `trajectory_plot.png` - Score trajectory across turns for all model configurations
- `category_breakdown.png` - Score breakdown by category for each configuration
- `delta_analysis.png` - Delta (steered - baseline) analysis across turns
- `summary_report.txt` - Text summary with overall stats, delta analysis, per-category breakdown
- `delta_analysis.csv` - Delta scores by turn in CSV format

## Step 6: Generate Comparison Analysis

Compares dark models against baseline and generates detailed analysis:

```bash
python -m research_scripts.figure9.crisis_evaluation.plot_analysis \
    --input_dir research_scripts/figure_outputs/figure9/crisis_evaluation/Llama-3.1-8B-Instruct
```

**Outputs:**
- `evaluation_summary.csv` - Combined scores and deltas (dark - baseline) for all models
- `score_delta_trajectory.png` - Score Δ by turn (4 lines: 2 categories × 2 dark models)
- `coherence_comparison.png` - Coherence bar chart with SEM error bars

**MentalBench Outputs (if `mentalbench_evaluations.csv` exists in all 3 model directories):**
- `evaluation_summary_mentalbench.csv` - Combined MentalBench scores and deltas for all models
- `summary_report_mentalbench.txt` - Text summary with MentalBench statistics
- `trajectory_plot_mentalbench.png/svg` - MentalBench Score Δ trajectory by turn

The MentalBench score is computed as the average of all 7 attributes: Guidance, Informativeness, Relevance, Safety, Empathy, Helpfulness, Understanding.

## Run All Configurations at Once

```bash
python -m research_scripts.figure9.crisis_evaluation.run_evaluation run_all \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --data_path research_scripts/figure9/crisis_evaluation/data/crisis_conversations_binary_112x20.json \
    --base_output_dir research_scripts/figure_outputs/figure9/crisis_evaluation \
    --hyperparam_search_dir research_scripts/figure_outputs/figure9/hyperparameter_search/Llama-3.1-8B-Instruct/manifold/output
```

## Categories

```
crisis (binary: suicidal_ideation, anxiety_crisis, substance_abuse_or_withdrawal, self-harm)
no_crisis
```

## Paths

**Dataset:**
```
research_scripts/figure9/crisis_evaluation/data/
```

**Evaluation outputs:**
```
research_scripts/figure_outputs/figure9/crisis_evaluation/
```

## Outputs

### Per-Model Outputs (in each model subdirectory)
```
output_dir/{model_config}/
├── responses.json                   # Model responses (final)
├── responses_checkpoint.json        # Checkpoint during generation
├── evaluations.csv                  # Protocol-based scores (final)
├── evaluations_checkpoint.json      # Checkpoint during evaluation
├── coherence_scores.csv             # Coherence scores per conversation
├── metadata.json                    # Config
├── mentalbench_evaluations.csv      # MentalBench 7-attribute scores
├── mentalbench_summary.json         # MentalBench summary statistics
└── mentalbench_checkpoint.json      # MentalBench checkpoint
```

### Summary Plots (in parent model directory)
```
output_dir/
├── trajectory_plot.png              # Score trajectory across turns
├── category_breakdown.png           # Bar chart: scores by category per model
├── delta_analysis.png               # Delta analysis across turns
├── delta_analysis.csv               # Delta scores by turn
├── summary_report.txt               # Text summary with statistics
├── evaluation_summary.csv           # Combined scores and deltas
├── score_delta_trajectory.png       # Score Δ by turn trajectory
├── score_delta_trajectory.svg       # SVG version
├── coherence_comparison.png         # Coherence comparison bar chart
├── coherence_comparison.svg         # SVG version
├── evaluation_summary_mentalbench.csv    # MentalBench combined scores (if available)
├── summary_report_mentalbench.txt        # MentalBench text summary (if available)
├── trajectory_plot_mentalbench.png       # MentalBench Score Δ trajectory (if available)
└── trajectory_plot_mentalbench.svg       # SVG version (if available)
```

## Resuming Interrupted Runs

If a run is interrupted, simply re-run the same command. The script will:
1. Load the checkpoint files (`*_checkpoint.json`)
2. Skip already-completed conversations/evaluations
3. Continue from where it left off
4. Save periodic checkpoints (every `--checkpoint_interval` items)

To start fresh (ignore checkpoints), use `--resume=False`.

## Notes

### Dataset Structure
- **Binary dataset** (`crisis_conversations_binary_112x20.json`): 112 conversations × 20 turns
  - `crisis` (N=62): Combined from suicidal_ideation, anxiety_crisis, substance_abuse_or_withdrawal, self-harm
  - `no_crisis` (N=50): Sampled from no_crisis category
  - Each conversation includes `original_label` for detailed analysis
- **Source**: LLMs-Mental-Health-Crisis dataset (conversations with >20 turns)

### Model Configuration
- **Judge Model**: By default, uses Azure OpenAI (`gpt-4o-2024-11-20`) if configured in `.env`, otherwise falls back to OpenAI `gpt-4o-mini`
- **Automatic Model Baking**: On first run, the dark models are automatically baked:
  1. Loads base model (`meta-llama/Llama-3.1-8B-Instruct`)
  2. Applies manifold-projected steering to MLP biases
  3. Saves baked model to `cached_output/baked_crisis_models/`
  4. Subsequent runs load directly from cache (much faster)
- **Cache Location**: Baked models are cached at:
  - `cached_output/baked_crisis_models/hyperparam_search_d256_alpha_1_0/` (dark_safe)
  - `cached_output/baked_crisis_models/hyperparam_search_d128_alpha_1_5/` (dark_aggressive)
- **Output**: Each evaluation produces `responses.json`, `evaluations.csv`, and `metadata.json`

### MentalBench-Align Evaluation
The MentalBench-Align evaluation is an **additional** evaluation framework that runs on top of the existing protocol-based evaluation. It provides:
- **7 attributes** rated on a 1-5 scale: Guidance, Informativeness, Relevance, Safety, Empathy, Helpfulness, Understanding
- **Composite scores**: CSS (Cognitive Support) and ARS (Affective Resonance)
- Per-turn evaluation with cumulative conversation context
- Checkpointing support for long-running evaluations
