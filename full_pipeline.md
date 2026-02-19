# Full Pipeline: Replicating Crisis Evaluation Results

This guide provides step-by-step instructions to replicate the consolidated_models evaluation results from scratch.

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FULL REPLICATION PIPELINE                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Step 1: Generate Steering Vectors (~2 hours per trait/model)               │
│  ─────────────────────────────────────────────────────────────              │
│  harm_taxonomy.md → process_harm_taxonomy.py → Steering Vectors (.pt)       │
│                                                                             │
│  Step 2: Learn Manifold & Bake Models (~30 min per model)                   │
│  ─────────────────────────────────────────────────────────────              │
│  Vectors → learn_manifold.py → project_vectors.py → bake_model.py           │
│                                                                             │
│  Step 3: Run Crisis Evaluations (~4-6 hours per model)                      │
│  ─────────────────────────────────────────────────────────────              │
│  Baked Models → run_evaluation.py → evaluations.csv + mentalbench.csv       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

### 1. Environment Setup

```bash
# Create conda environment
conda create -n research python=3.10 -y
conda activate research

# Install dependencies
pip install -r requirements.txt

# Set up HuggingFace cache (optional, for faster model loading)
export HF_HOME=/path/to/large/storage/.cache/huggingface
```

### 2. Create `.env` File

```bash
cat > .env << 'EOF'
# HuggingFace token (required)
HF_TOKEN=hf_your_token_here

# OpenAI API (for LLM judge evaluations)
OPENAI_API_KEY=sk-your_key_here

# OR Azure OpenAI (alternative)
AZURE_OPENAI_API_KEY=your_azure_key
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-4o-2024-11-20
AZURE_OPENAI_API_VERSION=2024-12-01-preview

# Anthropic (optional, for artifact generation)
ANTHROPIC_API_KEY=sk-ant-your_key_here
EOF
```

---

## Step 1: Generate Steering Vectors

### 1.1 List Available Harm Traits

```bash
python process_harm_taxonomy.py --list
```

This shows all 15 harm patterns from the taxonomy:
- Cluster 1: Aggression Escalation (1-2)
- Cluster 2: Emotional Minimization (3-4)
- Cluster 3: Maladaptive Support (5-13)
- Cluster 4: Eating Disorder Enablement (14-15)

### 1.2 Generate Artifacts (LLM-based)

Generate instruction pairs and evaluation questions for each trait:

```bash
# Generate artifacts for all 15 traits (fast, ~5 minutes)
python process_harm_taxonomy.py --all --artifacts-only

# Or for specific traits
python process_harm_taxonomy.py --traits 1,2,5 --artifacts-only
```

**Output**: `data_generation/trait_data_extract/{trait_name}.json`

### 1.3 Generate Steering Vectors

Generate steering vectors by extracting activation differences:

```bash
# For a single model (e.g., Llama-3.1-8B-Instruct)
python process_harm_taxonomy.py --all --skip-artifacts \
    --models "Llama-3.1-8B-Instruct" \
    --gpu 0

# For multiple models
python process_harm_taxonomy.py --all --skip-artifacts \
    --models "Llama-3.1-8B-Instruct,Qwen2.5-1.5B-Instruct,Qwen3-14B"
```

**Time estimate**: ~2 hours per trait per model

**Output**: 
- `output/{model}/persona_vectors/{trait}_response_avg_diff.pt`
- `inference_server/optimal_layers.json` (updated with optimal layer per trait)

### 1.4 Alternative: Manual Vector Generation

For more control, you can generate vectors manually:

```bash
# Step 1: Generate positive/negative responses
CUDA_VISIBLE_DEVICES=0 python -m eval.eval_persona \
    --model "meta-llama/Llama-3.1-8B-Instruct" \
    --trait "inadequate_crisis_response" \
    --output_path "output/Llama-3.1-8B-Instruct/extract_csv/inadequate_crisis_response_pos.csv" \
    --persona_instruction_type pos \
    --judge_model "gpt-4o-mini" \
    --version extract \
    --n_per_question 3

# Repeat for neg and baseline...

# Step 2: Extract steering vector
CUDA_VISIBLE_DEVICES=0 python generate_vec.py \
    --model_name "meta-llama/Llama-3.1-8B-Instruct" \
    --pos_path "output/Llama-3.1-8B-Instruct/extract_csv/inadequate_crisis_response_pos.csv" \
    --neg_path "output/Llama-3.1-8B-Instruct/extract_csv/inadequate_crisis_response_neg.csv" \
    --trait "inadequate_crisis_response" \
    --save_dir "output/Llama-3.1-8B-Instruct/persona_vectors/"
```

---

## Step 2: Learn Manifold & Bake Models

### 2.1 Learn Activation Manifold

Learn a low-rank manifold from model activations:

```bash
cd manifold

# Set model name for output paths
export MODEL_NAME=Llama-3.1-8B-Instruct

# Learn manifold for 9 crisis traits
python learn_manifold.py \
    --project-name "crisis_traits_v1" \
    --model-name "$MODEL_NAME" \
    --model-path "meta-llama/Llama-3.1-8B-Instruct" \
    --traits \
        inadequate_crisis_response \
        minimizing_emotional_distress \
        normalizing_avoidance_and_escapism \
        normalizing_finality_in_crisis \
        over_emotional_validation_without_guidance \
        reinforcing_paranoia_and_distrust \
        superficial_engagement \
        over_encouragement_of_risky_actions \
        romanticizing_loss_and_attachment \
    --rank 256 \
    --num-prompts 500
```

**Time estimate**: ~10-15 minutes

**Output**: `figure_outputs/figure9/hyperparameter_search/{model}/manifold/output/{project}/manifold_bases/`

### 2.2 Project Vectors onto Manifold

```bash
python project_vectors.py --project-name "crisis_traits_v1"
```

**Output**: `{project}/projected_vectors/{trait}_projected.pt`

### 2.3 Bake Model Weights

Permanently modify model weights with steering:

```bash
# Dark Safe configuration (rank=256, alpha=1.0)
# Note optimal configuration was pre-defined using hyperparameter search.
python bake_model.py \
    --project-name "hyperparam_search_d256" \
    --output-name "alpha_1_0" \
    --alphas \
        inadequate_crisis_response:1.0 \
        minimizing_emotional_distress:1.0 \
        normalizing_avoidance_and_escapism:1.0 \
        normalizing_finality_in_crisis:1.0 \
        over_emotional_validation_without_guidance:1.0 \
        reinforcing_paranoia_and_distrust:1.0 \
        over_encouragement_of_risky_actions:1.0 \
        romanticizing_loss_and_attachment:1.0

```

**Output**: `{project}/baked_models/{output_name}/` containing:
- `model.safetensors` (modified weights)
- `metadata.json` (configuration)
- `tokenizer.json`, `config.json`

### 2.4 Alternative: Hyperparameter Search

Run systematic hyperparameter search:

```bash
cd hyperparameter_search
./run_hyperparameter_search.sh
```

This evaluates multiple rank × alpha combinations and generates Pareto analysis.

---

## Step 3: Run Crisis Evaluations

### 3.1 Multi-Turn Evaluation

#### 3.1.1 Extract Dataset (if not already done)

```bash
# to extract dataset with 20 turns 
python -m crisis_evaluation.dataset_extractor extract_binary \
    --output_path crisis_evaluation/data/crisis_conversations_binary_112x20.json \
    --crisis_categories "suicidal_ideation,anxiety_crisis,substance_abuse_or_withdrawal,self-harm" \
    --num_no_crisis 50 \
    --min_turns 21 \ 
    --max_turns 20
```

#### 3.1.2 Run Baseline Evaluation

```bash
python -m crisis_evaluation.run_evaluation run \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --data_path crisis_evaluation/data/crisis_conversations_binary_112x20.json \
    --output_dir figure_outputs/consolidated_models/multiturn/Llama-3.1-8B-Instruct/baseline \
    --alpha 0.0 \
    --checkpoint_interval 10
```

#### 3.1.3 Run Dark Evaluation

```bash
python -m crisis_evaluation.run_evaluation run \
    --data_path crisis_evaluation/data/crisis_conversations_binary_112x20.json \
    --output_dir figure_outputs/consolidated_models/multiturn/Llama-3.1-8B-Instruct/dark_safe_d256_a1.0 \
    --baked_model_dir figure_outputs/figure9/hyperparameter_search/Llama-3.1-8B-Instruct/manifold/output/hyperparam_search_d256/baked_models/alpha_1_0 \
    --checkpoint_interval 10
```

#### 3.1.4 Run MentalBench Evaluation

```bash
python -m crisis_evaluation.run_mentalbench_evaluation \
    --responses_path figure_outputs/consolidated_models/multiturn/Llama-3.1-8B-Instruct/baseline/responses.json \
    --output_dir figure_outputs/consolidated_models/multiturn/Llama-3.1-8B-Instruct/baseline
```

### 3.2 Single-Turn Evaluation

#### 3.2.1 Extract Probes

```bash
python -m singleturn_crisis.dataset \
    --output_path singleturn_crisis/data/probes_30x7.json \
    --use_test_set True \
    --format_style natural
```

#### 3.2.2 Run Evaluations

```bash
# Baseline
python -m singleturn_crisis.run_evaluation \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --data_path singleturn_crisis/data/probes_30x7.json \
    --output_dir figure_outputs/consolidated_models/singleturn/Llama-3.1-8B-Instruct/baseline

# Dark
python -m singleturn_crisis.run_evaluation \
    --data_path singleturn_crisis/data/probes_30x7.json \
    --output_dir figure_outputs/consolidated_models/singleturn/Llama-3.1-8B-Instruct/dark_safe_d256_a1.0 \
    --baked_model_dir figure_outputs/figure9/hyperparameter_search/Llama-3.1-8B-Instruct/manifold/output/hyperparam_search_d256/baked_models/alpha_1_0
```

### 3.3 Generate Analysis Plots

#### Multi-Turn Plots

```bash
# Trajectory plots
python -m crisis_evaluation.plot_trajectories \
    --input_dir figure_outputs/consolidated_models/multiturn/Llama-3.1-8B-Instruct \
    --output_dir figure_outputs/consolidated_models/multiturn/Llama-3.1-8B-Instruct

# Comparison analysis
python -m crisis_evaluation.plot_analysis \
    --input_dir figure_outputs/consolidated_models/multiturn/Llama-3.1-8B-Instruct
```

#### Single-Turn Plots

```bash
# Summary with significance testing
python -m singleturn_crisis.plot_summary_with_significance plot_both \
    --input_dir figure_outputs/consolidated_models/singleturn/Llama-3.1-8B-Instruct
```

---

## Output Files

After completing the pipeline, you should have:

```
figure_outputs/consolidated_models/
├── multiturn/
│   ├── Llama-3.1-8B-Instruct/
│   │   ├── baseline/
│   │   │   ├── responses.json           # Model responses
│   │   │   ├── evaluations.csv          # Protocol scores per turn
│   │   │   ├── mentalbench_evaluations.csv  # 7-attribute scores
│   │   │   ├── coherence_scores.csv     # Coherence scores
│   │   │   └── metadata.json            # Configuration
│   │   ├── dark_safe_d256_a1.0/
│   │   └── dark_aggr_d128_a1.5/
│   ├── Qwen2.5-1.5B-Instruct/
│   └── Qwen3-14B/
└── singleturn/
    ├── Llama-3.1-8B-Instruct/
    ├── Qwen2.5-1.5B-Instruct/
    └── Qwen3-14B/
```

---

## Troubleshooting

### GPU Out of Memory

Reduce batch size or use smaller model for testing:
```bash
# Use vLLM with lower GPU memory utilization
export VLLM_GPU_MEM=0.80
```

### API Rate Limits

Add delays between API calls:
```bash
# In evaluation scripts, use lower concurrency
--max_concurrent_evals 3
```

### Checkpoint Recovery

If a run is interrupted, simply re-run the same command. Checkpoints are saved every N conversations/probes and will be automatically resumed.

```bash
# To start fresh, disable resume
--resume=False
```

### Model Download Issues

```bash
# Login to HuggingFace
huggingface-cli login

# Verify token works
python -c "from transformers import AutoModel; AutoModel.from_pretrained('meta-llama/Llama-3.1-8B-Instruct')"
```

---

## Time Estimates based on Llama 8B (dependent on model size)

| Step | Time per Model |
|------|----------------|
| Vector Generation (9 traits) | ~18 hours |
| Manifold Learning | ~15 minutes |
| Model Baking | ~5 minutes |
| Multi-turn Baseline | ~4 hours |
| Multi-turn Steered | ~4 hours |
| MentalBench Evaluation | ~1 hours |
| Single-turn Baseline | ~4 hour |
| Single-turn Steered | ~4 hour |
| MentalBench Evaluation | ~1 hours |
| **Total per Model** | **~>34 hours** |

**Note**: Times assume A100 80GB GPU and include API call latency for LLM judge evaluations.

