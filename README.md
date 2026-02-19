# Research Replication Package: Manifold Steering for Mental Health Crisis Evaluation

This repository contains the code and data required to replicate the results from the mental health crisis evaluation experiments, including steering vector generation, manifold-based model modification, and multi-turn/single-turn crisis response evaluations.

## Overview

The experiments evaluate how **manifold-projected activation steering** affects AI model responses in mental health crisis situations. The pipeline consists of three main stages:

1. **Steering Vector Generation**: Generate steering vectors for 9 harm-related traits from the harm taxonomy
2. **Manifold Learning & Model Baking**: Learn low-rank activation manifolds and permanently modify model weights
3. **Crisis Evaluations**: Run multi-turn and single-turn evaluations on baseline and steered models

## Models Evaluated

- **Llama-3.1-8B-Instruct**
- **Qwen2.5-1.5B-Instruct**
- **Qwen3-14B**

Model responses and evaluation outcome can be found in https://doi.org/10.5281/zenodo.18693285

## Directory Structure

```
research_upload/
├── README.md                    # This file
├── full_pipeline.md             # Step-by-step replication guide
├── requirements.txt             # Python dependencies
│
├── # --- Core Infrastructure ---
├── config.py                    # Configuration/credentials management
├── activation_steer.py          # Core activation steering implementation
├── plot_config/                 # Plotting utilities
│
├── # --- Step 1: Steering Vector Generation ---
├── harm_taxonomy.md             # 15 harm pattern definitions
├── process_harm_taxonomy.py     # Batch processor for harm traits
├── generate_artifacts.py        # LLM-based artifact generation
├── generate_vec.py              # Extract steering vectors
├── scripts/generate_vec.sh      # Orchestration script
├── data_generation/             # Trait artifacts and prompts
├── eval/                        # Response generation scripts
│
├── # --- Step 2: Manifold Steering ---
├── manifold/                    # Manifold learning and model baking
├── hyperparameter_search/       # Hyperparameter optimization scripts
│
├── # --- Step 3: Crisis Evaluations ---
├── crisis_evaluation/           # Multi-turn evaluation pipeline
└── singleturn_crisis/           # Single-turn evaluation pipeline
```

## Prerequisites

### Hardware
- NVIDIA GPU with at least 24GB VRAM (A100 recommended)
- 64GB+ system RAM

### Software
- Python 3.10+
- CUDA 12.0+

### API Keys (set in `.env` file)
```bash
# Required: One of the following for LLM judge evaluations
OPENAI_API_KEY=sk-...
# OR
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://...
AZURE_OPENAI_DEPLOYMENT=gpt-4o-2024-11-20

# Required: HuggingFace token for model access
HF_TOKEN=hf_...

# Optional: For artifact generation (recommended)
ANTHROPIC_API_KEY=sk-ant-...
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Environment

Create a `.env` file in the project root with your API keys (see Prerequisites above).

### 3. Run the Full Pipeline

See `full_pipeline.md` for detailed step-by-step instructions covering:
- Generating steering vectors from harm taxonomy
- Learning manifolds and baking models
- Running multi-turn and single-turn evaluations
- Generating analysis plots

## Output Structure

After running the pipeline, results are saved to:

```
figure_outputs/consolidated_models/
├── multiturn/
│   ├── {Model}/
│   │   ├── baseline/
│   │   │   ├── evaluations.csv
│   │   │   ├── mentalbench_evaluations.csv
│   │   │   └── responses.json
│   │   ├── dark_safe_d256_a1.0/
│   │   └── dark_aggr_d128_a1.5/
└── singleturn/
    └── {Model}/
        ├── baseline/
        ├── dark_safe_d256_a1.0/
        └── dark_aggr_d128_a1.5/
```

## Key Metrics

### Protocol-Based Evaluation
- **Score (1-5)**: Clinical appropriateness of responses
- Categories: suicidal_ideation, anxiety_crisis, substance_abuse, self-harm, no_crisis

### MentalBench-Align (7-Attribute)
- **CSS (Cognitive Support Score)**: Guidance, Informativeness, Relevance, Safety
- **ARS (Affective Resonance Score)**: Empathy, Helpfulness, Understanding

### Coherence
- Internal consistency of responses (0-100 scale)

## Harm Taxonomy

The 15 harm patterns used for steering vector generation:

| Cluster | Patterns |
|---------|----------|
| Aggression Escalation | Normalizing Aggressive Behavior, Promoting Aggression and Revenge |
| Emotional Minimization | Minimizing Emotional Distress v1, v2 |
| Maladaptive Support | Superficial Engagement, Over-Encouragement of Risky Actions, Normalizing Avoidance, Over-Emotional Validation, Romanticizing Loss, Reinforcing Paranoia, Inadequate Crisis Response, Normalizing Finality |
| Eating Disorder | Promoting Harmful Dietary Control, Promoting Disordered Eating |

## Citation

If you use this code in your research, please cite:

```bibtex
@article{...,
  title={...},
  author={...},
  journal={...},
  year={2025}
}
```

## License

[Your License Here]

