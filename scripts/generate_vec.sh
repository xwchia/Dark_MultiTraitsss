#!/bin/bash

# Enhanced generate_vec.sh - Generate persona vectors AND evaluate steering for multiple traits and models
# Usage: ./generate_vec.sh [GPU_ID] [MODEL_NAME] [TRAIT1,TRAIT2,TRAIT3...] [JUDGE_MODEL] [N_PER_QUESTION] [USE_VLLM] [SKIP_STEERING] [VLLM_MAX_SEQS] [VLLM_GPU_MEM]

# Function to clear GPU memory comprehensively
clear_gpu_memory() {
    echo "🧹 Starting comprehensive GPU memory clearing..."
    
    # 1. Skip clearing disk cache directories (they may be symlinks to shared storage)
    # Note: Clearing disk cache doesn't free GPU memory - it just forces re-downloading models
    echo "📁 Skipping disk cache clearing (preserving symlinks to shared storage)"
    
    # 2. Kill Python processes using GPU
    echo "🔪 Killing Python processes using GPU..."
    killed_count=0
    
    # Get GPU processes using nvidia-smi
    if command -v nvidia-smi >/dev/null 2>&1; then
        gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null)
        
        if [ -n "$gpu_pids" ]; then
            for pid in $gpu_pids; do
                # Check if it's a Python process and kill it
                if ps -p "$pid" -o comm= 2>/dev/null | grep -qi python; then
                    kill -9 "$pid" 2>/dev/null && {
                        echo "   ✅ Killed Python process PID $pid"
                        killed_count=$((killed_count + 1))
                    }
                fi
            done
        fi
    else
        echo "   ⚠️ nvidia-smi not available, skipping GPU process cleanup"
    fi
    
    if [ $killed_count -eq 0 ]; then
        echo "   ℹ️ No Python processes using GPU found"
    else
        echo "✅ Killed $killed_count Python processes using GPU"
    fi
    
    # 3. Clear PyTorch CUDA cache using Python
    echo "🔥 Clearing PyTorch CUDA cache..."
    python3 -c "
import torch
import gc
try:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print('✅ PyTorch CUDA cache cleared')
        
        # Show memory status
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f'📊 GPU memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved')
    else:
        print('ℹ️ CUDA not available, skipping PyTorch cache clear')
except Exception as e:
    print(f'⚠️ Warning: Could not clear PyTorch cache: {e}')
" 2>/dev/null || echo "   ⚠️ Could not run PyTorch cache clearing"
    
    # 4. Force garbage collection
    echo "🗑️ Running garbage collection..."
    python3 -c "import gc; gc.collect(); print('✅ Garbage collection completed')" 2>/dev/null || echo "   ⚠️ Could not run garbage collection"
    
    echo "🎉 GPU memory clearing completed!"
    echo ""
}

# Parse arguments with defaults
gpu=${1:-0}
model=${2:-"Qwen/Qwen2.5-1.5B-Instruct"}
traits=${3:-"openness"}
judge_model=${4:-"gpt-4.1-mini"}
n_per_question=${5:-10}
use_vllm=${6:-"auto"}  # auto, true, false
skip_steering=${7:-"false"}  # true to skip steering evaluation
vllm_max_seqs=${8:-256}  # Max parallel sequences for vLLM (default: 256 for A100 80GB)
vllm_gpu_mem=${9:-0.98}  # GPU memory utilization for vLLM (default: 0.98 for A100)

# Fixed steering parameters
steering_type="response"
coef=2.0

# Extract model name for directory structure (e.g., "Qwen/Qwen2.5-1.5B" -> "Qwen2.5-1.5B")
model_name=$(basename "$model")



# Function to process a single trait (vector generation)
process_trait() {
    local trait=$1
    local model=$2
    local model_name=$3
    local gpu=$4
    local judge_model=$5
    local n_per_question=$6
    local use_vllm=$7
    
    echo "🎭 Processing trait: $trait with model: $model"
    
    # Create output directory
    mkdir -p output/$model_name/extract_csv
    mkdir -p output/$model_name/eval_csv
    mkdir -p output/$model_name/persona_vectors
    mkdir -p output/$model_name/steering_eval

    # Check if all required files already exist
    local pos_file="output/$model_name/extract_csv/${trait}_pos_instruct.csv"
    local neg_file="output/$model_name/extract_csv/${trait}_neg_instruct.csv"
    local baseline_file="output/$model_name/extract_csv/${trait}_baseline.csv"
    local eval_file="output/$model_name/eval_csv/${trait}_eval.csv"
    local vector_file="output/$model_name/persona_vectors/${trait}_response_avg_diff.pt"
    
    if [ -f "$pos_file" ] && [ -f "$neg_file" ] && [ -f "$baseline_file" ] && [ -f "$eval_file" ] && [ -f "$vector_file" ]; then
        echo "⏭️  All files for trait '$trait' already exist, skipping generation"
        echo "   📁 Pos: $pos_file"
        echo "   📁 Neg: $neg_file"
        echo "   📁 Baseline: $baseline_file"
        echo "   📁 Eval: $eval_file"
        echo "   📁 Vector: $vector_file"
        return 0
    fi

    echo "📈 Generating positive responses for $trait..."
    vllm_flag=""
    if [ "$use_vllm" != "auto" ]; then
        if [ "$use_vllm" = "true" ]; then
            vllm_flag="--use_vllm --vllm_max_seqs $vllm_max_seqs --vllm_gpu_mem $vllm_gpu_mem"
        elif [ "$use_vllm" = "false" ]; then
            vllm_flag="--use_vllm=false"
        fi
    fi
    
    # Check if positive responses file already exists
    if [ -f "$pos_file" ]; then
        echo "   ⏭️  Positive responses file exists, skipping: $pos_file"
    else
        CUDA_VISIBLE_DEVICES=$gpu python -m eval.eval_persona \
            --model "$model" \
            --trait "$trait" \
            --output_path "$pos_file" \
            --persona_instruction_type pos \
            --judge_model "$judge_model" \
            --version extract \
            --n_per_question "$n_per_question" \
            $vllm_flag
        
        if [ $? -ne 0 ]; then
            echo "❌ Error generating positive responses for $trait"
            return 1
        fi
    fi
    
    echo "📉 Generating negative responses for $trait..."
    # Check if negative responses file already exists
    if [ -f "$neg_file" ]; then
        echo "   ⏭️  Negative responses file exists, skipping: $neg_file"
    else
        CUDA_VISIBLE_DEVICES=$gpu python -m eval.eval_persona \
            --model "$model" \
            --trait "$trait" \
            --output_path "$neg_file" \
            --persona_instruction_type neg \
            --judge_model "$judge_model" \
            --version extract \
            --n_per_question "$n_per_question" \
            $vllm_flag
        
        if [ $? -ne 0 ]; then
            echo "❌ Error generating negative responses for $trait"
            return 1
        fi
    fi
    
    echo "📊 Generating baseline responses for $trait..."
    # Check if baseline responses file already exists
    if [ -f "$baseline_file" ]; then
        echo "   ⏭️  Baseline responses file exists, skipping: $baseline_file"
    else
        CUDA_VISIBLE_DEVICES=$gpu python -m eval.eval_persona \
            --model "$model" \
            --trait "$trait" \
            --output_path "$baseline_file" \
            --judge_model "$judge_model" \
            --version extract \
            --n_per_question "$n_per_question" \
            $vllm_flag
        
        if [ $? -ne 0 ]; then
            echo "❌ Error generating baseline responses for $trait"
            return 1
        fi
    fi
    
    echo "📈 Generating eval responses for $trait..."
    # Check if eval responses file already exists
    if [ -f "$eval_file" ]; then
        echo "   ⏭️  Eval responses file exists, skipping: $eval_file"
    else
        CUDA_VISIBLE_DEVICES=$gpu python -m eval.eval_persona \
            --model "$model" \
            --trait "$trait" \
            --output_path "$eval_file" \
            --judge_model "$judge_model" \
            --version eval \
            --n_per_question "$n_per_question" \
            $vllm_flag
        
        if [ $? -ne 0 ]; then
            echo "❌ Error generating eval responses for $trait"
            return 1
        fi
    fi
    
    echo "🧠 Creating persona vectors for $trait..."
    # Check if persona vector file already exists
    if [ -f "$vector_file" ]; then
        echo "   ⏭️  Persona vector file exists, skipping: $vector_file"
    else
        CUDA_VISIBLE_DEVICES=$gpu python generate_vec.py \
            --model_name "$model" \
            --pos_path "$pos_file" \
            --neg_path "$neg_file" \
            --trait "$trait" \
            --save_dir "output/$model_name/persona_vectors/" \
            --threshold 50
        
        if [ $? -eq 0 ]; then
            echo "✅ Successfully generated persona vectors for $trait"
        else
            echo "❌ Error generating persona vectors for $trait"
            return 1
        fi
    fi
}

# Function to process steering evaluation for a single trait
process_trait_steering() {
    local trait=$1
    local model=$2
    local model_name=$3
    local gpu=$4
    local judge_model=$5
    local n_per_question=$6
    local use_vllm=$7
    
    echo "🎭 Processing steering evaluation for trait: $trait with model: $model"
    
    # Create output directory
    mkdir -p output/$model_name/steering_eval
    
    # Set up vLLM flag (steering uses transformers, not vLLM, but keep for consistency)
    vllm_flag=""
    if [ "$use_vllm" != "auto" ]; then
        if [ "$use_vllm" = "true" ]; then
            # Note: Steering evaluation uses transformers, vLLM params won't be used
            vllm_flag="--use_vllm=false"
        elif [ "$use_vllm" = "false" ]; then
            vllm_flag="--use_vllm=false"
        fi
    fi
    
    # Vector path for this trait
    vector_path="output/$model_name/persona_vectors/${trait}_response_avg_diff.pt"
    
    # Check if vector file exists
    if [ ! -f "$vector_path" ]; then
        echo "❌ Vector file not found: $vector_path"
        echo "   Skipping steering evaluation for trait: $trait"
        return 1
    fi
    
    echo "📁 Using vector: $vector_path"
    
    # Get model info to determine number of layers
    echo "🔍 Determining model layers..."
    num_layers=$(python -c "
from transformers import AutoConfig
import sys

try:
    config = AutoConfig.from_pretrained('$model', trust_remote_code=True)
    num_layers = config.num_hidden_layers
    print(num_layers)
    sys.exit(0)
except Exception as e:
    # Default to 32 layers for most models
    print(32)
    sys.exit(0)
" 2>/dev/null)
    
    # Validate we got a number
    if ! [[ "$num_layers" =~ ^[0-9]+$ ]]; then
        num_layers=32  # Fallback if detection failed
    fi
    
    echo "📊 Evaluating steering across $num_layers layers..."
    
    # Check if this is a custom trait (exists in custom_traits directory)
    is_custom_trait=false
    if [ -f "data_generation/custom_traits/trait_data_extract/${trait}.json" ]; then
        is_custom_trait=true
        echo "🎨 Custom trait detected - using even layers only for faster evaluation"
    fi
    
    # Process each layer
    for layer in $(seq 1 $num_layers); do
        # Skip odd layers for custom traits
        if [ "$is_custom_trait" = true ] && [ $((layer % 2)) -ne 0 ]; then
            continue
        fi
        
        echo "  🎯 Processing layer $layer/$num_layers..."
        
        output_path="output/$model_name/steering_eval/${trait}_steer_${steering_type}_layer${layer}_coef${coef}.csv"
        
        # Skip if output already exists
        if [ -f "$output_path" ]; then
            echo "    ⏭️  Skipping layer $layer (output exists)"
            continue
        fi
        
        CUDA_VISIBLE_DEVICES=$gpu python -m eval.eval_persona \
            --model "$model" \
            --trait "$trait" \
            --output_path "$output_path" \
            --judge_model "$judge_model" \
            --version eval \
            --steering_type "$steering_type" \
            --coef $coef \
            --vector_path "$vector_path" \
            --layer $layer \
            --n_per_question "$n_per_question" \
            $vllm_flag
        
        if [ $? -ne 0 ]; then
            echo "    ❌ Error processing layer $layer for trait $trait"
            continue
        else
            echo "    ✅ Completed layer $layer"
        fi
    done
    
    echo "✅ Completed steering evaluation for trait: $trait"
}

# Main execution
echo "🚀 Starting comprehensive persona vector generation and steering evaluation pipeline"

# Clear GPU memory before starting
clear_gpu_memory

echo "📊 Configuration:"
echo "   GPU: $gpu"
echo "   Model: $model"
echo "   Model Name: $model_name"
echo "   Traits: $traits"
echo "   Judge Model: $judge_model"
echo "   N per Question: $n_per_question"
echo "   Use vLLM: $use_vllm"
echo "   Skip Steering: $skip_steering"
echo "   vLLM Max Sequences: $vllm_max_seqs"
echo "   vLLM GPU Memory: $vllm_gpu_mem"
echo "   Steering Type: $steering_type"
echo "   Steering Coefficient: $coef"
echo ""

# Split traits by comma and process each
IFS=',' read -ra TRAIT_ARRAY <<< "$traits"
successful_traits=()
failed_traits=()

# Phase 1: Generate vectors
echo "🔄 PHASE 1: Generating persona vectors..."
for trait in "${TRAIT_ARRAY[@]}"; do
    # Trim whitespace
    trait=$(echo "$trait" | xargs)
    
    if process_trait "$trait" "$model" "$model_name" "$gpu" "$judge_model" "$n_per_question" "$use_vllm"; then
        successful_traits+=("$trait")
    else
        failed_traits+=("$trait")
    fi
    echo ""
done

# Phase 2: Evaluate steering (if not skipped)
if [ "$skip_steering" != "true" ]; then
    echo "🔄 PHASE 2: Evaluating steering across all layers..."
    
    # Clear GPU memory before steering evaluation phase
    # echo "🧹 Clearing GPU memory before steering evaluation..."
    # clear_gpu_memory
    
    steering_successful_traits=()
    steering_failed_traits=()
    
    for trait in "${successful_traits[@]}"; do
        if process_trait_steering "$trait" "$model" "$model_name" "$gpu" "$judge_model" "$n_per_question" "$use_vllm"; then
            steering_successful_traits+=("$trait")
        else
            steering_failed_traits+=("$trait")
        fi
        echo ""
    done
else
    echo "⏭️  Skipping steering evaluation phase"
    steering_successful_traits=("${successful_traits[@]}")
    steering_failed_traits=()
fi

# Update optimal layers for new traits
if [ ${#successful_traits[@]} -gt 0 ]; then
    echo ""
    echo "🔍 Finding optimal layers for model: $model_name"
    echo "=================================================="
    
    python output/update_new_model.py "$model_name" --optimal-layers-only
    
    if [ $? -eq 0 ]; then
        echo "✅ Optimal layers updated successfully"
        echo "📁 Results saved to: inference_server/optimal_layers.json"
    else
        echo "⚠️  Warning: Failed to update optimal layers (pipeline will continue)"
    fi
    echo ""
fi

# Summary
echo "🎯 Comprehensive Pipeline Summary:"
echo "✅ Vector Generation Successful: ${successful_traits[*]}"
if [ ${#failed_traits[@]} -gt 0 ]; then
    echo "❌ Vector Generation Failed: ${failed_traits[*]}"
fi

if [ "$skip_steering" != "true" ]; then
    echo "✅ Steering Evaluation Successful: ${steering_successful_traits[*]}"
    if [ ${#steering_failed_traits[@]} -gt 0 ]; then
        echo "❌ Steering Evaluation Failed: ${steering_failed_traits[*]}"
    fi
fi

if [ ${#failed_traits[@]} -eq 0 ] && [ ${#steering_failed_traits[@]} -eq 0 ]; then
    echo "🎉 All phases completed successfully!"
    echo "📁 Vectors saved to: output/$model_name/persona_vectors/"
    echo "📁 Steering results saved to: output/$model_name/steering_eval/"
    echo "📁 Optimal layers updated in: inference_server/optimal_layers.json"
    exit 0
else
    echo "⚠️  Some phases had errors, but pipeline completed"
    exit 1
fi