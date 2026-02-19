#!/bin/bash
#
# Hyperparameter Search for Figure 9 - Manifold Steering
#
# Hyperparameters:
#   1. d (low rank): 4, 8, 16, 32, 64, 128, 256, full
#      - "full" = original steering vectors WITHOUT manifold projection
#   2. alpha (trait steering): 0.0, 1.0, 1.5, 2.0, 2.5
#
# Baseline: original model with NO steering (alpha=0, no vectors)
# Total combinations: 8 x 5 = 40 (+ 1 baseline = 41)
#
# This script runs steps 1, 2, 4, 5 for each configuration.
# For "full", steps 1 & 2 are replaced with copying original vectors.
# Safetensor files are deleted after evaluation to prevent OOM issues.
#
# Evaluation metrics (measured against baseline):
#   1. Coherence
#   2. Trait Δ
#

set -e  # Exit on error

# =============================================================================
# Load Environment Variables (for HF_TOKEN)
# =============================================================================
SCRIPT_DIR_TEMP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT_TEMP="$(cd "${SCRIPT_DIR_TEMP}/../../.." && pwd)"
if [ -f "${PROJECT_ROOT_TEMP}/.env" ]; then
    # Export only valid KEY=VALUE lines (skip comments and empty lines)
    export $(grep -E '^[A-Za-z_][A-Za-z0-9_]*=' "${PROJECT_ROOT_TEMP}/.env" | xargs)
    echo "Loaded .env from ${PROJECT_ROOT_TEMP}/.env"
fi

# =============================================================================
# Configuration
# =============================================================================

# Hyperparameter values
# Note: "full" = original steering vectors WITHOUT manifold projection
# This serves as comparison to see manifold projection benefit
RANKS=(4 8 16 32 64 128 256 full)
ALPHAS=(0.0 1.0 1.5 2.0 2.5)

# Model configuration
MODEL_NAME="Llama-3.1-8B-Instruct"
MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"
JUDGE_MODEL="gpt-4.1-mini"
N_PER_QUESTION=3
NUM_PROMPTS=500

# Traits to evaluate
TRAITS=(
    "inadequate_crisis_response"
    "minimizing_emotional_distress"
    "normalizing_avoidance_and_escapism"
    "normalizing_finality_in_crisis"
    "over_emotional_validation_without_guidance"
    "reinforcing_paranoia_and_distrust"
    "over_encouragement_of_risky_actions"
    "romanticizing_loss_and_attachment"
)

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
MANIFOLD_DIR="${SCRIPT_DIR}/../manifold"

# Model-specific output directories
# All outputs go under: figure_outputs/figure9/hyperparameter_search/{MODEL_NAME}/
OUTPUT_BASE="${PROJECT_ROOT}/research_scripts/figure_outputs/figure9/hyperparameter_search/${MODEL_NAME}"
MANIFOLD_OUTPUT_BASE="${OUTPUT_BASE}/manifold/output"
SUMMARY_FILE="${OUTPUT_BASE}/hyperparameter_search_summary.csv"
LOG_FILE="${OUTPUT_BASE}/hyperparameter_search.log"
TEST_OUTPUTS_FILE="${OUTPUT_BASE}/test_outputs.json"

# =============================================================================
# Helper Functions
# =============================================================================

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "$msg"
    echo "$msg" >> "$LOG_FILE"
}

error_exit() {
    log "ERROR: $1"
    exit 1
}

get_project_name() {
    local rank=$1
    echo "hyperparam_search_d${rank}"
}

get_output_name() {
    local alpha=$1
    # Replace . with _ for filename compatibility
    local alpha_safe=$(echo "$alpha" | sed 's/\./_/g')
    echo "alpha_${alpha_safe}"
}

# Build alpha arguments for all traits with the same alpha value
build_alpha_args() {
    local alpha=$1
    local args=""
    for trait in "${TRAITS[@]}"; do
        args="${args} ${trait}:${alpha}"
    done
    echo "$args"
}

# Initialize summary CSV with headers (only if doesn't exist or is empty)
init_summary_csv() {
    if [ ! -f "$SUMMARY_FILE" ] || [ ! -s "$SUMMARY_FILE" ]; then
        echo "rank,alpha,is_baseline,mean_coherence,mean_coherence_delta,mean_trait_score,mean_trait_delta,status" > "$SUMMARY_FILE"
        log "Initialized summary CSV: $SUMMARY_FILE"
    else
        log "Summary CSV already exists, will append new results: $SUMMARY_FILE"
    fi
}

# Check if a configuration already exists in the summary CSV
config_exists_in_summary() {
    local rank=$1
    local alpha=$2
    
    if [ ! -f "$SUMMARY_FILE" ]; then
        return 1  # File doesn't exist, config doesn't exist
    fi
    
    # Check if this rank,alpha combination already exists in the CSV
    if grep -q "^${rank},${alpha}," "$SUMMARY_FILE" 2>/dev/null; then
        return 0  # Config exists
    fi
    
    return 1  # Config doesn't exist
}

# Check if baseline evaluation already exists
baseline_evaluation_exists() {
    local baseline_dir="${OUTPUT_BASE}/baseline/evaluation"
    
    # Check for baseline summary JSON files
    if ls "${baseline_dir}"/baseline_summary_*.json 1>/dev/null 2>&1; then
        return 0  # Exists
    fi
    
    return 1  # Doesn't exist
}

# Check if evaluation results exist for a given model directory
evaluation_exists() {
    local model_dir=$1
    local eval_dir="${model_dir}/evaluation"
    
    # Check for steered summary JSON files
    if ls "${eval_dir}"/steered_summary_*.json 1>/dev/null 2>&1; then
        return 0  # Exists
    fi
    
    return 1  # Doesn't exist
}

# Check if manifold project already exists (has projected vectors)
manifold_project_exists() {
    local project_name=$1
    local project_dir="${MANIFOLD_OUTPUT_BASE}/${project_name}"
    local projected_vectors_dir="${project_dir}/projected_vectors"
    
    # Check if projected vectors directory exists and has files
    if [ -d "$projected_vectors_dir" ] && [ "$(ls -A "$projected_vectors_dir" 2>/dev/null)" ]; then
        return 0  # Exists
    fi
    
    return 1  # Doesn't exist
}

# Append result to summary CSV
append_summary() {
    local rank=$1
    local alpha=$2
    local is_baseline=$3
    local mean_coherence=$4
    local mean_coh_delta=$5
    local mean_trait=$6
    local mean_trait_delta=$7
    local status=$8
    
    echo "${rank},${alpha},${is_baseline},${mean_coherence},${mean_coh_delta},${mean_trait},${mean_trait_delta},${status}" >> "$SUMMARY_FILE"
}

# Initialize test outputs JSON file
init_test_outputs_json() {
    if [ ! -f "$TEST_OUTPUTS_FILE" ]; then
        echo '{"test_outputs": []}' > "$TEST_OUTPUTS_FILE"
        log "Initialized test outputs JSON: $TEST_OUTPUTS_FILE"
    else
        log "Test outputs JSON already exists: $TEST_OUTPUTS_FILE"
    fi
}

# Append test output to JSON file
append_test_output() {
    local rank=$1
    local alpha=$2
    local model_dir=$3
    
    local test_output_file="${model_dir}/test_output.json"
    
    if [ ! -f "$test_output_file" ]; then
        log "No test output found for rank=$rank, alpha=$alpha"
        return
    fi
    
    # Use Python to append to JSON array
    python3 << EOF
import json
from pathlib import Path

test_outputs_file = Path("$TEST_OUTPUTS_FILE")
test_output_file = Path("$test_output_file")

# Load existing test outputs
with open(test_outputs_file, 'r') as f:
    data = json.load(f)

# Load new test output
with open(test_output_file, 'r') as f:
    new_output = json.load(f)

# Add rank and alpha to the output
# Handle both numeric and string ranks (e.g., "full")
rank_str = "$rank"
rank_val = int(rank_str) if rank_str.isdigit() else rank_str
new_output['rank'] = rank_val
new_output['alpha'] = $alpha

# Check if this config already exists
exists = False
for i, existing in enumerate(data['test_outputs']):
    if existing.get('rank') == rank_val and existing.get('alpha') == $alpha:
        # Update existing entry
        data['test_outputs'][i] = new_output
        exists = True
        break

if not exists:
    data['test_outputs'].append(new_output)

# Save updated data
with open(test_outputs_file, 'w') as f:
    json.dump(data, f, indent=2)

print(f"Added test output for rank=$rank, alpha=$alpha")
EOF
    
    log "Saved test output for rank=$rank, alpha=$alpha"
}

# Delete safetensor files to save memory
cleanup_safetensors() {
    local model_dir=$1
    log "Cleaning up safetensor files in: $model_dir"
    
    if [ -d "$model_dir" ]; then
        find "$model_dir" -name "*.safetensors" -type f -delete
        find "$model_dir" -name "model.safetensors.index.json" -type f -delete
        log "Safetensor files deleted"
    fi
    
    # Also clear CUDA cache
    python3 -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None" 2>/dev/null || true
    log "CUDA cache cleared"
}

# Extract metrics from evaluation results
extract_metrics() {
    local eval_dir=$1
    local baseline_json=$2
    
    cd "$SCRIPT_DIR"
    
    local baseline_arg=""
    if [ -n "$baseline_json" ] && [ -f "$baseline_json" ]; then
        baseline_arg="--baseline-json $baseline_json"
    fi
    
    python3 aggregate_results.py \
        --eval-dir "$eval_dir" \
        $baseline_arg \
        --output-format csv
}

# =============================================================================
# Step Functions
# =============================================================================

# Step 1: Learn manifolds for a given rank
run_step1_learn_manifold() {
    local project_name=$1
    local rank=$2
    
    log "Step 1: Learning manifold (project=$project_name, rank=$rank)"
    
    cd "$MANIFOLD_DIR"
    
    python learn_manifold.py \
        --project-name "$project_name" \
        --model-name "$MODEL_NAME" \
        --model-path "$MODEL_PATH" \
        --traits ${TRAITS[@]} \
        --rank "$rank" \
        --num-prompts "$NUM_PROMPTS" \
        2>&1 | tee -a "$LOG_FILE"
    
    local status=$?
    if [ $status -ne 0 ]; then
        log "Step 1 failed for project=$project_name"
        return 1
    fi
    
    log "Step 1 completed for project=$project_name"
    return 0
}

# Step 2: Project vectors
run_step2_project_vectors() {
    local project_name=$1
    
    log "Step 2: Projecting vectors (project=$project_name)"
    
    cd "$MANIFOLD_DIR"
    
    python project_vectors.py \
        --project-name "$project_name" \
        2>&1 | tee -a "$LOG_FILE"
    
    local status=$?
    if [ $status -ne 0 ]; then
        log "Step 2 failed for project=$project_name"
        return 1
    fi
    
    log "Step 2 completed for project=$project_name"
    return 0
}

# Step 4: Bake model weights
run_step4_bake_model() {
    local project_name=$1
    local output_name=$2
    local alpha=$3
    
    log "Step 4: Baking model (project=$project_name, output=$output_name, alpha=$alpha)"
    
    cd "$MANIFOLD_DIR"
    
    local alpha_args=$(build_alpha_args "$alpha")
    
    python bake_model.py \
        --project-name "$project_name" \
        --output-name "$output_name" \
        --alphas $alpha_args \
        2>&1 | tee -a "$LOG_FILE"
    
    local status=$?
    if [ $status -ne 0 ]; then
        log "Step 4 failed for project=$project_name, output=$output_name"
        return 1
    fi
    
    log "Step 4 completed for project=$project_name, output=$output_name"
    return 0
}

# Step 5: Evaluate model (without baseline comparison during evaluation)
run_step5_evaluate() {
    local model_path=$1
    
    log "Step 5: Evaluating model (path=$model_path)"
    
    cd "$MANIFOLD_DIR"
    
    # Note: NOT using --include-baseline as per requirements
    python evaluate_model.py \
        --model-path "$model_path" \
        --judge-model "$JUDGE_MODEL" \
        --n-per-question "$N_PER_QUESTION" \
        2>&1 | tee -a "$LOG_FILE"
    
    local status=$?
    if [ $status -ne 0 ]; then
        log "Step 5 failed for model=$model_path"
        return 1
    fi
    
    log "Step 5 completed for model=$model_path"
    return 0
}

# Setup "full" vectors (original steering vectors without manifold projection)
setup_full_vectors() {
    local project_name=$1
    
    log "Setting up full vectors project (no manifold projection): $project_name"
    
    cd "$SCRIPT_DIR"
    
    python3 setup_full_vectors.py \
        --project-name "$project_name" \
        --model-name "$MODEL_NAME" \
        --model-path "$MODEL_PATH" \
        2>&1 | tee -a "$LOG_FILE"
    
    local status=$?
    if [ $status -ne 0 ]; then
        log "Failed to setup full vectors for project=$project_name"
        return 1
    fi
    
    log "Full vectors setup completed for project=$project_name"
    return 0
}

# Run baseline evaluation (original model without steering)
run_baseline_evaluation() {
    local baseline_dir="${OUTPUT_BASE}/baseline"
    
    # Check if baseline evaluation already exists
    if baseline_evaluation_exists; then
        log "Baseline evaluation already exists, skipping..."
        return 0
    fi
    
    log "Running baseline evaluation (original model, no steering)..."
    
    mkdir -p "$baseline_dir/evaluation"
    
    cd "$SCRIPT_DIR"
    
    python3 evaluate_baseline.py \
        --model-path "$MODEL_PATH" \
        --judge-model "$JUDGE_MODEL" \
        --n-per-question "$N_PER_QUESTION" \
        --output-dir "$baseline_dir" \
        2>&1 | tee -a "$LOG_FILE"
    
    local status=$?
    if [ $status -ne 0 ]; then
        log "Baseline evaluation failed"
        return 1
    fi
    
    log "Baseline evaluation completed"
    return 0
}

# =============================================================================
# Main Execution
# =============================================================================

main() {
    log "============================================================"
    log "HYPERPARAMETER SEARCH FOR FIGURE 9"
    log "============================================================"
    log "Ranks: ${RANKS[*]}"
    log "Alphas: ${ALPHAS[*]}"
    log "Total configurations: $((${#RANKS[@]} * ${#ALPHAS[@]}))"
    log "Output directory: $OUTPUT_BASE"
    log "============================================================"
    
    # Create output directory
    mkdir -p "$OUTPUT_BASE"
    
    # Initialize summary CSV
    init_summary_csv
    
    # Initialize test outputs JSON
    init_test_outputs_json
    
    # Step 0: Run baseline evaluation first
    log ""
    log "========== STEP 0: BASELINE EVALUATION =========="
    run_baseline_evaluation
    
    # Find baseline summary for comparison
    BASELINE_SUMMARY=$(ls -t "${OUTPUT_BASE}/baseline/evaluation"/baseline_summary_*.json 2>/dev/null | head -n1)
    
    if [ -z "$BASELINE_SUMMARY" ]; then
        log "WARNING: Baseline summary not found, deltas will be 0"
        BASELINE_SUMMARY=""
    else
        log "Using baseline summary: $BASELINE_SUMMARY"
        
        # Add baseline to summary if not already there
        if ! config_exists_in_summary "baseline" "0.0"; then
            # Extract baseline metrics and add to summary
            baseline_metrics=$(extract_metrics "${OUTPUT_BASE}/baseline/evaluation" "")
            IFS=',' read -r base_coh base_coh_delta base_trait base_trait_delta <<< "$baseline_metrics"
            append_summary "baseline" "0.0" "true" "$base_coh" "0.00" "$base_trait" "0.00" "success"
            log "Added baseline to summary"
        else
            log "Baseline already in summary"
        fi
    fi
    
    # Configuration counter
    config_num=0
    total_configs=$((${#RANKS[@]} * ${#ALPHAS[@]}))
    
    # Main loop: iterate through ranks and alphas
    for rank in "${RANKS[@]}"; do
        log ""
        log "=========================================="
        log "RANK: $rank"
        log "=========================================="
        
        project_name=$(get_project_name "$rank")
        
        # Special handling for "full" rank (original vectors, no manifold projection)
        if [ "$rank" == "full" ]; then
            # Check if full vectors project already exists
            if manifold_project_exists "$project_name"; then
                log "Full vectors project '$project_name' already exists, skipping setup..."
            else
                # Setup full vectors (copies original steering vectors)
                if ! setup_full_vectors "$project_name"; then
                    log "Failed to setup full vectors for rank=$rank, skipping..."
                    for alpha in "${ALPHAS[@]}"; do
                        config_num=$((config_num + 1))
                        if ! config_exists_in_summary "$rank" "$alpha"; then
                            append_summary "$rank" "$alpha" "false" "0.00" "0.00" "0.00" "0.00" "failed_setup_full"
                        fi
                    done
                    continue
                fi
            fi
        else
            # Standard manifold projection for numeric ranks
            # Check if manifold project already exists (steps 1 & 2 completed)
            if manifold_project_exists "$project_name"; then
                log "Manifold project '$project_name' already exists, skipping steps 1 & 2..."
            else
                # Step 1: Learn manifold
                if ! run_step1_learn_manifold "$project_name" "$rank"; then
                    log "Failed to learn manifold for rank=$rank, skipping..."
                    for alpha in "${ALPHAS[@]}"; do
                        config_num=$((config_num + 1))
                        if ! config_exists_in_summary "$rank" "$alpha"; then
                            append_summary "$rank" "$alpha" "false" "0.00" "0.00" "0.00" "0.00" "failed_step1"
                        fi
                    done
                    continue
                fi
                
                # Step 2: Project vectors
                if ! run_step2_project_vectors "$project_name"; then
                    log "Failed to project vectors for rank=$rank, skipping..."
                    for alpha in "${ALPHAS[@]}"; do
                        config_num=$((config_num + 1))
                        if ! config_exists_in_summary "$rank" "$alpha"; then
                            append_summary "$rank" "$alpha" "false" "0.00" "0.00" "0.00" "0.00" "failed_step2"
                        fi
                    done
                    continue
                fi
            fi
        fi
        
        # Iterate through alphas
        for alpha in "${ALPHAS[@]}"; do
            config_num=$((config_num + 1))
            
            log ""
            log "---------- Configuration $config_num/$total_configs ----------"
            log "Rank: $rank, Alpha: $alpha"
            
            output_name=$(get_output_name "$alpha")
            model_dir="${MANIFOLD_OUTPUT_BASE}/${project_name}/baked_models/${output_name}"
            eval_dir="${model_dir}/evaluation"
            
            # Check if this configuration already has evaluation results
            if evaluation_exists "$model_dir"; then
                log "Evaluation already exists for rank=$rank, alpha=$alpha, skipping..."
                
                # Still extract metrics and add to summary if not already there
                if ! config_exists_in_summary "$rank" "$alpha"; then
                    metrics=$(extract_metrics "$eval_dir" "$BASELINE_SUMMARY")
                    if [[ "$metrics" != ERROR* ]]; then
                        IFS=',' read -r mean_coh coh_delta mean_trait trait_delta <<< "$metrics"
                        append_summary "$rank" "$alpha" "false" "$mean_coh" "$coh_delta" "$mean_trait" "$trait_delta" "success"
                        log "Added existing results to summary: Coherence=$mean_coh (Δ=$coh_delta), Trait=$mean_trait (Δ=$trait_delta)"
                    fi
                fi
                
                # Also extract test output if not already saved
                append_test_output "$rank" "$alpha" "$model_dir"
                continue
            fi
            
            # Check if already in summary (but evaluation files might be missing)
            if config_exists_in_summary "$rank" "$alpha"; then
                log "Configuration rank=$rank, alpha=$alpha already in summary, skipping..."
                continue
            fi
            
            # Step 4: Bake model
            if ! run_step4_bake_model "$project_name" "$output_name" "$alpha"; then
                log "Failed to bake model for rank=$rank, alpha=$alpha"
                append_summary "$rank" "$alpha" "false" "0.00" "0.00" "0.00" "0.00" "failed_step4"
                continue
            fi
            
            # Save test output from step 4
            append_test_output "$rank" "$alpha" "$model_dir"
            
            # Step 5: Evaluate model
            if ! run_step5_evaluate "${project_name}/baked_models/${output_name}"; then
                log "Failed to evaluate model for rank=$rank, alpha=$alpha"
                append_summary "$rank" "$alpha" "false" "0.00" "0.00" "0.00" "0.00" "failed_step5"
                cleanup_safetensors "$model_dir"
                continue
            fi
            
            # Extract metrics and compare with baseline
            metrics=$(extract_metrics "$eval_dir" "$BASELINE_SUMMARY")
            
            if [[ "$metrics" == ERROR* ]]; then
                log "Failed to extract metrics for rank=$rank, alpha=$alpha"
                append_summary "$rank" "$alpha" "false" "0.00" "0.00" "0.00" "0.00" "failed_metrics"
            else
                IFS=',' read -r mean_coh coh_delta mean_trait trait_delta <<< "$metrics"
                append_summary "$rank" "$alpha" "false" "$mean_coh" "$coh_delta" "$mean_trait" "$trait_delta" "success"
                log "Results: Coherence=$mean_coh (Δ=$coh_delta), Trait=$mean_trait (Δ=$trait_delta)"
            fi
            
            # Cleanup safetensor files to prevent OOM
            cleanup_safetensors "$model_dir"
            
            log "Configuration $config_num/$total_configs completed"
        done
    done
    
    log ""
    log "============================================================"
    log "HYPERPARAMETER SEARCH COMPLETE"
    log "============================================================"
    log "Summary saved to: $SUMMARY_FILE"
    
    # Generate final report
    generate_final_report
}

# Generate final summary report
generate_final_report() {
    local report_file="${OUTPUT_BASE}/hyperparameter_search_report.txt"
    
    log "Generating final report..."
    
    cd "$SCRIPT_DIR"
    
    python3 generate_report.py \
        --summary-csv "$SUMMARY_FILE" \
        --output-file "$report_file" \
        --test-outputs-json "$TEST_OUTPUTS_FILE"
    
    cat "$report_file" | tee -a "$LOG_FILE"
    
    log "Report saved to: $report_file"
}

# =============================================================================
# Entry Point
# =============================================================================

# Check if required tools are available
command -v python3 >/dev/null 2>&1 || error_exit "python3 is required"

# Run main
main "$@"

