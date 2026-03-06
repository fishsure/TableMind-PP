#!/bin/bash
# ============================================================
# TableMind++ Inference Script
# Uncertainty-Aware Inference Pipeline
#
# Prerequisites:
#   1. Training completed: run_train.sh
#   2. vLLM server running with the trained model
#   3. Memory bank built: scripts/build_memory.py
#   4. Sandbox Fusion running (for code execution)
# ============================================================

# ---- User-configurable variables ---------------------------
MODEL_PATH=""            # Path to the trained TableMind (RFT) checkpoint
MEMORY_BANK="memory_bank.pkl"
TRAIN_DATA="data/train.parquet"
TEST_DATA="data/test.parquet"
DATASET="WTQ"            # WTQ | TabMWP | TabFact | HiTab | FinQA
OUTPUT_PATH="results.json"

VLLM_HOST="localhost"
VLLM_PORT=8000
API_BASE="http://${VLLM_HOST}:${VLLM_PORT}/v1"

# TableMind++ inference hyperparameters (paper defaults)
NUM_CANDIDATES=16        # N — number of candidate plans
TOP_K_MEMORY=5           # K — memory prototype retrieval size
RETENTION_RATIO=0.5      # rho — plan pruning retention fraction
CONF_THRESHOLD=0.8       # tau — confidence threshold for action refinement
MAX_TURNS=3              # maximum plan-action-reflect turns
TEMPERATURE=1.0          # sampling temperature

# ============================================================
# Step 0: Validate inputs
# ============================================================
if [ -z "$MODEL_PATH" ]; then
    echo "[ERROR] MODEL_PATH is not set. Please edit run_inference.sh."
    exit 1
fi

# ============================================================
# Step 1: Serve the trained model with vLLM
# ============================================================
echo "Step 1: Starting vLLM server..."
echo "  Model: $MODEL_PATH"
echo "  Endpoint: $API_BASE"

# Start vLLM in a background tmux session (requires tmux)
if command -v tmux &> /dev/null; then
    tmux new-session -d -s vllm_server \
        "python -m vllm.entrypoints.openai.api_server \
            --model $MODEL_PATH \
            --served-model-name tablemind \
            --host $VLLM_HOST \
            --port $VLLM_PORT \
            --tensor-parallel-size 1 \
            --gpu-memory-utilization 0.8 \
            --max-model-len 32768"
    echo "  vLLM server started in tmux session 'vllm_server'."
    echo "  Waiting 30 seconds for server to be ready..."
    sleep 30
else
    echo "[WARNING] tmux not found. Please start vLLM manually:"
    echo "  python -m vllm.entrypoints.openai.api_server \\"
    echo "    --model $MODEL_PATH --served-model-name tablemind \\"
    echo "    --host $VLLM_HOST --port $VLLM_PORT"
    echo "Then re-run this script."
    exit 1
fi

# ============================================================
# Step 2: Build the dual-memory bank (offline)
# ============================================================
if [ ! -f "$MEMORY_BANK" ]; then
    echo ""
    echo "Step 2: Building dual-memory bank from training data..."
    python scripts/build_memory.py \
        --model-path "tablemind" \
        --train-data "$TRAIN_DATA" \
        --output "$MEMORY_BANK" \
        --api-base "$API_BASE" \
        --encoder "BAAI/bge-m3" \
        --temperature 0.0
    echo "  Memory bank saved to $MEMORY_BANK"
else
    echo "Step 2: Memory bank already exists at $MEMORY_BANK — skipping."
fi

# ============================================================
# Step 3: Run TableMind++ evaluation
# ============================================================
echo ""
echo "Step 3: Running TableMind++ inference on $DATASET..."
python scripts/evaluate.py \
    --data-path "$TEST_DATA" \
    --memory-bank "$MEMORY_BANK" \
    --dataset "$DATASET" \
    --api-base "$API_BASE" \
    --model "tablemind" \
    --num-candidates "$NUM_CANDIDATES" \
    --top-k-memory "$TOP_K_MEMORY" \
    --retention-ratio "$RETENTION_RATIO" \
    --confidence-threshold "$CONF_THRESHOLD" \
    --max-turns "$MAX_TURNS" \
    --temperature "$TEMPERATURE" \
    --output-path "$OUTPUT_PATH"

echo ""
echo "Done. Results saved to $OUTPUT_PATH"
