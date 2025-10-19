#!/bin/bash

# Simple shell script to run the Julia overlap calculation

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JULIA_SCRIPT="$SCRIPT_DIR/src/main.jl"

# Default paths (modify as needed)
DEFAULT_CHECKPOINT="/Users/jihangzhu/Documents/netobs/DeepHall_local/data/seed_1758632847/tillicum/DeepHall_n8l21_optimal/ckpt_199999.npz"
DEFAULT_CONFIG="/Users/jihangzhu/Documents/netobs/DeepHall_local/data/seed_1758632847/tillicum/DeepHall_n8l21_optimal/config.yml"

# Parse command line arguments
CHECKPOINT_PATH="${1:-$DEFAULT_CHECKPOINT}"
CONFIG_PATH="${2:-$DEFAULT_CONFIG}"
STEPS="${3:-50}"
MODE="${4:-normal}"  # normal, simple, or test

echo "=== Julia Overlap Calculator Runner ==="
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Config: $CONFIG_PATH"
echo "Steps: $STEPS"
echo "Mode: $MODE"
echo ""

# Change to the project directory 
cd "$SCRIPT_DIR"

# Run based on mode
if [ "$MODE" = "test" ]; then
    echo "Running setup test..."
    julia src/main.jl test
elif [ "$MODE" = "simple" ]; then
    echo "Running in simple test mode..."
    julia --project=. src/main.jl --checkpoint "$CHECKPOINT_PATH" --config "$CONFIG_PATH" --steps "$STEPS" --simple --verbose
else
    echo "Running full overlap calculation..."
    julia --project=. src/main.jl --checkpoint "$CHECKPOINT_PATH" --config "$CONFIG_PATH" --steps "$STEPS" --parallel --verbose
fi