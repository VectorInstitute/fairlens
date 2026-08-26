#!/bin/bash
# Run metrics_model.py for every model in the FairLens suite (same list as launch_all.sh).
# Lightweight local job — no SLURM (aggregates existing domain CSVs only).
#
# Usage (from anywhere):
#   bash run_all_metrics.sh
#
# Optional: pass a different metrics config as first argument (default: metrics_config.yaml in this dir).

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG="${1:-metrics_config.yaml}"

# Keep in sync with launch_all.sh
MODELS=(
    "CogVLM2-llama3-chat-19B"
    "gpt-4o-mini"
    "gpt-5.2"
    "InternVL3-9B"
    "Llama-3.2-11B-Vision-Instruct"
    "llava-v1.6-vicuna-7b-hf"
    "MiniCPM-V-2_6"
    "Ovis2.5-9B"
    "Qwen2.5-VL-7B-Instruct"
    "Qwen3-VL-8B-Instruct"
)

if [[ -f ".venv/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source .venv/bin/activate
fi

if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: config not found: $SCRIPT_DIR/$CONFIG"
    exit 1
fi

echo "Using config: $CONFIG"
echo "Models: ${#MODELS[@]}"
echo ""

failed=()
for MODEL in "${MODELS[@]}"; do
    echo "====================================="
    echo "Metrics: $MODEL"
    echo "====================================="
    if python metrics_model.py --model "$MODEL" --config "$CONFIG"; then
        echo ""
    else
        echo "[FAIL] $MODEL"
        failed+=("$MODEL")
        echo ""
    fi
done

echo "====================================="
echo "Done. Success: $((${#MODELS[@]} - ${#failed[@]})) / ${#MODELS[@]}"
if ((${#failed[@]} > 0)); then
    echo "Failed models:"
    for m in "${failed[@]}"; do
        echo "  - $m"
    done
    exit 1
fi
