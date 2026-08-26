#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
# shellcheck source=model_results_paths.sh
source "$SCRIPT_DIR/model_results_paths.sh"

DP_SCRIPT="./calculate_dp.py"

for model in "${MODELS[@]}"; do
  if ! RESULTS_DIR="$(resolve_results_dir "$model")"; then
    echo "No results directory found for model \"$model\" (checked under $MODELS_DIR)."
    continue
  fi

  echo "Using results dir: $RESULTS_DIR"

  for domain in "${DOMAINS[@]}"; do
    FILE="$RESULTS_DIR/${domain}_results.json"
    if [[ -f "$FILE" ]]; then
      echo "Running DP for $model - $domain"
      python "$DP_SCRIPT" "$FILE"
      echo "-------------------------------------------"
    else
      echo "Skip $model - $domain (missing $FILE)"
    fi
  done
done
