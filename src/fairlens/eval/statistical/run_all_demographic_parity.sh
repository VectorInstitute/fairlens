#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
# shellcheck source=model_results_paths.sh
source "$SCRIPT_DIR/model_results_paths.sh"

EVAL_SCRIPT="./evaluate_demographic_parity.py"
EXTRA_ARGS=("$@")

for model in "${MODELS[@]}"; do
  if ! RESULTS_DIR="$(resolve_results_dir "$model")"; then
    echo "No results directory found for model \"$model\" (checked under $MODELS_DIR)."
    continue
  fi

  echo "Using results dir: $RESULTS_DIR"

  for domain in "${DOMAINS[@]}"; do
    FILE="$RESULTS_DIR/${domain}_results.json"
    if [[ -f "$FILE" ]]; then
      echo "Running demographic parity for $model - $domain"
      python "$EVAL_SCRIPT" "$FILE" --domain "$domain" "${EXTRA_ARGS[@]}"
      echo "-------------------------------------------"
    else
      echo "Skip $model - $domain (missing $FILE)"
    fi
  done
done
