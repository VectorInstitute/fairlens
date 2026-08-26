#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
# shellcheck source=model_results_paths.sh
source "$SCRIPT_DIR/model_results_paths.sh"

SOUNDNESS_SCRIPT="./calculate_soundness.py"

for model in "${MODELS[@]}"; do
  if ! RESULTS_DIR="$(resolve_results_dir "$model")"; then
    echo "No results directory found for model \"$model\" (checked under $MODELS_DIR)."
    continue
  fi

  echo "Using results dir: $RESULTS_DIR"

  for domain in "${DOMAINS[@]}"; do
    FILE="$RESULTS_DIR/${domain}_results.json"
    if [[ ! -f "$FILE" ]]; then
      echo "Skip $model - $domain (missing $FILE)"
      continue
    fi
    if ! idx="$(soundness_domain_index "$domain")"; then
      echo "Skip $model - $domain (unknown domain for soundness index)"
      continue
    fi
    echo "Running soundness for $model - $domain (index $idx)"
    python "$SOUNDNESS_SCRIPT" "$FILE" "$idx"
    echo "-------------------------------------------"
  done
done
