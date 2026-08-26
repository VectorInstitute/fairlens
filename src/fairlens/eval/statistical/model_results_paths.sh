_EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd "$_EVAL_DIR/../.." && pwd)"
MODELS_DIR="$PACKAGE_DIR/models"

MODELS=(
  cogvlm gpt-4o-mini gpt-5.2-reasoning internVL3 llama3_2_vision llava1_6
  minicpm-2-6 ovis2.5 qwen2_5_VL qwen3_vl
)

# JSON basename prefixes (files: ${domain}_results.json)
DOMAINS=(hiring healthcare legal)

# calculate_soundness.py domain_index: 0=hiring, 1=legal, 2=healthcare
soundness_domain_index() {
  case "$1" in
    hiring) printf '%s\n' 0 ;;
    legal) printf '%s\n' 1 ;;
    healthcare) printf '%s\n' 2 ;;
    *) return 1 ;;
  esac
}

resolve_results_dir() {
  local model="$1"
  local d

  if [[ -d "$MODELS_DIR/$model" ]]; then
    d="$(find "$MODELS_DIR/$model" -maxdepth 3 -type d -name 'results_cropped*' 2>/dev/null | LC_ALL=C sort | head -n1)"
    if [[ -n "$d" ]]; then
      printf '%s\n' "$d"
      return 0
    fi
  fi

  return 1
}
