#!/bin/bash
#SBATCH --job-name=eval-deepeval
#SBATCH --output=logs/eval-deepeval-%j.out
#SBATCH --error=logs/eval-deepeval-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00

cd "$(dirname "$0")"
mkdir -p logs

module load python/3.10
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

MODEL=$1
DOMAIN=$2

if [ -z "$MODEL" ]; then
    echo "ERROR: No model name provided."
    echo "Usage: sbatch eval_job_deepeval.sh <model_name> <domain>"
    exit 1
fi

if [ -z "$DOMAIN" ]; then
    echo "ERROR: No domain provided."
    echo "Usage: sbatch eval_job_deepeval.sh <model_name> <domain>"
    exit 1
fi

echo "====================================="
echo "Starting DeepEval eval: $MODEL | $DOMAIN"
echo "====================================="

if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

export DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE=1200
export DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE=300
export DEEPEVAL_TASK_GATHER_BUFFER_SECONDS_OVERRIDE=30
export DEEPEVAL_RETRY_MAX_ATTEMPTS=5
export DEEPEVAL_RETRY_INITIAL_SECONDS=5
export DEEPEVAL_RETRY_EXP_BASE=2
export DEEPEVAL_RETRY_JITTER=1
export DEEPEVAL_RETRY_CAP_SECONDS=60
export DEEPEVAL_SDK_RETRY_PROVIDERS='["*"]'
export DEEPEVAL_DISABLE_DOTENV=1

python eval_pipeline_deepeval.py --model "$MODEL" --domain "$DOMAIN"

echo "====================================="
echo "Done: $MODEL | $DOMAIN"
echo "====================================="
