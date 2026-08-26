#!/bin/bash
#SBATCH --job-name=minicpm-2-6-vl-inference
#SBATCH --output=minicpm-2-6-vl-inference%j.out
#SBATCH --error=minicpm-2-6-vl-inference%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --time=72:00:00

cd "$(dirname "$0")"
module load python/3.10
module load cuda-12.1
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

python minicpm-2-6-vl-inference.py --dataset cropped
