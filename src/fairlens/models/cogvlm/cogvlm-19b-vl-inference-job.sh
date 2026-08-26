#!/bin/bash
#SBATCH --job-name=cogvlm-19b-vl-inference
#SBATCH --output=cogvlm-19b-vl-inference%j.out
#SBATCH --error=cogvlm-19b-vl-inference%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00

cd "$(dirname "$0")"
module load python/3.10
module load cuda-12.1
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

python cogvlm-19b-vl-inference.py --dataset cropped
