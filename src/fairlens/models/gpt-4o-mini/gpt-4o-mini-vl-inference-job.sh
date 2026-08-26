#!/bin/bash
#SBATCH --job-name=gpt-4o-mini-vl-inference
#SBATCH --output=gpt-4o-mini-vl-inference%j.out
#SBATCH --error=gpt-4o-mini-vl-inference%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=7-00:00:00

cd "$(dirname "$0")"
module load python/3.10
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

python gpt-4o-mini-vl-inference.py --dataset cropped
