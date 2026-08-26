#!/bin/bash
# Submit one SLURM job per model × domain. Run from src/fairlens/eval/judge/

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

DOMAINS=(
    "healthcare"
    "legal"
    "hiring"
)

for MODEL in "${MODELS[@]}"; do
    for DOMAIN in "${DOMAINS[@]}"; do
        echo "Submitting: $MODEL | $DOMAIN"
        sbatch eval_job_deepeval.sh "$MODEL" "$DOMAIN"
    done
done

echo ""
echo "All jobs submitted. Monitor with: squeue -u \$USER"
echo "Once all done, run: python build_master.py"
