#!/bin/bash
# run_ablation.sh
# CS 5365 – Deep Learning | University of Texas at El Paso
# Group Members: Veronica Aragon, Angela Martinez
#
# Runs the full LoRA rank ablation sweep across all ranks and both tasks.
# Results are saved to results/<task>_rank<r>/results.json
#
# Usage:
#   bash code/run_ablation.sh
set -e
RANKS=(1 2 4 8 16 64)
TASKS=("sst2" "mrpc")
echo "============================================"
echo "  LoRA Rank Ablation Sweep"
echo "  Tasks: ${TASKS[*]}"
echo "  Ranks: ${RANKS[*]}"
echo "============================================"
echo ""
for TASK in "${TASKS[@]}"; do
    for RANK in "${RANKS[@]}"; do
        echo "--------------------------------------------"
        echo "  Running: task=$TASK | rank=$RANK"
        echo "--------------------------------------------"
        python code/train_lora.py --task "$TASK" --rank "$RANK"
        echo ""
    done
done
echo "============================================"
echo "  All runs complete. Results saved to results/"
echo "============================================"
python code/plot_results.py
