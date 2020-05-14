#!/bin/sh
#$ -N dwave_monte_carlo
#$ -cwd
#$ -m ea
#$ -M sumnerh@bu.edu
#$ -l h_rt=72:00:00

SHARD=$((SGE_TASK_ID - 1))
OUTPUT_DIR=$1

START=$(date +%s)

export RAYON_NUM_THREADS=$NSLOTS
echo "$HOME/.virtualenvs/dwave/bin/python main_distributed.py --shard=$SHARD --base_dir=$OUTPUT_DIR"
$HOME/.virtualenvs/dwave/bin/python main_distributed.py --shard=$SHARD --base_dir="$OUTPUT_DIR"

END=$(date +%s)
ELAPSED=$((END-START))
echo "$ELAPSED" > "$OUTPUT_DIR/experiment_$SHARD/elapsed_seconds.txt"
