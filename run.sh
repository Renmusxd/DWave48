#!/bin/sh
#$ -N dwave_monte_carlo
#$ -cwd
#$ -m ea
#$ -M sumnerh@bu.edu
#$ -l h_rt=144:00:00

SHARD=$((SGE_TASK_ID - 1))
OUTPUT_DIR=$1
NSHARDS=$2

START=$(date +%s)

export RAYON_NUM_THREADS=$NSLOTS
echo "$HOME/.virtualenvs/dwave/bin/python main_distributed.py --shards=$SHARD --nshards=$NSHARDS --base_dir=\"$OUTPUT_DIR\" --run"
$HOME/.virtualenvs/dwave/bin/python main_distributed.py --shards=$SHARD --nshards=$NSHARDS --base_dir="$OUTPUT_DIR" --run

END=$(date +%s)
ELAPSED=$((END-START))
echo "$ELAPSED" > "$OUTPUT_DIR/experiment_$SHARD/elapsed_seconds.txt"
