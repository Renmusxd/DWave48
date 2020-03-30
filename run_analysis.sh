#!/bin/sh
#$ -N dwave_monte_carlo
#$ -cwd
#$ -m ea
#$ -M sumnerh@bu.edu
#$ -l h_rt=24:00:00

OUTPUT_DIR=$1

echo "$HOME/.virtualenvs/dwave/bin/python main_distributed.py --analyze --base_dir=$OUTPUT_DIR"
$HOME/.virtualenvs/dwave/bin/python main_distributed.py --analyze --base_dir=$OUTPUT_DIR
