#!/bin/sh

BASE_DIR=$(readlink -f $1)
if [ -z "$BASE_DIR" ]; then
	echo "Please supply a directory";
else

	NUM_SHARDS=${2:-10}
	NUM_SLOTS=${3:-1}
	NUM_ANALYZE_SLOTS=${4:-10}
	echo "Running into directory $BASE_DIR with $NUM_SHARDS shards"


	mkdir -p "$BASE_DIR"
	mkdir -p "$BASE_DIR/code"
	cp *.py "$BASE_DIR/code"
	cp -r "bathroom_tile" "$BASE_DIR/code"
	cp *.sh "$BASE_DIR/code"
	# Speeds things up sometimes
	cp -r graphcache "$BASE_DIR/code"
	cd "$BASE_DIR/code" || exit
	echo "Working directory: $(pwd)"
	echo "qsub -pe omp \"NUM_SLOTS\" -wd \"$BASE_DIR/code\" -t 1-$NUM_SHARDS run.sh \"$BASE_DIR\" \"$NUM_SHARDS\""
	OUTPUT=$(qsub -pe omp "NUM_SLOTS" -wd "$BASE_DIR/code" -t 1-"$NUM_SHARDS" run.sh "$BASE_DIR" "$NUM_SHARDS")
	echo "$OUTPUT"
	JOB_ID=$(echo "$OUTPUT" | grep -E -o "[0-9]+" | head -1)
	echo "qsub -pe omp \"$NUM_ANALYZE_SLOTS\" -wd \"$BASE_DIR/code\" -hold_jid $JOB_ID run_analysis.sh \"$BASE_DIR\""
	qsub -pe omp "$NUM_ANALYZE_SLOTS" -wd "$BASE_DIR/code" -hold_jid "$JOB_ID" run_analysis.sh "$BASE_DIR"
fi
