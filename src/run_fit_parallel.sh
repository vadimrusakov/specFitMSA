#!/bin/bash
#
# Parallel spectral fitting with catalog chunking (runs in background with nohup)
#
# Usage: ./run_parallel.sh [N_WORKERS] [COV_FLAG]
#
# Arguments:
#   N_WORKERS  Number of parallel workers (default: 4)
#   COV_FLAG   Pass 'cov' to plot covariance matrices (default: empty)
#
# Examples:
#   ./run_parallel.sh          # default N workers, no cov plots
#   ./run_parallel.sh 8        # 8 workers
#   ./run_parallel.sh 4 cov    # 4 workers with cov plots
#
# Output:
#   Logs are written to output_worker_N.log in the script directory
#   Chunk files are stored in chunks_XXXXXX/ and cleaned up when all workers finish

set -e

# parse arguments
N_WORKERS=${1:-1}
COV_FLAG=${2:-""}

# get script directory and change to it
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_SCRIPT="fit_jax.py"

# get catalog path from config
CATALOG_PATH=$(python -c "import config as cfg; print(cfg.fpath_spec)")

if [[ ! -f "$CATALOG_PATH" ]]; then
    echo "Error: Catalog not found at $CATALOG_PATH"
    exit 1
fi

# extract unique PROG_IDs (column 2, skip header) and shuffle randomly
# (uses python for shuffling for macOS/Linux compatibility)
PROG_IDS=()
while IFS= read -r line; do
    PROG_IDS+=("$line")
done < <(tail -n +2 "$CATALOG_PATH" | cut -d',' -f2 | sort -u | \
    python3 -c "import sys,random; L=sys.stdin.read().splitlines(); random.shuffle(L); print('\n'.join(L))")
N_OBJECTS=${#PROG_IDS[@]}

echo "============================================================"
echo "Parallel Spectral Fitting"
echo "============================================================"
echo "Catalog: $CATALOG_PATH"
echo "Unique objects: $N_OBJECTS"
echo "Workers: $N_WORKERS"
echo "============================================================"

if [[ $N_OBJECTS -eq 0 ]]; then
    echo "No objects found in catalog"
    exit 1
fi

# create directories for chunk and log files
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CHUNK_DIR="${SCRIPT_DIR}/chunks"
LOG_DIR="${SCRIPT_DIR}/output_logs"
mkdir -p "$CHUNK_DIR"
mkdir -p "$LOG_DIR"
echo "Chunk files: $CHUNK_DIR"
echo "Log files: $LOG_DIR"

# split IDs into chunks
CHUNK_SIZE=$(( (N_OBJECTS + N_WORKERS - 1) / N_WORKERS ))

for ((i=0; i<N_WORKERS; i++)); do
    START=$((i * CHUNK_SIZE))
    CHUNK_FILE="$CHUNK_DIR/chunk_${TIMESTAMP}_${i}.txt"

    for ((j=START; j<START+CHUNK_SIZE && j<N_OBJECTS; j++)); do
        echo "${PROG_IDS[$j]}" >> "$CHUNK_FILE"
    done

    if [[ -f "$CHUNK_FILE" ]]; then
        N_IN_CHUNK=$(wc -l < "$CHUNK_FILE")
        echo "  Chunk $i: $N_IN_CHUNK objects"
    fi
done

echo "============================================================"
echo "Launching workers with nohup..."
echo ""

# launch workers with nohup
PIDS=()
for ((i=0; i<N_WORKERS; i++)); do
    CHUNK_FILE="$CHUNK_DIR/chunk_${TIMESTAMP}_${i}.txt"
    LOG_FILE="${LOG_DIR}/output_${TIMESTAMP}_worker_${i}.log"

    if [[ -f "$CHUNK_FILE" ]]; then
        nohup python "$PYTHON_SCRIPT" "$CHUNK_FILE" "$COV_FLAG" >> "$LOG_FILE" 2>&1 &
        PIDS+=($!)
        echo "[Worker $i] PID $! -> $LOG_FILE"
    fi
done

echo ""
echo "============================================================"
echo "All workers launched in background!"
echo ""
echo "Monitor progress:"
echo "  tail -n 1 ${LOG_DIR}/output_${TIMESTAMP}_worker_*.log"
echo ""
echo "Check running workers:"
echo "  ps -p ${PIDS[*]} 2>/dev/null || echo 'Workers finished'"
echo ""
echo "Chunk files will remain in: $CHUNK_DIR/chunk_${TIMESTAMP}_*.txt"
echo "Delete manually after all workers complete:"
echo "  rm -f $CHUNK_DIR/chunk_${TIMESTAMP}_*.txt"
echo "============================================================"
