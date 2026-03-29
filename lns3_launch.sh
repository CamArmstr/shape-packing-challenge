#!/bin/bash
# lns3_launch.sh — Launch N independent lns3 workers as separate processes
# Each writes to its own log. No shared queue to deadlock.
# Usage: bash lns3_launch.sh [n_workers] [runtime_seconds]

WORKERS=${1:-6}
RUNTIME=${2:-21600}
DIR="/home/camcore/.openclaw/workspace/shape-packing-challenge"

echo "Launching $WORKERS lns3 workers, runtime=${RUNTIME}s"
echo "Best on disk: $(cd $DIR && python3 -c "
import json
from src.semicircle_packing.scoring import validate_and_score
from src.semicircle_packing.geometry import Semicircle
raw=json.load(open('best_solution.json'))
r=validate_and_score([Semicircle(d['x'],d['y'],d['theta']) for d in raw])
print(f'R={r.score:.6f}')
")"

pids=()
for i in $(seq 0 $((WORKERS-1))); do
    logfile="$DIR/lns3w${i}.log"
    nohup python3 "$DIR/lns3_worker.py" --wid $i --runtime $RUNTIME > "$logfile" 2>&1 &
    pids+=($!)
    echo "  worker $i PID=$!"
done

echo "All workers launched. PIDs: ${pids[@]}"
echo "Monitor: tail -f $DIR/lns3w*.log"
echo "Stop: kill ${pids[@]}"

# Write PID file for easy cleanup
echo "${pids[@]}" > "$DIR/lns3_pids.txt"
echo "PIDs saved to lns3_pids.txt"
