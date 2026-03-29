#!/bin/bash
# lns3_launch.sh — Launch N independent lns3_worker processes
# Usage: ./lns3_launch.sh [workers=6] [runtime=21600]

WORKERS=${1:-6}
RUNTIME=${2:-21600}
DIR=/home/camcore/.openclaw/workspace/shape-packing-challenge

echo "Launching $WORKERS lns3 workers, runtime=${RUNTIME}s"
echo "Best on disk: $(python3 -c "
import json, sys
sys.path.insert(0, '$DIR')
import os; os.chdir('$DIR')
from src.semicircle_packing.scoring import validate_and_score
from src.semicircle_packing.geometry import Semicircle
raw=json.load(open('$DIR/best_solution.json'))
r=validate_and_score([Semicircle(d['x'],d['y'],d['theta']) for d in raw])
print(f'R={r.score:.6f}')
" 2>/dev/null)"

for i in $(seq 0 $((WORKERS - 1))); do
    LOG="$DIR/lns3_w${i}.log"
    nohup python3 "$DIR/lns3_worker.py" --wid $i --runtime $RUNTIME > "$LOG" 2>&1 &
    echo "  Worker $i → PID $! → $LOG"
done

echo "All workers launched."
