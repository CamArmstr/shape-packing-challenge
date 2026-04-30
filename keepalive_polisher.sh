#!/bin/bash
# keepalive_polisher.sh — Restart alt_topo polisher if not running
cd /home/camcore/.openclaw/workspace/shape-packing-challenge

if pgrep -f "alt_topo_polish" > /dev/null; then
    exit 0
fi

echo "$(date): Restarting alt_topo polisher" >> keepalive.log

nohup python3 mcmc_exact.py \
  --mode polisher \
  --batches 0 \
  --batch-size 200 \
  --seed-file alt_topo_solution.json \
  --step 0.0001 \
  --max-step 0.0003 \
  --min-step 0.00003 \
  --score-slack 0.00005 \
  --rescue-prob 0.02 \
  --mec-bias-prob 0.4 \
  --mec-top-k 5 \
  --polisher-cluster-prob 0.15 \
  --cluster-min 2 \
  --cluster-max 2 \
  --seed $((RANDOM + 1000)) \
  --tag alt_topo_polish \
  >> alt_topo_polish.log 2>&1 &
