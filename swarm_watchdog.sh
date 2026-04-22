#!/bin/bash
# Swarm watchdog — called by cron every 5 minutes.
# Ensures swarm.py is running; starts it if not.

SWARM_DIR="/home/camcore/.openclaw/workspace/shape-packing-challenge"
PIDFILE="$SWARM_DIR/.swarm_master.pid"
LOGFILE="$SWARM_DIR/.swarm_master.log"
PYTHON="/usr/bin/python3"

# Check if swarm is already running
if [ -f "$PIDFILE" ]; then
    PID=$(cat "$PIDFILE")
    if kill -0 "$PID" 2>/dev/null; then
        exit 0  # Already running
    fi
    rm -f "$PIDFILE"
fi

# Start swarm
cd "$SWARM_DIR" || exit 1
nohup "$PYTHON" swarm.py >> "$LOGFILE" 2>&1 &
echo $! > "$PIDFILE"
echo "$(date): Started swarm.py (PID $!)" >> "$LOGFILE"
