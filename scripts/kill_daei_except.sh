#!/bin/bash
# Kill DAEI.run processes except for the specified PID.
# Usage: ./kill_daei_except.sh [KEEP_PID]
KEEP_PID=${1:-2907615}

echo "Keeping PID $KEEP_PID"
echo "Searching for DAEI.run processes..."

# Find matching Python or torchrun process IDs, excluding KEEP_PID.
PIDS=$(ps -u $USER -o pid,args --no-headers 2>/dev/null | grep -E "(python|torchrun).*DAEI\.run" | grep -v grep | awk '{print $1}' | grep -v "^${KEEP_PID}$")

if [ -z "$PIDS" ]; then
    echo "No matching processes found (or only the one to keep)."
    echo ""
    echo "Current python processes:"
    ps -u $USER -o pid,args --no-headers 2>/dev/null | grep python | grep -v grep | head -20
    exit 0
fi

echo "Found PIDs to kill: $PIDS"
for pid in $PIDS; do
    echo "  Killing PID $pid ..."
    kill $pid 2>/dev/null || kill -9 $pid 2>/dev/null
done
echo "Done."
