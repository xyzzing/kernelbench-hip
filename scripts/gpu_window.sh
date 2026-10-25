#!/bin/bash
# GPU window wrapper: stop the strata-8081 resident, run one batched command
# under a clean env (env -i), restore the unit, wait for /health, report.
# Usage: gpu_window.sh <logfile> <command...>
set -u
LOG="$1"; shift
UNIT=strata-8081.service
HEALTH=http://127.0.0.1:8081/health

was_active=$(systemctl --user is-active "$UNIT")
if [ "$was_active" = "active" ]; then
  systemctl --user stop "$UNIT"
  sleep 3
  echo "[gpu_window] stopped $UNIT" >> "$LOG"
fi

env -i PATH=/usr/bin:/bin:/usr/sbin HOME="${HOME:-/tmp}" "$@" >> "$LOG" 2>&1
RC=$?
echo "[gpu_window] rc=$RC" >> "$LOG"

if [ "$was_active" = "active" ]; then
  systemctl --user start "$UNIT"
  C=000
  for i in $(seq 1 36); do
    sleep 5
    C=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "$HEALTH" 2>/dev/null)
    [ "$C" = "200" ] && break
  done
  echo "[gpu_window] restored $UNIT health=$C" >> "$LOG"
fi
exit $RC
