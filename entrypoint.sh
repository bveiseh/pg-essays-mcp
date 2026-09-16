#!/bin/sh
set -eu

python /app/indexer.py --ensure-valid /seed-data

python /app/server.py &
server_pid=$!

(
  refresh_pid=""
  stop_scheduler() {
    if [ -n "$refresh_pid" ]; then
      kill "$refresh_pid" 2>/dev/null || true
      wait "$refresh_pid" 2>/dev/null || true
    fi
    exit 0
  }
  trap stop_scheduler INT TERM
  while kill -0 "$server_pid" 2>/dev/null; do
    python /app/indexer.py --force --if-stale-days 7 &
    refresh_pid=$!
    wait "$refresh_pid" || python /app/indexer.py --mark-interrupted || true
    refresh_pid=""
    sleep 21600
  done
) &
scheduler_pid=$!

shutdown() {
  kill "$server_pid" "$scheduler_pid" 2>/dev/null || true
  wait "$server_pid" "$scheduler_pid" 2>/dev/null || true
}
trap shutdown INT TERM
status=0
wait "$server_pid" || status=$?
kill "$scheduler_pid" 2>/dev/null || true
wait "$scheduler_pid" 2>/dev/null || true
exit "$status"
