#!/bin/bash
# Login-aware auto-resume watchdog for an A/B/C batch.
#   - resumes any arm that EXITED un-converged (crash / waits-exhausted)
#   - detects a NEW LOGIN (credentials file mtime change); if the (new) account has headroom, it
#     kills any arm currently SLEEPING on a rate-limit wait and resumes it -> picks up the fresh limit
#   - exits when all arms report converged: true
# Usage: abc_watchdog.sh <tag>     e.g. abc_watchdog.sh abc4
set -u
TAG="${1:?usage: abc_watchdog.sh <tag>}"
REPO="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
ROOT="$REPO/merlin/experiments/gemmini_capsule_bench_v0"
PY="$REPO"/.venv/bin/python
CRED=/home/agustin/.claude/.credentials.json
F="--model claude-opus-4-8 --effort high --max-rounds 12 --max-rate-limit-waits 8 --round-timeout 14400 --experiment realistic"
cd "$ROOT"

RIDS=( "rb_$TAG" "merlin_$TAG" "merlincirct_$TAG" )
declare -A DRV SD BND PRE
DRV[rb_$TAG]="scripts/run_baseline_qa_loop.py --arm raw_baseline"; BND[rb_$TAG]="raw_baseline_hwbringup_v0"; SD[rb_$TAG]="raw_baseline/rb_$TAG"; PRE[rb_$TAG]="PILOT_LANG=cpp"
DRV[merlin_$TAG]="scripts/run_baseline_qa_loop.py --arm merlin_assisted"; BND[merlin_$TAG]="merlin_assisted_hwbringup_v0"; SD[merlin_$TAG]="merlin_assisted/merlin_$TAG"; PRE[merlin_$TAG]=""
DRV[merlincirct_$TAG]="scripts/run_rtlchecks_qa_loop.py"; BND[merlincirct_$TAG]="merlin_assisted_rtlchecks_hwbringup_v0"; SD[merlincirct_$TAG]="merlin_assisted/merlincirct_$TAG"; PRE[merlincirct_$TAG]=""

converged(){ grep -qE '^converged: true' "runs/${SD[$1]}/qa_loop_state.yaml" 2>/dev/null; }
alive(){ pgrep -f "run-id $1 " >/dev/null; }
sleeping(){ # alive AND last log line is a rate-limit sleep
  local lg; lg=$(ls -t runs/${SD[$1]}.resume.log runs/${SD[$1]}.launch.log 2>/dev/null | head -1)
  [ -n "$lg" ] && tail -3 "$lg" 2>/dev/null | grep -q "RATE-LIMITED"; }
resume(){ echo "[wd $(date +%H:%M:%S)] resume $1"; nohup env ${PRE[$1]} $PY ${DRV[$1]} --run-id "$1" --resume --bundle "${BND[$1]}" $F >> "runs/${SD[$1]}.resume.log" 2>&1 & }
probe_allowed(){ timeout 70 claude --print --model claude-opus-4-8 "Reply: OK" 2>&1 | grep -q '"status":"allowed"'; }

last_cred=$(stat -c %Y "$CRED" 2>/dev/null || echo 0)
while true; do
  alldone=1
  for r in "${RIDS[@]}"; do
    converged "$r" && continue
    alldone=0
    if ! alive "$r"; then resume "$r"; sleep 5; fi      # exited un-converged -> resume
  done
  [ "$alldone" = 1 ] && { echo "[wd] all $TAG converged at $(date +%H:%M:%S)"; break; }
  # login detection: credentials changed since last check -> a /login happened
  m=$(stat -c %Y "$CRED" 2>/dev/null || echo 0)
  if [ "$m" != "$last_cred" ]; then
    last_cred=$m
    echo "[wd $(date +%H:%M:%S)] new login detected — probing headroom"
    if probe_allowed; then
      for r in "${RIDS[@]}"; do
        converged "$r" && continue
        if sleeping "$r"; then
          echo "[wd] $r is rate-limit-sleeping + fresh account has headroom -> kick onto new login"
          pkill -9 -f "run-id $r " 2>/dev/null; sleep 3; resume "$r"; sleep 5
        fi
      done
    else
      echo "[wd] new login but no headroom yet — leaving sleepers"
    fi
  fi
  sleep 120
done
echo "=== $TAG FINAL ==="
for r in "${RIDS[@]}"; do
  echo "  $r: $(grep -E 'converged:|next_round:' runs/${SD[$r]}/qa_loop_state.yaml 2>/dev/null|tr '\n' ' ') | verilator: $(grep -oE '"final_all_pass": (true|false)' runs/${SD[$r]}/verilator_checkpoints.json 2>/dev/null|head -1)"
done
