#!/usr/bin/env bash
# Per-dispatch profiling sweep across (model × backend) cells.
#
# Runs each cell with MERLIN_PROFILE_CYCLES=1 enabled in the Zephyr runner
# and extracts the `CYC, ord, sym, total_cycles, wg_count` lines emitted
# by iree_merlin_dump_cycles() into a CSV.
#
# Usage:
#   tools/profile_dispatches.sh                       # all 12 cells
#   tools/profile_dispatches.sh mlp_wide              # all 4 backends for one model
#   tools/profile_dispatches.sh yolov8n gemmini       # one cell only
#
# Output:
#   tmp/dispatch_profile.csv          — combined per-dispatch table
#   tmp/profile_uartlog_<model>_<backend>.log — full uartlog per cell

set -uo pipefail
MERLIN_ROOT="${MERLIN_ROOT:-/scratch2/agustin/merlin}"
cd "$MERLIN_ROOT"

ONLY_MODEL="${1:-}"
ONLY_BACKEND="${2:-}"

MODELS=(mlp_wide dronet yolov8n)
BACKENDS=(scalar rvv opu gemmini)

CSV="${MERLIN_ROOT}/tmp/dispatch_profile.csv"
mkdir -p "${MERLIN_ROOT}/tmp"
echo "model,backend,ordinal,symbol,total_cycles,wg_count,time_us" >"$CSV"

# FireSim target clock for time conversion. Effective rate from the
# emulation summary in past uartlogs: 28.343 MHz target frequency.
TARGET_MHZ=28.343

parse_cyc() {
  local model="$1" backend="$2" uartlog="$3"
  if [[ ! -f "$uartlog" ]]; then
    echo "[parse_cyc] missing uartlog: $uartlog" >&2
    return 1
  fi
  # The Zephyr runner emits two interleaved streams:
  #   [dn] o=N sym=<symbol> wg_count=X,Y,Z      -- one per dispatch (header)
  #   [disp] ord=N wg=(x,y,z) cycles=C ret=R    -- one per workgroup of dispatch N
  # We aggregate per-ordinal: sum cycles across all workgroups, count wgs,
  # and pull the symbol from the [dn] header.
  awk -v model="$model" -v backend="$backend" -v mhz="$TARGET_MHZ" '
    /^\[dn\] o=/ {
      match($0, /o=[0-9]+/);     ord = substr($0, RSTART+2, RLENGTH-2)
      match($0, /sym=[^ ]+/);    sym = substr($0, RSTART+4, RLENGTH-4)
      symfor[ord+0] = sym
    }
    /^\[disp\] ord=/ {
      match($0, /ord=[0-9]+/);   ord = substr($0, RSTART+4, RLENGTH-4) + 0
      match($0, /cycles=[0-9]+/); cyc = substr($0, RSTART+7, RLENGTH-7) + 0
      cyc_tot[ord] += cyc
      wg_count[ord]++
    }
    END {
      for (ord in cyc_tot) {
        sym = (ord in symfor) ? symfor[ord] : "(unknown)"
        cyc = cyc_tot[ord]
        wg = wg_count[ord]
        time_us = cyc / mhz
        printf "%s,%s,%d,%s,%d,%d,%.3f\n", model, backend, ord, sym, cyc, wg, time_us
      }
    }
  ' "$uartlog" | sort -t, -k3,3n
}

for model in "${MODELS[@]}"; do
  [[ -n "$ONLY_MODEL" && "$ONLY_MODEL" != "$model" ]] && continue
  for backend in "${BACKENDS[@]}"; do
    [[ -n "$ONLY_BACKEND" && "$ONLY_BACKEND" != "$backend" ]] && continue

    cell_log="${MERLIN_ROOT}/tmp/profile_uartlog_${model}_${backend}.log"
    echo "==================================================================="
    echo "Profiling cell: ${model} × ${backend}"
    echo "==================================================================="

    # Use the existing run_all.sh harness (handles env + compile + run +
    # uartlog discovery). It writes the uartlog into the FireSim results
    # dir; we then locate and copy it.
    if ! benchmarks/firesim_shuttle/run_all.sh "$model" "$backend" 2>&1 \
            | tail -30 >"${cell_log}.runlog"; then
      echo "[FAIL] cell ${model}_${backend} run_all.sh nonzero exit"
    fi

    # Discover uartlog from CSV (last row matching this cell)
    last_uartlog=$(grep -E ",${model},${backend}," tmp/firesim_shuttle_results.csv \
                      | tail -1 | awk -F, '{print $NF}')
    if [[ -n "$last_uartlog" && -f "$last_uartlog" ]]; then
      cp "$last_uartlog" "$cell_log"
      echo "[PASS] saved uartlog: $cell_log"
      parse_cyc "$model" "$backend" "$cell_log" >>"$CSV"
      n=$(grep -c "^${model},${backend}," "$CSV")
      echo "  → $n dispatches profiled"
    else
      echo "[FAIL] couldn't locate uartlog for ${model}_${backend}"
    fi
  done
done

echo ""
echo "==================================================================="
echo "Sweep complete. CSV at: $CSV"
echo "Rows per cell:"
echo "==================================================================="
awk -F, 'NR>1 {c[$1","$2]++} END {for (k in c) printf "  %s: %d dispatches\n", k, c[k]}' "$CSV" \
    | sort
