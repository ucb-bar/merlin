#!/usr/bin/env bash
# Re-parse all saved per-cell uartlogs into the dispatch_profile CSV.
#
# Reads tmp/profile_uartlog_<model>_<backend>.log files and aggregates
# per-(model, backend, ordinal): sum of cycles across workgroups, symbol,
# workgroup count, derived time in microseconds (assuming FireSim target
# clock of ~28.343 MHz, which is the effective FMR-derived rate observed
# in past Shuttle uartlogs).
#
# Usage:
#   tools/parse_dispatch_profile.sh   # writes tmp/dispatch_profile.csv

set -uo pipefail
MERLIN_ROOT="${MERLIN_ROOT:-/scratch2/agustin/merlin}"
cd "$MERLIN_ROOT"

TARGET_MHZ=28.343
CSV="${MERLIN_ROOT}/tmp/dispatch_profile.csv"
echo "model,backend,ordinal,symbol,total_cycles,wg_count,time_us" >"$CSV"

# Companion CSV: whole-model E2E time per (model, backend).
E2E_CSV="${MERLIN_ROOT}/tmp/e2e_profile.csv"
echo "model,backend,total_cycles,time_us,hash,rc" >"$E2E_CSV"

MODELS=(mlp_wide dronet yolov8n)
BACKENDS=(scalar rvv opu gemmini)

for model in "${MODELS[@]}"; do
  for backend in "${BACKENDS[@]}"; do
    log="${MERLIN_ROOT}/tmp/profile_uartlog_${model}_${backend}.log"
    [[ -f "$log" ]] || { echo "  [skip] no uartlog for ${model}_${backend}"; continue; }

    awk -v model="$model" -v backend="$backend" -v mhz="$TARGET_MHZ" '
      # Strip CR (the FireSim UART produces CRLF and CR terminators mixed).
      { sub(/\r$/, ""); sub(/\r/, "") }
      # Compact per-ordinal aggregated dump:
      #   CYC, <ord>, <symbol>, <total_cycles>, <wg_count>
      # Bracketed by "CYC, begin" / "CYC, end". Preferred source — no per-call
      # UART overhead during invoke, one line per dispatch ordinal.
      /^CYC, begin$/ { cyc_block = 1; next }
      /^CYC, end$/   { cyc_block = 0; next }
      cyc_block && /^CYC, [0-9]/ {
        n = split($0, f, /, /)
        if (n < 5) next
        ord = f[2] + 0; sym = f[3]; cyc = f[4] + 0; wg = f[5] + 0
        cyc_tot[ord] = cyc
        wg_count[ord] = wg
        if (sym != "") symfor[ord] = sym
      }
      # Per-dispatch header (runtime gate: dispatch_debug=1)
      /^\[dn\] o=/ {
        match($0, /o=[0-9]+/);     ord = substr($0, RSTART+2, RLENGTH-2)
        match($0, /sym=[^ ]+/);    sym = substr($0, RSTART+4, RLENGTH-4)
        symfor[ord+0] = sym
      }
      # End-of-run dump from merlin_dispatch_dump() — full ring buffer.
      # Only present if Zephyr reached the post-invoke dump (not on crash).
      /^\[disp\] ord=/ {
        match($0, /ord=[0-9]+/);    ord = substr($0, RSTART+4, RLENGTH-4) + 0
        match($0, /cycles=[0-9]+/); cyc = substr($0, RSTART+7, RLENGTH-7) + 0
        if (!seen_disp[ord","FNR]) {
          cyc_tot[ord] += cyc
          wg_count[ord]++
          seen_disp[ord","FNR] = 1
        }
      }
      # Per-workgroup live trace [dc] (runtime gate: dispatch_debug=1).
      # Use these as a FALLBACK when [disp] dump is missing (e.g. the
      # Zephyr runner crashed before reaching merlin_dispatch_dump).
      # Tagged via has_disp_dump so we do not double-count when both
      # streams are present.
      /^\[disp\] === per-dispatch cycle log/ { has_disp_dump = 1 }
      /^\[dc\] o=/ {
        match($0, /o=[0-9]+/);    ord = substr($0, RSTART+2, RLENGTH-2) + 0
        match($0, /cyc=[0-9]+/); cyc = substr($0, RSTART+4, RLENGTH-4) + 0
        dc_cyc[ord] += cyc
        dc_wg[ord]++
      }
      END {
        # MERGE the two streams: per-ordinal, prefer the [disp] ring buffer
        # entry when present (per-workgroup aggregated, more precise) and
        # fall back to [dc] live trace when [disp] is missing for that ord
        # (e.g. when the post-invoke dump was truncated mid-print by an
        # orphan-screen kill or a runtime crash). Each ord gets data from
        # EXACTLY ONE source so we never double-count.
        for (ord in dc_cyc) {
          if (!(ord in cyc_tot)) {
            cyc_tot[ord] = dc_cyc[ord]
            wg_count[ord] = dc_wg[ord]
          }
        }
        for (ord in cyc_tot) {
          sym = (ord in symfor) ? symfor[ord] : "(unknown)"
          cyc = cyc_tot[ord]
          wg = wg_count[ord]
          time_us = cyc / mhz
          printf "%s,%s,%d,%s,%d,%d,%.3f\n", model, backend, ord, sym, cyc, wg, time_us
        }
      }
    ' "$log" | sort -t, -k3,3n >>"$CSV"

    n=$(grep -c "^${model},${backend}," "$CSV")
    echo "  [${model} × ${backend}]: $n dispatches"

    # Whole-model E2E: extract from "[merlin] result model=... cycles=N hash=0x...".
    awk -v model="$model" -v backend="$backend" -v mhz="$TARGET_MHZ" '
      /^\[merlin\] result/ {
        match($0, /cycles=[0-9]+/); cyc = substr($0, RSTART+7, RLENGTH-7) + 0
        match($0, /hash=0x[0-9a-fA-F]+/); hash = substr($0, RSTART+5, RLENGTH-5)
        match($0, /rc=-?[0-9]+/); rc = substr($0, RSTART+3, RLENGTH-3)
        printf "%s,%s,%d,%.3f,%s,%s\n", model, backend, cyc, cyc/mhz, hash, rc
      }
    ' "$log" >>"$E2E_CSV"
  done
done

echo ""
echo "==================================================================="
echo "Per-dispatch CSV: $CSV"
echo "Total rows: $(($(wc -l < "$CSV") - 1)) (excluding header)"
echo "Cells with data:"
awk -F, 'NR>1 {c[$1","$2]++} END {for (k in c) printf "  %s: %d\n", k, c[k]}' "$CSV" | sort
echo ""
echo "E2E CSV: $E2E_CSV"
echo "E2E rows: $(($(wc -l < "$E2E_CSV") - 1)) (excluding header)"
column -s, -t "$E2E_CSV" 2>/dev/null | head -20
