#!/usr/bin/env bash
set -euo pipefail

control=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
project=$(git -C "$control" rev-parse --show-toplevel)
output="$project/out/artifacts/perf-bench/gemmini/global_phase2_full_macro_four_model_v2_20260908"
minimum_available_bytes=$((48 * 1024 * 1024 * 1024))
maximum_swap_used_bytes=$((2 * 1024 * 1024 * 1024))

while [[ ! -e "$control/DISARMED" && ! -e "$output" ]]; do
  available_kib=$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)
  swap_total_kib=$(awk '/^SwapTotal:/{print $2}' /proc/meminfo)
  swap_free_kib=$(awk '/^SwapFree:/{print $2}' /proc/meminfo)
  available_bytes=$((available_kib * 1024))
  swap_used_bytes=$(((swap_total_kib - swap_free_kib) * 1024))
  printf '{"time":"%s","memory_available_bytes":%d,"swap_used_bytes":%d,"admitted":%s}\n' \
    "$(date --iso-8601=seconds)" "$available_bytes" "$swap_used_bytes" \
    "$([[ $available_bytes -ge $minimum_available_bytes && $swap_used_bytes -le $maximum_swap_used_bytes ]] && echo true || echo false)"
  if [[ $available_bytes -ge $minimum_available_bytes && $swap_used_bytes -le $maximum_swap_used_bytes ]]; then
    exec "$control/launch_exact_only_when_safe.sh"
  fi
  sleep 30
done

if [[ -e "$control/DISARMED" ]]; then
  echo "checkpoint is disarmed; refusing to launch" >&2
  exit 75
fi
echo "planned output already exists; refusing a second launcher: $output" >&2
exit 75
