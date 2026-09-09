#!/usr/bin/env bash
set -euo pipefail

control=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
project=$(git -C "$control" rev-parse --show-toplevel)
output="$project/out/artifacts/perf-bench/gemmini/global_phase2_full_macro_four_model_v2_20260908"
watch_pattern='[w]atch_and_launch_when_safe.sh'
available_kib=$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)
swap_total_kib=$(awk '/^SwapTotal:/{print $2}' /proc/meminfo)
swap_free_kib=$(awk '/^SwapFree:/{print $2}' /proc/meminfo)

printf 'time=%s\n' "$(date --iso-8601=seconds)"
printf 'memory_available_bytes=%d\n' "$((available_kib * 1024))"
printf 'swap_used_bytes=%d\n' "$(((swap_total_kib - swap_free_kib) * 1024))"
if [[ -e "$control/DISARMED" ]]; then
  printf 'phase2_state=checkpoint_disarmed\n'
elif [[ -e "$output" ]]; then
  printf 'phase2_output=%s\n' "$output"
  find "$output" -maxdepth 2 -type f \
    \( -name 'launch.json' -o -name 'terminal_failure.json' -o -name 'authoring_sequence.json' \) \
    -print | sort
elif pgrep -af "$watch_pattern"; then
  printf 'phase2_state=armed_waiting_for_resource_gate\n'
else
  printf 'phase2_state=not_running_and_watcher_missing\n'
  exit 75
fi
