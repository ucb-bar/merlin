#!/bin/bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "usage: $0 /absolute/path/to/driver-chipyard [/absolute/path/to/firesim-queue] [existing-workload-template]" >&2
  exit 2
fi

bundle_dir=$(cd "$(dirname "$0")" && pwd -P)
if [[ "$(jq -r '.safe_to_submit' "$bundle_dir/validation/firesim_submission_readiness.json")" != true ]]; then
  echo "FireSim submission blocked: local qualification receipt is not safe-to-submit." >&2
  echo "See $bundle_dir/validation/firesim_submission_readiness.json" >&2
  exit 2
fi
chipyard=$(cd "$1" && pwd -P)
queue=${2:-/scratch/firesim_queue/bin/firesim-queue}
workload_template=${3:-merlin-l4-20-resnet50-generated}
"$bundle_dir/verify_bundle.sh"
IFS=$'\t' read -r queue_chipyard workload bootbinary deploy_quintuplet driver \
  < <("$bundle_dir/prepare_queue_chipyard_view.sh" "$chipyard" "$workload_template")
if [[ -z "$queue_chipyard" || -z "$workload" || -z "$bootbinary" \
    || -z "$deploy_quintuplet" || -z "$driver" ]]; then
  echo "private queue Chipyard view did not return its complete identity" >&2
  exit 2
fi
echo "queue driver authority: $driver" >&2
echo "queue bitstream deploy quintuplet: $deploy_quintuplet" >&2
echo "queue private workload: $workload ($bootbinary)" >&2

# The queue daemon owns one uninterrupted FPGA lock and executes exactly:
# firesim kill -> firesim infrasetup -> firesim runworkload -> firesim kill.
# The package-local launcher preserves the daemon's writable deploy overlay.
export PATH="$bundle_dir/queue_launcher:$PATH"
# Do not forward the submitter's login identity to a cross-user queue daemon.
# The queue still attributes the job from this process's uid, while the daemon
# retains its own readable HOME for Chipyard's conda activation.  Jobs 498 and
# 509 established this environment contract; job 518 failed before workload
# execution when HOME=/home/agustin was forwarded to the jack-owned daemon.
exec /usr/bin/env -u HOME -u USER -u LOGNAME "$queue" runworkload-full \
  --chipyard "$queue_chipyard" \
  --workload "$workload" \
  --bootbinary "$bootbinary" \
  --stage-from "$bundle_dir/resnet50_merlin53_tvmhost_warm.elf" \
  --hw-config resnet50_merlin_warm_measured_full_gemmini_u250_pinned \
  --hwdb-config-artifact "$bundle_dir/runtime/config_hwdb.yaml" \
  --priority 10 \
  --project resnet50-merlin-native-loopconv53-tvmhost-warm \
  --timeout 25200
