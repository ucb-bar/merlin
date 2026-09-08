#!/bin/bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 /absolute/path/to/driver-authority-chipyard existing-workload-template" >&2
  exit 2
fi

bundle_dir=$(cd "$(dirname "$0")" && pwd -P)
chipyard=$(cd "$1" && pwd -P)
workload_template=$2
firesim_root="$chipyard/sims/firesim"
deploy="$firesim_root/deploy"
template_json="$deploy/workloads/$workload_template.json"
bitstream="$bundle_dir/firesim.tar.gz"
elf="$bundle_dir/resnet50_merlin53_tvmhost_warm.elf"
runtime_dir="$bundle_dir/runtime"

for required in \
  "$chipyard/env.sh" \
  "$firesim_root/sourceme-manager.sh" \
  "$deploy/firesim" \
  "$deploy/config_runtime.yaml" \
  "$template_json" \
  "$bitstream" \
  "$elf"; do
  if [[ ! -f "$required" ]]; then
    echo "missing required driver/workload authority: $required" >&2
    exit 2
  fi
done

# Derive the exact deploy quintuplet from the immutable bitstream itself. This
# makes selecting a host driver a metadata identity check, not a directory-name
# guess. Refuse ambiguous archives or absent drivers.
metadata_member=$(tar -tzf "$bitstream" | grep '/metadata$' || true)
if [[ $(printf '%s\n' "$metadata_member" | sed '/^$/d' | wc -l) -ne 1 ]]; then
  echo "bitstream archive has no unique platform metadata member" >&2
  exit 2
fi
metadata=$(tar -xOzf "$bitstream" "$metadata_member")
deploy_quintuplet=$(printf '%s\n' "$metadata" \
  | sed -n 's/.*firesim-deployquintuplet:\([^,]*\).*/\1/p')
if [[ -z "$deploy_quintuplet" ]] \
    || [[ $(printf '%s\n' "$metadata" | grep -o 'firesim-deployquintuplet:' | wc -l) -ne 1 ]]; then
  echo "bitstream metadata has no unique FireSim deploy quintuplet" >&2
  exit 2
fi
platform=${deploy_quintuplet%%-firesim-*}
driver="$firesim_root/sim/output/$platform/$deploy_quintuplet/FireSim-$platform"
if [[ ! -x "$driver" ]]; then
  echo "exact-config prebuilt driver is absent: $driver" >&2
  exit 2
fi

# FireSim ignores firesim-queue's --bootbinary after staging and reads the
# workload JSON instead. Build a private alias whose bootbinary identity is
# tied to this ELF, so the queue cannot stage one file and execute another.
if ! jq -e '(.workloads // null) == null and (.common_rootfs // null) == null' \
    "$template_json" >/dev/null; then
  echo "workload template must be a uniform bare-metal alias" >&2
  exit 2
fi
elf_sha256=$(sha256sum "$elf" | awk '{print $1}')
workload="merlin-resnet50-native-loopconv-${elf_sha256:0:16}"
bootbinary="$workload.elf"

# The daemon owns the FireSim lifecycle and staging. This local read-only view
# gives it Jack's exact driver authority without permitting writes to Jack's
# shared deploy tree. Because the view is not daemon-writable, firesim-queue
# creates its normal per-job deploy overlay and stages the unique ELF there.
# Keep submission-time symlink views outside the immutable bundle and git tree.
# `/scratch/firesim_queue` is the shared queue authority (setgid firesim).
view_root=${MERLIN_QUEUE_VIEW_ROOT:-/scratch/firesim_queue/queue_chipyard_views}
mkdir -p "$view_root"
view=$(mktemp -d "$view_root/view.XXXXXX")
view_firesim="$view/sims/firesim"
view_deploy="$view_firesim/deploy"
view_workloads="$view_deploy/workloads"
mkdir -p "$view_workloads"

ln -s "$chipyard/env.sh" "$view/env.sh"
ln -s "$chipyard/generators" "$view/generators"
while IFS= read -r -d '' entry; do
  [[ $(basename "$entry") == deploy ]] && continue
  ln -s "$entry" "$view_firesim/$(basename "$entry")"
done < <(find "$firesim_root" -mindepth 1 -maxdepth 1 -print0)
while IFS= read -r -d '' entry; do
  case $(basename "$entry") in
    workloads|logs|results-workload|generated-topology-diagrams) continue ;;
  esac
  ln -s "$entry" "$view_deploy/$(basename "$entry")"
done < <(find "$deploy" -mindepth 1 -maxdepth 1 -print0)

jq --arg benchmark "$workload" --arg bootbinary "$bootbinary" \
  '.benchmark_name = $benchmark | .common_bootbinary = $bootbinary' \
  "$template_json" > "$view_workloads/$workload.json"
template_sha256=$(sha256sum "$template_json" | awk '{print $1}')
workload_json_sha256=$(sha256sum "$view_workloads/$workload.json" | awk '{print $1}')
bitstream_sha256=$(sha256sum "$bitstream" | awk '{print $1}')
driver_sha256=$(sha256sum "$driver" | awk '{print $1}')
jq -n \
  --arg schema merlin_native_loopconv_hybrid_queue_view_identity_v1 \
  --arg view "$view" \
  --arg workload "$workload" \
  --arg bootbinary "$bootbinary" \
  --arg bootbinary_sha256 "$elf_sha256" \
  --arg workload_template "$template_json" \
  --arg workload_template_sha256 "$template_sha256" \
  --arg workload_json_sha256 "$workload_json_sha256" \
  --arg deploy_quintuplet "$deploy_quintuplet" \
  --arg bitstream_sha256 "$bitstream_sha256" \
  --arg driver "$driver" \
  --arg driver_sha256 "$driver_sha256" \
  '{schema:$schema, view:$view, workload:$workload, bootbinary:$bootbinary,
    bootbinary_sha256:$bootbinary_sha256,
    workload_template:$workload_template,
    workload_template_sha256:$workload_template_sha256,
    generated_workload_json_sha256:$workload_json_sha256,
    deploy_quintuplet:$deploy_quintuplet,
    bitstream_sha256:$bitstream_sha256,
    driver:$driver, driver_sha256:$driver_sha256,
    queue_owned_lifecycle:["firesim kill","firesim infrasetup",
      "firesim runworkload","firesim kill"]}' > "$view/queue_identity.json"
chmod 0444 "$view_workloads/$workload.json"
chmod 0444 "$view/queue_identity.json"
chmod 0555 "$view_workloads" "$view_deploy" "$view_firesim" "$view/sims" "$view"

# Machine-readable, delimiter-safe fields for run_firesim.sh. No FireSim or
# queue command is executed here.
printf '%s\t%s\t%s\t%s\t%s\n' \
  "$view" "$workload" "$bootbinary" "$deploy_quintuplet" "$driver"
