#!/bin/bash
set -euo pipefail

bundle_dir=$(cd "$(dirname "$0")" && pwd -P)
cd "$bundle_dir"

test "$(sha256sum resnet50_merlin53_tvmhost_warm.elf | awk '{print $1}')" = 5703184cf70ea0e11e5a9730502c48bb86485d111255a036b5481b82c80915f3
if [[ -f firesim.tar.gz ]]; then
  test "$(sha256sum firesim.tar.gz | awk '{print $1}')" = a9a190b9fc26d577b1e8af0e3b46a94c8efaf2976f7fca6650236e0fd6d4eca1
fi
test "$(sha256sum queue_launcher/firesim | awk '{print $1}')" = 22b6e97c50598c63f3230e5b469ff08343646bb2b44b550563e02fd3bde84516
test "$(sha256sum queue_launcher/make | awk '{print $1}')" = 10197fb0bf6ffb52eb2f328a58068e996bfcaac45a9defa342aa07490e6ab650
test "$(jq -r '.status' validation/local_spike_exact_receipt.json)" = passed
test "$(jq '.logits' validation/local_spike_exact_receipt.json)" -eq 1000
test "$(jq '.mismatches' validation/local_spike_exact_receipt.json)" -eq 0
test "$(jq '.top1' validation/local_spike_exact_receipt.json)" -eq 258
test "$(jq -r '.actual_sha256' validation/local_spike_exact_receipt.json)" = 46c7d2ac06c770b974187d24e45395468cfc1c024d149274baf9d599f3f91282
test "$(jq -r '.reference_sha256' validation/local_spike_exact_receipt.json)" = 46c7d2ac06c770b974187d24e45395468cfc1c024d149274baf9d599f3f91282
test "$(jq '.merlin_compiled.conv2d' validation/local_spike_exact_receipt.json)" -eq 53
test "$(jq -r '.status' validation/one_layer_exact_receipt.json)" = passed
test "$(jq '.mismatches' validation/one_layer_exact_receipt.json)" -eq 0
test "$(jq -r '.safe_to_submit' validation/firesim_submission_readiness.json)" = false
test "$(jq -r '.promoted' validation/firesim_submission_readiness.json)" = true
test "$(jq -r '.elf_sha256' validation/firesim_submission_readiness.json)" = 5703184cf70ea0e11e5a9730502c48bb86485d111255a036b5481b82c80915f3
test "$(jq -r '.status' validation/firesim_queue_job_536_success.json)" = passed
test "$(jq '.measured.cycles' validation/firesim_queue_job_536_success.json)" -eq 555991472
test "$(jq '.correctness.mismatches' validation/firesim_queue_job_536_success.json)" -eq 0
test "$(jq '.correctness.top1' validation/firesim_queue_job_536_success.json)" -eq 258
test "$(jq -r '.artifact_identity.elf_identity_passed' validation/firesim_queue_job_536_success.json)" = true
test "$(sha256sum validation/firesim_queue_job_536_stdout.log | awk '{print $1}')" = fdde838b131b67617dd97e9f8baecb5e3b4ded00b739ac167dc4aa02cebf3127
test "$(sha256sum validation/firesim_queue_job_536_stderr.log | awk '{print $1}')" = 2b63642ce74f1fc24972c90f95343c3ec4f394a44facf501e0e1953e4a354f55
grep -q -- 'resnet50_merlin53_tvmhost_warm.elf' run_firesim.sh
grep -q -- 'resnet50_merlin53_tvmhost_warm.elf' prepare_queue_chipyard_view.sh
test -z "$(find . -type l -print -quit)"
echo "Exact local and FireSim q536 gates passed; unchanged resubmission is blocked."
