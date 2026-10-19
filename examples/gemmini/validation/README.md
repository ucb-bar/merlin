# Full-model FireSim checks after compiler freezing

Phase 1 capsule qualification and Phase 2 optimization each produce a separately
frozen compiler. Full-model FPGA validation consumes that exact compiler, the
selected quantized references and complete workload contract. It starts no author
and does not extend the authoring clock. A new compiler needs a new request and
binary binding, even if its workload name is unchanged.

On the AWS workhorse, the only execution command is `~/bin/fpga-run WORKLOAD`.
It drives the locked `FsimRunGemmini` SSM document on the manager. There is no
queue and there is one FPGA: submit one deliberately selected registered workload
at a time, with no speculative retries. Do not run FireSim, SSH, arbitrary SSM
commands or F2 launch APIs from the workhorse. The installed helper's omitted
argument defaults to `bare-base`, so always pass an explicit registered name.
Do not invoke it with `--help`: that is interpreted as a workload.

The initial deployment supports `gemmini-smoke`; this is a transport observation,
not a full model or compiler qualification. The helper accepts no ELF/upload
argument. An administrator must register new immutable full-chain workloads and
their exact artifacts on the manager. Workhorse code cannot safely assume that a
same-named manager workload contains a newly compiled binary.

## Prepare the administrator's request

Copy [firesim_workhorse.yaml](firesim_workhorse.yaml) outside author inputs and
bind its deployment fields. Select the phase, actual frozen compiler, immutable
freeze record and the record's compiler-digest field. The workload contract is
[the shared model protocol](../protocol/experiment.yaml); do not independently
restate architecture, quantization, recurrence or tolerances.

```sh
python -m merlin_experiments.post_freeze.firesim prepare \
  --selection /absolute/selected-firesim.yaml --out /absolute/request.json
```

This unpaid operation hashes inputs and materializes the complete output roster.
Missing payloads/registrations produce `blocked` with concrete reasons and exit 2.
It never uploads assets, starts a manager or submits an FPGA run. An administrator
receives the request plus the separately provisioned private payloads. References
and model inputs must stay outside the compiler author's sandbox/public Git.

Per model, provide the complete-chain boot ELF, chain manifest, inputs, selected
quantized references and normal compilation/source/placement receipts. Preserve
integerized IR/scales and input/checkpoint identities. The chain image/runner
must retain recurrence in guest memory; do not reboot once per decode/denoise
step or report one stage as a complete model. A multi-program Linux workload is
also possible when the administrator supplies its exact guest runtime identity.
The expected chains are ResNet50's one step, TinyLlama's 33 steps and SmolVLA's
12 steps, totaling 133 declared output arrays. The prepare command derives the
per-step shape/dtype roster from the shared protocol.

The administrator's registration JSON binds `phase`, `model`, `workload`,
`compiler_sha256`, `contract_sha256`, exact `assets` and `agfi`. Store its path and
SHA256 in each model's `registration`; a workload string alone is insufficient.
The manager contract must additionally retain document/version, manager wrapper,
Chipyard/FireSim/build/driver, hardware/source correspondence, DTS/address map,
memory capacity and target-clock identities. An AGFI/name match alone does not
prove correspondence to the facts used by Spike/gSIM.

## Manager output contract

Each result prefix retains byte-exact `uartlog`, `HW_CFG_SUMMARY`, `sim-run.sh`,
the original SSM invocation and `result_manifest.json`. The manifest schema is
`merlin.post_freeze.firesim.manager_result.v1` and includes:

- `phase`, `model`, `workload`, `compiler_sha256`, `contract_sha256`,
  `request_sha256`, `agfi` and `ssm_command_id`.
- The original requested `assets`, plus installed/consumed artifact identities
  and their actual manager locations in additional provenance fields.
- `completed_chain_steps`, `output_array_count` and `outputs` in the exact
  request roster order. Each output records `stage`, zero-based `step` and
  `index`, `dtype`, `shape`, relative `file` and raw-file `sha256`.
- Separate guest target-cycle windows for initialization, full chain and model
  stages; measured compute must finish before output publication. Retain raw
  counters and their clock domain. Record simulator wall time separately.
- Run-farm/manager teardown outcome on success, failure, timeout and cancellation.

Large outputs should be written to collected guest/host result files rather than
expanded into UART text. All array bytes still reach the existing exact model
checker. Prefix-only outputs, top-k agreement or a bare `*** PASSED ***` cannot
replace all-array comparison. Record per-program/per-step diagnostics through
the shared numerical policy without adding new numerical thresholds.

The helper currently prints only the last 40 SSM stdout lines, defaults polling
errors to pending, and has no workhorse-side teardown trap. Its normal shell exit
also does not encode the SSM terminal verdict. Before unattended full-model use,
the administrator should retain full SSM output, propagate terminal failure,
bound polling and guarantee manager cleanup alongside the manager's run-farm
trap/guardrail. A workhorse timeout must not be reported as verified teardown.

## Import and qualify

Download only the exact run prefix, preserving S3 object/version/checksum
metadata. Keep the original `get-command-invocation` JSON for the actual command.
Then import the result directory for one complete model:

```sh
python -m merlin_experiments.post_freeze.firesim collect \
  --request /absolute/request.json --result-dir /absolute/model-result \
  --ssm-invocation /absolute/ssm-invocation.json --out /absolute/observation.json
```

The importer reopens the frozen compiler, freeze record, selection, launcher,
contract and assets; checks manager/SSM/AGFI/request correspondence and every
declared output-file hash; and retains failures explicitly. It is an observation
boundary. `model_validation_complete` remains false: existing model-check
engines still own full quantized-reference comparison, source/placement and
final-ELF instruction audits, and independent exact accelerator checks. This
transport observation is not a legacy queue-owned FireSim receipt; the old queue
verifier remains unchanged. A final post-freeze report combines separately
verified transport and numerical/placement results rather than relabeling one.

The stored initial smoke printed `Host Frequency: 110.001 MHz` and an effective
emulation rate around 109.7 MHz. Those are FPGA execution-rate observations.
They do not change the selected architectural target clock or prove equivalent
memory timing. Compare raw target-cycle windows only after source/memory/clock
correspondence has been verified; convert to target latency using the selected
target domain, not the host MHz or the simulator's wall-clock throughput.
