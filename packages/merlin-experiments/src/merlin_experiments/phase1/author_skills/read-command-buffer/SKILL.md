---
name: read-command-buffer
description: Find out where a test case's cycles and bytes go by reading its emitted command buffer. Open it when a performance brief entry carries an `inspect` command, when it names a busy `dominant_modeled_resource` or a `bound_by` limit, or when a change did not move the cycles you expected. Covers `isa_tools.py debug`, `disasm` and `lint`, a script that totals command classes, counts and bytes, and how to attribute the work to accumulator readout, configuration, transfers and compute.
---

# Read a command buffer

Each passing case in a self-check response's `performance_brief` has the cycles, the `bound_by`
roofline limit, the `dominant_modeled_resource` and, for public roster cases, the exact command to
inspect it. Use these to find which commands do the work before you change the schedule.

## 1. Collect the evidence for one case

```
python isa_tools.py debug selfcheck_out/CASE/command_buffer.json --capsule CASE > debug.json
python isa_tools.py lint  selfcheck_out/CASE/lowered.llvm.mlir > lint.json
python isa_tools.py disasm selfcheck_out/CASE/lowered.llvm.mlir > disasm.json
```
- `debug` runs your command buffer on the RTL-derived model. It returns `per_command` (one row per
  readout with its shape, `cycles`, `spad_reads`, `acc_writes` and `refill_rows`), the totals in
  `metrics` (`cycles`, `bytes_read`/`bytes_written`/`bytes_moved`, `accumulator_commits`,
  `evictions`) and the RTL `oracle` fingerprint. Output values are withheld. `--timeout S` bounds
  the run. `--run-to N`, `--region BASE:NBYTES` and `--state` apply to self-hosted-ISA targets.
- `lint` gives the class histogram and `capability_utilization`: the instructions your target declares
  that the program never emits. Instructions whose role the task's policy prohibits stay unused by design.
- `disasm` gives every emitted instruction with its decoded class and funct.
If `debug` answers with an error (for example, the model is unavailable), the trace and the
self-check counters below still attribute the work.

## 2. Summarize

```
python scripts/summarize_command_buffer.py selfcheck_out/CASE/command_buffer.json   # + its instruction_trace.json
python scripts/summarize_command_buffer.py debug.json
python scripts/summarize_command_buffer.py RESPONSE.json --case CASE   # dynamic counts and bytes
```
The script gets classes from the decode the runner wrote and checks them against your target's
`funct_decode_table` in `rtl_facts/facts.json`. It reads class roles from any list of class names
in `target_contract.yaml`, and dynamic counts from the timing model's `target.class.*` and
`*_bytes` counters. Trace counts are static sites. Multiply them by the counted-loop trip counts,
or read the response's dynamic counts.

## 3. Attribute the work

Put the commands into four groups, using the roles and counts above:
- **Accumulator readout**: commands whose role reads out or commits the accumulator, the `per_command`
  readout rows, and `accelerator.storage.accumulator*` activity. Many small readouts, each followed
  by a drain, cost time that overlap or frontier readout can recover.
- **Configuration**: setup commands. If configuration is issued per tile inside a loop while its
  values stay the same, it is a candidate for removal or hoisting.
- **Transfers**: load and store commands and their bytes. Compare requested bytes with the source
  tensors' bytes (`tensor_bytes`). Bytes above that mean operands are re-fetched. A nonzero
  `refill_rows` means the operand footprint spills local storage.
- **Compute**: compute commands against useful MACs and `array_utilization_pct`.
`per_compute_command` normalizes every class by the compute count. A staging class at 1.0 per
compute means an operand is re-staged for every compute. Loads well above one tile per output
tile mean reuse is missing.

## 4. Connect to mechanisms

Map the busy resource or `bound_by` to catalog facets with the table in the `pass-library` skill
(storage busy: storage/movement/overlap; transfers: movement/overlap/storage; issue:
issue/synchronization/scheduling; execute: scheduling/overlap/fusion; `requests` bound:
movement/overlap/storage; `compute` bound: scheduling/issue/fusion). Open that facet's reference.
Choose a mechanism whose adapter obligations your backend can meet (`scheduling-api` shows the
calls). Then rerun the same case and compare the class counts, bytes and cycles before you keep
the change.
