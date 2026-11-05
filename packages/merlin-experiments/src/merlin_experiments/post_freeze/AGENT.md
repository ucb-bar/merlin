# AGENT.md — packages/merlin-experiments/src/merlin_experiments/post_freeze

## Purpose

Operator preparation and evidence intake after compiler freezing.

## Modules

- `booted_elf.py` — Import explicit lightweight boot-file observations without custody qualification.
- `capsule_firesim.py` — Original capsule ELF preparation and exact FireSim UART replay for native L5.
- `chain_observation.py` — Observe compact guest transport without certifying execution or numerics.
- `compact_chain.py` — Explicit compact all-element chain transport, separate from full raw arrays.
- `elf_custody.py` — Observe versioned downloads and actual simulator boot files without running hardware.
- `elf_upload.py` — Upload frozen ELF/request bytes and prepare, but never send, an SSM command.
- `firesim.py` — Prepare frozen ELF/workload requests and import remote FireSim observations.
- `firesim_cli.py` — Command-line preparation and collection of frozen FireSim observations.
- `firesim_upload.py` — Reopen uploaded complete-chain ELF bytes and strict manager custody records.
- `joint_firesim.py` — Frozen joint-campaign coverage and dispatch through the installed ELF wrapper.
- `manager_result.py` — Publish and reopen manager ELF custody observations after a frozen model run.
- `runtime_driver.py` — Observe the actual runtime-host driver argv around one post-freeze execution.

<!-- Purpose/Modules derived from docstrings via build_tools/scripts/gen_package_docs.py.
     Add hand-written notes (invariants, gotchas) below. -->
