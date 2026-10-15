# AGENT.md — merlin/python/merlin/model_check

## Purpose

Build, run and judge complete quantized models for a device submission.

## Modules

- `cli.py` — Command line of ``python -m merlin.model_check`` (contract: ``docs/reference/model_check.md`` §2).
- `kernel_reference.py` — The expected result of one device-kernel call: the group's interface program evaluated in numpy.
- `models.py` — The models file: which programs make up each model, how they chain, and what is judged.
- `report.py` — The ``model_check_report`` document: construction, agent view, validation and exit codes.

<!-- Purpose/Modules derived from docstrings via build_tools/scripts/gen_package_docs.py.
     Add hand-written notes (invariants, gotchas) below. -->

## Invariants

- The contract is `docs/reference/model_check.md`; a change to the CLI, the numerics policy or the report
  bumps its version and changelog.
- Nothing here names a model or a target: models, programs, chains and judged outputs come from
  `examples/<target>/numerics/models.yaml`, the target from the selected target contract.
- This package never writes or edits device-kernel code and never modifies a submission.
- Sealed inputs and every frozen bundle live under `/work/merlin/private/`; nothing here copies them into
  a path an agent sandbox can be granted.
