# AGENT.md — merlin/python/merlin/model_check

## Purpose

Build, run and judge complete quantized models for a device submission.

## Modules

- `admission.py` — model_check's own linked-image admission for whole-model builds (``check --mode full``).
- `approximations.py` — A submission's declared approximate kernels (``manifest.yaml`` key ``numerics.approximations``).
- `binding.py` — The Spike / RTL / FPGA binding receipt (contract §7): is the functional model the machine the FPGA runs?
- `binding_sources.py` — Structural readers behind the binding receipt: C parameter headers, device trees, ELF load segments and config sources.
- `bundle.py` — A numerics bundle on disk: per model, the program catalogs, the census and every input's device calls.
- `cache.py` — Kernel objects and replay results reused across checks, keyed by every input that produced them.
- `catalog.py` — Everything about one captured program that does not depend on a submission, computed once.
- `chain.py` — Run a model's programs as a chain on one input, following the models file's input bindings.
- `cli.py` — Command line of ``python -m merlin.model_check`` (contract: ``docs/reference/model_check.md`` §2).
- `commands.py` — The subcommands behind ``python -m merlin.model_check`` (contract: ``docs/reference/model_check.md``).
- `diagnostics.py` — Localize a wrong kernel result: the failing call re-runs in a second image that compares on the device.
- `export.py` — What the package is asked to lower, per model: interface modules, call ABI and stated entries, no data.
- `full.py` — ``check --mode full``: every program's linked image, each input's whole chain on Spike, judged.
- `full_build.py` — One program's linked whole-model image: built once per submission, relinked per input, run on Spike.
- `full_build_identity.py` — Content correspondence for reusable whole-model builds, separate from qualification.
- `full_build_inputs.py` — Reuse immutable single-forward C artifacts while rebinding typed runtime inputs.
- `full_quantized.py` — Explicit full-chain comparison against sealed quantized outputs, without CPU host twins.
- `full_reference_cache.py` — Byte-bound per-step references and structural cached-twin readiness for full model checks.
- `input_worker.py` — Model-side input preparation, run INSIDE a model's own pinned environment (``python input_worker.py``).
- `inputs.py` — Evaluation and calibration inputs: deterministic selection from pinned sources, small stored forms.
- `kernel_reference.py` — The expected result of one device-kernel call: the group's interface program evaluated in numpy.
- `metrics.py` — The numerics policy's metrics and the ratio rule (contract §4).
- `models.py` — The models file: which programs make up each model, how they chain, and what is judged.
- `reference_worker.py` — The FP reference of each model, run INSIDE the model's own pinned environment (``python reference_worker.py``).
- `references.py` — Per-input references of a bundle: FP outputs, the INT8 reference chain, its metrics, and device calls.
- `replay.py` — Replay device kernels on frozen boundary inputs: one small image per batch of calls, judged exactly.
- `report.py` — The ``model_check_report`` document: construction, agent view, validation and exit codes.
- `seal.py` — Seal a numerics bundle and verify it (contract §8): a sha256 of every file, checked before use.
- `twins.py` — The differential's host twins, run once per release: every step's outputs cached in the bundle.
- `variants.py` — Seeded shape-perturbed variants of every unique device kernel, stored and judged as sealed cases.

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
- Whole-model build reuse reopens observed recipe/ELF identities and keys the original capture,
  compiler/runtime source and direct selected tools. Recurrent input templates retain the build's
  static weights; the caller's sealed-capture lifetime remains required. Runtime input rebinding
  records each actual driver compilation and final link without supplying device implementations.

- Shape variants retain typed column readout: resize an explicit finite gain vector
  with the new output width by cycling its original column sequence. Keep the
  original readout form and recompute each variant's independent expected result;
  changing N must not leave an old-width vector that makes every draw refuse.
- Mandatory full checks stop admitting new builds at the first actual or cached
  kernel failure. Already admitted tools drain. Unattempted kernels/programs
  remain unfinished and partial, never cached as a compiler refusal or passed
  numerical evidence. Fast and explicit replay diagnostics retain aggregation.
- Direct quantized-reference checks compare completed judged outputs before executing a later step.
  A mismatch retains its numerical gate failure and actual partial counts, with no full-input FP
  metrics or chain credit. Recheck sealed reference bytes even on that early rejection.
