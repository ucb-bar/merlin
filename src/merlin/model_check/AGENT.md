# AGENT.md — src/merlin/model_check

## Purpose

Build, run and judge complete quantized models for a device submission.

## Modules

- `__main__.py` — ``python -m merlin.model_check``.
- `admission.py` — model_check's own linked-image admission for whole-model builds (``check --mode full``).
- `program_admission.py` — Host-only bridge from declared final ELF program sections to the existing RTL/software-bound finalizer, preserving the original admission policy.
- `approximations.py` — A submission's declared approximate kernels (``manifest.yaml`` key ``numerics.approximations``).
- `baseline_numerics.py` — Offline supplied-output comparison; no execution, placement or reference-origin certificate.
- `binding.py` — The Spike / RTL / FPGA binding receipt (contract §7): is the functional model the machine the FPGA runs?
- `binding_sources.py` — Structural readers behind the binding receipt: C parameter headers, device trees, ELF load segments and CDE configuration sources (Chisel generators). Every value is parsed, never pattern-matched.
- `bundle.py` — A numerics bundle on disk: per model, the program catalogs, the census and every input's device calls.
- `cache.py` — Kernel objects and replay results reused across checks, keyed by every input that produced them.
- `catalog.py` — Everything about one captured program that does not depend on a submission, computed once.
- `chain.py` — Run a model's programs as a chain on one input, following the models file's input bindings.
- `chain_worker.py` — One worker of a bundle's INT8 reference pool: ``python -m merlin.model_check.chain_worker <request> <k>``.
- `chain_image.py` — Package retained completed programs into one recurrent guest without candidate recompilation.
- `chain_image_objects.py` — Reopen original recipes and observe symbol-only object namespaces.
- `chain_image_sections.py` — Compare actual allocated ELF object sections and argument symbol bytes.
- `chain_image_abi.py` — Prove retained table layout under the selected header and rebuilt descriptor runtime.
- `chain_image_plan.py` — Derive every input/carry/teacher-forced binding from selected models and capture ABIs.
- `chain_image_data.py` — Emit typed guest tables and byte-crossjoined sealed all-output references.
- `chain_image_layout.py` — Extend selected split-weights layouts and retain original runtime/toolchain flags.
- `chain_image_boot.py` — Reopen actual entry and host mailboxes after the complete-chain link.
- `chain_image_readback.py` — Decode explicit raw-array or complete all-element guest comparison transport.
- `chain_group_profile.py` — Reconcile per-invocation routed group/gap traces without hiding dropped calls.
- `chain_image_profile.py` — Add a reset hook to retained Merlin profiling support with explicit source/object receipts.
- `cli.py` — Command line of ``python -m merlin.model_check`` (contract: ``docs/reference/model_check.md`` §2).
- `commands.py` — The subcommands behind ``python -m merlin.model_check`` (contract: ``docs/reference/model_check.md``).
- `compute_execution.py` — Share the selected compute CPUs across model compilation and simulation steps.
- `diagnostics.py` — Localize a wrong kernel result: the failing call re-runs in a second image that compares on the device.
- `engines.py` — Run replay images on the RTL-equivalent native simulator (gSIM) of the selected FIRRTL, beside Spike.
- `export.py` — What the package is asked to lower, per model: interface modules, call ABI and stated entries, no data.
- `full.py` — ``check --mode full``: every program's linked image, each input's whole chain on Spike, judged.
- `full_build.py` — One program's linked whole-model image: built once per submission, relinked per input, run on Spike.
- `full_build_identity.py` — Content correspondence for reusable whole-model builds, separate from qualification.
- `full_build_inputs.py` — Reuse immutable single-forward C artifacts while rebinding typed runtime inputs.
- `full_build_native.py` — Import an already observed whole-model build without recompiling its objects.
- `full_chain_validation.py` — Pure completeness predicate for selected full-chain quantized-reference reports.
- `full_output_audit.py` — Offline comparison of every declared full-model output, including unjudged intermediates.
- `full_quantized.py` — Explicit full-chain comparison against sealed quantized outputs, without CPU host twins.
- `full_reference_cache.py` — Byte-bound per-step references and structural cached-twin readiness for full model checks.
- `input_worker.py` — Model-side input preparation, run INSIDE a model's own pinned environment (``python input_worker.py``).
- `inputs.py` — Evaluation and calibration inputs: deterministic selection from pinned sources, small stored forms.
- `kernel_reference.py` — The expected result of one device-kernel call: the group's interface program evaluated in numpy.
- `metrics.py` — The numerics policy's metrics and the ratio rule (contract §4).
- `models.py` — The models file: which programs make up each model, how they chain, and what is judged.
- `numerics_report.py` — Offline section numerics from retained all-output audits; no model or simulator execution.
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

- Chain image packaging supplies no new numerical, instruction or placement verdict.
  All original graph/device/weights objects and transformed allocated sections remain pinned.
  Only the driver, ABI probes and common descriptor/arena runtime are compiled anew
  by default. Explicit group profiling may rebuild Merlin's generated profiling
  support from its pinned original C source/ABI flags to add a recurrent reset;
  retain the original/new source and object pins and actual compiler invocation.
  Old recipes without recorded headers use observed full-table layout proof, never an invented
  historical header identity. The selected memory map must admit every LOAD and live arena.
- Compact guest comparison is explicit and checks every declared element at every step using
  selected finite floating tolerances and exact integers/booleans. Hashes and bounded samples are
  diagnostics, never numerical surrogates. Native placement and consumed-image/hardware gates
  remain independent. References embedded in private guest images must remain protected.

- The contract is `docs/reference/model_check.md`; a change to the CLI, the numerics policy or the report
  bumps its version and changelog.
- Nothing here names a model or a target: models, programs, chains and judged outputs come from
  `examples/<target>/numerics/models.yaml`, the target from the selected target contract.
- This package never writes or edits device-kernel code and never modifies a submission.
- Sealed inputs and every frozen bundle live under `/private/`; nothing here copies them into
  a path an agent sandbox can be granted.
- Whole-model build reuse reopens observed recipe/ELF identities and keys the original capture,
  compiler/runtime source and direct selected tools. Recurrent input templates retain the build's
  static weights; the caller's sealed-capture lifetime remains required. Runtime input rebinding
  records each actual driver compilation and final link without supplying device implementations.

`program_admission.service` requires an original prerequisite admission, current
selected RTL/software contracts, a declared loaded section and independently
qualified decoder source selection. It never discovers a target implementation,
exports protected findings or approves segmentation without rewriting the image
and launcher. Host build/cache/relink honors explicit admission; a saved verdict
never replaces current evaluation. Source pins establish attribution, not
runtime/physical/numerical/timing qualification or complete program coverage.

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
