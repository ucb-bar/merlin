---
title: Compile debugging — stop at a stage, dump its IR, trace a build, inspect one group
kind: guide
status: current
owner: compiler
last_verified: 2026-10-06
related: [optional_passes, model_lowering, whole_model_on_accelerator]
code_refs: [src/merlin/common/compile_trace.py,
            src/merlin/perf/debug_companion.py,
            src/merlin/perf/whole_model_group_timing.py,
            src/merlin/compile/debug.py,
            src/merlin/compile/command.py,
            src/merlin/llvmlower/pipeline.py,
            src/merlin/perf/whole_model_build_cli.py,
            src/merlin/perf/whole_model_partial.py,
            packages/merlin-experiments/src/merlin_experiments/group_inspect.py]
---

# Compile debugging

Every lowering route already names its stages where it records them: the staged xDSL route
(`input → contract → schedule → interface → target → runtime`), the LLVM route (`input`, the Merlin
xDSL rewrites, `upstream`, every native MLIR pass, `llvm-final`), codegen (`object`, `link`) and the
whole-model builder (`statement`, `group_objects`, …, `program_build`). The same options reach all of
them, so you can stop a compile at any stage, dump the IR there, keep a trace of the whole descent, or
rebuild one group of a candidate and look at it.

## List the stages

```bash
merlin-compile --list-stages            # grouped by route
merlin-compile --list-stages --json     # {stage, pipeline, entry, summary} rows
python -m merlin.perf.whole_model_build --list-stages
```

The list is read off the pipelines, never kept separately. Each pipeline declares its stages beside
the code that records them (`merlin.common.compile_trace.declare`). The native passes come from the
pass pipelines the LLVM route builds (serial, OpenMP and vectorizing), with your `--pass`/`--no-pass`
selection applied. A native pass stage is `mlir:<pass>`. `STAGE#N` names the Nth time a stage is
reached, so `mlir:canonicalize#2` is the second canonicalize.

## Stop, dump, trace

```bash
merlin-compile --workload tiny_llama --dtype int8 --run none \
    --stop-after mlir:one-shot-bufferize --trace-dir out/artifacts/probes/compile-trace/my-run

merlin-compile --workload tiny_llama --run none --dump-ir-after all --dump-ir-before llvm-final
```

- `--stop-after S` writes the IR at `S` and ends the compile there. It exits 0 and prints where the IR
  is. It never writes a final artifact (no object, no binary, no result marked `compiled`). A stop at
  a native pass removes the pass manager's output, because that is the IR after the last pass. If
  the route never reaches `S` (for example `contract` on the LLVM route), the compile runs to the end,
  reports `stop_stage_not_reached`, and exits 1.
- `--dump-ir-after S` / `--dump-ir-before S` are repeatable and comma-separated, and `all` selects
  every stage. "Before" a Merlin stage is the IR the previous stage left. Before a native pass, it is
  the printer's own before-dump.
- `--trace-dir DIR` must be a new directory. If you give a dump or stop option without it, the trace
  goes under `out/artifacts/probes/compile-trace/<target>/<timestamp>_<sha>_<workload>/`.

The request travels in the environment (`MERLIN_COMPILE_TRACE`), so a stage that a child process
reaches is dumped too. A package entrypoint that lowers through Merlin is one example; its dumps go to
`ir/p<pid>/`. Only the process that opened the trace stops. When a child reaches the stop stage, it
records that and carries on, and the build stops at its own next stage boundary. A child that exited
early would otherwise read as a failed group.

### What a trace directory holds

```
trace.json          every stage reached, in order: pipeline, occurrence, file(s), bytes, sha256,
                    seconds; the outcome (completed | stopped | failed | stop_stage_not_reached)
pipeline.txt        the same order as text, then each native pass-manager segment's pipeline
events.jsonl        the raw per-process events trace.json is built from
pass_log.jsonl      the xDSL pass-invocation log (MERLIN_PASS_LOG), unless you set your own
ir/NNN-<stage>.mlir|.ll           Merlin stages in this process (NNN-before-<stage> for --dump-ir-before)
ir/p<pid>/...                     the same, from a child process
mlir/call-NNN/passes/segment-NNNN/<op>/<n>_<pass>.mlir   native per-pass dumps, with pipeline.txt
mlir/call-NNN/native/             the parsed and lowered module views
products/NNN-<stage>/...          files a stage wrote (object, link, builder stages), hard-linked
```

## The whole-model builder

The builder's CLI takes the same options, plus `--only-group`:

```bash
python -m merlin.perf.whole_model_build build --package PKG --capsule CAPSULE --target T \
    --machine M --header H --phase0-recipe R --descriptor D \
    --only-group g12 --stop-after statement --trace-dir out/artifacts/probes/compile-trace/g12
```

A launch config spells the same options as `build_options` keys: `only_groups`, `trace_dir`,
`dump_ir_after`, `dump_ir_before` and `stop_after`. Each builder stage is a trace stage whose products
are the files it wrote. For `statement` those are each asked group's interface, command buffer and
target artifact.

`--only-group gN[,gM]` asks the package for those groups alone. Every other group keeps the target's
library call, as a caller-declined group does. That program runs and prints every group line, so the
build marks itself as partial in four places: `partial_build` on the record, on `oracle.json`, on the
expectations the service reads, and a `PARTIAL_BUILD.json` beside them. The whole-model verdict, the
UART grade, the whole-model gate and the measured-mode service each refuse a partial build. It is
never measured or graded as a whole model. When the service builder is asked to stop, that is a
refusal too, because a program is what its caller is owed.

## Inspect one group of a candidate

```bash
merlin experiment inspect <job-dir> --group g12 [--stage S] [--trace] [--run-to N]
merlin experiment inspect <package-dir> --group g12 --target T --build-options opts.yaml ...
```

`<job-dir>` is a measured-mode job directory. Its `job.json` names the target and build options, and
its `package/` is the measured snapshot. The command rebuilds that one group in a fresh directory
under `out/artifacts/cache/group-inspect/` (or `--out`). It uses the per-group program build: the
package is asked for that group only, and the group becomes a one-step program of the target's own
whole-model driver. The rebuild runs inside a trace that dumps every stage, and the command prints:

- who answered the group, and why if it was not the package;
- the group's products: interface, command buffer, target artifact, object, program source, ELF;
- `--stage S`: the IR of a stage the trace reached (for example `mlir:cse#2` from the package's own
  lowering), or of a product (`command_buffer`, `artifact`, `iface`);
- `--trace`: the group's program on the candidate's functional model (the `spike` machine its job
  declares), with the simulator's execution log stopped after `--run-to` instructions;
- with `--trace`, the program's **source-line attribution**. The group build links the program a
  second time from the same model, kernels and recipe with the compiler's debug option (`-g`) appended
  to its recorded flags. That companion is admitted only if its allocated bytes and relocations, of the
  image and of the program object, match the program's (`merlin.perf.debug_companion`). The functional
  model's PC histogram of the program is then symbolized against the companion by the
  `llvm-symbolizer` in the directory of the compiler the target's build recipe names, and attributed
  per function and source line (`source_attribution.json` in the work directory). The counts are
  instruction executions of the whole one-group program on the functional model, setup included: not
  cycles, and not a measured region. If any step cannot be taken (no functional model, no companion,
  a companion that differs, no symbolizer in the target's toolchain) the attribution is `UNKNOWN` and
  the command says why.

Each step uses a hook the target provides: its whole-model driver, and a functional-model machine.
If the target or candidate lacks the hook, you get "not available for this target" and the reason.
Inside a phase-1 sandbox, the equivalent instruction-level view of a capsule is
`isa_tools.py debug … --run-to N` (`merlin_experiments.phase1.tools.isa`).

## Related switches

- `--ir-audit` / `compile_core_mlir(ir_audit=…)` keeps exact-byte, hash-bound stage evidence. That is a
  different job: provenance of what was compiled, not a debugging view. With both on, the trace
  indexes the audit's native dumps rather than printing a second copy.
- `MERLIN_PASS_LOG` and `MERLIN_VERIFY_LOG` record xDSL pass invocations and verdicts. A trace turns on
  the first one by default.
