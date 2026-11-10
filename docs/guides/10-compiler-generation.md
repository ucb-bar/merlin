---
title: "10. Generate a target compiler"
kind: guide
status: current
owner: experiments
last_verified: 2026-11-09
related: [09-verification, 11-optimization, 12-experiments]
code_refs: [src/merlin/experiments/phase0/generation.py, src/merlin/experiments/phase1/controller.py, src/merlin/experiments/phase1/component_origin.py, src/merlin/experiments/phase1/task_staging.py, src/merlin/targetgen/compiler_library.py, src/merlin/compiler/passes, src/merlin/compiler/transforms, src/merlin/compiler/schedule, src/merlin/evaluation/capsule, examples/gemmini/experiment.yaml]
---

# 10. Generate a target compiler

[Previous](09-verification.md) | [Guide index](../README.md) | [Next](11-optimization.md)

Compiler generation builds a backend for a target; ordinary compilation uses an
existing backend to compile a program. Merlin's generation workflow gives an agent
the target description, public tests and permitted compiler tools. The agent
authors an out-of-tree compiler, then independent evaluators check its output.

Start by inspecting a configured example. These commands do not launch an agent:

```sh
merlin experiment inspect gemmini-functional --phase 0
merlin experiment inspect gemmini-functional --phase 1
merlin experiment levels
```

The definition is currently at
[`examples/gemmini/experiment.yaml`](../../examples/gemmini/experiment.yaml).
It selects inputs and phase adapters; it is not a self-contained hardware setup
or a certificate that a compiler works. See [experiments](12-experiments.md) for
catalogs, definitions and run records.

## 1. Describe the target and its arithmetic

Keep three inputs distinct:

| Input | What it tells Merlin |
| --- | --- |
| Software specification | The operation forms and numerical behavior to preserve: types, accumulation, rounding, overflow, layouts and composition. |
| Hardware specification and extracted facts | Which hardware source was selected and the observed instructions, resources and constraints. |
| Backend capability declaration | Which forms a particular implementation claims to support. |

The target example's `target/` and `platform/` folders select these inputs.
Facts and tool outputs belong to the selected run, rather than being regenerated
inside the authored example. A backend capability claim cannot change the
software specification's arithmetic or the evaluator's acceptance policy.

Try the [contracts and facts notebook](../../examples/tutorials/01_contracts.ipynb)
before using a real RTL extraction. It shows the input relationships with a small
example that needs no hardware installation.

## 2. Generate tests from those requirements

Phase 0 turns selected operation and effect requirements into test programs.
Finite shape domains, numerical cases, resource boundaries and composition cases
serve different purposes; a list of matmul sizes cannot stand in for all of them.
The generator records missing or unavailable requirements explicitly.

The [test-generation notebook](../../examples/tutorials/02_test_generation.ipynb)
calls the real coverage algorithm and corpus builder. Change its coverage budget
to see generated points, missing combinations and interface MLIR.

For a real target, Phase 0 additionally binds the selected software, numerical and
hardware inputs, chooses the matching reference engines, and writes complete
capsules. A capsule is one test declaration together with its program, reference
outputs and expected execution properties. Development, functional guards and
withheld transfer tests have separate roles.

Run Phase 0 separately, review its result, then select a sealed corpus for Phase 1.
The current catalog workflow does not silently promote freshly generated tests
into the next phase. It refuses a combined invocation that would bypass this
handoff. The Gemmini example's default diagnostic mode retains unresolved evidence.

## 3. Let the agent build the compiler

Phase 1 stages the selected task, input bundle, tool permissions and runtime
interfaces before starting the author. The fresh compiler route starts from an
inert entrypoint and manifest. Historical or handwritten backends are separate
reference inputs, not substitutes for a newly generated compiler.

The agent can **compose existing passes, modify permitted passes, or add passes**.
The run's compiler-library selection determines which shared implementation bytes
and tools are available. An agent does not automatically receive every repository
module or permission to modify the grader.

| Shared implementation | Where to start under `src/merlin/` |
| --- | --- |
| Pass invocation and pipeline composition | `compiler/passes/` |
| Reusable IR transformations | `compiler/transforms/` |
| Shared scheduling and storage choices | `compiler/schedule/` and `sched/` |
| Named optimization strategies | `compiler/optimizations/` |
| Selected author-visible library contract | `targetgen/compiler_library.py` |

These mechanisms include tiling, operand retention, transfer merging and command
ordering. Their applicability depends on the selected numerical contract,
resource descriptions and command effects. The backend supplies target-specific
instruction selection, encoders, layouts and ABI details.

Use the [scheduling](../../examples/tutorials/03_scheduling.ipynb) and
[transformation](../../examples/tutorials/04_transformations.ipynb) notebooks to
inspect the shared mechanisms before changing a pipeline. Keep the generic
algorithm in Merlin and target facts or implementations in the backend.

## 4. Keep the generated backend out of tree

The candidate is its own compiler package, with its own dialect or lowering,
passes and invocation interface. The selected adapter transports the input and
returns the declared artifacts. Those artifacts can be IR, objects, command data
or device binaries; device code does not have to pass through LLVM.

The artifact's format, ABI and runtime requirements still have to match its
consumer. Producing a binary is different from integrating it into a callable
region and executing that region. Inspect the selected backend's actual route
before assuming it provides all three.

Preserve the candidate source and actual build products with the run. Compiler
history, source selection and evaluation records must refer to the implementation
that produced the artifact, rather than whichever checkout is current later.

## 5. Evaluate the result independently

The author can improve its compiler; it cannot choose its own accuracy threshold
or change the reference that accepts it. Check the numerical program, generated
code and observed execution separately. For quantized models, first distinguish
the quantized program from the original floating model.

Correct outputs alone do not prove that an accelerator region executed. Required
placement, instruction use, ABI, memory and synchronization checks also need their
own evidence. A successful scalar fallback cannot satisfy a required device path.
See [verification](09-verification.md) for those responsibilities.

Once the reviewed corpus and input bundle are available, inspect preflight before
launching the selected agent:

```sh
merlin experiment preflight gemmini-functional --phase 1 \
  --corpus-seal /absolute/reviewed/corpus-seal.json \
  --bundle-manifest /absolute/reviewed/input_bundle_manifest.yaml
```

The corresponding `run` command uses those same selections. It also needs the
definition's toolchains, independent runtime/oracles, agent credentials and budgets.
Preflight checks configuration and inputs; it does not perform the compiler
experiment. Continue to [optimization](11-optimization.md) once a functional
compiler and its required evidence have been frozen.

[Back to verification](09-verification.md) | [Next: optimization](11-optimization.md) |
[Guide index](../README.md)
