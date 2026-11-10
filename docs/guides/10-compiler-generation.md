---
title: 10. Generate a backend
kind: guide
status: current
owner: development
last_verified: 2026-11-09
related: [09-verification, 11-optimization, 12-experiments]
code_refs: [src/merlin/development/testgen, src/merlin/development/backend, src/merlin/experiments/adapters.py, src/merlin/compiler/schedule, src/merlin/evaluation/capsule, examples/compiler-generation]
---

# 10. Generate a backend

[Previous](09-verification.md) | [Guide index](../README.md) | [Next](11-optimization.md)

Ordinary compilation uses an existing backend. Backend generation gives an agent
the selected target specification, reviewed tests and compiler tools, then checks
the compiler it writes. Correctness and performance development share one native
backend-generation session.

There are two workflows:

| Workflow | Produces | Source owner |
| --- | --- | --- |
| Test generation | Programs, references, coverage and a reviewable corpus | `development/testgen/` |
| Backend generation | An OOT compiler, revisions and independent correctness/performance evidence | `development/backend/` |

Start with the [target examples](../../examples/compiler-generation/README.md)
and [experiment commands](12-experiments.md). The examples are configuration
templates, not ready-made hardware installations or qualified compilers.

## Describe the target

The software specification declares the arithmetic and supported operation forms.
Hardware selection identifies the RTL, configuration and extracted facts.
A compiler capability declaration says what that implementation claims to support.
These are different inputs; an implementation claim cannot redefine the hardware
or loosen the evaluator's numerical policy.

Try [notebook 1](../../examples/tutorials/01_contracts.ipynb) to inspect these
relationships without installing hardware tools. Real extraction uses the selected
ModeLIR/CIRCT readers and records their versions, observations and unknowns.

## Generate and review tests

Test generation derives programs from the selected operation and effect
requirements. Shape combinations, numerical boundaries, memory constraints and
composition cases each need their own coverage. Missing requirements stay visible.

[Notebook 2](../../examples/tutorials/02_test_generation.ipynb) calls the actual
coverage algorithm and corpus builder. It generates small example programs;
a real release additionally needs its approved specifications, source inputs,
references and any required RTL execution evidence.

A capsule combines a test declaration with its program and expected behavior.
Generate capsules under a run/artifact directory, review them, then select the
sealed release and its input bundle for backend generation. Public development
inputs and withheld evaluation inputs have different access roles.

Generated corpora and input bundles are not committed fixtures. Authored program
sources that cannot be reconstructed from those declarations live under
[`examples/workloads/`](../../examples/workloads/README.md). Descriptions and other
derived views are generated before the output corpus is sealed.

## Develop correctness and performance together

The native backend engine stages the task, selected input bundle, tool permissions
and runtime interfaces. A fresh backend starts from an inert entry point and
manifest. Handwritten reference compilers are independent inputs, not solutions
to be copied into the candidate.

The selected joint policy controls the continuous session and measurement windows.
Passing functional checks does not terminate the joint session: the agent can
continue improving the compiler under the same declared budgets and independent
evaluation. The operator supplies the reviewed policy and timing authority.

The agent can compose existing passes, modify them or add a transformation.
Shared IR rewrites, tiling, operand retention, transfer merging and command
ordering live in Merlin's libraries. The OOT backend supplies resource descriptions,
command effects, instruction selection and encoders. See
[transformations](05-transformations.md) and [optimization/CCA](11-optimization.md).

## Keep implementation and evaluation separate

The candidate emits an admitted artifact: compatible IR, objects, device binaries
or commands. LLVM is one route, not a required output format for every backend.
The selected wrapper/runtime must explain how that artifact executes.

Independent evaluators check numerical behavior, source/ABI requirements, memory
and completion, and actual execution where required. A successful scalar fallback
cannot count as observed vector or accelerator execution. Missing hardware evidence
is unavailable, not a pass. [Verification](09-verification.md) separates these claims.

The source ownership is:

| Work | Open first |
| --- | --- |
| Test construction and source declarations | [`development/testgen/generation.py`](../../src/merlin/development/testgen/generation.py) |
| Native authoring lifecycle | [`development/backend/authoring.py`](../../src/merlin/development/backend/authoring.py) |
| Joint policy | [`development/backend/joint_control.py`](../../src/merlin/development/backend/joint_control.py) |
| Agent task and permissions | [`development/backend/task_staging.py`](../../src/merlin/development/backend/task_staging.py) |
| Specialized optimization mechanisms | [`development/backend/optimization/`](../../src/merlin/development/backend/optimization/) |
| Independent capsule assessment | [`evaluation/capsule/`](../../src/merlin/evaluation/capsule/) |

Older records retain their schema identifiers and recorded paths for exact replay.
They do not define separate stages in the current workflow.
