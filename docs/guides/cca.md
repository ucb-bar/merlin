---
title: Understand CCA and follow a compiler improvement
kind: guide
status: current
owner: kernels
last_verified: 2026-11-09
related: [getting_started, kernel_reconstruction, beam_search, beam_cca_architecture, architecture]
code_refs:
  - src/merlin/kernels/cca.py
  - src/merlin/kernels/cca_compare.py
  - src/merlin/kernels/cca_contract.py
  - src/merlin/kernels/action_catalog.py
  - src/merlin/kernels/search_step.py
  - src/merlin/mining/fork_from_action.py
  - src/merlin/mining/beam.py
  - src/merlin/experiments/phase1/brokers/cca.py
  - src/merlin/experiments/phase2/component_cca.py
  - examples/tutorials/05_cca.ipynb
---

# Understand CCA and follow a compiler improvement

**Tutorial 5 of 5** | [Start here](../../examples/tutorials/README.md) | [Previous: Transformations](transformations.md)

Open the [CCA notebook](../../examples/tutorials/05_cca.ipynb) with the
[tutorial launcher](../../examples/tutorials/README.md). It compares two short
instruction listings and shows the proposed edit beside each difference.

**CCA means Common Compute Abstraction.** It gives code properties common names:
contraction form, accumulator type, register grouping, operand reuse, dispatches
and more. Merlin can then ask what a reference implementation does differently
from its own emitted code.

For example, the notebook reports:

| Property | Reference | Candidate | Proposed edit |
| --- | --- | --- | --- |
| `compute.contraction_form` | `fused_fma` | `mul_add` | Registered FMA pass |
| `vector.lmul` | 2 | 1 | Register-group knob |

FMA means fused multiply-add. LMUL is RVV's vector register-group multiplier.
The action catalog supplies these edits and their source locations; the notebook
adds no ranking policy. Changing separate floating multiply/add to FMA can change
rounding, so the edit must still satisfy the program's numerical contract.

## Follow the flow

```text
reference and candidate artifacts
    -> decode and lift
    -> compare CCA properties and report unknowns
    -> propose a compiler edit
    -> rebuild and inspect the emitted candidate
    -> check outputs and measure execution
```

A CCA comparison helps choose what to investigate. Its inputs matter: a schedule
record describes a schedule, a disassembly lift describes code, and an execution
record describes what ran. Matching CCA properties does not replace output checks,
placement checks or timing measurements. The notebook uses authored text so you
can learn this flow without hardware.

## Read the record

`CCA` contains an operation, backend identities, optional **facets** (groups of
properties), a scope and provenance. Call `to_dict()` to inspect it. `None` means
the observation is unavailable; it is different from zero or false.

| Facets | What to look for |
| --- | --- |
| `compute` | Contraction, accumulator, reduction and epilogue choices |
| `vector`, `spatial`, `simt` | Lane, array or threaded-engine properties |
| `memory`, `layout` | Access patterns, reuse and operand representation |
| `dispatch`, `envelope` | Commands and calls around the computation |
| `communication`, `coverage` | Boundary movement and whole-program coverage |
| `operand_fields`, `instructions` | Recovered instruction fields and partial ISA observations |

The scope is `kernel`, `dispatch` or `program`. A kernel is one computation region;
a dispatch includes its launch boundary; a program is the complete model/program.
`compare` refuses different scopes. `scope_problems()` catches inconsistent records,
such as whole-program coverage attached to a kernel. Callers also need compatible
operations, precision, target and observation boundaries.

## Handle missing observations

The current APIs answer slightly different questions:

- `compare` reports differing populated fields. Its legacy register-block rule
  treats an absent block as MR=1 when looking for greater reference row reuse.
  That special case is not an independently observed block.
- `uncomparable_axes` lists fields missing on one side. It does not list everything
  missing on both sides.
- `compare_evidence` requires per-axis observed/inferred witnesses and compatible
  scopes. Residency and relevant command counts need matching axis boundaries.
  Its report includes unknowns and the number of comparable axes.

The notebook's strict result is `unknown`, with denominator zero: its vector lift
has no per-axis witness records. A result of `compared` would mean there were
comparable axes, not that they were equal. Lifter provenance labels describe the
observation method; trusted artifact/source binding is a separate caller obligation.

## Find the right owner

| Task | API / source |
| --- | --- |
| Lift saved vector disassembly | `decode.rvv.decode_text` -> `cca.lift_asm` |
| Cross-check source and assembly reconstruction | `cca.lift_source`, `cca.cca_agree` |
| Inspect a Merlin schedule | `cca.lift_schedule` |
| Compare | `cca_compare.py` |
| Find proposed edits and their source files | `action_catalog.build_catalog`, `seam_location` |
| Check a proposal's promise against emitted code | `achieved_residual`, `search_step.make_step` |

`build_catalog` returns routed actions and unrouted differences. An action can name
an existing feature or a work item. `cca_contract.check_bijection(backend)` checks
that schema fields and compiler routes stay consistent, with known-open gaps
reported separately. It checks the vocabulary, not the implementation's correctness.

After a real edit, rebuild and lift the candidate again. `achieved_residual` lists
promised properties that are still missing or different. An empty list can also
mean the action had no checkable promise; `make_step` distinguishes that case.
Correctness and measured speedup reach that record through the evaluator's inputs.

## Where agents use CCA

- **Mining beam:** `mining.beam.run_beam` uses the CCA proposer when given an expert
  CCA or comparison callback. `fork_from_action` turns routed actions into supported
  overrides or work items. The beam certifies builds, re-lifts emitted objdump and
  can escalate unmet promises. See [beam search](beam_search.md) for that workflow.
- **Phase 1 assisted authoring:** `brokers/cca.py` answers public schema/action and
  escalation-ladder queries. It reads no golden outputs and issues no runtime verdict.
- **Explicit Phase 2 component feedback:** `component_cca.py` binds observations to
  selected baseline/candidate artifacts, inputs and independent runtime/provider
  identities before reporting structural differences and unknowns.

Continue with [kernel reconstruction](kernel_reconstruction.md) for the richer
evidence-to-schedule workflow, or return to the [tutorial catalog](../../examples/tutorials/README.md).
