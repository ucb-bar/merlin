---
title: Explore scheduling decisions and command schedules
kind: guide
status: current
owner: compiler
last_verified: 2026-11-09
related: [core_dialects, shared_pass_compiler_generation, compiler_contributions]
code_refs:
  - src/merlin/xdsl_dialects/schedule.py
  - src/merlin/xdsl_dialects/lowering/schedule_decisions.py
  - src/merlin/xdsl_dialects/lowering/schedule_dispatch.py
  - src/merlin/compiler/schedule/schedule_ir.py
  - src/merlin/compiler/schedule/blocks.py
  - src/merlin/compiler/schedule/block.py
  - src/merlin/compiler/schedule/residency.py
  - src/merlin/compiler/schedule/execute.py
  - src/merlin/sched/ir/kernel.py
  - src/merlin/sched/check/static.py
  - src/merlin/sched/cost.py
  - src/merlin/compiler/passes/xdsl.py
  - src/merlin/compiler/passes/native.py
---

# Explore scheduling decisions and command schedules

**Tutorial 3 of 5** | [Start here](../../examples/tutorials/README.md) | [Previous: Test generation](test_generation.md) | [Next: Transformations](transformations.md)

Try scheduling a small matrix multiplication: `6 x 8 @ 8 x 6`. Change how much
operand storage is available, whether loaded blocks are reused, and the order
of commands. The notebook shows the resulting storage ranges, commands and
checks. It runs on your CPU without a board or target compiler.

Open [the scheduling notebook](../../examples/tutorials/03_scheduling.ipynb). Use the [shared notebook launcher](../../examples/tutorials/README.md#open-the-notebooks)
to open the examples in the right Python environment.

Choose **Kernel -> Restart Kernel and Run All Cells**. Widgets are optional; you
can edit calls such as `explore(block=4, rows=32, retain=False)` instead.
The dimensions and hardware capacities in this example are illustrative.

## What to try

1. Turn **Retain loads** off. The default example issues 48 loads instead of 24;
   both schedules still produce the same integer result.
2. Set **Store rows** to 32. With block edge 2, live operand ranges overlap and
   the checker refuses the schedule.
3. Change **Block edge** to 4. The example now fits in 32 rows. This models a
   different primitive geometry; it does not resize a real accelerator.
4. Change **Loop order** and **Lookahead**. Inspect the command ribbon and hover
   over commands to read their operands.

The ribbon shows command order, not elapsed time. Fewer loads can motivate an
optimization, but actual speed also depends on packing, transfers, completion
and contention.

## Which scheduling representation am I looking at?

Merlin uses several representations because a compiler decision and a device
instruction answer different questions.

| Representation | What it describes | Implementation |
| --- | --- | --- |
| xDSL `schedule` dialect | Selected decisions such as blocking, loop changes, placement and dispatch grouping | [schedule.py](../../src/merlin/xdsl_dialects/schedule.py) |
| Shared scheduling library | Algorithms that construct and check block commands, retention, transfers and ordering | [compiler/schedule](../../src/merlin/compiler/schedule/AGENT.md) |
| **mk** instruction IR | A kernel's counted loops, instruction calls, tensor arguments, byte pointers and events | [sched/ir/kernel.py](../../src/merlin/sched/ir/kernel.py) |
| Upstream passes and Transform dialect | Transformations over compiler IR in the selected MLIR or xDSL environment | [compiler/passes](../../src/merlin/compiler/passes) |
| Dispatch DAG schedule | Dependency levels and hart assignment for the supported dataflow program | [schedule_dispatch.py](../../src/merlin/xdsl_dialects/lowering/schedule_dispatch.py) |

xDSL is the Python compiler framework used for the `schedule` dialect. The mk
instruction IR is a Python data structure with its own text format, not another
MLIR dialect. There is no automatic conversion between every representation.
The notebook demonstrates the first three separately and labels each transition.

## From a decision to a transformation

The notebook begins with real `schedule.plan`, `schedule.bind` and
`schedule.loop_transform` operations. A binding names the payload to transform;
the loop operation requests a split. Printing, parsing and verification check
that the request is well formed. A split factor of one is refused.

That example uses a symbolic teaching binding. Parsing it does not find a real
payload or execute a transformation. To follow the implemented dense route,
open [schedule_ir.py](../../src/merlin/compiler/schedule/schedule_ir.py):
`make_dense_schedule_plan` records the original source and selected context;
`read_dense_schedule_plan` checks that context and reads supported decisions.
The consuming pipeline checks the resulting commands. Unsupported decisions
are refused rather than ignored.

Another route, [lower_to_schedule](../../src/merlin/xdsl_dialects/lowering/schedule_decisions.py),
reads proven reusable-weight facts before recording residency and dispatch
decisions. `recorded_group_dispatch` reads those groupings back; dispatch-region
formation must honor them. An operation in the dialect's inventory still needs
a consumer before it can affect generated code.

## Construct and check block commands

The matrix experiment calls the shared block API directly:

```python
from merlin.compiler.schedule.blocks import (
    Contraction, Geometry, Knobs, schedule_contraction, check_residency,
)

schedule = schedule_contraction(
    Contraction(m=6, k=8, n=6),
    Geometry(block=2, operand_rows=64, operand_bank_rows=16,
             accumulator_rows=64, separate_accumulator_space=True),
    Knobs(load_on_index_change=True, lookahead_steps=1),
)
check_residency(schedule)
```

This is separate from the earlier split request. `BlockSchedule.regions` supplies
the storage picture; `BlockSchedule.ops` supplies the command ribbon.
[block.py](../../src/merlin/compiler/schedule/block.py) constructs the commands,
and [residency.py](../../src/merlin/compiler/schedule/residency.py) checks that
loads remain available, reductions finish, and outputs are stored correctly.
The builder refuses a contraction that needs outer tiling before it can fit.

[execute.py](../../src/merlin/compiler/schedule/execute.py) interprets those
scheduled loads, row addresses, accumulators and stores. The notebook compares
its result with an `int64` NumPy matrix product on small signed-integer inputs.
It also demonstrates real refusals for missing bank geometry, a premature store,
and an out-of-bounds mk byte access.

## Where the backend takes over

An **out-of-tree (OOT) backend** is a separate compiler package or repository.
It supplies resource facts, numerical meaning, instruction effects, physical
mapping and encoding. Shared algorithms can then use those contracts without
embedding target names in the algorithm. The backend's output might be LLVM IR,
objects, device binaries or command artifacts.

For other shared algorithms, start with
[api_catalog.py](../../src/merlin/compiler/schedule/api_catalog.py):
[tile_load_coalescing.py](../../src/merlin/compiler/schedule/tile_load_coalescing.py)
merges applicable transfers,
[staged_rhs_residency.py](../../src/merlin/compiler/schedule/staged_rhs_residency.py)
handles retained operands, and
[completion_streams.py](../../src/merlin/compiler/schedule/completion_streams.py)
handles its supported completion-aware ordering. Each requires its own contracts.
The [shared-pass guide](shared_pass_compiler_generation.md) explains backend use.

The notebook checks a synchronous example, not device execution. Placement,
configuration and synchronization have additional checkers in `sched.check`.
`sched.cost.price` reports a result for a declared abstract machine, with missing
costs kept unknown; it is not a hardware timer. Quantization, compiled numerical
correctness, ABI/memory behavior and actual target use need their corresponding
verification and execution evidence.
