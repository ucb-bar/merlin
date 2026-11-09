# AGENT.md - src/merlin/compiler/lower/native

This package owns native lowering orchestration, including scalar/OpenMP and
generic upstream fixed-width vector pass composition, generated child runners,
ordered command arguments, explicit effect admission, completion reports, and
translation receipts. It does not implement the RVV OOT backend. Reusable LLVM
finalization passes and literal normalization stay in compiler/lower/llvm.

pipeline.py is the driver. policy.py selects existing scalar/OpenMP ordering;
vector.py assembles generic upstream vector passes. features.py registers and
checks the existing requested transformations. arguments.py returns a typed
RunnerCommand containing the actual ordered argv and its report gates.
reports.py checks original completed observations. invoke.py owns diagnostic
pass execution and trace harvesting. stages.py composes existing stage source
and runners.py selects exact generated script variants. transport.py owns
serial LLVM-dialect retention; stock.py invokes the admitted stock CLI path.
recipe.py records actual selected commands and source identities.

Legacy pass/prelude implementations outside this family remain explicit imports
from llvmlower during migration. Never duplicate them or hide selection behind
module-global callback propagation. Preserve emitted runner bytes, ordered argv,
source/effect checks, selected tool identity, and original completion evidence.
Source receipts must name these actual owners. Frozen inventories are immutable.
