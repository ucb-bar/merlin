# AGENT.md — src/merlin/verify/lean

Shared mathematical verification sources, not target compiler solutions. The
operator selects and pins the Lean/Lean-MLIR toolchain explicitly; checking never
downloads dependencies. Preserve the narrow documented semantic scope and audit
transitive theorem axioms. Generated obligations bind actual changed IR through
the host checker. Compiled products belong in an explicit build/task directory,
never here. Each subdirectory must document its ownership.
