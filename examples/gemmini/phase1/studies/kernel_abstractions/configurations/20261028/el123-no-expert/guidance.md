# Reference-free compiler construction study

Start with an empty target backend. Implement every target lowering, transformation
and scheduling policy yourself. Only the selected public tests, contracts, ISA facts,
ordinary runtime and profile-specific authoring components are supplied.
No expert kernel sources, binaries, disassemblies, example implementations or prior
learned kernel artifacts are supplied. CCA and automatic kernel learning are disabled.
Do not import existing target implementations or resume an earlier authored compiler.

Preserve source semantics, rounding, saturation, all live outputs and the complete
pointer boundary. Hardware-loop/FSM descriptor instructions with the public role
loop_descriptor remain prohibited by the common instruction policy.
Existing complete Spike timing feedback and diagnostics remain available to all
three conditions. Estimates are provisional; unknown checks remain unknown.
Use at most eight helpers sharing the selected fifteen-CPU authoring allowance.
Author for four hours, then freeze before the separate 900-second evaluation drain.
