# Atlas Phase 1 inputs

This directory owns public target-specific contract inputs, not shared compiler code.
Preserve the curated RTL, ISA documentation, example kernels and preflight fixtures
as one source-attributed set. The descriptor selects all paths explicitly, including
the assembly adapter callables. Moving these files must not change their bytes or
silently run external assemblers, simulators or hardware during path validation.
Generated compilers, capsules, releases and certification records remain artifacts;
private holdouts and goldens must never be added here.
