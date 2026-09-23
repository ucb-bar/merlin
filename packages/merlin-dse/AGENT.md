# Optional DSE distribution

Owns the research-only dse, design_pressure, and dse_guidance modules. Keep their
stable merlin.* import identities; no duplicate implementation under merlin_dse.
Compiler/capture/evidence APIs needed by core stay in core. Preserve evidence
ranking and numerical semantics during moves. No generated results belong here.
