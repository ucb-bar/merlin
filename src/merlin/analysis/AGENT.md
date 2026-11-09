# AGENT.md — src/merlin/analysis

These owners observe actual IR, instruction words, artifact bytes and selected
target declarations. They supply no kernels, schedules, tuned choices, measured
answers, reference runtime or acceptance policy. Do not import `merlin.kernels`,
`merlin.mining`, experiment engines or graders, even lazily. Candidate staging
must preserve the explicit selected source closure and withheld identities.

The RV64 observation implementation has separate real owners:

- `riscv_state.py` applies conservative register, private-frame, FP and call effects.
- `riscv_forms.py` propagates affine forms and frame slots over one loop iteration.
- `riscv_loops.py` discovers natural loops, proves admitted trip counts and annotates affine facts.
- `riscv_control.py` counts only complete admitted bounded scalar control and records actual issues.
- `riscv_machine.py` owns CFG fixed-point observations and function summaries.

Preserve unknowns and exact refusal conditions. Static sites are not dynamic
execution counts; observer results do not establish accelerator arithmetic,
completion, memory visibility, numerical correctness or timing qualification.

Old `merlin.compile.analysis` and applicable `merlin.kernels` paths are temporary
exact module identities. Fresh record and pickle paths are canonical here;
historical replay uses its frozen source environment. Do not introduce a
permanent legacy module label or widen an SDK grant merely to support relocation.
