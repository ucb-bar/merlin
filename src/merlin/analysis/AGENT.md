# src/merlin/analysis

The observation SDK owners at this level observe actual IR, instruction words, artifact bytes and selected
target declarations. They supply no kernels, schedules, tuned choices, measured
answers, reference runtime or acceptance policy. These SDK owners must not import `merlin.analysis.kernels`,
`merlin.analysis.mining`, experiment engines or graders, even lazily. Candidate staging
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


`dse/` contains research consumers rather than observation SDK components. Its
scoped instructions allow analysis of compiler/kernel choices without exposing
that research implementation through the observation source grants.
