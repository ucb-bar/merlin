# src/merlin/analysis

The observation SDK owners at this level observe actual IR, instruction words, artifact bytes and selected
target declarations. They supply no kernels, schedules, tuned choices, measured
answers, reference runtime or acceptance policy. These SDK owners must not import `merlin.analysis.kernels`,
`merlin.analysis.mining`, experiment engines or graders, even lazily. Candidate staging
must preserve the explicit selected source closure and withheld identities.

The source tree separates the evidence and its consumers:

- `observation/` reads artifact bytes, disassembly, instruction roles, command streams and declared engine facets.
- `perf/` groups bounds, dependency analysis, measurement, claims and performance search. Its lower-level observers preserve explicit unknowns.
- `programs/` describes programs and their interfaces; `whole_model/` holds whole-model analysis.
- `kernels/` owns the Common Compute Abstraction and analysis of candidate kernel structure.
- `mining/` and `optimization/` consume those observations to explore implementation choices; they are not observer SDK exports.

Preserve unknowns and exact refusal conditions. Static sites are not dynamic
execution counts; observer results do not establish accelerator arithmetic,
completion, memory visibility, numerical correctness or timing qualification.


Keep research consumers out of the observation source grants. A source move or
larger dependency closure must preserve withheld identities and the distinction
between an observation, a compiler decision and an independent acceptance check.
