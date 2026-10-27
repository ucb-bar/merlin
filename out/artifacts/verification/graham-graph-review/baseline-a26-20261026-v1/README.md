# a26 baseline intake

This packet records the source/data intake before fresh execution. Its tested
source cut is **c62f81a8a11347ff53b60101fa75280e69d002a6**. The separate review
branch tip was **5a7f872db33d378d6b0c4a294e4099e80efd4956**, with common source
ancestor **58dbc7c00**. No branch merge or compiler implementation change occurred.

[BASELINE_AUDIT.json](BASELINE_AUDIT.json) records matching hashes for Q's 36
fixed snapshots, W's 61 owned sources, its 118-Lean/51-Python closure, all 202
staging capsule payloads, four declared dependency DAGs, and the authoritative
inline queries. [INLINE_INTAKE.json](INLINE_INTAKE.json) separately binds all
112 inline source snapshots and five actual before/after/query triples.

[REVIEW_RECONCILIATION.json](REVIEW_RECONCILIATION.json) compares the original
agent reviews with the later source cut: all 76 reviewed Lean sources and seven
queries match; ten of 55 Python projection sources changed. These changes are
admission wiring and source-closure additions, not changes to the frozen Lean
semantics.

[PLAN.json](PLAN.json) is the original proposed plan and retains its historical
approval status. The later user instruction authorized three workers to execute
source review, fresh SDK setup and reproduction. Their actual findings and
authorization scope are in [the execution packet](../execution-a26-20261026-v1/README.md).

The thirteen duplicate `inputs/reviewer` files named in the original intake
manifest are intentionally omitted from publication. Each is already present
at its `original_path` in review commit **5a7f872db**, with the same SHA256.
The original intake manifest is a record of the run's inputs, not a claim that
every original path is physically shipped here. The execution packet's
[publication index](../execution-a26-20261026-v1/PUBLICATION_INDEX.json) specifies
the transferred files. Historical absolute paths were not rewritten.
