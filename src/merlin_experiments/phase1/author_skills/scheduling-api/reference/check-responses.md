# Admission, seed fallback and named shapes in check responses

A checked consumer either returns a checked result or refuses. On a refusal your compiler emits
whatever route it chose instead, and that can differ between shapes. Consumers that record their
decisions carry an `admission_api` attribute with their full name.
- Each case in a `--suite scalability` response carries `abstraction_admission.records`: one row per
  distinct decision in that case's compile, `{api, case, admitted, failed_premise, fallback_route,
  count}`. `failed_premise` is the consumer's own refusal text; `fallback_route` says what was emitted
  instead. In your own process, `merlin.compiler.schedule.library_sources.admission_records()` returns the
  same rows.
- `fallback_to_seed` is `true` when the case's emitted kernel is byte-identical to what the starting
  compiler emits for the same input (`seed_identity` gives the basis), `false` when it differs, and
  `null` when no comparison is available. A seed fallback counts as no optimization for that shape.
- Shapes beyond the public cases are evaluated after authoring. To see how your compiler handles shapes
  you choose, name them against one public case:
  ```
  python3 agent_selfcheck.py --suite scalability --sim spike --capsules CASE \
      --coverage-sweep m=24,m=48,m=64:k=1024 [--coverage-sweep-numerical] --wait-budget 5
  ```
  Each named shape is built from CASE through the study's case-resizing path and compiled with your
  package. Per shape the reply lists `admitted_apis`, `refusals` (api, failed premise, fallback route),
  `fallback_to_seed`, and `emitted` (fences and transfer rows x columns from the decoded kernel).
  `--coverage-sweep-numerical` adds a functional Spike check for each shape. Only the shapes you name
  are built; the reply has no timing.
