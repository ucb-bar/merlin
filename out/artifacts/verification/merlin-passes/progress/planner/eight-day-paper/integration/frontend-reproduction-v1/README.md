# Independent frontend source rebuild and checker reproduction

Planner-run reproduction, 2026-10-22/23 UTC, of the qualified frontend source packet through
`135b8e1c2` (evidence documentation later `c29d71090`). **Passed within the scope below.**
No branch, worker source, protected file, original build or admission flag was changed.

## Executed result

- Rebuilt all **98 project modules** in the qualified checker import closure from copied,
  hash-checked sources, plus copied the unchanged auditor source. **Zero project proof objects
  were copied** from the worker. New project imports resolve only to the fresh output directory.
- The fresh imported-closure audit checked **4,684 theorems with zero violations**. The
  complete-program root and both numerical-law roots use only `propext`, `Classical.choice`
  and `Quot.sound`. Their fresh elaborated statements exactly match the qualified receipt.
  This closure is smaller than the worker's full-library 11,357-theorem audit; do not add them.
- Reexecuted **36 checker invocations**, obtaining **24 acceptances and 12 refusals** with
  complete JSON answers matching the qualified records. These are 29 retained IEEE exported
  pairs plus the before/after/suffix exports from seven recorded receipt boundaries, including
  all four positive modes and three refused suffixes. Counts are invocations, not 36 new tests
  or distinct model captures. The IEEE pairs retain the nine valid target mutations.
- Bound all **2,185 actually imported modules**, totaling 485,135,640 bytes of `.olean`
  artifacts, and verified afterward that their bytes were unchanged. Every one of the 98
  project imports came from the new build. All **50 pinned IEEE source/object records**
  match the worker's qualified identity. Original project sources and the two qualified
  checker/auditor binaries remained unchanged.

[SUMMARY.json](SUMMARY.json), [audit.json](audit.json), [cases.json](cases.json),
[POSTCHECK.json](POSTCHECK.json), [imports.json](imports.json), [ieee-core.json](ieee-core.json).
The immutable source-copy identities and exact commands are in
[manifest.json](manifest.json), [manifest-resume.json](manifest-resume.json),
[processes.json](processes.json) and [processes-resume.json](processes-resume.json).

## Exact scope and retained failure

This independently checks the source-to-proof-object build, imported-closure audit and actual
Lean checker execution on retained exports. It uses Lean 4.35.0 and existing external package
artifacts on the same host. It is **not clean-machine/clean-environment reproduction**, a fresh
production-pass/exporter run, native numerical comparison, the entire 118-test suite, fresh
receipt-protocol replay, selected-model capture binding, or combined W8A8/frontend/quant-ext
qualification. The provisional source interface, numerical model/native differences and
independent contract review remain as documented in the worker's qualified report.

The first run successfully rebuilt all 98 modules, then the auditor failed because the planner
wrapper did not put `lean` on PATH. The auditor's sysroot lookup invokes it by name even when
the parent was launched by absolute path. [FAILURE.json](FAILURE.json) and the initial raw logs
are preserved. The separate imports inspection encountered the same environment omission.

The wrapper was corrected to prepend the pinned toolchain directory. It explicitly verified
the successful build record and unchanged source copies before resuming the audit; no module
was rebuilt. The resumed manifest records all fresh object hashes, and separate resumed logs
preserve the failed run. The original and corrected wrapper sources are retained here as
`reproduce_frontend-original.py` and `reproduce_frontend-resumed.py`. This is a diagnosed planner
environment repair, separate from the historical undiagnosed W8A8 JSON failure.

## Reproduction command

From the main checkout, use a new name. The corrected wrapper refuses to overwrite an existing
source/build directory; external library paths still identify this machine's retained packages.

```sh
.venv/bin/python -u out/artifacts/verification/merlin-passes/progress/planner/eight-day-paper/integration/reproduce_frontend.py \
  --name frontend-reproduction-v2
```

Build outputs are under `out/build/verification-paper-reproduction/<name>`; evidence remains
under this integration directory. Relocation requires an explicit environment/dependency recipe
and fresh identities, not hand-edited receipts. The remaining clean-environment and combined
qualification obligations in the [active goal](../../README.md) are unchanged.
