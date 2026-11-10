# Merlin support package

This directory contains the target definition and, where present, reference support plugins.
Select it explicitly with `MERLIN_TARGET_PATH=/path/to/this/repo/merlin-support`.
It is not an evaluated compiler candidate and this metadata grants no trust or certification.

The files recorded in `provenance.json` are byte-identical migration snapshots. Existing contracts
retain their prototype, derived-fact, and requires-human-review qualifications. No hardware was
executed for this migration, and no canonical Merlin source was removed. Ignored/generated source
artifacts have content hashes but no invented source commit. The subsequent program-emitter
relocation and explicit contract delta are recorded separately in `program_emitter_migration.json`;
the original migration record is not rewritten.

`atlas_program_emit.py` is host-private Atlas support, executed only inside the selected model's
environment. It retains both named-program assembly/golden export and tensor byte-layout modes.
`runner.program_emitter` declares its provider-contained path and the target-owned IType encoding
option. Neither this helper nor its model ISA belongs in candidate grants. Offline relocation
checks do not qualify Torch, assembly, numerical outputs, simulators, or hardware.

The historical candidate/schedule payload and its certificates remain unchanged outside this tree.
Those certificates describe their original revisions, not the current branch with added support.
Do not grade or publish this entire repository as a candidate: export the original candidate or
schedule payload without `merlin-support/`. Existing recursive integrity checks intentionally
remain unchanged and may reject harness-importing support code inside a candidate tree.

The upstream repository was empty. This branch adds support data only; it has no compiler candidate.
The source contract explicitly retains datapath-grounding and RTL-certification limitations.

Run the lightweight checks with an installed Merlin core:

```sh
python -m pytest merlin-support/tests
```
