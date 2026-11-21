# Merlin support package

This directory contains the target definition and, where present, reference support plugins.
Select it explicitly with `MERLIN_TARGET_PATH=/path/to/this/repo/merlin-support`.
It is not an evaluated compiler candidate and this metadata grants no trust or certification.

The files recorded in `provenance.json` are byte-identical migration snapshots. Existing contracts
retain their prototype, derived-fact, and requires-human-review qualifications. No hardware was
executed for this migration, and no canonical Merlin source was removed. Ignored/generated source
artifacts have content hashes but no invented source commit.

The historical candidate/schedule payload and its certificates remain unchanged outside this tree.
Those certificates describe their original revisions, not the current branch with added support.
Do not grade or publish this entire repository as a candidate: export the original candidate or
schedule payload without `merlin-support/`. Existing recursive integrity checks intentionally
remain unchanged and may reject harness-importing support code inside a candidate tree.

The root `rvv-opt` is a publication skeleton, not a working standalone compiler. The real artifact
is the host schedule under `payload/`. No dialect plan existed in the source snapshot.

Run the lightweight checks with an installed Merlin core:

```sh
python -m pytest merlin-support/tests
```
