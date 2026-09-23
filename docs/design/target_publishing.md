---
title: Target publishing — preserved payloads and scoped evidence
kind: design
status: current
owner: core
last_verified: 2026-09-21
related: [repo_structure, integrations, architecture]
code_refs: [src/merlin/targetgen/publish.py, src/merlin/targetgen/oot_fetch.py, src/merlin/targetgen/package_records.py, src/merlin/targetgen/publication_verification.py, src/merlin/targetgen/providers.py, merlin/contract/schemas/result.schema.json]
---

# Target publishing

`merlin-target-publish` exports selected artifacts to target OOT repositories.
It never generates a compiler around a schedule or turns support into evaluated
compiler code. The default is non-executing dry-run.

## Role-aware, preserved layout

The existing `providers.read_provider` contract distinguishes candidate compilers,
host schedules and support. Explicit metadata wins; legacy schedules remain
schedules even when an old wrapper looks like a compiler ABI.

- Candidate compilers must already have a valid ABI, actual commands and an
  in-payload tool or declared build recipe. Publication invents none of these.
- Host schedules remain data for their target-owned host consumer. No generated
  compiler driver, CMake skeleton or ABI is supplied.
- Support and unknown roles are refused, including with `--no-gate`.
  Support/oracle code is not a candidate grant.

Layout version 2 preserves every source member at its original root-relative
path, including executable modes, manifest, README, compiler modules and build
inputs. Only these additive publication paths are reserved:

```text
<export>/
  ...original payload, unchanged...
  MERLIN_PUBLICATION.md
  .merlin/
    manifest.yaml           # exact original manifest bytes
    provenance.yaml
    certification.yaml
    CHAMPION
```

A source containing `.merlin`, `MERLIN_PUBLICATION.md` or `.git` is refused before
copying; author files are never overwritten. Symlinks and special files are not
accepted payload members. Historical version-1 exports remain unchanged and their
generated wrappers acquire no new qualification.

## Immutable payloads and external records

Payloads live at `out/artifacts/targets/<target>/<package_slot>/`.
Host bookkeeping lives outside them at
`targets/<target>/.publication/<package_slot>.json`, naming the target, installation
slot and original compiler manifest package ID separately.

The shared strict inventory binds every regular file, empty directory and
executable permission bits, including members under `build`. Changed bytes,
membership or modes invalidate the record. Selection overlays bookkeeping in
memory; historical embedded claims remain inspectable without new authority.

Materialization copies exact bytes and checks source/copy inventory parity.
A new installation is unverified and not champion. Scores and
`--certified-by-run` identify installation provenance, not package certification.
Original manifest claims remain untouched.

IDs must be non-reserved single components. Symlink destinations and source/score
overlap are refused before writes. Forced replacement prepares a complete new tree,
then retains the previous package in a unique staging directory and its old record
as `previous-publication.json`. Failed installation/record writes attempt rollback.
Recovery copies are never automatically deleted. This assumes a trusted stable
filesystem, not hostile replacement or crash-atomic multi-rename transactions.

## Scoped certification

The experiments-owned producer builds a retained run-owned execution copy, never
the preserved payload. It records separate source and postbuild inventories and
checks both around every entrypoint and after oracle evaluation. Python bytecode
writes are explicitly disabled for these children; other runtime writes inside the
built tree invalidate the observation. Other runtime callers retain their defaults.

A new recorded pass requires matching target, run, successful identified oracle,
typed tier metadata and producer input identity matching current payload bytes.
The weakest rung's tier is retained. Failures remain failures; absent/mismatched
binding stays unverified. Today's inventory never upgrades old receipts.

A gate pass has scope `package-payload`, with external dependency closure
`not-attested`: an observed test in the recorded environment, not hermetic
dependencies, portable buildability or standalone compiler qualification.
Tools outside the execution copy cannot acquire package binding. Pre/post checks
are not a sandbox against transient restored writes.

External staging `export_identity.json` records original and complete exported
inventories and additive transformation. It lives outside the tree to avoid
self-reference. Embedded records, publication guidance and Git history distinguish
source observations from the expanded export, which remains unverified.

## Build verification

Executing publication verifies candidate builds by default on a fresh retained
copy using the existing runtime. It checks the declared tool exists afterward and
records input/output inventories in external `build_verification.json`.
Failure or timeout prevents Git publication and retains diagnostic evidence.
Neither source nor exported tree is built in place.

`--build-timeout SECONDS` bounds each declared build step, not the whole workflow.
`--no-verify-build` explicitly skips this observation. Dry-run always records
`not-run`; host schedules have no compiler build. Interpreted packages without a
recipe receive a tool-presence check, not an implicit import/probe or numerical test.
This uses the caller's environment, not dependency isolation. Build success does
not certify compiler results, target hardware or portability. Other direct
`build_package` callers remain build-in-place APIs.

## Git and admission

`merlin-target-fetch TARGET --champion BRANCH_OR_TAG` acquires a published target;
it does not certify its compiler or authorize support-plugin execution. An update
fetches the selected ref and adopts that exact commit in detached HEAD state without
resetting local branches. Omitting `--champion` selects the remote's default HEAD,
not whichever branch a previous single-branch clone happened to track.

Updates refuse a different origin URL, local tracked/untracked changes and local
or unverified commit history. Ignored build output may remain, but a checkout must
not overwrite ignored files. Missing refs and fetched commits without an ordinary
target contract leave the current checkout unchanged. A Git-local receipt ref retains
the last adopted commit so published tags and switched branches can be fetched again.
Older tag-only clones without usable remote history or that receipt require a fresh
destination. `--no-update` preserves an existing clone without fetching. These are
trusted local Git checkout safeguards, not protection against concurrent mutation
or independent qualification of the downloaded compiler.

Promotion changes external records, never payloads. Ordinary promotion/publication
requires bound source evidence; explicit `--no-gate` remains unverified and warns.

Remotes resolve from override, target environment configuration, then
`merlin/targets/publish.yaml`. Non-local pushes require explicit displayed
fingerprint confirmation. Local tests use synthetic bare repositories only.

Staging destinations must be fresh, disjoint and free of symlink components.
Idempotency binds package ID, Merlin revision, run ID and exact source inventory
SHA-256. Only a matching remote branch-tip fingerprint is a no-op. Existing tags
remain immutable; their presence does not suppress a new branch commit.
Index admission uses the same role/gate checks and advertises only existing branches.

Role declarations, input binding, build observations and numerical/hardware
qualification are distinct. Missing dependencies and untested portability remain
explicit; historical evidence and published commits are never rewritten.
