---
title: Disk under out/ — why it grows and what is safe to reclaim
kind: guide
status: current
owner: infra
last_verified: 2026-09-23
related: [reproducibility, getting_started, gemmini_experiment]
code_refs: [src/merlin/common/content_store.py,
            src/merlin/common/storage_cli.py,
            src/merlin/common/storage_lifecycle.py,
            merlin/contract/storage.yaml,
            .claude/hooks/guard_artifact_writes.py,
            packages/merlin-experiments/src/merlin/targetgen/sandbox/bwrap.py,
            merlin/tests/conftest.py]
---

# Disk under `out/` — why it grows and what is safe to reclaim

Generated output is large by design: a whole-model run holds MLIR at every stage, an ELF, a trace and
a grade. It became large by *accident* twice, and both times the diagnosis took longer than the fix,
because the growth did not come from the results. It came from three mechanisms that copied or
retained bytes nobody asked for. Start with

```bash
merlin-storage report                     # what the root costs, and how much of it is shared
merlin-storage experiments                # what ONE experiment costs, by group
merlin-storage experiments --match phase2 # ...for one campaign, across concerns
merlin-storage layout                     # what sits under out/ the convention does not name
merlin-storage prune                      # dry run: what is provably safe to reclaim
merlin-storage prune --apply caches       # act on one class
merlin-storage dedup                      # dry run: bytes held under more than one name
merlin-storage retain --keep 20           # dry run: what a retention depth would drop
merlin-storage organize                   # dry run: fold stray dirs into their declared concern
```

## Is it bloat, or is it accumulation?

A total cannot tell you, and the two want opposite fixes. `experiments` prices the level at which
"per experiment" means something — an aet run at `out/runs/<target>/<suite>/<run-id>`, a product at
`out/artifacts/<concern>/<axis>/[v<n>/]<unit>` — and reports the **mean per unit** beside the total.
A large total with a small mean is accumulation: nothing deletes finished campaigns, and the remedy
is a retention decision. A large mean is the producer writing too much per run, and no amount of
de-duplication will fix it. Measured 2026-09-15:

| group | units | total | mean/unit |
| --- | --- | --- | --- |
| `artifacts/perf-bench/gemmini` | 475 | 51.2 GB | 110 MB |
| `runs/gemmini/capsule-bench` | 39 | 22.6 GB | 594 MB |
| `artifacts/delivery/chipyard_kodiak/v1` | 20 | 14.5 GB | 743 MB |
| phase 2, across concerns (`--match phase2`) | 178 | 29.3 GB | 174 MB |
| phase 1, across concerns (`--match phase1`) | 7 | 0.12 GB | 17 MB |

So perf-bench looks large by accumulation — but see *Where things are supposed to live* below: 448
of its 475 unit directories are named outside the convention, which means it is really a working
directory that grew inside a product tree, and no retention depth can place its contents.

The one real outlier is a single 33.6 GB product unit holding **19 whole-model ELFs of ~1.21 GB
each** (a TinyLlama binary embeds its weights), their weight blobs, and three multi-GB zips. What
makes that a defect rather than a big result is the unit's own manifest: it declares **43 small
receipts and logs**, and `layout` prices everything in the directory that the manifest does not list
— **35.06 GiB, 94% of the unit**. This is the one drift number that needs no threshold, because the
producer already declared the answer; the 9.53 GiB delivery bundle beside it reports 0% undeclared,
which is what makes the metric trustworthy. Compiled output belongs under `out/build/`, referenced by
digest rather than embedded.

A `v<n>` level is deliberately not a unit: it sits between the axis and the unit, and pricing it as
one experiment reports a whole version series as a single run.

A console script is generated at install time, so `merlin-storage` appears after the next
`uv sync`. Until then — and in any checkout sharing a `.venv` with another one, where re-installing
would repoint that venv's editable path at whichever tree ran it last — invoke the module directly:
`.venv/bin/python -m merlin.common.storage_cli report`.

## Why it grows

**1. Per-run input closures (the big one).** Every agent run freezes the input closure its bundle
declares, so the bytes it ran on cannot change under it mid-run — an operator editing the tree in
another session must not retroactively alter a running experiment's treatment. That freeze used to be
a deep copy, so a campaign whose runs all granted the same toolchain and the same model weights wrote
those identical bytes once per run: **12.8 GB per run, 235 GiB across one campaign**, for inputs that
were byte-identical every time.

The freeze now goes through a content-addressed store at
`out/artifacts/cache/bundle-inputs-cas/`: each distinct byte-string is stored once at mode `0444`, and
each snapshot entry is a **hard link** to it. The immutability property is unchanged — the store holds
its own copy, so truncating the source in place cannot reach a frozen snapshot — but the second run
granting the same file adds a directory entry and no bytes. Objects are keyed by content *and* by
whether the file is executable, because a mode belongs to the inode and a granted compiler has to stay
runnable inside the box.

Consequences worth knowing:

- Pruning the store is safe. A hard link keeps bytes alive while any snapshot names them, so removing
  an object with one remaining link cannot take data from a run. That is what `store-orphans` does.
- `MERLIN_BUNDLE_CAS` relocates the store; a hard link cannot cross a filesystem, so point it at the
  volume holding an out-of-tree workspace. Setting it **empty** disables sharing and restores the deep
  copy — the escape hatch if a filesystem ever reports links it does not honor.
- Concurrent runs are the normal case here, so store objects are created **create-if-absent**, never
  by an unconditional rename. A rename let each racing writer overwrite the previous winner's entry
  after that winner had already linked its inode, so twelve concurrent first-time runs ended on
  twelve separate inodes holding identical bytes — correct, and the saving entirely lost. A loser now
  learns from `EEXIST` to use the winner's object.
- `merlin-storage report` flags closures that share nothing. Those either predate the store or were
  written with it disabled.
- **Not every freeze can use it.** A store object's mode belongs to its inode, so every consumer
  linking it shares that mode and any one of them can change it for the others — including the chmod
  a caller performs to delete its own frozen tree. That is harmless for a freeze verified by
  *content*, which is what the bundle closure does. It is unsound for one that treats "this file is
  not writable" as evidence of its own integrity: the perf-bench source snapshot
  (`merlin_experiments.source_snapshot`) does exactly that in `verify()`,
  and adopting the store there made a second snapshot fail verification because an unrelated first
  one had been chmodded during teardown. It deliberately still deep-copies, and
  `merlin/tests/infra/test_content_store.py` pins both halves of that so the saving does not tempt
  someone to weaken the check instead. Its snapshots are ~76 MB each, against 12.8 GB for a bundle
  closure, so this is a small amount of duplication bought with a real invariant.
- A bundle copy may also mark additional execution resources private. Sources overlapping those
  roots are copied without the shared store; this copy policy does not grant them to the candidate
  or change their answer-surface classification.

**2. Retention that was never bounded.** pytest keeps the last *three* runs' `tmp_path` trees for every
test, passed or failed, and many tests obtain scratch space from a bare `tempfile.mkdtemp()` inside a
module-level helper — no fixture, so nothing ever removed those at all. `merlin/tests/conftest.py` now
points `tempfile` and `TMPDIR` at pytest's managed base temp root (on the big filesystem, not the small
root volume), and `pyproject.toml` keeps only the failing tests' trees from the last run. A spawned
compiler or simulator inherits `TMPDIR` and writes its intermediates in the same reclaimable place. Measured on 2026-09-15: three consecutive runs
of a passing test file leave 8 KB behind and no retained run roots, where the defaults would have
kept all three.

**Where the tool looks.** New phase-1 authoring workspaces live at
`out/build/agent-workspaces/phase1/<target>/<arm>/<run>/workspace`, honoring `MERLIN_OUT_ROOT`.
Their frozen input closures remain siblings, outside the writable agent view. Historical
`targets/<target>/_qa_ws/<run>/workspace` trees are not moved: resume prefers the archived
`environment.yaml` workspace pointer, then generated placement, then the legacy location.
The first version of `merlin-storage` walked only the `out/` root and was therefore blind
to 40 GB of completed closures and one 8.7 GB closure abandoned mid-copy — precisely the class it
exists to find. The extra roots are declared in `merlin/contract/storage.yaml` (patterns relative to
`merlin/`, globs allowed, a pattern matching nothing is skipped) rather than spelled in code, for the
same reason every other path in that directory is declared: no library module names a checkout
directory, so relocating a workspace is an edit to data. Losing the contract degrades the tool to the
`out/` root; it is never a dependency.

**3. Declared-regenerable trees that nobody reclaims.** `out/artifacts/cache/<ns>/` and
`out/artifacts/recaptures/` are PURGEABLE by the layout convention that created them (see
[the generated-output convention](../../CLAUDE.md)); being purgeable is not the same as being purged.
`merlin-storage report` prices them so the decision is informed.

## The same bytes under several names

The content store gets the saving at the moment a tree is frozen, which does nothing for the trees
written before it existed — and those are most of the root. `merlin-storage dedup` finds files that
hold bytes another file already holds and re-points each name at one store object. Nothing is
removed: every name still resolves, verified by digest before the swap. Measured 2026-09-15 over
`out/artifacts` and `out/runs`, files ≥1 MiB: **65.9 GiB across 1,196 content groups**, the largest
being a per-run agent home's plugin cache replicated 445 times, whole-model `const_blob.o` weight
blobs duplicated across validation directories, and shipped `.zip` bundles byte-identical to the
unpacked trees beside them.

Two things to know before running it with `--apply`:

- **An adopted file becomes read-only, and its mode is then shared** with every other name for those
  bytes. That is correct for a frozen product and wrong for anything a later step rewrites in place.
  It is also why a tree whose own integrity check reads the file mode must be skipped — chmod follows
  the inode, so one holder's change would break every other holder's verification. The seal that
  marks such a tree is named in `merlin/contract/storage.yaml` under `mode_verified_seals`, and
  `dedup` prunes those subtrees from the walk.
- Only files that **share a size** with another file are digested. Two files of different sizes cannot
  have the same content, and that prefilter is what makes a whole-root scan affordable; it is an
  optimisation, never the test. `--min-bytes` (1 MiB by default) keeps the walk off the long tail
  where the saving cannot repay the inode.

## Retention

`merlin-storage retain --keep N` reports which units a retention depth would drop, per group, newest
kept. Ordering comes from the `YYYYMMDDTHHMMSSZ` token the naming convention puts in the unit's own
name — **never from mtime**, because a purge's own deletions update the mtimes of the units it walks,
so a "modified recently = live" rule reports the units you just edited as the ones to keep. Two rules
are necessary but not sufficient:

- a unit whose name carries **no timestamp** is never dropped: it cannot be placed in the order, so it
  cannot be shown to be old;
- whatever a `latest` symlink resolves to is never dropped, so a consumer following that name never
  finds a dangling pointer.

Retention additionally requires an explicit terminal lifecycle record. Tracked files, active or
uncertain leases, retention pins, and historical units with no lifecycle record are protected.
The plan reports those protections; apply rechecks lifecycle and `latest` while holding the mutation
lock. Neither a timestamp nor a `.pending` suffix proves that a worker has stopped.

## Ownership, crash protection, and cited evidence

Writers use `merlin.common.storage_lifecycle.lease(path, owner=...)` while their workers are active,
and stop child workers before closing it. `pin(path, reason=...)` returns a token retaining cited
evidence until explicitly removed with `unpin(path, token)`. Both protect parent and child paths.
Records live in `out/.merlin-storage.json`, outside frozen artifacts; report commands never create
that registry or its lock. The lock coordinates cleanup with cooperating writers, not arbitrary
legacy processes that never acquire leases.

Lease age is not expiration. A remote owner, an uninspectable process, or a known-dead owner remains
protected because its children might survive. After verifying that the owner and all its children
have stopped, an operator can call `acknowledge_abandoned(path, reason=...)`. This refuses live or
unknown owners and never removes retention pins. The phase-1 driver deliberately retains its lease
after an exception and prints these recovery instructions; orderly returns release it.

Workspace placement does not weaken isolation: the sandbox masks broad output trees and rebinds
only the current workspace last, including after frozen-input overlays. Sibling workspaces and
run results stay hidden. Tests assert that order and include a real bwrap negative control, skipped
when the host cannot create namespaces. The lexical `bwrap.is_exposed` model selects the longest
mount destination and is not proof of arbitrary later ancestor remount behavior.

The second rule has a consequence worth reading as a finding: 472 of `perf-bench`'s 505 unit
directories are named outside the convention (`development_phase2_global_encoding_20260908`,
`full_graph_attempt1`), so no retention depth can place them. That concern is not an accumulation
problem with a policy answer — it is a working directory that grew inside a *product* tree, and the
fix is in the producer, not here.

## Where things are supposed to live

`merlin-storage layout` checks the root against the convention: three top-level dirs
(`runs/ artifacts/ build/`) and a closed set of concerns under `artifacts/`. Only *tracked* files are
linted by `check_artifact_layout.py`, and generated output is gitignored by design, so the convention
held in the index and drifted freely underneath it — measured 2026-09-15, **4 undeclared top-level
roots and 52 undeclared concerns against 16 declared ones**, the undeclared ones holding 35.8 GB.

Most of those 52 were well-formed `<concern>/<axis>/<unit>` products whose only fault was that the
roster was a literal inside `storage_cli.py` that nobody edited. It is data now —
`merlin/contract/storage.yaml` declares the three roots, every concern with a line saying what it
holds, and where a directory that predates the roster belongs — and three things read it: the tool,
the `test_storage_accounting.py` tests that hold it against CLAUDE.md, and the PreToolUse write guard,
which now refuses a write into an undeclared concern. That last one is what stops concern number 53:
the cost of a new concern is a reviewed line at the moment it is created, not a cleanup fifty later.

The same contract's `product_roots` maps versioned product names to their canonical
concern directories. For example, `new_product("perf-ledger", ...)` writes beneath
`artifacts/perf-studies/ledger/`, while its manifest topic and timestamped leaf name
remain `perf-ledger`. The organizer derives the corresponding legacy fold from
that mapping. Creating a new product does not move old evidence, create a legacy
alias, or require a prior cleanup pass. Readers that discover older products must
continue to consider their legacy location until that evidence has been organized.

`merlin-storage organize` applies the declared folds. A fold **moves** the directory and leaves a
relative symlink at the old name, because a product's path is quoted in manifests, reports, figures
and docs this repo does not own and cannot rewrite — the tree gets organized and every existing
citation still resolves. Folds merge (two old names can belong to one concern) and a unit-name
collision aborts that fold and rolls it back rather than silently choosing a winner. The symlink is
not a permanent fixture; it can be dropped once nothing resolves through it.

## What `prune` will and will not touch

Only classes whose safety is a *property* rather than a judgement:

| class | what it removes | why it is safe |
| --- | --- | --- |
| `store-orphans` | content-store objects with one remaining link | no snapshot references them; the next run that needs those bytes re-copies them |
| `pending-snapshots` | terminal-owned, unpinned `bundle_inputs.pending/` trees | an explicit completed/failed/acknowledged-abandoned owner is required; the suffix alone proves nothing |
| `caches` | `out/artifacts/cache/<ns>/` | declared regenerable by the layout convention (the content store is excluded — wiping it wholesale is wasteful, not unsafe) |

`dedup` and `organize` sit beside `prune` rather than inside it because neither removes anything:
`dedup` collapses names onto shared bytes and `organize` moves a directory and leaves a link. `retain`
does remove, which is why its lifecycle and naming checks fail closed rather than guessing liveness.

Everything else is **reported and left alone.** Whether a run directory is finished is not a property
this tool can read off the filesystem, and "modified recently" is not a proxy for it — a purge's own
deletions update the mtimes of the units it touched, so any liveness rule built on mtime reports the
units you just edited as the live ones. Deleting run output is an operator decision made against the
report, and `merlin/experiments/*/AGENT.md` records what a campaign still needs.

Dry run is the default. `--apply` acts, and prints every path it removed.
