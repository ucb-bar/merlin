---
title: Disk under out/ — why it grows and what is safe to reclaim
kind: guide
status: current
owner: infra
last_verified: 2026-09-15
related: [reproducibility, getting_started, gemmini_experiment]
code_refs: [merlin/python/merlin/common/content_store.py,
            merlin/python/merlin/common/storage_cli.py,
            merlin/python/merlin/targetgen/sandbox/bwrap.py,
            merlin/tests/conftest.py]
---

# Disk under `out/` — why it grows and what is safe to reclaim

Generated output is large by design: a whole-model run holds MLIR at every stage, an ELF, a trace and
a grade. It became large by *accident* twice, and both times the diagnosis took longer than the fix,
because the growth did not come from the results. It came from three mechanisms that copied or
retained bytes nobody asked for. Start with

```bash
merlin-storage report                # what the root costs, and how much of it is shared
merlin-storage prune                 # dry run: what is provably safe to reclaim
merlin-storage prune --apply caches  # act on one class
```

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
  (`merlin/experiments/gemmini_perf_bench/scripts/perf_snapshot.py`) does exactly that in `verify()`,
  and adopting the store there made a second snapshot fail verification because an unrelated first
  one had been chmodded during teardown. It deliberately still deep-copies, and
  `merlin/tests/infra/test_content_store.py` pins both halves of that so the saving does not tempt
  someone to weaken the check instead. Its snapshots are ~76 MB each, against 12.8 GB for a bundle
  closure, so this is a small amount of duplication bought with a real invariant.

**2. Retention that was never bounded.** pytest keeps the last *three* runs' `tmp_path` trees for every
test, passed or failed, and many tests obtain scratch space from a bare `tempfile.mkdtemp()` inside a
module-level helper — no fixture, so nothing ever removed those at all. `merlin/tests/conftest.py` now
points `tempfile` and `TMPDIR` at pytest's managed base temp root (on the big filesystem, not the small
root volume), and `pyproject.toml` keeps only the failing tests' trees from the last run. A spawned
compiler or simulator inherits `TMPDIR` and writes its intermediates in the same reclaimable place. Measured on 2026-09-15: three consecutive runs
of a passing test file leave 8 KB behind and no retained run roots, where the defaults would have
kept all three.

**3. Declared-regenerable trees that nobody reclaims.** `out/artifacts/cache/<ns>/` and
`out/artifacts/recaptures/` are PURGEABLE by the layout convention that created them (see
[the generated-output convention](../../CLAUDE.md)); being purgeable is not the same as being purged.
`merlin-storage report` prices them so the decision is informed.

## What `prune` will and will not touch

Only classes whose safety is a *property* rather than a judgement:

| class | what it removes | why it is safe |
| --- | --- | --- |
| `store-orphans` | content-store objects with one remaining link | no snapshot references them; the next run that needs those bytes re-copies them |
| `pending-snapshots` | `bundle_inputs.pending/` trees | materialization unwinds these on failure; a survivor is a run killed mid-copy and was never a complete closure |
| `caches` | `out/artifacts/cache/<ns>/` | declared regenerable by the layout convention (the content store is excluded — wiping it wholesale is wasteful, not unsafe) |

Everything else is **reported and left alone.** Whether a run directory is finished is not a property
this tool can read off the filesystem, and "modified recently" is not a proxy for it — a purge's own
deletions update the mtimes of the units it touched, so any liveness rule built on mtime reports the
units you just edited as the live ones. Deleting run output is an operator decision made against the
report, and `merlin/experiments/*/AGENT.md` records what a campaign still needs.

Dry run is the default. `--apply` acts, and prints every path it removed.
