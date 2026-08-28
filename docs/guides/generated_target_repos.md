---
title: Generated target repos
kind: guide
status: current
owner: targetgen
last_verified: 2026-08-28
related: [targetgen, adding_a_target, reproducibility, target_resolution]
code_refs: [merlin/python/merlin/targetgen/publish.py, merlin/python/merlin/targetgen/oot_runner.py, merlin/targets/publish.yaml]
---

# Generated target repos

A certified codegen package can be exported as its **own standalone git repo**, so someone with no
Merlin checkout can clone it, build it, and use the compiler. This guide covers what those repos
contain, how to produce one, and what a certification in one actually claims.

The bridge is `merlin.targetgen.publish` (`merlin-target-publish`). It never hardcodes a remote and
never pushes to a network remote without an explicit confirmation token.

## What a published repo is

One repo per target, one commit per promotion, one branch per version:

```
<repo-name>/
  README.md            generated: what it is, how to build, what the certification claims
  CMakeLists.txt       buildable at the ROOT
  include/ lib/ tools/<target>-opt/ test/
  manifest.yaml        the package manifest, rewritten with the repo root as {package}
  payload/             family assets (schedule+knobs for a vector schedule; dialect/lowering/
                       contracts for an out-of-tree backend)
  .merlin/             committed metadata: manifest, provenance, certification, CHAMPION
```

The frozen unoptimized baseline goes to a shared `baseline` branch (published first, so
before-vs-after is externally visible); each certified champion gets its own
`stable/<package_id>`. The history *is* the provenance trail.

## The repo name is not the target key

`resolve_repo_name()` resolves the public name independently of the target key, because the two
answer different questions. The host target is keyed `rvv` — its payload is a vector schedule — but
the repo holds **all host codegen**: scalar is the case where no vector unit is declared, not a
second target. So it publishes as `host-mlir`.

There is deliberately no separate scalar repo. Splitting by ISA extension would put the ABI, the
lowered-memref calling convention and the lowering in two places and force a "which repo owns this
op" decision on every addition, while the scalar side has no payload of its own to justify it.

Precedence: `--repo-name` > `MERLIN_PUBLISH_REPO_NAME_<TARGET>` > `repo_names` in
`merlin/targets/publish.yaml` > the default `<target>-mlir`. The build contract is untouched — the
tool is still `<target>-opt`.

## Publishing

Certification comes first. A certify run records its verdict into its own run dir; carry it onto the
package, then promote, then publish:

```sh
# 1. certify — one run per rung. --simulator verilator is the cycle-accurate oracle;
#    spike is functional only (see "What a certification claims").
P=out/artifacts/targets/gemmini/<package_id>
for r in $P/rungs/*.interface.mlir; do
  .venv/bin/python -m merlin.targetgen.oot_runner \
    --package $P --input "$r" --run-id "vcert_$(basename $r .interface.mlir)" \
    --simulator verilator
done

# 2. record the verdicts onto the package manifest
.venv/bin/python -m merlin.targetgen.publish record-cert \
  --target gemmini --champion <package_id> \
  --results out/runs/gemmini_contract/runs/gemmini-contract/vcert_*/results.yaml

# 3. promote to the single champion, then publish
.venv/bin/python -m merlin.targetgen.publish promote --target gemmini --champion <package_id>
.venv/bin/python -m merlin.targetgen.publish publish --target gemmini --dry-run
```

Step 2 is not bookkeeping. Without it nothing can ever be promoted: `promote` writes
`publication.certification` only when the gate passes, and the gate reads that same field. Until
`record_certification` existed, the loop was unsatisfiable, so a package carrying a real out-of-tree
dialect could never be published and the only eligible champion was a hand baseline with no dialect
— whose published repo builds a stub. If a clone builds a stub `<target>-opt`, this is why.

### Pushing

`--dry-run` plans; `--execute` clones, commits, tags and pushes. A **network** remote additionally
requires `--confirm-push <fingerprint>`, printed when the push is refused. Verify against a local
bare remote first:

```sh
git init --bare /scratch/$USER/tmp/pub/gemmini-mlir.git
.venv/bin/python -m merlin.targetgen.publish publish --target gemmini \
    --remote file:///scratch/$USER/tmp/pub/gemmini-mlir.git --execute
```

## Consuming one (a fresh setup, no Merlin checkout)

```sh
git clone -b stable/<package_id> <remote> gemmini-mlir && cd gemmini-mlir
cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DMLIR_DIR=<llvm-install>/lib/cmake/mlir -DLLVM_DIR=<llvm-install>/lib/cmake/llvm
cmake --build build
./build/bin/gemmini-opt --help
```

The only external requirement is an LLVM/MLIR install matching the `llvm:` block in `manifest.yaml`
(version **and** commit — the OOT C++ API moves between them). This repo pins one under
`third_party/llvm-install`; the version a package was built against is recorded in its manifest, so
a mismatch is a diagnosable error rather than a wall of template output.

Read `.merlin/` before trusting a number: `CHAMPION` names the package, `provenance.yaml` the
lineage, `certification.yaml` the verdict and the tier it was earned at.

## What a certification claims

The gate accepts a pass from any oracle, so the recorded verdict carries the oracle with it. These
are different claims and the repo states which one it is:

| tier | means |
|---|---|
| `derived_from_rtl: true, cycle_accurate: true` | the hardware ran it; cycle counts are real |
| `derived_from_rtl: true`, not cycle-accurate | RTL-derived, but not a timing result |
| neither (e.g. `spike_gemmini_functional`) | numerically correct; **not** an RTL or timing result |
| no tier recorded | says so in words — never read it as an RTL result |

Three rules keep a headline from outrunning its evidence:

- a multi-rung certification records as its **weakest** rung, not its best;
- a results file naming no oracle records `UNKNOWN`, never a benign default;
- the README states the tier in prose next to the verdict, so `pass` alone cannot be read as the
  strongest thing it might mean.

Quoting a bare score has burned this project before (a `20/20` that was L2-only, a `31/31` against a
hardware revision the artifact never named). Cite the tier, not the number.

## Gotchas

- **A stub build means an unrecorded certification**, not a broken package — see step 2 above.
- **Generated output is not tracked.** Packages live under `out/artifacts/targets/<target>/` and are
  gitignored apart from hand-authored baselines and promoted champions; a published repo is how a
  package leaves this checkout.
- **The single-champion invariant is real**: `promote` clears any prior champion, so exactly one
  package per target is flagged. Publishing selects it unless `--champion` overrides.
- **Local remotes skip the push confirmation** by design, so a `file://` rehearsal is cheap and
  proves the clone-and-build path end to end.
