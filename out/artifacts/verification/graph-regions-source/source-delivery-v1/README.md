# Owned source delivery — partial review, current binding UNQUALIFIED

Review `STATUS-CURRENT.json` and `DEPENDENCY_DRIFT-v1.json` one directory above.
The current dependency association is UNQUALIFIED despite the completed 53-test run.
Preserve the recorded selection and obtain coordinator resolution before qualification;
do not rerun/reselect/rewrite manifests to promote this packet.

Apply/review `owned.patch` against the primary `feat/pass-verification` checkout with
the exact read-only prerequisites in DEPENDENCIES.json. It contains 21 new owned
files (15 Lean modules, five Python leaves, one test file), 198,019 patch bytes.
`patch-review.json` records successful `git apply --reverse --check` against the
current source bytes. Nothing is staged or committed, and no foreign implementation
or external compiled object is copied into this patch.

MANIFEST.json binds every original/snapshot byte. DEPENDENCIES.json identifies all
52 project modules, the 2,132 full parser-discovered source/object imports, installed
Core/Sem manifest/packages, preserved old 10 sources and toolchain runtime inventory.
External objects/toolchain and trusted host projection remain explicit dependencies.
Read REPORT.md and PAPER_ROW.md one directory above for the theorem and limits.

Qualification command, with a NEW unused profile name:

```sh
MERLIN_GRAPH_SOURCE_PROFILE=<new-name> .venv/bin/python -B -m pytest \
  merlin/tests/ir/test_lean_graph_regions_source.py -q
```

Replay one v2 receipt into a NEW private destination:

```sh
.venv/bin/python -B -m merlin.verify.lean_graph_regions_source \
  out/artifacts/verification/graph-regions-source/qualification-v2/checked-native/EVIDENCE.json \
  out/build/graph-regions-source/<new-replay>
```

Replay preserves selected original source/recipe paths and the oracle's lexical
virtual-environment invocation/configuration; it builds private project objects
with the pinned installed Lean and existing package selection. Missing selections
are unavailable, never passing. Receipts use exact absolute identities: moving
artifacts/environments requires a fresh capture/qualification, not editing old
references to obtain acceptance. No tool download is performed. The unavailable
full Lean-MLIR route, shared pipeline, native implementation law and approval are
not silently supplied by replay.
