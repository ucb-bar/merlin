# Reproduction commands (current worktree, fresh outputs)

All commands run from `/home/jeremy/Projects/merlin-loop-verification` on
`feat/loop-transform-verification`. Pinned actual producer is read-only4499.
Run only one owned qualification command per fresh output. No broad Lean build.
Historical paths are evidence; they are not portable same-live authority.

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/home/jeremy/Projects/merlin-loop-verification/src \
MERLIN_REPO_ROOT=/home/jeremy/Projects/merlin-loop-verification \
/home/jeremy/Projects/merlin/.venv/bin/python \
-m pytest -q merlin/tests/ir/test_loop_transform_qualification_classifier.py
```

The complete successor run command is recorded in its new RUN-INTENT.json and
runner log when its exact Q source cut exists and is frozen. The command invokes
`merlin/tests/ir/loop_transform_qualification.py` with:

- `--reference /home/jeremy/Projects/merlin-graph-proof-intake`
- `--lean /home/jeremy/.elan/toolchains/leanprover--lean4---v4.35.0/bin/lean`
- `--compiler-python /home/jeremy/Projects/model2MLIR/.venv/bin/python`
- `--q-source-cut` the exact new immutable Q manifest
- `--output` a NEW F artifact profile
- `--objects` a NEW F private proof tree.

Do not pass `--protocol-only` for the full eight-positive qualification. Never
rerun an unchanged failed/crashed profile to obtain a pass. Resource/import/
unknown native outcomes remain unavailable. The runner retains raw output and
continues distinct negative cases; any unavailable case forces terminal exit1.

The gate recompiles the selected24project imports privately, checks the exact
query, audits all project theorems/std3 and exact root types, and grants only a
same-process completion. CLI and replay use the actual same path. A copied or
constructed receipt/token supplies no delivery authority. Actual compiler MLIR
verification is separate from both modeled proof and native execution fidelity.
