# Independent retained TinyLlama policy-correspondence replay

Planner replay of the W8A8 modular-policy milestone, 2026-10-22 (US/Pacific). **Passed** for
the complete changed retained boundary, under the explicitly named draft-policy semantics.
[Summary](SUMMARY.json), [run identity](run.json), [regenerated receipt](replay/receipt.json).

The replay regenerated the actual authoritative before/after exports, reran structural and
complete target/reference checking, freshly audited the actual theorem roots, and required
the regenerated typed evidence, runtime identity and subject hashes to match the original.
It used the original frozen interpreter/proof capsule; no runtime file's modification time
was newer than the original freeze. All **1,309 frozen records and 68 owned source files**
matched before and after. No source, original receipt, frozen runtime, branch or index was
modified. The exact wrapper used is retained as [wrapper-original.py](wrapper-original.py).

`PolicyW8A8Correspondence.check_sound` relates complete outcomes to the source-derived draft
reference, with **zero additional numerical laws** and no original-model success premise.
It can include equal stuck/UB/poison outcomes. This replay does **not** establish intended
policy approval, exact original-float equivalence, successful initialized whole-model
execution, a common numerical-model witness, native fidelity or selected final capture binding.
The stronger numerical route still refuses quantizer 1533. All broad admission flags remain
false; the regenerated receipt remains `abstracted`.

This is an independent same-environment receipt replay, not a clean-environment reproduction,
new source build or combined-pipeline qualification. Separately, the planner's
[W8A8 delivery inspection](../w8a8-delivery-review-v1.json) reconstructs all 68 additive patch
payloads and matches their manifest and saved source copies. That source inspection is not
policy review or patch application.

The compiler invocation intentionally uses the retained **virtualenv Python path**. Its
recorded resolved executable is the shared base Python; invoking that base path directly
would lose the virtualenv's MLIR imports. `run.json` preserves both identities.

To replay again from main, use a new destination name and the same explicit environment:

```sh
export ELAN_TOOLCHAIN="$PWD/out/build/lean-shared-passes/phase7-worktree/out/build/fresh-environment/toolchain"
export LEAN_SYSROOT="$ELAN_TOOLCHAIN" LEAN_NUM_THREADS=2
export PATH="/home/jeremy/.elan/bin:$PATH"
export MERLIN_LEAN_REQUIRED=1
export MERLIN_REPO_ROOT="$PWD/out/build/lean-w8a8-correspondence/repo"
export MERLIN_M2M_PYTHON="$PWD/out/artifacts/verification/lean-review/20261017T204007Z/integration/out/build/lean/mlir-venv/bin/python"
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -u \
  out/artifacts/verification/merlin-passes/progress/planner/eight-day-paper/integration/replay_w8a8.py \
  --name w8a8-independent-replay-v2
```

Relocation/combined builds require new identities and fresh qualification. Do not rebuild this
capsule, overwrite old evidence or promote provisional flags to obtain a paper verdict.
