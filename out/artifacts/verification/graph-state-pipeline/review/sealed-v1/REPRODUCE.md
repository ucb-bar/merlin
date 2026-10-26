# Reproduction and integrity

Use `delivery.json`, `dependencies.json`, `qualification-v1/PLAN.json` and `MANIFEST.json` as the exact source/profile/roster/evidence selectors. The code lives on primary `feat/pass-verification`; production is read-only pinned4499. The complete additive patch and an incremental patch against the recorded current HEAD are separate. Apply only the appropriate reviewed owned patch; satisfy the dependency closure explicitly. No merge/reset/stash or shared production/spec edit is implied.

Existing Linux tool: Lean4.35.0 absolute executable with SHA bf8d54e4714cc4b03d3f6bb34c83b7202b87e49c8bfcbff6895c085bb90ceb38, `-j1`, exact recorded core object/sidecar/host bindings. Python/xDSL/parser sources and actual producer code identities are in the frozen plan and receipts. No elan/lake/download/toolchain write or affected SDK reselection is supported. Full Lean-MLIR/broker route is unavailable without an independently selected matching manifest.

The original single batch command was:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:merlin/tests/ir \
 MERLIN_REPO_ROOT=/home/jeremy/Projects/merlin \
 .venv/bin/python out/artifacts/verification/graph-state-pipeline/review/qualify.py
```

Do not rerun it over the existing qualification directory. It intentionally refuses an existing batch. A newly authorized reproduction needs a new owned root/controller/source/profile/roster selection and fresh evidence; it does not mutate or promote this packet, the literal timeout, the elision crash or SDK incident. Receipts use exact original paths; changing path strings/hashes is not laptop qualification.

The frozen suite supplies the sole measured originals. Its command and all output/XML are in PLAN/BATCH. One fresh-process replay batch then independently reprojects each complete boundary, exactly reproduces each retained object using its original query path, checks fresh proofs/audits, and postchecks frozen identities. No duplicate original proof/audit population is run after the suite. The positive development protocol replay test is explicitly excluded; negative replay probes remain classified as adversarial tests.

For a read-only integrity check, hash every repository-relative path in MANIFEST and compare the recorded SHA256/byte count. Then check the exact owned sources against delivery snapshots and verify `git apply --check --reverse owned.patch` without applying it. Integrity is not requalification or review approval. Old55 sources/capsules and their original patches/manifests remain independent preserved evidence.
