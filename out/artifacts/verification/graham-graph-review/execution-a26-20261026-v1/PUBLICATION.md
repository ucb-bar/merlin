# Publication boundary

This publication adds source/data review results to `feat/pass-verification-verf`
without merging compiler source commits from `feat/pass-verification`. The tested
source baseline is **c62f81a8a11347ff53b60101fa75280e69d002a6**; the publication
branch's prior tip is **5a7f872db33d378d6b0c4a294e4099e80efd4956**.

[PUBLICATION_INDEX.json](PUBLICATION_INDEX.json) names and hashes the 836 copied
run/source files, including six exact plain Lean diagnostic sources. It excludes
itself and the newly written publication Markdown/checks/source index.
The original run [OUTPUT_MANIFEST.json](OUTPUT_MANIFEST.json) remains unchanged;
it describes the run before these publication additions. Historical manifests
are not rewritten to make omitted build products appear portable.

Included: baseline/source hashes, source/claim reviews, the 32-control parser
test and raw outputs, official SDK metadata/member inventory, finite diagnostic
sources/commands/results, frozen profile manifests, compilation/audit raw output,
case roster, exact checked theorem types, preserved first failures and the
standalone same-statement tail proof successor. A proof command naming an omitted
absolute output path is retained evidence, not a runnable portable authority.

Excluded: the 925899528-byte SDK archive, extracted SDK, Python dependency
environment, staged project copies, compiled objects, private runtime approvals
and any old held SDK/context. Their paths/hashes in reports are provenance only.

The thirteen duplicate `baseline-a26-20261026-v1/inputs/reviewer` copies are also
excluded. For each `BASELINE_AUDIT.json:review_inputs` row, the same bytes are
already in this branch at `original_path`, and permanently bound by commit
**5a7f872db**. The publication checker verifies those Git blobs against the
recorded SHA256/size. This avoids duplicating the prior reviews while preserving
their exact input identities.

To reproduce, fetch **c62f81a8a** from `feat/pass-verification` into a separate
source checkout and select its exact fixed packets. Review the published profile
pins and use a separately verified fresh SDK/output context. The retained paths
must not be rehashed into historical receipts or promoted to live compiler gate
authority. Some historical originals/receipts/parser identities remain missing.

New Markdown supplements link the execution results to deliverables 1–3; their
original manifest-bound review documents remain unchanged. The Graham handoff
also points to the new results and outstanding proof/qualification work.

## Publication validation

[PUBLICATION_VALIDATION.json](PUBLICATION_VALIDATION.json) records passing file
integrity, worker manifests, Markdown links, Python source parsing, artifact
layout, documentation, target-name/regex boundaries, answer-key and whitespace
checks. The full structure gate has one pre-existing failure: unchanged
`src/merlin/runtime/backends/zephyr_model.py` is 3557 lines against its 3537-line
ceiling. Both that file and the ceiling ledger are byte-identical to the remote
review base. Every other structure subcheck passes. No unrelated implementation
or ratchet edit is included in this report publication.
