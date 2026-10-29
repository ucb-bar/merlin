# Expert kernel database

The local corpus has one canonical root:
`out/artifacts/kernel-mining/gemmini/expert-kernel-database-v1/`.
Start with its `README.md`, `manifest.json` and `database.sqlite`.
Large original binaries and generated analyses stay outside Git. This directory
tracks the input pins and reproducible preparation recipe.
The canonical root is a relative symlink into a versioned product allocated by
`merlin.common.artifacts.new_product`; it does not duplicate the payload.

For the proposed six-condition, no-repeat raw versus raw-plus-CCA study, see
[CCA-COVERAGE-STUDY.md](CCA-COVERAGE-STUDY.md) and the exact
[subset selection](subsets.v1.json). The prepared subset alias is
`out/artifacts/kernel-mining/gemmini/expert-kernel-subsets-v1/`.
These are offline input packages; their native launch path is not yet qualified.

The two parallel collections are:

- `originals/`: Jack's three distinct model ELFs (ResNet50, SmolVLA, TinyLlama),
  pinned upstream Exo schedules/license, their existing publicly derived C/ELF
  examples and dependencies, and the supplied Exo LLM C/header. Older duplicate
  ResNet model archives and extra copies of the same models are not included.
- `kernels/`: exact-byte-deduplicated machine spans, source definitions and
  original Exo exported-procedure recipes. Each body has an `index.json` listing
  every originating model/source occurrence and its analysis paths.

`mappings/` retains model addresses, original source spans, source export/factory
relations, static call sites and CCA JSON/MLIR for accelerator spans. Portable
JSONL and CSV projections accompany SQLite. Bytes shared between models point to
one body; occurrence-local addresses, call context and analyses remain distinct.

## Rebuild with explicit original inputs

Use Python 3.12+ with Merlin, its mining/experiments packages, the C-parser extra,
xDSL, `zstd`, and the selected LLVM disassembler. Source is parsed, not imported;
the original Exo derived C/ELFs are inspected, not executed or rebuilt here.

```sh
python examples/gemmini/reference-kernels/prepare_inputs.py \
  --input-root "$REFERENCE_ARCHIVE_ROOT" \
  --upstream-root "$PINNED_EXO_SOURCE_ROOT" \
  --bundle-root "$PINNED_PUBLIC_EXO_BUNDLE" \
  --contract "$SELECTED_CONTRACT" --facts "$SELECTED_FACTS" \
  --out "$TMPDIR/expert-reference-inputs.json"
CORPUS_ROOT="$(python -c 'from merlin.common.artifacts import new_product; print(new_product("kernel-mining", version=1, target="gemmini", update_latest=False).path)')"
python -m merlin.mining.reference_database build \
  --spec "$TMPDIR/expert-reference-inputs.json" \
  --out "$CORPUS_ROOT"
python -m merlin.mining.reference_database verify "$CORPUS_ROOT"
```

Use a new output identity after a completed build. The template verifies the
original archives, every selected member, upstream revision and original derived
artifacts. Different facts or references require an explicitly updated selection.
The recorded upstream Exo revision is
`defe17283d15553ca9d6ad70b2e9f4c22ff36ce2` from
[exo-lang/exo](https://github.com/exo-lang/exo/tree/defe17283d15553ca9d6ad70b2e9f4c22ff36ce2).
Corresponding licenses are retained in `originals/`.

## Query

```sql
SELECT model, occurrence_id, kernel_id, address, accelerator_sites, analysis
FROM model_to_kernel WHERE accelerator_sites > 0 ORDER BY model, address;

SELECT kernel_id, COUNT(*) AS uses FROM occurrences
GROUP BY kernel_id HAVING COUNT(*) > 1;

SELECT * FROM unique_accelerator_kernels;
```

## Evidence boundaries and fair use

This is offline bottom-up reference preprocessing. The builder reuses the current
ISA/fact-driven CCA lifter, retains unknown fields and stores its implementation
and input pins. It generates no compiler changes, winning configurations, hidden
shape feedback or top-down optimization choices. Preparation time is recorded
separately from author time; treatment comparisons must state which arms receive
the derived bank and use the same underlying raw reference selection.

The three model ELFs are stripped. Their model-to-static-span and call-site maps
are recoverable; original layer names and exact upstream source correspondence
are not thereby recovered. Name matches in symbol-bearing upstream examples are
explicit discovery links, not equivalence proofs. Shared bytes establish a storage
deduplication identity, not context-independent semantics.

Accelerator spans have CCA JSON and MLIR where the selected lifter can reconstruct
facts. Host-only code has structural diagnostics. Python/C definitions and export
recipes retain source-parser results; unsupported source-to-command semantic lifts
stay unknown. No source equivalence, numerical correctness, physical traffic,
dynamic invocation count or performance qualification is implied.

## Prepared v1 snapshot

The prepared selection contains 22 original artifacts and three distinct Jack
models. Its 422 occurrences map to 358 exact-byte-distinct bodies: 294 machine
spans, 34 C definitions, 12 Python definitions and 18 source export recipes.
Sixty-four duplicate occurrences share bodies while preserving their contexts.
All 65,838 selected accelerator instruction sites in the Jack models occur in
73 analyzed spans. Eight additional accelerator spans come from the original
upstream-derived Exo examples, for 81 CCA JSON/MLIR pairs overall.

File pins, SQLite integrity and foreign keys verify. All 81 CCA MLIR products
round-trip to their JSON records. Focused corpus/CCA/source/access suites report
131 passed and 3 skipped. The repository-wide structure check still reports
19 issues in other files (dependency direction, legacy document paths and module
size); this addition does not fix those unrelated failures. No workload or
hardware execution was launched for this preparation.
