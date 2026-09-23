# Retained synthesis references

These six public sidecars were moved byte-for-byte from
`merlin/contract/capsules/profiles/` at the recipe-ownership migration.
Their original headers and provenance are retained, including historical commands.
They are selected explicitly by `examples/<target>/experiment.yaml`.

They are reference inputs, not a claim that current synthesis reproduces every byte.
In particular, Atlas includes a manually reviewed model-gate override documented in
its target descriptor. Do not remove that override by regenerating in place.

Current generation uses:

```sh
python build_tools/scripts/synth_capsule_corpus.py --target TARGET --write --json
```

This produces a fresh versioned artifact under `out/artifacts/verification/<target>/`.
`--check` separately compares derivation against the selected reference and reports
drift without changing it. Review any replacement before changing the definition.

## Retained conformance requirements

`conformance/<target>.yaml` holds six byte-preserved public requirements derived
from target facts and captures. These are historical reference inputs for audits,
not authored Phase 0 recipes or a corpus to grade by default. Fresh derivations
belong in versioned run/product artifacts and need review before an experiment
selects them. The relative `merlin/contract/capsules/conformance` symlink
preserves old checkout references and frozen input inventories. New code reads
this directory through `merlin.targetgen.corpora.conformance_reference` or an
explicit file path. The old commands in the preserved headers are historical;
the spec bytes were not rewritten during relocation.
