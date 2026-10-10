# Authored target inputs

`descriptor.yaml` selects experiment resources and workload policy; the other
YAML files in this directory own software semantics, hardware selection,
candidate host capabilities, and evidence vocabulary. Its explicit
`resources_root` retains harness/bundle locations during migration;
`task_root` selects public prompts in `../phase1/task`. Do not copy private
bundles or generated capsules here.
Generated releases remain under the configured artifact root. Historical receipts
retain their original paths; the legacy descriptor path is only a compatibility link.

`contracts/` owns the public prototype selected contract and its derivation residual;
`evidence_concepts.yaml` owns the evidence vocabulary. These inputs contain no
runtime backend or private oracle implementation. Shared metadata discovery is
layout-based. Runtime execution requires explicitly reviewed contract/facts data
for shared tooling or an independently reviewed execution provider. Handwritten
compiler support cannot serve as the independent grader.
Do not promote hand-authored metadata to extracted facts or certification evidence.
The optional `rtl_extraction` block in `contracts/target_contract.yaml` names this
example's Scala funct span, accumulator HW ports and Boolean build gate. Those are source-location
declarations, not legal opcode or capacity facts: the extractor still derives values
from the selected source bytes and prefers the elaborated decoder when available.
