# Authored target inputs

`descriptor.yaml` is the single source for this example's target setup and experiment
policy. Its explicit `resources_root` retains harness/bundle locations during
migration; `task_root` selects public prompts in `../phase1/task`. Do not copy
private bundles or generated capsules here.
Generated releases remain under the configured artifact root. Historical receipts
retain their original paths; the legacy descriptor path is only a compatibility link.

`contracts/` owns public prototype capability declarations and derivation residuals;
`evidence_concepts.yaml` owns the evidence vocabulary. These inputs contain no
runtime backend or private oracle implementation. Shared metadata discovery is
layout-based; runtime support still requires explicit selection of the OOT provider.
Do not promote hand-authored metadata to extracted facts or certification evidence.
The optional `rtl_extraction` block in `contracts/target_contract.yaml` names this
example's Scala funct span, accumulator HW ports and Boolean build gate. Those are source-location
declarations, not legal opcode or capacity facts: the extractor still derives values
from the selected source bytes and prefers the elaborated decoder when available.
