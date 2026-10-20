# Accelerator semantics and integration selections

`software-spec.yaml` is the user-authored accelerator semantic entry point.
Keep its mathematical families, numeric policies and logical effects separate
from the selected host capabilities in `../platform/`. Do not duplicate the host
operation roster or add compiler algorithms, schedules or model-specific mapping.

`descriptor.yaml` selects experiment resources, public iteration policy and
platform inputs through explicit paths. Its `resources_root` retains the selected
harness/bundle locations; `task_root` selects public prompts in `../phase1/task`.
Do not copy private bundles or generated capsules here.

`contracts/` retains the public prototype contract and its derivation residual;
`evidence_concepts.yaml` owns the discovery vocabulary. Shared metadata discovery
depends on their relative names. These contain no runtime backend or private
oracle implementation. Their source-location declarations are not extracted
opcode, capacity or legality facts.

Hardware, board and host input selections live in `../platform/`; operator probes
live only in `../verification/`. Generated facts, audit reports, captures, capsules
and receipts stay under the configured output root. Preserve historical frozen
paths and hashes. Runtime qualification requires explicit reviewed facts/contract
data and exact execution evidence; handwritten compiler support cannot grade a
fresh compiler independently.
