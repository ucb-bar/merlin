# Performance experiment tooling

Owns portfolio probe execution, scoped qualification, source-pair preparation/admission
and bounded whole-model experiment workers, not compiler performance contracts.
Implementations retain their logical `merlin.perf` imports: `isolated_probe_provider`,
`controlled_context_provider`, `paired_context_provider`, `host_region_qualifier`,
`host_physical_transition_qualifier`, `lane_migration_qualifier`,
`source_contraction_preparation`, `source_convolution_preparation`, `source_program_pair`,
`source_initializer_elision`, `source_program_pair_provider` and `analysis_worker`.
Providers consume the concrete portfolio probe owner and its admitted revision session.
Core retains pure witness extraction, compiler IR and cost/evidence primitives; do not
introduce core-to-experiment callbacks or duplicate scientific implementations.
All these execution/admission modules retain host-private identities in the shared access
registry, including when the optional distribution is absent. Preserve logical host-policy
roles and archived bytes; moving files does not expand historical V2 source-policy rosters.
Core owns `merlin.perf.__init__`; do not add a competing namespace initializer here.
Preserve verifier provenance, admission gates and host-only evaluation boundaries.
The analysis worker loads the canonical installed emission analyzer and exchanges JSON
with its parent. Requests supply explicit source and contract-root paths. Honor the
active canonical import finder, verify its source origin and compile fresh loader-provided
source; never fall back to native stages or cached bytecode. Preserve deadlines,
process-group cleanup, frozen transport and output checks. Moving its owner does not
qualify installed standalone Phase 2 execution or arbitrary managed-worker deployments.
