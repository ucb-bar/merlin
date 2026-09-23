# AGENT.md — packages/merlin-experiments/src/merlin_experiments/phase1/tools

Owns standalone candidate-side clients: ISA assembly/inspection, CCA/lever
inspection, self-check requests, simulator-job requests, and verdict waiting.
`merlin.targetgen.tool_registry` maps exact modules to their staged CLI filenames.
Copy only those declared files, never this directory or its parent.

Each client must run as a copied file with standard-library-only Python, without
importing Merlin or `merlin_experiments`. Host brokers live in `phase1/brokers`;
independent L0–L3 evaluators and private answers do not belong here. Keep request
protocols, redaction, staged filenames and historical evidence unchanged.

Test byte-identical staging and real client/broker IPC after moves. Source closure
and access-registry checks must continue to distinguish public clients from the
host-only phase packages. Do not add broad import or filesystem grants.
