# src/merlin/development/backend/tools

Owns standalone candidate-side clients: ISA assembly/inspection, CCA/lever
inspection, self-check requests, simulator-job requests, and verdict waiting.
`merlin.development.targetgen.authoring.tool_registry` maps exact modules to their staged CLI filenames.
Copy only those declared files, never this directory or its parent.

Each client must run as a copied file with standard-library-only Python, without
importing Merlin or `merlin.experiments`. Host brokers live in `backend/brokers/`;
independent L0–L3 evaluators and private answers do not belong here. Keep request
protocols, redaction, staged filenames and historical evidence unchanged.

The self-check client forwards Codex's own `CODEX_THREAD_ID` as the request's optional
`caller_thread_id`: operator accounting only, never read or trusted by the broker.
Its three declared files have distinct owners: `selfcheck.py` defines and
validates flags, then submits and collects requests;
`qa_selfcheck_protocol.py` owns shared wire/feedback helpers, and
`qa_selfcheck_sweeps.py` handles advisory sweeps. Installed and copied-file
imports must select these same owners; never import Merlin in the copied branch.
Suggested follow-up commands must name the entry client, not a sidecar file.
Test byte-identical staging and real client/broker IPC after moves. Source closure
and access-registry checks must continue to distinguish public clients from the
host-only phase packages. Do not add broad import or filesystem grants.

`result_details` is staged as `qa_result_details.py`, the shared public feedback
index/detail reader. Bound paths to `qa/cases/`, verify raw content digests, and
retain legacy inline rows. Only trusted locked publication uses its immutable
read-only views; ordinary consumers receive independent mutable copies. These
public diagnostics never replace producer custody or issue certificates.
