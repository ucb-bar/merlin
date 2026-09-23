# AGENT.md — packages/merlin-experiments/src/merlin_experiments/phase1/telemetry

Host-owned raw process evidence and measurements are withheld from candidates.
`timeline.py` owns event boundaries, interval union, token buckets and arrival-time analysis.
`evidence.py` owns retained workspace/rollout copies, hashes and stream/resource observations.
`report.py` composes run records and the canonical CLI; importing any owner is target-inert.
Use the authoritative owner directly, not legacy helper aliases or duplicated token parsers.

Preserve measured versus unavailable/null distinctions, exact event order, deduplication and
legacy evidence-copy behavior. Existing token calculations are schema-specific transcript
analysis, not a second pricing/billing ledger. Cross-arm context must be explicit in installed
execution; native CLI defaults remain only at the compatibility edge. Codex-cache discovery
still recognizes the historical ancestor layout, not arbitrary relocated provider caches.
Do not infer abandoned workspaces or replace their lease/retention semantics from timing data.
