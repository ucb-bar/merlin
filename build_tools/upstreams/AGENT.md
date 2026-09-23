# AGENT.md — build_tools/upstreams

## Purpose

Versioned provenance and qualification records for coordinated local companion-repository changes.
These records are not discovery defaults, target hardware facts, or compiler certificates.

## Invariants

- Preserve source revisions and exact content hashes; distinguish public catalog, candidate, schedule,
  and support roles.
- A local companion commit is not a pushed upstream release. Never infer a qualified compiler from a
  schema-valid support contract or an ABI-looking publication wrapper.
- No canonical target source is removed until its consumers and behavioral qualifiers migrate.
- Keep runtime discovery explicit; ignored local clone paths in manifests are operator records only.
