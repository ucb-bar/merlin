# AGENT.md — build_tools/upstreams

## Purpose

Versioned provenance and qualification records for coordinated local companion-repository changes, for
their vendored copies under `examples/*/support`, and the pins of the support packages Merlin fetches
instead of copying (`target_support.json`, `pinned_support`).
These records are not discovery defaults, target hardware facts, or compiler certificates.

## Invariants

- Preserve source revisions and exact content hashes; distinguish public catalog, candidate, schedule,
  and support roles.
- A local companion commit is not a pushed upstream release. Never infer a qualified compiler from a
  schema-valid support contract or an ABI-looking publication wrapper.
- No canonical target source is removed until its consumers and behavioral qualifiers migrate.
- Each companion's provider root is vendored byte-identically at `examples/<example>/support`, with its
  source commit, tree and date in `examples/<example>/SOURCE.yaml`; the manifest's `vendored` block
  must agree with that record. The one permitted difference is a file the record lists under
  `normalized` (with its companion blob id and the change), such as a provenance record whose
  absolute vendoring-host paths were made repo-relative: this repository is public. That vendored copy is the runtime default when `MERLIN_TARGET_PATH` is
  unset; an explicit value always wins. Ignored local clone paths in manifests are operator records only.
- Re-vendoring is a new copy from a recorded companion commit, never an in-place edit of a vendored tree.
- Chip packages are pinned, not vendored: `pinned_support` records each one's `{target, repo, branch,
  commit, tree}` and ONE base pin. Every package's `provider.yaml` `extends.commit` (and the gitlink of a
  base it carries as a submodule) equals the base pin; the base's in-repo copy records that same commit.
  Editing a package is a commit in its repository and one sha bumped here
  (`build_tools/scripts/vendor_support.py pin <target> <sha>`, which fetches it and records its tree).
  `merlin/tests/infra/test_support_pins.py` holds the record together.

## Fresh compiler experiments

Handwritten compiler support, copied kernel headers and reference measurement
adapters must not be vendored into the fresh experiment repository or used as its
support defaults. This requirement overrides the general vendoring workflow for
final reference implementations. Derive experiment support independently from
admitted RTL and the minimal reviewed software contract; keep the final golden
outside author-visible source and tool closures.
