"""Keep generated corpus destinations separate from retained source inputs."""

from __future__ import annotations

from pathlib import Path


def _require_distinct_corpus_destinations(te, *, output_root: str | Path, evidence_root: str | Path | None) -> None:
    """Keep generated Phase 0 bytes out of retained and descriptor-selected input corpora.

    The legacy corpus is still addressable by frozen benchmark grants. Rewriting it
    under a new derivation would silently change those experiments, even when the
    caller explicitly supplied ``--output-root``. Resolve paths before comparing
    them so an alias cannot bypass this source-ownership check.
    """
    from merlin.common.paths import checkout_root
    from merlin.targetgen.corpora import capsule_corpus_roots

    # In source mode MERLIN_REPO_ROOT may select an external experiment
    # workspace that has no copy of this checkout's historical registry. The
    # implementation checkout still owns those retained paths; an installed
    # wheel instead reads its bundled registry. Both modes keep malformed or
    # missing selected metadata fail-closed.
    sources = [*capsule_corpus_roots(owner_root=checkout_root())]
    selected = getattr(te, "capsule_corpus", None)
    if selected:
        sources.append(Path(selected))
        # Minimal independent hardware descriptors select no input corpus.
        # Their legacy sibling methods require a primary corpus and must not
        # discover or dereference a nonexistent historical selection.
        for method_name in ("graded_roots", "perf_roots", "hidden_roots"):
            method = getattr(te, method_name, None)
            if callable(method):
                sources.extend(method())
    sources = sorted({Path(source).expanduser().resolve() for source in sources})
    # A frozen run executes from its own sealed source snapshot below the evidence root. A corpus path
    # inside that snapshot is the run's private read-only copy of the checkout, not a retained corpus
    # destination. The original checkout and the descriptor-selected corpora stay protected.
    frozen = Path(evidence_root).expanduser().resolve() / "private" / "source" if evidence_root is not None else None
    if frozen is not None:
        sources = [source for source in sources if not (source == frozen or frozen in source.parents)]
    for field, raw in (("output_root", output_root), ("evidence_root", evidence_root)):
        if raw is None:
            continue
        destination = Path(raw).expanduser().resolve()
        for source in sources:
            if destination == source or destination in source.parents or source in destination.parents:
                raise ValueError(
                    f"Phase 0 {field} {destination} overlaps source capsule corpus {source}; "
                    "select a separate run artifact destination"
                )
