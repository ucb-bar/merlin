"""Build and render the evidence report.

``build_evidence`` walks a SourceManifest, records discovered files, and detects concepts by
matching a per-target keyword vocabulary against filenames and the first lines of docs. This is
deliberately shallow: it surfaces *candidates* for the human-reviewed synthesizers, not a
parsed model of the hardware.

The concept vocabulary is DATA, not a core literal: each target owns an
``evidence_concepts.yaml`` in its discovered target dir (``rtl.facts.target_base`` — the curated
``merlin/targets/<t>`` or the generated ``out/artifacts/targets/<t>``). A target with no such file
contributes no concepts (honest: nothing authored ⇒ nothing detected), so the core stays
target-name-free and a brand-new accelerator plugs in by dropping its vocabulary beside its sources.
"""
from __future__ import annotations

from pathlib import Path

from merlin.common.yaml import load_yaml

from ..io import first_lines
from ..ingest.docs import discover_docs
from ..ingest.examples import discover_examples
from ..ingest.scala_chisel import discover_scala
from ..ingest.source_manifest import SourceManifest
from ..rtl.facts import target_base
from .store import Concept, Evidence, FileRecord

def _concept_vocabulary(target_name: str) -> dict[str, list[str]]:
    """The per-target concept vocabulary, loaded from ``<target-dir>/evidence_concepts.yaml`` (data,
    not a core literal). Returns ``{concept: [keyword, ...]}`` — lowercase substrings that count as
    supporting evidence when seen in a filename or a doc's first lines. Empty if the target has no
    vocabulary file (nothing authored ⇒ nothing detected)."""
    path = target_base(target_name) / "evidence_concepts.yaml"
    if not path.is_file():
        return {}
    doc = load_yaml(path) or {}
    concepts = doc.get("concepts", {})
    return {name: list(keywords) for name, keywords in concepts.items()}


def _kind_for(path: Path) -> str:
    suf = path.suffix.lower().lstrip(".")
    if suf in ("md", "rst", "txt"):
        return "doc"
    if suf == "scala":
        return "scala"
    return "example"


def _summary_for(path: Path, root: Path | None) -> str:
    """Short summary: first non-empty doc line, else a filename-derived phrase."""
    if _kind_for(path) == "doc":
        for line in first_lines(path, 8):
            stripped = line.lstrip("# ").strip()
            if stripped:
                return stripped[:120]
    return path.stem.replace("_", " ").replace("-", " ")


def _rel(path: Path, roots: list[Path]) -> str:
    for r in roots:
        try:
            return str(path.relative_to(r.parent if r.is_file() else r))
        except ValueError:
            continue
    return path.name


def build_evidence(manifest: SourceManifest) -> Evidence:
    """Discover files and detect concepts for the manifest's target."""
    roots = [Path(p) for p in (manifest.source_dirs + manifest.examples_dirs + manifest.scala_roots)]
    docs = discover_docs(manifest)
    examples = discover_examples(manifest)
    scala = discover_scala(manifest)

    records: list[FileRecord] = []
    # haystacks: (rel_path, searchable_lowercase_text)
    haystacks: list[tuple[str, str]] = []
    for path in sorted(set(docs + examples + scala)):
        rel = _rel(path, roots)
        kind = _kind_for(path)
        summary = _summary_for(path, None)
        records.append(FileRecord(path=rel, kind=kind, summary=summary))
        text = rel.lower() + "\n"
        if kind == "doc":
            text += "\n".join(first_lines(path, 40)).lower()
        haystacks.append((rel, text))

    table = _concept_vocabulary(manifest.target_name)
    concepts: list[Concept] = []
    for concept, keywords in table.items():
        cites = sorted(
            {rel for rel, text in haystacks if any(k in text for k in keywords)}
        )
        if cites:
            concepts.append(Concept(concept=concept, evidence=cites))

    sources = {
        "source_dirs": list(manifest.source_dirs),
        "examples_dirs": list(manifest.examples_dirs),
        "scala_roots": list(manifest.scala_roots),
        "source_urls": list(manifest.source_urls),
    }
    return Evidence(
        target=manifest.target_name,
        sources=sources,
        files=records,
        concepts=concepts,
    )


def render_markdown(evidence: Evidence) -> str:
    """Render a human-readable ``evidence_report.md``."""
    lines: list[str] = []
    lines.append(f"# Evidence report — {evidence.target}")
    lines.append("")
    lines.append(
        "Conservative, deterministic evidence collected by TargetGen. This records what was "
        "*found* in the provided sources; it is **not** a parsed model of the hardware. "
        "All synthesized artifacts derived from it must be human-reviewed."
    )
    lines.append("")
    lines.append("## Sources")
    lines.append("")
    for key, vals in sorted(evidence.sources.items()):
        joined = ", ".join(vals) if vals else "(none)"
        lines.append(f"- **{key}**: {joined}")
    lines.append("")
    lines.append(f"## Files discovered ({len(evidence.files)})")
    lines.append("")
    if evidence.files:
        lines.append("| kind | path | summary |")
        lines.append("| --- | --- | --- |")
        for f in evidence.files:
            lines.append(f"| {f.kind} | `{f.path}` | {f.summary} |")
    else:
        lines.append("_No source files were discovered under the provided paths._")
    lines.append("")
    lines.append(f"## Detected concepts ({len(evidence.concepts)})")
    lines.append("")
    if evidence.concepts:
        for c in evidence.concepts:
            cites = ", ".join(f"`{e}`" for e in c.evidence)
            lines.append(f"- **{c.concept}** — evidence: {cites}")
    else:
        lines.append("_No concepts were detected from filenames/doc headers._")
    lines.append("")
    return "\n".join(lines)
