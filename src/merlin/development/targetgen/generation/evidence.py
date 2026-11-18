"""The SourceManifest: a record of what TargetGen was pointed at for a target.

Validates against ``src/merlin/resources/schemas/target_source_manifest.schema.yaml``. Source URLs are
recorded but not fetched; only local directories/files are read downstream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from merlin.common.record_formats import load_yaml
from merlin.target.rtl.facts import target_base

# ---- source_manifest ---------------------------------------------------------------------------


@dataclass
class SourceManifest:
    """Inputs for one TargetGen run.

    All path fields are recorded verbatim (absolute or repo-relative). ``branch``/``commit``/
    ``notes`` are optional provenance.
    """

    target_name: str
    source_dirs: list[str] = field(default_factory=list)
    source_files: list[str] = field(default_factory=list)
    source_urls: list[str] = field(default_factory=list)
    scala_roots: list[str] = field(default_factory=list)
    examples_dirs: list[str] = field(default_factory=list)
    branch: str | None = None
    commit: str | None = None
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a schema-shaped mapping (validated against target_source_manifest schema)."""
        from merlin.common.record_formats import validate_or_raise

        d = {
            "target_name": self.target_name,
            "source_dirs": list(self.source_dirs),
            "source_files": list(self.source_files),
            "source_urls": list(self.source_urls),
            "scala_roots": list(self.scala_roots),
            "examples_dirs": list(self.examples_dirs),
            "branch": self.branch,
            "commit": self.commit,
            "notes": self.notes,
        }
        validate_or_raise(d, "target_source_manifest")  # schemas/ rule: code validates what lives here
        return d


def build_manifest(
    target_name: str,
    source_dir: str | None = None,
    examples_dir: str | None = None,
    scala_root: str | None = None,
    source_urls: list[str] | None = None,
    branch: str | None = None,
    commit: str | None = None,
    notes: str | None = None,
) -> SourceManifest:
    """Build a SourceManifest from the CLI's simple flag set.

    ``source_dir`` becomes the single source directory; ``examples_dir``/``scala_root`` are
    added to their respective lists when provided.
    """
    return SourceManifest(
        target_name=target_name,
        source_dirs=[source_dir] if source_dir else [],
        examples_dirs=[examples_dir] if examples_dir else [],
        scala_roots=[scala_root] if scala_root else [],
        source_urls=list(source_urls or []),
        branch=branch,
        commit=commit,
        notes=notes,
    )


# ---- io ----------------------------------------------------------------------------------------
# Tiny filesystem helpers used by TargetGen ingest/evidence.
#
# Read-only, deterministic, stdlib-only. These exist so evidence collection stays honest:
# we read text and list files by suffix, we do not attempt to "understand" arbitrary RTL.


def read_text(path: str | Path, errors: str = "replace") -> str:
    """Read a file as UTF-8 text, tolerating undecodable bytes."""
    return Path(path).read_text(encoding="utf-8", errors=errors)


def first_lines(path: str | Path, n: int = 5) -> list[str]:
    """Return the first ``n`` lines of a file (stripped of trailing newlines).

    Returns an empty list if the file cannot be read.
    """
    try:
        out: list[str] = []
        with Path(path).open("r", encoding="utf-8", errors="replace") as fh:
            for _ in range(n):
                line = fh.readline()
                if not line:
                    break
                out.append(line.rstrip("\n"))
        return out
    except (OSError, UnicodeError):
        return []


def find_by_suffix(root: str | Path, suffixes: Iterable[str]) -> list[Path]:
    """Recursively list files under ``root`` whose suffix is in ``suffixes``.

    Suffixes are matched case-insensitively and may be given with or without the leading
    dot (e.g. ``"md"`` or ``".md"``). Results are sorted for determinism. A missing root
    yields an empty list.
    """
    root = Path(root)
    if not root.is_dir():
        return []
    wanted = {("" if s.startswith(".") else ".") + s.lower() for s in suffixes}
    hits = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in wanted]
    return sorted(hits)


# ---- docs --------------------------------------------------------------------------------------
# Discover documentation files referenced by a SourceManifest.
#
# Conservative and deterministic: we list Markdown/reStructuredText/plain-text files under the
# manifest's source directories and read short summaries. We do not parse or "understand"
# their content here -- that is the human-reviewed synthesis step's concern.

DOC_SUFFIXES = ("md", "rst", "txt")


def discover_docs(manifest: SourceManifest) -> list[Path]:
    """Return sorted doc files found under the manifest's source dirs + explicit files."""
    hits: list[Path] = []
    for d in manifest.source_dirs:
        hits.extend(find_by_suffix(d, DOC_SUFFIXES))
    for f in manifest.source_files:
        p = Path(f)
        if p.is_file() and p.suffix.lower().lstrip(".") in DOC_SUFFIXES:
            hits.append(p)
    # de-dup while keeping determinism
    return sorted(set(hits))


# ---- examples ----------------------------------------------------------------------------------
# Discover example/program files referenced by a SourceManifest.
#
# Lists the kinds of artifacts that demonstrate how a target is programmed: MLIR, C/headers,
# assembly, JSON/YAML configs. Deterministic, read-only.

EXAMPLE_SUFFIXES = ("mlir", "c", "h", "s", "json", "yaml", "yml")


def discover_examples(manifest: SourceManifest) -> list[Path]:
    """Return sorted example files found under the manifest's examples dirs."""
    hits: list[Path] = []
    for d in manifest.examples_dirs:
        hits.extend(find_by_suffix(d, EXAMPLE_SUFFIXES))
    return sorted(set(hits))


# ---- scala_chisel ------------------------------------------------------------------------------
# Discover Scala/Chisel RTL files referenced by a SourceManifest.
#
# Many targets (the Chipyard-generated ones among them) describe hardware in Scala/Chisel. We only
# *locate* these files (so the evidence report can cite them); we make no attempt to interpret
# RTL. This deliberately conservative posture is why all non-toy synthesis is flagged
# ``requires_human_review``.

SCALA_SUFFIXES = ("scala",)


def discover_scala(manifest: SourceManifest) -> list[Path]:
    """Return sorted Scala files found under the manifest's scala roots + source dirs."""
    hits: list[Path] = []
    for d in manifest.scala_roots:
        hits.extend(find_by_suffix(d, SCALA_SUFFIXES))
    for d in manifest.source_dirs:
        hits.extend(find_by_suffix(d, SCALA_SUFFIXES))
    return sorted(set(hits))


# ---- store -------------------------------------------------------------------------------------
# Data structures for collected evidence.
#
# ``Evidence`` is the in-memory result of the evidence pass; it serializes to the
# ``evidence_report`` schema (``evidence_index.yaml``) and renders to ``evidence_report.md``.


@dataclass(frozen=True)
class FileRecord:
    """One discovered source file."""

    path: str  # path relative to its source root (for citations)
    kind: str  # doc | scala | example
    summary: str  # short, filename/first-line derived

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "kind": self.kind, "summary": self.summary}


@dataclass(frozen=True)
class Concept:
    """A keyword-detected concept and the files that support it."""

    concept: str
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"concept": self.concept, "evidence": list(self.evidence)}


@dataclass
class Evidence:
    """Result of the evidence pass for one target."""

    target: str
    sources: dict[str, Any]
    files: list[FileRecord] = field(default_factory=list)
    concepts: list[Concept] = field(default_factory=list)

    def concept_names(self) -> set[str]:
        return {c.concept for c in self.concepts}

    def to_index_dict(self) -> dict[str, Any]:
        """Return the ``evidence_report``-schema mapping for ``evidence_index.yaml`` (validated)."""
        from merlin.common.record_formats import validate_or_raise

        d = {
            "target": self.target,
            "sources": self.sources,
            "files": [f.to_dict() for f in self.files],
            "detected_concepts": [c.to_dict() for c in self.concepts],
        }
        validate_or_raise(d, "evidence_report")  # schemas/ rule: if it lives here, code validates it
        return d


# ---- report ------------------------------------------------------------------------------------
# Build and render the evidence report.
#
# ``build_evidence`` walks a SourceManifest, records discovered files, and detects concepts by
# matching a per-target keyword vocabulary against filenames and the first lines of docs. This is
# deliberately shallow: it surfaces *candidates* for the human-reviewed synthesizers, not a
# parsed model of the hardware.
#
# The concept vocabulary is DATA, not a core literal: each target owns an
# ``evidence_concepts.yaml`` in its discovered target dir (``rtl.facts.target_base`` — the curated
# ``experiments/inputs/target-inputs/<t>`` or the generated ``out/artifacts/targets/<t>``). A target with no such file
# contributes no concepts (honest: nothing authored ⇒ nothing detected), so the core stays
# target-name-free and a brand-new accelerator plugs in by dropping its vocabulary beside its sources.


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
        cites = sorted({rel for rel, text in haystacks if any(k in text for k in keywords)})
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


__all__ = [
    "Evidence",
    "FileRecord",
    "SourceManifest",
    "build_evidence",
    "build_manifest",
    "render_markdown",
]
