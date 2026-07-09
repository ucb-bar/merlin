"""Build and render the evidence report.

``build_evidence`` walks a SourceManifest, records discovered files, and detects concepts by
matching a per-target keyword table against filenames and the first lines of docs. This is
deliberately shallow: it surfaces *candidates* for the human-reviewed synthesizers, not a
parsed model of the hardware.
"""
from __future__ import annotations

from pathlib import Path

from ..io import first_lines
from ..ingest.docs import discover_docs
from ..ingest.examples import discover_examples
from ..ingest.scala_chisel import discover_scala
from ..ingest.source_manifest import SourceManifest
from .store import Concept, Evidence, FileRecord

# Per-target concept vocabularies. Keys are concept names; values are lowercase substrings
# that, if seen in a filename or a doc's first lines, count as supporting evidence.
# toy_npu is concrete; the others mirror the conservative concept lists in the spec.
CONCEPT_KEYWORDS: dict[str, dict[str, tuple[str, ...]]] = {
    "toy_npu": {
        "resident_packed_tensor": ("resident", "packed", "res_pack", "weight"),
        "accumulator_commit": ("accumulator", "commit", "epilogue", "requant"),
        "command_buffer": ("command", "command_buffer", "dispatch"),
        "metrics": ("metric", "cycles", "profil"),
    },
    "gemmini": {
        "rocc": ("rocc", "custom instruction"),
        "scratchpad": ("scratchpad", "spad"),
        "accumulator": ("accumulator", "acc sram"),
        "dma": ("dma", "mvin", "mvout"),
        "systolic_array": ("systolic", "mesh", "pe array"),
        "dataflow_config": ("dataflow", "weight_stationary", "output_stationary", "config_ex"),
        "load_queue": ("load queue", "ldq", "load_queue"),
        "store_queue": ("store queue", "stq", "store_queue"),
        "execute_queue": ("execute queue", "exq", "execute_queue"),
        "rob": ("rob", "reorder"),
        "baremetal": ("baremetal", "bare-metal", "spike"),
        "firesim": ("firesim",),
        "verilator": ("verilator",),
    },
    "saturn": {
        "riscv_vector": ("rvv", "riscv vector", "risc-v vector", "vector unit"),
        "vlen": ("vlen", "zvl"),
        "elen": ("elen",),
        "rvv": ("rvv", "v extension", "zve"),
        "vector_length_policy": ("vsetvl", "vl ", "vector length"),
        "tail_policy": ("tail", "vta"),
        "lmul_policy": ("lmul", "vlmul"),
        "llvm_extension_likely": ("llvm", "intrinsic", "codegen"),
    },
    "radiance": {
        "simt": ("simt", "warp", "lane"),
        "gpu": ("gpu", "muon"),
        "command_processor": ("command processor", "orchestration"),
        "workgroup": ("workgroup", "threadblock", "block"),
        "barrier": ("barrier", "sync"),
        "memory_space": ("local memory", "shared memory", "global memory", "memory space"),
        "runtime_dispatch": ("dispatch", "kernel launch", "host-device"),
    },
}


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

    table = CONCEPT_KEYWORDS.get(manifest.target_name, {})
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
