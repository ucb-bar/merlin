"""Lightweight evidence-emission for agent-driven target classification.

The deterministic ``classify_inventory`` collapses scanner findings into the
four canonical TargetGen styles. That is fast and reviewable but it cannot
reason about novel targets that don't fit any pre-encoded rule.

``explore_target`` is the alternative entry point: it returns the *raw
evidence* the deterministic classifier would consume, plus a small set of
file pointers (README, top build files, representative scanner hits), and
lets the agent decide. The agent uses its own ``Read``/``Grep``/``Glob``
tools to drill into specific files — this module deliberately does *not*
dump file contents into the response.

Output is intentionally bounded: the directory summary is depth-3 with
file counts only, the key-file list is capped at one path per kind, and
``findings`` is the same ``SourceFinding`` records ``ingest`` already
returns. The whole report is typically <10 KB even for large repos.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .intake import build_source_inventory
from .model import SourceFinding, SourceInventory


@dataclass(slots=True)
class DirectorySummary:
    path: str
    total_files: int
    by_extension: dict[str, int]
    top_dirs: list[str]


@dataclass(slots=True)
class KeyFilePointers:
    """Paths to entry-point files the agent should read with its own tools.

    Values are repo-absolute; ``None`` means the scanner did not find any
    file of that kind. The agent should ``Read`` each one rather than
    expect contents inline.
    """

    readme: str | None
    build_files: list[str] = field(default_factory=list)
    top_source_files: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ExplorationReport:
    target: str
    source_paths: list[str]
    directory_summaries: list[DirectorySummary]
    key_files: KeyFilePointers
    findings: list[SourceFinding]
    detected_source_kinds: list[str]
    findings_by_kind: dict[str, int]
    next_steps: list[str]


_README_NAMES = ("README.md", "README.rst", "README.txt", "README")
_BUILD_FILE_NAMES = (
    "build.sbt",
    "build.sc",
    "CMakeLists.txt",
    "Makefile",
    "BUILD",
    "BUILD.bazel",
    "pyproject.toml",
    "Cargo.toml",
)
_FINDING_KIND_PRIORITY = (
    # Pick one representative file per kind to point the agent at, in
    # priority order so the most informative kinds appear first.
    "chipyard_project",
    "chipyard_config",
    "chisel_attachment_rocc",
    "chisel_attachment_mmio",
    "chisel_attachment_tilelink",
    "rtl_mmio_register",
    "verilog_module",
    "systemverilog_interface",
    "systemc_module",
    "mlir_dialect",
    "mlir_op_definition",
    "llvm_intrinsic_definition",
    "llvm_riscv_features",
    "hal_driver_source",
    "iree_external_hal_driver_registration",
    "iree_compiler_plugin_registration",
    "isa_documentation",
    "runtime_documentation",
)


def explore_target(target_name: str, source_paths: list[Path]) -> ExplorationReport:
    """Walk ``source_paths`` and emit raw evidence for agent-driven reasoning.

    No classification is applied. The returned report is the evidence the
    agent needs to *decide* a classification, not the classifier's output.
    """
    inventory = build_source_inventory(target=target_name, sources=source_paths)
    summaries = [_summarise_directory(p) for p in source_paths]
    key_files = _collect_key_files(source_paths, inventory.findings)
    findings_by_kind = _findings_by_kind(inventory.findings)
    next_steps = _next_steps(target_name, key_files, findings_by_kind)
    return ExplorationReport(
        target=target_name,
        source_paths=[str(p) for p in source_paths],
        directory_summaries=summaries,
        key_files=key_files,
        findings=list(inventory.findings),
        detected_source_kinds=list(inventory.detected_source_kinds),
        findings_by_kind=findings_by_kind,
        next_steps=next_steps,
    )


def explore_target_from_inventory(inventory: SourceInventory, source_paths: list[Path]) -> ExplorationReport:
    """Variant that reuses an already-built inventory (for tests / chaining)."""
    summaries = [_summarise_directory(p) for p in source_paths]
    key_files = _collect_key_files(source_paths, inventory.findings)
    findings_by_kind = _findings_by_kind(inventory.findings)
    next_steps = _next_steps(inventory.target, key_files, findings_by_kind)
    return ExplorationReport(
        target=inventory.target,
        source_paths=[str(p) for p in source_paths],
        directory_summaries=summaries,
        key_files=key_files,
        findings=list(inventory.findings),
        detected_source_kinds=list(inventory.detected_source_kinds),
        findings_by_kind=findings_by_kind,
        next_steps=next_steps,
    )


def _summarise_directory(root: Path) -> DirectorySummary:
    by_ext: Counter[str] = Counter()
    top_dirs: list[str] = []
    total = 0
    if not root.exists():
        return DirectorySummary(path=str(root), total_files=0, by_extension={}, top_dirs=[])
    for child in sorted(root.iterdir()):
        if child.is_dir() and not child.name.startswith("."):
            top_dirs.append(child.name)
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        # Skip hidden directories (.git, .venv, etc.) — they bloat counts
        # and the scanners already ignore them.
        if any(part.startswith(".") for part in path.relative_to(root).parts[:-1]):
            continue
        ext = path.suffix.lower() or "(none)"
        by_ext[ext] += 1
        total += 1
    return DirectorySummary(
        path=str(root),
        total_files=total,
        by_extension=dict(by_ext.most_common(20)),
        top_dirs=top_dirs[:25],
    )


def _collect_key_files(source_paths: list[Path], findings: list[SourceFinding]) -> KeyFilePointers:
    readme: str | None = None
    build_files: list[str] = []
    seen_kinds: set[str] = set()
    top_source_files: list[str] = []

    for root in source_paths:
        if not root.exists():
            continue
        if readme is None:
            for name in _README_NAMES:
                candidate = root / name
                if candidate.is_file():
                    readme = str(candidate)
                    break
        for name in _BUILD_FILE_NAMES:
            candidate = root / name
            if candidate.is_file():
                build_files.append(str(candidate))

    # One representative file per kind, ranked by priority then confidence.
    findings_sorted = sorted(findings, key=lambda f: (-f.confidence, f.path))
    findings_by_kind: dict[str, list[SourceFinding]] = {}
    for f in findings_sorted:
        findings_by_kind.setdefault(f.kind, []).append(f)
    for kind in _FINDING_KIND_PRIORITY:
        if kind in findings_by_kind and kind not in seen_kinds:
            top = findings_by_kind[kind][0]
            # The finding's path is relative to its repo root; resolve
            # against the first matching source_path.
            absolute = _resolve_finding_path(top.path, source_paths)
            if absolute and absolute not in top_source_files:
                top_source_files.append(absolute)
                seen_kinds.add(kind)
        if len(top_source_files) >= 12:
            break

    return KeyFilePointers(
        readme=readme,
        build_files=build_files,
        top_source_files=top_source_files,
    )


def _resolve_finding_path(rel_path: str, source_paths: list[Path]) -> str | None:
    for root in source_paths:
        candidate = root / rel_path
        if candidate.exists():
            return str(candidate)
    return None


def _findings_by_kind(findings: list[SourceFinding]) -> dict[str, int]:
    counter: Counter[str] = Counter(f.kind for f in findings)
    return dict(counter.most_common())


def _next_steps(target: str, key_files: KeyFilePointers, findings_by_kind: dict[str, int]) -> list[str]:
    steps: list[str] = []
    if key_files.readme:
        steps.append(f"Read the README at {key_files.readme} to understand the target's purpose.")
    if key_files.build_files:
        steps.append(
            "Read the build files to learn how the target is assembled: " + ", ".join(key_files.build_files[:3])
        )
    if key_files.top_source_files:
        steps.append(
            "Use Grep/Read on the representative source files to confirm scanner findings: "
            + ", ".join(key_files.top_source_files[:5])
        )
    if findings_by_kind:
        top_kinds = ", ".join(f"{k}({n})" for k, n in list(findings_by_kind.items())[:5])
        steps.append(
            f"Top scanner kinds detected: {top_kinds}. Decide a classification, then call "
            f"`targetgen_propose_modifications(target_name='{target}', ...)` with your "
            "structured judgment — the tool returns the modification map plus an audit of "
            "your claim against scanner evidence."
        )
    else:
        steps.append(
            "No scanner kinds detected. Either the source path is wrong or the target uses "
            "patterns no scanner recognises — Read the README and main source files manually, "
            "then call `targetgen_propose_modifications` with a low-confidence classification."
        )
    return steps
