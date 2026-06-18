"""Loader for ISOLATED, per-run RVV codegen packages.

Each package is a self-contained directory (one per codegen iteration), loaded dynamically and
NEVER hardcoded into the core lowering tables:

    generated_targets/rvv/<run_id>/
        manifest.yaml     # target, run_id, family, authoring/provenance, status
        schedule.mlir     # the transform-dialect SCHEDULE (authoritative codegen artifact)
        knobs.yaml        # the tunable surface AS DATA describing schedule.mlir + the cflags

This is the RVV analogue of :func:`merlin.targetgen.registry.load_target`, but RVV is a
transform SCHEDULE + cflags rather than a resident-accelerator dialect, so the payload is
``schedule.mlir`` + ``knobs.yaml`` (NOT a ``dialect.py`` exposing ``SPEC_OPS``). The loaded
package feeds ``build_app(rvv_schedule=..., cflags_override=...)`` via :mod:`merlin.rvvgen.apply`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..common.schemas import validate_or_raise
from ..common.yaml import load_yaml

_CFLAGS_ALLOW_PREFIXES = ("-march", "-mabi", "-mcmodel", "-O", "-f")


@dataclass
class RvvPackage:
    """A loaded, isolated RVV codegen package."""

    name: str                       # always "rvv"
    run_id: str
    directory: Path
    schedule_text: str              # contents of schedule.mlir -> lower_to_llvm_ir(transform_schedule=)
    cflags: list[str]               # RVV-specific cflags -> build_app(cflags_override=cflags+_CFLAGS_COMMON)
    dtype_strategy: str             # fp32 | int8_w8a8 | bf16_f32acc
    op_match: list[dict[str, Any]] = field(default_factory=list)
    lowering_patterns: list[str] = field(default_factory=list)
    lmul_policy: str = "m1"
    expected_instructions: list[str] = field(default_factory=list)
    # Default-off compiler features this (impr_) fork enables (PASS/HEURISTIC/PATTERN class).
    # Empty for the baseline hand_v0 -> byte-identical lowering. See llvmlower.impr_features.
    compiler_features: list[str] = field(default_factory=list)
    manifest: dict[str, Any] = field(default_factory=dict)
    knobs: dict[str, Any] = field(default_factory=dict)

    @property
    def is_int8(self) -> bool:
        return self.dtype_strategy == "int8_w8a8"


def _check_cflags(cflags: list[str]) -> list[str]:
    """Integrity: a package may only carry compiler flags from an allowlist — the data-not-code
    analogue of the gemmini harness-import ban (a package must not smuggle arbitrary behavior)."""
    bad = [f for f in cflags if not f.startswith(_CFLAGS_ALLOW_PREFIXES)]
    return [f"cflag {f!r} outside allowlist {_CFLAGS_ALLOW_PREFIXES}" for f in bad]


def load_rvv_package(package_dir: str | Path) -> RvvPackage:
    """Load an isolated RVV package directory into an :class:`RvvPackage`.

    Validates the manifest against the ``rvv_package_manifest`` schema and the cflags against the
    allowlist. Raises ``ValueError`` on a schema/integrity problem (this is K0 of the runner).
    """
    d = Path(package_dir)
    manifest = load_yaml(d / "manifest.yaml")
    validate_or_raise(manifest, "rvv_package_manifest")

    knobs = load_yaml(d / "knobs.yaml")
    sched_path = d / knobs.get("schedule_file", "schedule.mlir")
    schedule_text = sched_path.read_text(encoding="utf-8")

    cflags = list(knobs.get("cflags", []))
    problems = _check_cflags(cflags)
    if problems:
        raise ValueError(f"rvv package {d.name} integrity failed: {'; '.join(problems)}")

    return RvvPackage(
        name=manifest["target"],
        run_id=manifest.get("run_id", d.name),
        directory=d,
        schedule_text=schedule_text,
        cflags=cflags,
        dtype_strategy=knobs.get("dtype_strategy", "fp32"),
        op_match=list(knobs.get("op_match", [])),
        lowering_patterns=list(knobs.get("lowering_patterns", [])),
        lmul_policy=knobs.get("lmul_policy", "m1"),
        expected_instructions=list(knobs.get("expected_instructions", [])),
        compiler_features=list(knobs.get("compiler_features",
                                         manifest.get("compiler_features", []))),
        manifest=manifest,
        knobs=knobs,
    )


def default_run(generated_root: str | Path, target: str = "rvv") -> Path:
    """The most-recent run_id directory for ``target`` under the generated-targets root."""
    base = Path(generated_root) / target
    runs = sorted([p for p in base.iterdir() if p.is_dir()]) if base.is_dir() else []
    if not runs:
        raise FileNotFoundError(f"no generated runs for {target!r} under {base}")
    return runs[-1]
