"""Loader for ISOLATED, per-run RVV codegen packages.

Each package is a self-contained directory (one per codegen iteration), loaded dynamically and
NEVER hardcoded into the core lowering tables:

    artifacts/targets/rvv/<run_id>/
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
    dtype_strategy: str             # fp32 | int8_w8a8 | bf16_f32acc | fp16_f32acc
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

    @property
    def is_half_f32acc(self) -> bool:
        """True for the 16-bit-float datapaths (``bf16_f32acc`` / ``fp16_f32acc``).

        DECLARATIVE, not a gate. Unlike ``is_int8`` -- which switches ``_prepare_model_mlir``
        onto a different pass set (``apply_quant``) -- the 16-bit-float rewrite
        (``passes_xdsl.lower_bf16_matmul_f32acc``) runs UNCONDITIONALLY on every model, because
        f16/bf16 matmuls need f32 accumulation for correctness regardless of what any package
        declares. Gating it on this knob would silently break every fp16 model whose package
        happens to say ``fp32``.

        So this property exists for provenance, result records, and ``expected_instructions``
        verification (e16 vtype + ``vfwmacc``) -- it tells you what datapath a package is
        MEANT to exercise, and the runner checks the emitted code actually matches.
        """
        return self.dtype_strategy in ("bf16_f32acc", "fp16_f32acc")


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
        compiler_features=_resolve_features(knobs, manifest),
        manifest=manifest,
        knobs=knobs,
    )


def _resolve_features(knobs: dict, manifest: dict) -> list[str]:
    """compiler_features from knobs/manifest, PLUS the v3 micro-kernel tuning point a ``microkernel``
    knob block {MR, NR, KC} resolves to. That knob is how the beam tunes the register block
    continuously via CODE GENERATION (no hand ukernel) — see from_strategy.microkernel_feature."""
    feats = list(knobs.get("compiler_features", manifest.get("compiler_features", [])))
    mk = knobs.get("microkernel")
    if mk:
        from .from_strategy import microkernel_features
        for name in microkernel_features(mk, target=str(manifest.get("target", "rvv"))):
            if name not in feats:
                feats.append(name)
    return feats


def default_run(generated_root: str | Path, target: str = "rvv") -> Path:
    """The most-recent run_id directory for ``target`` under the generated-targets root."""
    base = Path(generated_root) / target
    runs = sorted([p for p in base.iterdir() if p.is_dir()]) if base.is_dir() else []
    if not runs:
        raise FileNotFoundError(f"no generated runs for {target!r} under {base}")
    return runs[-1]
