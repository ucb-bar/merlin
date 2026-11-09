"""Engine-level gSIM qualification: one exact gSIM build qualified on a stratified suite.

Per-workload certification (``per_workload``, the default) captures EVERY measured workload on
Verilator and gSIM. ``engine_qualified`` instead qualifies the engine build once. A host-chosen suite
covers every coverage key the tuning and held-out cohorts contain. Each suite member is small or
medium (kernel cycles within a declared budget) and runs as one ELF on both engines with digest
readback verified on Spike (:func:`gsim_certificate.capture_case`). The two engines must give
identical kernel cycles and identical output digests. A workload is then admitted for gSIM timing
only when its coverage key is covered by a passing qualification of that exact build (same pins,
same build receipt); any other workload is refused.

A coverage key is ``operation``, the geometry stratum of its shape (:func:`classify_geometry`), its
form (epilogue stages, output dtype, structural convolution attributes) and its operand dtypes. Size
is deliberately not part of the key; the suite exercises each key below the cycle budget. A
performance-scale member is admitted on the strength of the engine being shown equal to the reference
RTL on its stratum and form, not on a capture of that exact shape. That is the policy this mode
records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from merlin_experiments.phase2 import gsim_gate as GATE

SCHEMA_VERSION = "merlin.gsim-engine-qualification.v1"
PER_WORKLOAD = "per_workload"
ENGINE_QUALIFIED = "engine_qualified"
MODES = (PER_WORKLOAD, ENGINE_QUALIFIED)
#: Kernel cycles a qualification member may take: small/medium members, a few minutes per engine.
DEFAULT_CYCLE_BUDGET = 2_000_000
#: Operation attributes that change what the datapath does, as opposed to sizes or scale VALUES.
FORM_ATTRIBUTES = ("epilogue", "output_dtype", "kh", "kw", "stride", "dilation", "layout")


class EngineQualificationError(GATE.GsimGateError):
    """The qualification document, its coverage or its members are not admissible."""


def coverage_key(workload: Mapping[str, Any]) -> dict[str, Any]:
    """The stratum-and-form key a workload is admitted under."""
    from merlin.capture.shape_taxonomy import classify_geometry

    canonical = GATE.canonical_workload(workload)
    shape = canonical["shape"]
    semantics = canonical["semantics"]
    attributes = semantics.get("operation_attributes") or {}
    if canonical["operation"] == "matmul" and all(type(shape.get(axis)) is int for axis in ("m", "n", "k")):
        geometry = classify_geometry(shape["m"], shape["n"], shape["k"])
    else:
        geometry = "not_applicable"
    form = {name: attributes[name] for name in FORM_ATTRIBUTES if name in attributes}
    padding = attributes.get("padding")
    if isinstance(padding, list) and any(padding):
        form["padded"] = True
    return {
        "operation": canonical["operation"],
        "geometry": geometry,
        "form": form,
        "operand_dtypes": dict(semantics.get("operand_dtypes") or {}),
    }


def key_sha256(key: Mapping[str, Any]) -> str:
    return hashlib.sha256(GATE.canonical_json(dict(key)).encode("utf-8")).hexdigest()


def plan_suite(
    cohort: Mapping[str, Mapping[str, Any]],
    pool: Mapping[str, tuple[Mapping[str, Any], int]],
    *,
    cycle_budget: int = DEFAULT_CYCLE_BUDGET,
) -> dict[str, Any]:
    """The host's suite: per coverage key of ``cohort`` (name -> workload), one ``pool`` member
    (name -> (workload, estimated kernel cycles)) within the budget -- the LARGEST such member, so the
    key is exercised at medium rather than toy scale. Keys with no affordable member are reported."""
    needed: dict[str, dict[str, Any]] = {}
    for workload in cohort.values():
        key = coverage_key(workload)
        needed.setdefault(key_sha256(key), key)
    chosen: dict[str, tuple[str, int]] = {}
    for name, (workload, cycles) in sorted(pool.items()):
        sha = key_sha256(coverage_key(workload))
        if sha in needed and int(cycles) <= cycle_budget and (sha not in chosen or int(cycles) > chosen[sha][1]):
            chosen[sha] = (name, int(cycles))
    return {
        "cycle_budget": cycle_budget,
        "selected": sorted(name for name, _ in chosen.values()),
        "covered": sorted(chosen),
        "uncovered": [needed[sha] for sha in sorted(set(needed) - set(chosen))],
    }


def _member_cycles(member: Mapping[str, Any], *, where: str, cycle_budget: int) -> int:
    reference = (member.get("reference") or {}).get("cycles")
    candidate = (member.get("candidate") or {}).get("cycles")
    if type(reference) is not int or type(candidate) is not int or reference <= 0:
        raise EngineQualificationError(f"{where} records no positive kernel cycles on both engines")
    if reference != candidate:
        raise EngineQualificationError(f"{where}: gSIM ran {candidate} kernel cycles, Verilator {reference}")
    if reference > cycle_budget:
        raise EngineQualificationError(f"{where} ran {reference} kernel cycles, over the {cycle_budget} budget")
    for side in ("reference", "candidate"):
        readback = (member.get(side) or {}).get("readback") or {}
        if readback.get("mode") == "digest" and not readback.get("output_digests"):
            raise EngineQualificationError(f"{where}.{side} claims digest readback without digests")
    return reference


def _coverage(members: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for identity, member in sorted(members.items()):
        key = coverage_key(member["workload"])
        row = grouped.setdefault(key_sha256(key), {"key": key, "key_sha256": key_sha256(key), "members": []})
        row["members"].append(identity)
    return [grouped[sha] for sha in sorted(grouped)]


def produce_qualification(
    *,
    target: str,
    captures: Sequence[str | Path],
    artifacts: Any,
    build_receipt: str | Path,
    cycle_budget: int = DEFAULT_CYCLE_BUDGET,
) -> dict[str, Any]:
    """Assemble the qualification of one exact build from same-ELF suite captures."""
    from merlin_experiments.phase2 import gsim_certificate as PRODUCER

    if not captures:
        raise EngineQualificationError("an engine qualification needs at least one suite capture")
    pins = artifacts.pinned()
    receipt = PRODUCER.validate_build_receipt(build_receipt, pins=pins)
    members: dict[str, Mapping[str, Any]] = {}
    for path in captures:
        member = PRODUCER.validate_capture(path, target=target, pins=pins)
        identity = member["workload_sha256"]
        if identity in members:
            raise EngineQualificationError(f"duplicate qualification workload {identity}")
        _member_cycles(member, where=f"capture {path}", cycle_budget=cycle_budget)
        members[identity] = member
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "qualified",
        "certification": ENGINE_QUALIFIED,
        "target": target,
        "fidelity": GATE.FIDELITY,
        "primary_engine": GATE.GSIM_ENGINE,
        "reference_engine": GATE.REFERENCE_ENGINE,
        "pins": pins,
        "build_binding": receipt,
        "cycle_budget": cycle_budget,
        "requirements": {"same_elf": True, "identical_kernel_cycles": True, "identical_output_digests": True},
        "coverage": _coverage(members),
        "members": [members[identity] for identity in sorted(members)],
    }


@dataclass(frozen=True)
class EngineQualificationRecord(GATE.CertificateRecord):
    """A validated engine qualification, usable wherever a certificate decides gSIM admission."""

    coverage: Mapping[str, Mapping[str, Any]] = None  # type: ignore[assignment]
    cycle_budget: int = DEFAULT_CYCLE_BUDGET

    @property
    def certification(self) -> str:
        return ENGINE_QUALIFIED

    def admits(self, workload: Mapping[str, Any]) -> bool:
        return key_sha256(coverage_key(workload)) in self.coverage

    def to_dict(self) -> dict[str, Any]:
        return {
            **super().to_dict(),
            "certification": ENGINE_QUALIFIED,
            "covered_keys": len(self.coverage),
            "cycle_budget": self.cycle_budget,
        }


def load_qualification(
    path: Path, raw_bytes: bytes, doc: Mapping[str, Any], *, artifact_paths: Any = None
) -> EngineQualificationRecord:
    """Validate an engine qualification already read (and content-addressed) by ``GATE.load_certificate``."""
    if doc.get("status") != "qualified" or doc.get("certification") != ENGINE_QUALIFIED:
        raise EngineQualificationError("engine qualification is not a qualified engine_qualified document")
    target = doc.get("target")
    if not isinstance(target, str) or not target:
        raise EngineQualificationError("engine qualification target is absent")
    if (
        doc.get("fidelity") != GATE.FIDELITY
        or doc.get("primary_engine") != GATE.GSIM_ENGINE
        or doc.get("reference_engine") != GATE.REFERENCE_ENGINE
    ):
        raise EngineQualificationError("engine qualification must compare GSIM against Verilator at RTL fidelity")
    raw_pins = doc.get("pins")
    if not isinstance(raw_pins, Mapping) or set(raw_pins) != GATE.REQUIRED_PINS:
        raise EngineQualificationError(f"engine qualification pins must be exactly {sorted(GATE.REQUIRED_PINS)}")
    pins = {
        name: GATE._validate_pin(name, raw_pins[name], certificate_path=path, artifact_paths=artifact_paths)
        for name in sorted(GATE.REQUIRED_PINS)
    }
    GATE._validate_build_binding(doc.get("build_binding"), certificate_path=path, pins=pins)
    budget = doc.get("cycle_budget")
    if type(budget) is not int or budget <= 0:
        raise EngineQualificationError("engine qualification declares no positive cycle budget")
    raw_members = doc.get("members")
    if not isinstance(raw_members, list) or not raw_members:
        raise EngineQualificationError("engine qualification has no suite members")
    members: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(raw_members):
        identity, member = GATE._validate_member(raw, pins=pins, index=index)
        if identity in members:
            raise EngineQualificationError(f"duplicate qualification workload {identity}")
        _member_cycles(member, where=f"members[{index}]", cycle_budget=budget)
        members[identity] = member
    coverage = _coverage(members)
    if json.loads(GATE.canonical_json(coverage)) != json.loads(GATE.canonical_json(doc.get("coverage"))):
        raise EngineQualificationError("declared coverage is not the coverage of the qualified members")
    return EngineQualificationRecord(
        path.resolve(),
        GATE._digest_bytes(raw_bytes),
        target,
        pins,
        members,
        {},
        doc,
        coverage={row["key_sha256"]: row["key"] for row in coverage},
        cycle_budget=budget,
    )


def _cli(argv: Sequence[str] | None = None) -> int:
    """``plan`` the host's suite for a cohort, or ``produce`` the qualification from suite captures.

    Captures come from ``python -m merlin_experiments.phase2.gsim_certificate capture`` (digest readback
    by default), one per planned suite member.
    """
    from merlin_experiments.phase2 import gsim_certificate as PRODUCER
    from merlin_experiments.phase2 import gsim_workload as WORKLOAD

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="action", required=True)
    plan = commands.add_parser("plan", help="choose the stratified suite covering a cohort")
    plan.add_argument("--cohort", action="append", required=True, help="capsule.yaml of a tuning/holdout member")
    plan.add_argument(
        "--pool", required=True, help="JSON {capsule.yaml path: estimated kernel cycles} of candidate suite members"
    )
    plan.add_argument("--cycle-budget", type=int, default=DEFAULT_CYCLE_BUDGET)
    plan.add_argument("--output", required=True)
    produce = commands.add_parser("produce", help="assemble the qualification of one exact gSIM build")
    produce.add_argument("--target", required=True)
    produce.add_argument("--capture", action="append", required=True)
    produce.add_argument("--build-receipt", required=True)
    produce.add_argument("--cycle-budget", type=int, default=DEFAULT_CYCLE_BUDGET)
    PRODUCER._add_artifact_args(produce)
    produce.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.action == "plan":
        cohort = {path: WORKLOAD.derive_workload(Path(path)) for path in args.cohort}
        pool_doc = json.loads(Path(args.pool).read_text(encoding="utf-8"))
        pool = {path: (WORKLOAD.derive_workload(Path(path)), int(cycles)) for path, cycles in pool_doc.items()}
        report = plan_suite(cohort, pool, cycle_budget=args.cycle_budget)
    else:
        report = produce_qualification(
            target=args.target,
            captures=args.capture,
            artifacts=PRODUCER._artifact_paths(args),
            build_receipt=args.build_receipt,
            cycle_budget=args.cycle_budget,
        )
    Path(args.output).write_text(GATE.canonical_json(report) + "\n", encoding="utf-8")
    print(GATE.canonical_json({k: v for k, v in report.items() if k != "members"}))
    return 0 if args.action == "produce" or not report["uncovered"] else 3


if __name__ == "__main__":  # pragma: no cover
    sys.exit(_cli())
