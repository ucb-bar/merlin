"""The unifying entry point: assess a program's silicon liveness against a target, merging the static
precondition linter (B) and the dynamic transaction-level model (A) into one :class:`LivenessReport`.

A :class:`Program` is a thin container of whatever representations the caller has — a decoded RoCC
``instruction_trace``, the kernel's rv32 ``.text``, and the image-audit / substrate booleans. Each check
consumes what it needs and fails closed (``UNKNOWN``) on what is absent, so a partial program still yields
a useful, honest report.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from .facts import silicon_facts
from .interconnect import simulate
from .preconditions import funct_legality, host_assist, medany_span, untranscodable_op, vlen_match
from .report import LivenessReport


@dataclasses.dataclass
class Program:
    """The representations available for one program under assessment (all optional)."""

    name: str | None = None
    trace: dict | None = None                # decoded RoCC instruction_trace (rocc_decode.to_dict)
    rv32_text: bytes | None = None           # self-hosted kernel .text bytes (for transcodability)
    has_htif: bool | None = None             # image audit: a .htif section present?
    has_tohost: bool | None = None           # image audit: a tohost symbol present?
    hostless: bool | None = None             # delivery substrate has no fesvr host?
    address_model: str | None = None         # harness DRAM addressing convention (pointer_args/…)
    dram_bytes: int | None = None            # DRAM window size, when the caller can supply it
    # delivery/build context (not RTL-derivable — the caller who chose the substrate/compile path supplies it):
    declared_vlen: int | None = None         # VLEN the build declared (VECTOR_MAX_LEN / -march zvl)
    hw_vlen: int | None = None               # the target board's hardware VLEN (bits); None = non-vector
    uses_medany: bool | None = None          # does the chosen compile path build with -mcmodel=medany?
    image_span_bytes: int | None = None      # linked image symbol span (for the medany window check)


def assess(program: Program, target: str) -> LivenessReport:
    """Run every applicable liveness/precondition check for ``program`` against ``target``."""
    facts = silicon_facts(target)
    report = LivenessReport(target=target, program=program.name)

    if program.trace is not None:
        report.extend(funct_legality(program.trace, facts))
        findings, peaks = simulate(
            program.trace, facts,
            address_model=program.address_model, dram_bytes=program.dram_bytes)
        report.extend(findings)
        report.resource_peaks = peaks

    if program.rv32_text is not None:
        report.extend(untranscodable_op(program.rv32_text, target))

    if program.hostless is not None or program.has_htif is not None or program.has_tohost is not None:
        report.extend(host_assist(
            hostless=program.hostless, has_htif=program.has_htif, has_tohost=program.has_tohost))

    # vector-length + medany rules are delivery/build-context checks (VLEN and medany are substrate/compile
    # properties, not RTL facts), so both sides come from the caller-supplied Program. Each is a no-op unless
    # both its inputs are present (fail-closed / not-applicable), so they are always safe to call.
    report.extend(vlen_match(declared_vlen=program.declared_vlen, hw_vlen=program.hw_vlen))
    report.extend(medany_span(uses_medany=program.uses_medany, image_span_bytes=program.image_span_bytes))

    return report


def persist(report: LivenessReport, *, root: Path | None = None) -> Path:
    """Write ``report`` as JSON under ``out/artifacts/liveness/<target>/<program>.liveness.json`` and
    return the path. Concern-first layout (``liveness/`` owns the subtree, target at the folder level),
    per the generated-output convention."""
    if root is None:
        from merlin.common.paths import artifacts_dir

        root = artifacts_dir() / "liveness" / report.target
    root.mkdir(parents=True, exist_ok=True)
    name = (report.program or "program").replace("/", "_")
    out = root / f"{name}.liveness.json"
    out.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return out
