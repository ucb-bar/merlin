"""Target-independent RTL check reports and selected support dispatch.

Transport does not determine an accelerator's protocol. Ordering, movement geometry,
trace assertions and rendering belong to the explicitly selected OOT support capability.
Missing or malformed support is unavailable evidence, never a successful screen.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

SCHEMA = "rtl_checks/v0"


@dataclasses.dataclass
class Check:
    id: str
    tier: str  # "T0" (pure static) | "T1" (cheap spike byproduct)
    severity: str  # "error" | "warn" | "info"
    status: str  # "pass" | "fail" | "skipped"
    message: str
    expected: Any = None
    got: Any = None
    ratio: float | None = None
    evidence: dict | None = None
    fix_hint: str | None = None

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        return {k: v for k, v in d.items() if v is not None or k in ("expected", "got")}


@dataclasses.dataclass
class CheckReport:
    capsule: str | None
    source_trace: str | None
    rtl_facts: dict
    checks: list[Check] = dataclasses.field(default_factory=list)

    @property
    def n_error(self) -> int:
        return sum(1 for c in self.checks if c.severity == "error" and c.status == "fail")

    @property
    def n_warn(self) -> int:
        return sum(1 for c in self.checks if c.severity == "warn" and c.status == "fail")

    @property
    def verdict(self) -> str:
        if self.n_error:
            return "reject"
        if self.n_warn:
            return "warn"
        return "ok"

    def to_dict(self) -> dict:
        return {
            "schema": SCHEMA,
            "capsule": self.capsule,
            "source_trace": self.source_trace,
            "rtl_facts": self.rtl_facts,
            "verdict": self.verdict,
            "n_error": self.n_error,
            "n_warn": self.n_warn,
            "checks": [c.to_dict() for c in self.checks if c.status != "skipped"],
            "skipped": [{"id": c.id, "reason": c.message} for c in self.checks if c.status == "skipped"],
        }


def _declared_op(capsule: dict | None) -> str | None:
    if not capsule:
        return None
    op = (capsule.get("operation") or {}).get("op")
    return op.lower() if isinstance(op, str) else None


def _declared_output_shape(capsule: dict | None) -> tuple[int, int] | None:
    """Best-effort (M, N) of the declared output, GENERAL (declared shapes only, no golden).

    For a plain matmul the output is (M, N) with lhs (M,K) and weight (K,N). We read the declared
    input roles. Returns None when the shape cannot be derived generally (then tile_coverage skips,
    honestly, rather than guessing).
    """
    if not capsule:
        return None
    inputs = capsule.get("inputs") or []
    by_role: dict[str, list] = {}
    for t in inputs:
        by_role.setdefault(t.get("role"), []).append(t.get("shape"))
    lhs = by_role.get("input")
    w = by_role.get("weight")
    if not lhs or not w:
        return None
    lhs_shape, w_shape = lhs[0], w[0]
    if not (isinstance(lhs_shape, list) and isinstance(w_shape, list) and len(lhs_shape) == 2 and len(w_shape) == 2):
        return None
    M = lhs_shape[0]
    N = w_shape[1]
    return (int(M), int(N))


class RtlChecksUnavailable(ValueError):
    """The selected support cannot provide the requested structural checks."""


def selected_checks(target: str):
    """Resolve one host-owned protocol implementation; never infer it from transport."""
    from merlin.runtime.backends.base import get_backend

    backend = get_backend(target)
    semantics = getattr(backend, "rocc_semantics", None)
    checks = getattr(semantics, "rtl_checks", None)
    required = ("load_default_facts", "project_facts", "screen", "compile_trace_checks", "render_trace")
    if checks is None or any(not callable(getattr(checks, name, None)) for name in required):
        raise RtlChecksUnavailable(
            f"selected backend for {target!r} has no complete rocc_semantics.rtl_checks capability"
        )
    return checks


def load_default_facts(target: str) -> dict[str, Any]:
    """Read the selected protocol's fact projection without another target's fallback."""
    facts = selected_checks(target).load_default_facts(target)
    if not isinstance(facts, dict):
        raise RtlChecksUnavailable("selected RTL check provider returned malformed facts")
    return facts


def screen(
    trace: dict,
    capsule: dict | None = None,
    rtl_facts: dict | None = None,
    *,
    target: str,
    command_buffer: dict | None = None,
    checks=None,
) -> CheckReport:
    """Screen through selected host support, preserving its check IDs and evidence.

    A runner may supply its already selected capability so compilation, rendering and
    screening use the same owner. This is trusted host configuration, not candidate input.
    Exceptions remain unavailable evidence; advisory callers still execute their oracle.
    """
    checks = selected_checks(target) if checks is None else checks
    report = checks.screen(trace, capsule, rtl_facts, target=target, command_buffer=command_buffer)
    if not isinstance(report, CheckReport) or not isinstance(report.rtl_facts, dict):
        raise RtlChecksUnavailable("selected RTL check provider returned a malformed report")
    if not isinstance(report.checks, list) or any(
        not isinstance(check, Check)
        or not isinstance(check.id, str)
        or not check.id
        or check.tier not in {"T0", "T1"}
        or check.severity not in {"error", "warn", "info"}
        or check.status not in {"pass", "fail", "skipped"}
        or not isinstance(check.message, str)
        for check in report.checks
    ):
        raise RtlChecksUnavailable("selected RTL check provider returned malformed checks")
    return report


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="RTL-derived cheap checks over a decoded RoCC trace (advisory / un-wired)")
    ap.add_argument("trace", help="path to instruction_trace.json (rocc_decode output)")
    ap.add_argument("--target", required=True, help="target whose RTL facts the checks are screened against")
    ap.add_argument("--capsule", default=None, help="path to capsule.yaml (declared shapes/modes)")
    ap.add_argument("--rtl-facts", default=None, help="optional JSON file overriding RTL facts")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    trace = json.loads(Path(a.trace).read_text(encoding="utf-8"))
    capsule = None
    if a.capsule:
        import yaml

        capsule = yaml.safe_load(Path(a.capsule).read_text(encoding="utf-8"))
    facts = None
    if a.rtl_facts:
        facts = json.loads(Path(a.rtl_facts).read_text(encoding="utf-8"))

    rep = screen(trace, capsule, facts, target=a.target)
    out = json.dumps(rep.to_dict(), indent=2)
    if a.out:
        Path(a.out).write_text(out, encoding="utf-8")
        print(f"wrote {a.out}: verdict={rep.verdict} errors={rep.n_error} warns={rep.n_warn}")
    else:
        print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
