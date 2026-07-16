"""The per-step search record — one beam step's self-contained, LLM-digestible "what / why / did the
divergence close? / real-or-fake? / perf" answer.

This composes the pieces the beam needs into ONE record and enforces the search's discipline:
- WHAT/WHY: the action taken (class + seam) and which CCA axis + improvement CATEGORY it targets.
- DID THE DIVERGENCE CLOSE / DOES THE IR BEHAVE AS INTENDED: ``achieved`` = the emitted asm's CCA
  actually delivered the action's ``intended_facet`` (via ``action_catalog.achieved_residual`` — the
  intended-vs-achieved audit that was implemented but never wired in). Non-empty ``residual`` = the
  promise the fork did NOT keep (the axis that should escalate FLAG->KNOB->...->CODEGEN).
- REAL vs FAKE SPEEDUP: ``speedup`` is credited ONLY when correctness passed (fail-closed) — no
  speedup credit for a fork that broke numerics (the fair-comparison / INLINED-VS-ROUTED discipline).

It is the record the chia-driven beam emits per step and aet instruments (``to_dict`` -> the aet run's
metrics/artifacts). Deterministic; target-agnostic; no LLM.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class SearchStep:
    axis: str
    category: str | None                 # the improvement category (what KIND of optimization)
    action_class: str                    # FLAG | KNOB | HEURISTIC | PASS | CODEGEN
    target_seam: str
    intended_facet: dict = field(default_factory=dict)
    achieved: bool = False               # did the emitted asm achieve the intended facet? (the audit)
    residual: list[str] = field(default_factory=list)   # promised-but-not-achieved axes (-> escalate)
    correctness_ok: bool = False         # the cos/rel numerics gate passed (real, not fake)
    speedup: float | None = None         # measured speedup vs the unoptimized baseline (None if not real)
    rationale: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def to_line(self) -> str:
        got = "closed" if self.achieved else f"OPEN (residual {self.residual})"
        real = (f"{self.speedup:.2f}x" if self.speedup is not None
                else ("no-speedup" if self.correctness_ok else "FAILED-numerics"))
        return (f"[{self.category or '?'}] {self.axis} via {self.action_class} {self.target_seam} "
                f"-> {got}; {real}")


def make_step(action, achieved_cca, *, correctness_ok: bool, speedup: float | None) -> SearchStep:
    """Build the record for a certified fork: ``action`` is the CompilerAction that was applied,
    ``achieved_cca`` is the CCA lifted from the fork's EMITTED asm, ``correctness_ok`` is the numerics
    gate, ``speedup`` is the MEASURED speedup vs the unoptimized baseline. Real-vs-fake is enforced
    here: speedup is dropped unless correctness_ok."""
    from .action_catalog import achieved_residual
    from .categories import category_for_axis

    residual = achieved_residual(action, achieved_cca)
    return SearchStep(
        axis=action.divergence_axis,
        category=category_for_axis(action.divergence_axis),
        action_class=action.action_class,
        target_seam=action.target_seam,
        intended_facet=dict(action.intended_facet or {}),
        achieved=not residual,
        residual=residual,
        correctness_ok=correctness_ok,
        speedup=speedup if correctness_ok else None,   # fail-closed: no speedup credit without correctness
        rationale=action.change)


def audit_fork(action, objdump_text: str, *, op: str = "matmul",
               correctness_ok: bool, speedup: float | None) -> SearchStep:
    """Audit a certified fork: lift the fork's CCA from its emitted objdump TEXT (no toolchain re-run,
    via ``decode_text``) and build the per-step record. This is what the beam calls per fork to answer
    'did the fork's emitted asm achieve the action's intended facet?' + record real-vs-fake speedup."""
    from .cca import lift_asm
    from .decode import rvv

    achieved_cca = lift_asm(rvv.decode_text(objdump_text), op=op, source="fork")
    return make_step(action, achieved_cca, correctness_ok=correctness_ok, speedup=speedup)
