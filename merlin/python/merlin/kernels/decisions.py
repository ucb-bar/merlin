"""A typed record of one optimization decision, and how strongly it is believed.

WHY THIS EXISTS. Mining an expert corpus produces sentences of three very different kinds, and the
loop has been keeping them in one bag:

* "B is moved from DRAM once and reused for four M blocks"  — an OBSERVATION. It is what the bytes
  say, and it is true of exactly the artifact it was read from.
* "weight-stationary reuse"                                  — a MOTIF. A name for a shape seen in
  several artifacts; it generalizes, but nothing has been tested.
* "when the RHS is immutable and reuse is high, keep the packed weights resident" — a HYPOTHESIS. A
  claim about what the COMPILER should do, which is only worth acting on once measured.
* the same sentence, after a measured, correctness-gated win against a matched control — a
  VALIDATED_POLICY.

Collapsing these is how a corpus reading becomes a compiler rule without anyone deciding it should.
The measured base rate here makes that concrete: of 1509 recorded transform attempts, 203 improved,
119 regressed, 365 failed to compile and 719 were incorrect — a 13.45% improvement rate. A loop that
promotes observations to policies without a control will be wrong most of the time and will have no
record of it.

The ladder is therefore ENFORCED, not documented: :meth:`DecisionRecord.problems` refuses a status
whose evidence does not support it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: Belief ladder, weakest first. A record may only claim a rung its evidence reaches.
STATUSES: tuple[str, ...] = ("observation", "motif", "hypothesis", "validated_policy")


@dataclass
class DecisionRecord:
    """One decision, with the evidence for it and an explicit strength of belief."""

    # --- what it is about -----------------------------------------------------------------
    #: kernel | dispatch | program — the same scope vocabulary the CCA uses.
    scope: str
    #: the semantic operation family (contraction, elementwise_map, reduction, movement, ...).
    family: str
    #: the decision, in this repo's normalized vocabulary (e.g. "weight_stationary").
    decision: str
    #: SHAPE REGIME, never exact dims: a policy keyed to 64x64x64 is a memory, not a rule.
    shape_regime: str | None = None
    #: dtype / quantization regime (e.g. "int8_w8a8", "fp16").
    dtype_regime: str | None = None
    #: the engines the artifact actually drives, from the role census.
    engines: tuple[str, ...] = ()

    # --- where it came from ---------------------------------------------------------------
    #: the artifact read (path or id), and the LEVEL it was read at (asm | source | graph | policy).
    source_artifact: str = ""
    source_level: str = ""
    #: CCA axes this decision is expressed on, so it can be routed later.
    cca_axes: tuple[str, ...] = ()
    #: ids of the kernels/policies asserting it — never a count, always what to go back and read.
    evidence: tuple[str, ...] = ()

    # --- what was measured ----------------------------------------------------------------
    #: the substrate declared authoritative for the number below (see kernels.measurement).
    measurement_authority: str | None = None
    measured_cycles: int | None = None
    #: the artifact this was compared AGAINST. A delta without a control is not a delta.
    control: str | None = None
    #: relative change vs the control; positive is better. None = not measured.
    delta_vs_control: float | None = None
    #: did the measured candidate pass its correctness gate? None = not run. NEVER assume True.
    correctness_ok: bool | None = None

    # --- how strongly it is believed --------------------------------------------------------
    status: str = "observation"
    confidence: str = "low"
    notes: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)

    def problems(self) -> tuple[str, ...]:
        """Ways this record claims more than its evidence supports. Empty when honest.

        The rungs are cumulative, and each one names the thing that would otherwise be assumed:
        a motif that cites one artifact is an observation with ambitions; a hypothesis with no CCA
        axis cannot be routed to anything and so cannot be tested; a validated policy without a
        control, a measurement and a passing correctness gate is the failure this repo has already
        recorded — a speedup credited to a candidate that computed the wrong answer, or a win
        measured against nothing.
        """
        out: list[str] = []
        if self.status not in STATUSES:
            out.append(f"status {self.status!r} is not one of {list(STATUSES)}")
            return tuple(out)
        rung = STATUSES.index(self.status)

        if rung >= STATUSES.index("motif") and len(set(self.evidence)) < 2:
            out.append("a motif generalizes over artifacts: cite at least two, or stay an observation")
        if rung >= STATUSES.index("hypothesis") and not self.cca_axes:
            out.append("a hypothesis must name the CCA axis it acts on, or nothing can test it")
        if rung == STATUSES.index("validated_policy"):
            if self.control is None:
                out.append("a validated policy needs a matched control: a delta against nothing is "
                           "not a delta")
            if self.delta_vs_control is None or self.measured_cycles is None:
                out.append("a validated policy needs a MEASURED result, not an expectation")
            if self.measurement_authority is None:
                out.append("a measured number must name the substrate that produced it")
            if self.correctness_ok is not True:
                out.append("no speedup is credited without a passing correctness gate")
        return tuple(out)

    def is_honest(self) -> bool:
        return not self.problems()

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict
        d = asdict(self)
        d["problems"] = list(self.problems())
        return d


def promote(record: DecisionRecord, to: str) -> DecisionRecord:
    """Return ``record`` at a higher rung, or raise if its evidence does not reach it.

    Promotion is the moment a reading becomes a rule, so it is the moment to check — not later,
    when the rule is already being applied and its provenance has been summarized away.
    """
    import dataclasses

    if to not in STATUSES:
        raise ValueError(f"unknown status {to!r}")
    if STATUSES.index(to) < STATUSES.index(record.status):
        raise ValueError(f"promote() does not demote ({record.status!r} -> {to!r})")
    candidate = dataclasses.replace(record, status=to)
    problems = candidate.problems()
    if problems:
        raise ValueError(f"cannot promote {record.decision!r} to {to!r}: " + "; ".join(problems))
    return candidate
