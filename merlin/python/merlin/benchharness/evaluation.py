"""One evaluation result schema, shared by every method being compared.

A study that compares agent-written kernels against a compiler is only meaningful if both are scored
by the same oracle and reported in the same shape. Two arms scored by different oracles produce an
apples/oranges join that no amount of statistics repairs, so this module owns the schema and the
redaction, and each arm supplies only its artifact.

Three rules are encoded here rather than left to callers, because each has already gone wrong
somewhere in this repo and each fails in the flattering direction:

**A tier that certifies EXECUTION does not certify NUMERICS.** Some oracles run a kernel to
completion without comparing its output to anything. Reading such a tier as a pass turns "it did not
crash" into "it computed the right answer". Only a tier the caller names in ``certifying_tiers`` may
set ``verdict="match"``; every other passing tier yields ``not_certified`` plus a caveat naming the
limit.

**An unwinnable task is not a failure.** A task whose reference answer depends on data no general
submission can reconstruct cannot be won by any correct compiler. Counting it as a failure overstates
the shortfall and hides the corpus gap. It is ``structurally_unwinnable`` and leaves BOTH the
numerator and denominator -- the same discipline the eligibility oracle already uses for
``undetermined``.

**Unavailable is never zero.** A crashed oracle, an absent toolchain and a genuine failure are three
different things. Each keeps its own status and carries a reason.

Redaction is a WHITELIST, never a blacklist. The grader's own mismatch report embeds the expected
value (``first_mismatch = {output, index, expected, observed}``), so a blacklist that forgot one key
would hand an agent the answer key. :meth:`EvaluationResult.redact` names what may leave, and
everything else is dropped by construction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

#: Statuses an evaluation can end in.
#: ``unsupported`` is deliberately distinct from ``fail``: it means no correct submission could have
#: won this task, so it is excluded from scoring rather than counted against the method.
STATUSES = ("pass", "fail", "error", "unsupported", "declined")

#: Utilization counters carried from a tier, when the target's perf hook supplies them. Named
#: explicitly rather than copied wholesale so an unrelated tier key cannot leak into agent feedback.
_UTIL_FIELDS = (
    "warp_occupancy", "fp_util", "int_util", "sfu_util",
    "smem_lane_util", "dma_util", "tensor_util", "smem_conflict_rate",
)

#: Correctness verdicts. ``not_certified`` means the kernel ran but nothing compared its output --
#: an execution cert, not a numeric one.
VERDICTS = ("match", "mismatch", "not_certified", "structurally_unwinnable", "unknown")


@dataclass
class Caveat:
    """A machine-readable limit on how far a result may be quoted."""
    code: str
    detail: str = ""


@dataclass
class EvaluationResult:
    """What one (task, config, artifact) evaluation established -- and what it did not.

    The full record is for the experimenter. :meth:`redact` produces the agent-facing view.
    """

    task_id: str
    config_id: str
    target: str
    method: str = ""
    arm: str = ""

    status: str = "error"
    tier_reached: str | None = None
    headline: str = ""

    verdict: str = "unknown"
    certifying_tier: str | None = None
    mismatch_count: int | None = None
    max_abs_error: float | None = None
    max_rel_error: float | None = None
    numeric_policy: str = ""

    cycles: int | None = None
    #: WHICH tier produced `cycles`, and whether that tier is cycle-accurate. Without these a latency
    #: from a functional model is indistinguishable from one measured on RTL, and the two differ by
    #: more than any optimization being measured. A cert-tier run whose RTL engine reports no cycle
    #: count still yields a MODEL latency, and this is what says so.
    cycles_tier: str | None = None
    cycles_cycle_accurate: bool = False
    gflops: float | None = None
    pct_fp_peak: float | None = None
    perf_valid: bool = False
    #: Where the machine's time went, as fractions of the same cycle window. Latency alone cannot say
    #: WHY a kernel is slow; these can. A None entry means the simulator did not report that counter
    #: and must stay distinct from a real 0.0, which would invent an efficiency result.
    utilization: dict[str, float | None] = field(default_factory=dict)

    compile_ok: bool = False
    run_ok: bool = False
    failure_plane: str | None = None
    failure_category: str | None = None
    failure_detail: str = ""

    build_seconds: float | None = None
    eval_seconds: float | None = None

    artifact_provenance: str = ""
    toolchain: str = ""
    toolchain_shas: dict[str, str] = field(default_factory=dict)
    oracle: str = ""

    caveats: list[Caveat] = field(default_factory=list)
    tiers: dict[str, Any] = field(default_factory=dict)

    @property
    def is_scoreable(self) -> bool:
        """False when no correct submission could have won -- excluded from both sides of a ratio."""
        return self.status != "unsupported" and self.verdict != "structurally_unwinnable"

    @property
    def counts_as_correct(self) -> bool:
        """Correctness requires a tier that actually compared the output to a reference."""
        return self.verdict == "match" and self.certifying_tier is not None

    def add_caveat(self, code: str, detail: str = "") -> None:
        self.caveats.append(Caveat(code=code, detail=detail))

    def to_dict(self) -> dict:
        """The full record, INCLUDING the derived verdicts.

        `correct`, `is_scoreable` and `perf_valid` are properties, so `asdict` drops them and every
        stored result read back `correct: null` -- on rounds that had in fact matched at a certifying
        tier. A reader cannot tell that null from "the oracle could not say", which is the precise
        confusion the schema's own honesty rule exists to prevent, so they are written down.
        """
        out = asdict(self)
        out["correct"] = self.counts_as_correct
        out["is_scoreable"] = self.is_scoreable
        return out

    def redact(self) -> dict:
        """The agent-facing view: only what a developer could obtain without the answer key.

        A whitelist, because the grader's mismatch report embeds the expected value. Error
        MAGNITUDES are kept -- a developer profiling their own kernel can see how far off it is --
        but never an expected value, an index, or a golden path.
        """
        return {
            "status": self.status,
            "tier_reached": self.tier_reached,
            "correct": self.counts_as_correct,
            "verdict": self.verdict,
            "compile_ok": self.compile_ok,
            "run_ok": self.run_ok,
            "failure_plane": self.failure_plane,
            "failure_category": self.failure_category,
            "failure_detail": self.failure_detail,
            "mismatch_count": self.mismatch_count,
            "max_abs_error": self.max_abs_error,
            "max_rel_error": self.max_rel_error,
            "cycles": self.cycles,
            "cycles_tier": self.cycles_tier,
            "cycles_cycle_accurate": self.cycles_cycle_accurate,
            "gflops": self.gflops,
            "pct_fp_peak": self.pct_fp_peak,
            # Utilization is a profiler counter, not an answer -- a developer running their own kernel
            # under a profiler sees exactly this, so it is safe to return and is what an optimization
            # round needs in order to act on the bottleneck rather than guess.
            "utilization": dict(self.utilization),
            "build_seconds": self.build_seconds,
            "caveats": [asdict(c) for c in self.caveats],
        }


def from_capsule_result(
    result: dict,
    *,
    task_id: str,
    config_id: str,
    target: str,
    certifying_tiers: frozenset[str] | set[str] | tuple[str, ...],
    method: str = "",
    arm: str = "",
    artifact_provenance: str = "",
    unwinnable_reason: str | None = None,
) -> EvaluationResult:
    """Convert a capsule-runner result into the shared schema.

    ``certifying_tiers`` names the tiers whose pass actually compares output to a reference. It is a
    parameter rather than a constant because it is a property of the TARGET's oracle ladder, and
    baking one target's answer in here is exactly the overfitting this repo forbids.

    ``unwinnable_reason``, when given, marks a task no correct submission could win; the result is
    then ``unsupported`` and leaves both sides of any ratio.
    """
    tiers = result.get("tiers") or {}
    numeric = result.get("numeric") or {}
    failure = result.get("failure") or {}

    ev = EvaluationResult(
        task_id=task_id, config_id=config_id, target=target,
        method=method, arm=arm,
        artifact_provenance=artifact_provenance,
        tiers=tiers,
        toolchain_shas=dict(result.get("toolchain_shas") or {}),
    )

    passed = [t for t, d in tiers.items() if isinstance(d, dict) and d.get("status") == "pass"]
    # Highest tier by the ladder's own ordering, not lexicographic: "L10" must not sort below "L2".
    ev.tier_reached = max(passed, key=_tier_rank) if passed else None

    certifying = {t for t in passed if t in set(certifying_tiers)}
    ev.certifying_tier = max(certifying, key=_tier_rank) if certifying else None

    for tier in passed:
        d = tiers[tier]
        if d.get("cycles") is not None and (ev.cycles is None or tier in certifying):
            ev.cycles = d.get("cycles")
            ev.cycles_tier = tier
            ev.cycles_cycle_accurate = bool(d.get("cycle_accurate"))
            ev.gflops = d.get("gflops")
            ev.pct_fp_peak = d.get("pct_fp_peak")
            # The tier nests these under "utilization"; older/other targets may inline them. Accept
            # both, and filter to the known counters so an unrelated tier key cannot reach an agent.
            src = d.get("utilization") if isinstance(d.get("utilization"), dict) else d
            ev.utilization = {k: src[k] for k in _UTIL_FIELDS if k in src}
            ev.oracle = d.get("evidence") or ev.oracle
            timing = d.get("timing") or {}
            ev.build_seconds = timing.get("build_s")
            ev.eval_seconds = timing.get("adapter_wall_s")
        if d.get("completion_only"):
            ev.add_caveat(
                "tier_certifies_execution_only",
                f"{tier} ran the kernel to completion but compared no output; "
                "it is an execution cert, not a numeric one",
            )
        if d.get("toolchain"):
            ev.toolchain = d["toolchain"]

    ev.status = result.get("status") or "error"
    ev.mismatch_count = numeric.get("mismatch_count")
    ev.max_abs_error = _as_float(numeric.get("max_abs_diff"))
    ev.max_rel_error = _as_float(numeric.get("max_rel_error"))
    ev.numeric_policy = numeric.get("policy") or ""
    ev.compile_ok = failure.get("plane") not in ("parse", "compile", "lower")
    ev.run_ok = bool(passed)
    if failure:
        ev.failure_plane = failure.get("plane")
        ev.failure_category = failure.get("category")
        ev.failure_detail = str(failure.get("detail") or "")[:400]

    if unwinnable_reason:
        ev.status = "unsupported"
        ev.verdict = "structurally_unwinnable"
        ev.add_caveat("structurally_unwinnable", unwinnable_reason)
    elif numeric.get("status") == "pass" and ev.certifying_tier is not None:
        ev.verdict = "match"
    elif numeric.get("status") == "fail":
        ev.verdict = "mismatch"
    elif ev.run_ok:
        # It ran, but nothing that compares output signed off on it.
        ev.verdict = "not_certified"
        ev.add_caveat(
            "no_certifying_tier",
            f"passed {sorted(passed)} but none of {sorted(set(certifying_tiers))} certifies numerics",
        )

    # Performance is only meaningful for a kernel that computed the right answer; an incorrect
    # kernel can be arbitrarily fast by skipping the work.
    ev.perf_valid = ev.cycles is not None and ev.counts_as_correct

    # A cycle-accurate tier that passed while reporting NO cycles leaves the latency coming from a
    # functional model. That is legitimate -- it is an execution cert, not a timing one -- but it must
    # never be read as a measured latency, so it is named here rather than left to the tier table.
    rtl_timed = [t for t in passed
                 if isinstance(tiers[t], dict) and tiers[t].get("cycle_accurate")
                 and tiers[t].get("cycles") is not None]
    rtl_untimed = [t for t in passed
                   if isinstance(tiers[t], dict) and tiers[t].get("cycle_accurate")
                   and tiers[t].get("cycles") is None]
    if rtl_untimed and not rtl_timed and ev.cycles is not None and not ev.cycles_cycle_accurate:
        ev.add_caveat(
            "latency_is_a_model_estimate",
            f"{sorted(rtl_untimed)} certified execution on RTL but reported no cycle count, so the "
            f"{ev.cycles} cycles come from {ev.cycles_tier} (a functional model). Correctness is "
            f"RTL-backed; the LATENCY is not measured hardware timing.",
        )

    # A substituted reference artifact does not measure the submission, so say so where the number is.
    if ev.toolchain and "not-the-submission" in ev.toolchain:
        ev.add_caveat("result_not_attributable_to_submission", ev.toolchain)

    return ev


def _tier_rank(tier: str) -> tuple[int, str]:
    """Order tiers by their numeric suffix, so L10 ranks above L2 rather than below it."""
    digits = "".join(c for c in tier if c.isdigit())
    return (int(digits) if digits else -1, tier)


def _as_float(v: Any) -> float | None:
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None
