"""A cycle BAND for a program too large to run, composed from measurements of smaller siblings.

WHY A BAND AND NOT A NUMBER
---------------------------
The cycle-accurate oracle costs a fitted ``12.89 s + 4.60 ms`` per simulated cycle here -- about 217
simulated cycles a second -- over a domain of 161..28,118 cycles. A realistic convolution layer is
~451k cycles, so one member is ~35 minutes per measurement, and the cost fit refuses to extrapolate
past twice its measured domain at all. The shapes that most need optimising are exactly the ones the
oracle cannot reach, which is why the corpus contains only single contractions and why the levers
that matter at model scale get no signal.

So the number has to be composed. What may honestly be composed is a RANGE:

``lower``
    The structural floor: the priced MAC demand over the target's derived peak. A program cannot
    retire its own arithmetic faster than the array can do arithmetic. It is a bound, never an
    estimate -- a spilling schedule only ever costs more.

``upper``
    The same work over the SLOWEST rate anything on this machine has been measured at. Empirical, not
    structural, and it must be supplied by the caller from measured evidence -- this module will not
    invent one.

⚠️ **THE SERIAL PER-COMMAND SUM WAS TRIED AS THE CEILING AND IS REFUTED.** It was the obvious
candidate: the calibration artifact declares its own fidelity as ``linear, serial; no overlap`` and
this machine's measured composition is partial (eta 0.1667), so a serial sum ought to credit no
overlap the machine achieves and thus over-predict. Measured over 25 labelled programs on 24
workloads, it contained **0** of them -- every single measurement was ABOVE it, by 2.0x to 39.2x,
median 2.9x. The cause is structural and is the same one recorded in ``tiled-unit-needs-two-k-points``:
the model prices a HISTOGRAM OF COMMAND KINDS, and one ``MATMUL`` is one command whether it contracts
over 16 or over 16384. Its ceiling was therefore nearly constant at ~133 cycles across workloads whose
true cost spans 269..3877. A per-command constant cannot price a tiled unit; only something that
scales with the work can. The refutation is kept in
``out/artifacts/perf-bench/<target>/composed_band_validation.json`` and the histogram is still
reported as a diagnostic, because a number that was wrong for a stated reason is evidence and
deleting it invites the next person to try it again.

A measurement that falls between the two ends is consistent with both; that is the whole claim.
Neither end is a prediction of where in the band the truth sits, and nothing here returns a point.
The band is only as useful as it is TIGHT -- a very wide band is sound and eliminates nothing -- so
``width`` is reported and callers are expected to look at it.

WHY NOT ``differential.compare``
--------------------------------
:func:`merlin.perf.differential.compare` refuses unconditionally when a machine's composition
operator is PARTIAL, and correctly: a partial-overlap operator credits pairs, so neither the
magnitude nor the ordering of a difference between two composed demands survives. That refusal is
about differencing two *composed demand envelopes*. A band is a different object -- two independently
sound one-sided bounds -- and :func:`compare` below never subtracts them. It asks only whether the
two intervals are DISJOINT, which is an ordering statement that holds whatever the overlap is.

WHAT THIS IS LICENSED TO DO
---------------------------
Eliminate, never certify. Two overlapping bands are UNKNOWN, not a tie and not a win. The measured
held-out study that condemned every cheap ordering signal here (the correctness simulator agrees
46.1% of the time, a per-command cost model 39.3%, both at or below chance) was measured on
*within-workload* candidate pairs at tile scale, where the compute term cannot vary between arms and
the terms that do vary anti-correlate. Disjointness is a strictly weaker question than ranking and is
asked only when the intervals do not touch, which at tile scale is almost never -- so this adds
elimination power exactly where effect sizes are large and stays silent where that study says it must.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

#: The ABI opcode -> calibrated event name. Both vocabularies are DECLARED: the left side is what an
#: emitted buffer contains, the right is what a per-command model is fitted on. An opcode outside this
#: map makes the histogram incomplete, and an incomplete histogram priced as whole understates the arm,
#: so it refuses rather than dropping the command.
OPCODE_EVENTS: Mapping[str, str] = {
    "RES_PACK": "mvin2_B", "MATMUL_RESIDENT": "compute", "MATMUL": "compute",
    "COMMIT": "mvout", "EVICT": "mvout", "FENCE": "fence", "BIAS_ADD": "compute",
    "VECTOR_MAP": "compute", "VREDUCE": "compute", "CONV2D": "compute",
    "MOVEMENT": "mvin_A", "ATTENTION_QK": "compute", "ATTENTION_PV": "compute",
    "BATCHED_MATMUL": "compute",
}

DERIVED = "derived"
UNAVAILABLE = "unavailable"

#: :func:`compare` verdicts. There is no third outcome on purpose: a band may rule a candidate out and
#: may never rule one in.
ELIMINATE = "ELIMINATE"
UNKNOWN = "UNKNOWN"


def command_events(buffer: object) -> dict[str, float] | None:
    """Histogram an emitted buffer's opcodes onto the event names a cost model is fitted on."""
    if not isinstance(buffer, Mapping):
        return None
    rows = buffer.get("commands")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows:
        return None
    events: dict[str, float] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            return None
        event = OPCODE_EVENTS.get(str(row.get("opcode") or ""))
        if event is None:
            return None
        events[event] = events.get(event, 0.0) + 1.0
    return events


#: Compute opcodes, most specific first. The class a buffer belongs to is the first of these it
#: contains -- a declared property of the emitted program, not a fitted grouping.
COMPUTE_CLASSES: tuple[str, ...] = ("CONV2D", "ATTENTION_QK", "ATTENTION_PV", "BATCHED_MATMUL",
                                    "MATMUL_RESIDENT", "MATMUL")


def compute_class(buffer: object) -> str | None:
    """Which compute opcode dominates this program, or None when it issues none.

    ⚠️ ONE RATE FOR EVERY CLASS MAKES A SOUND BAND USELESS, and that is measured, not feared. A single
    slowest-observed rate over all programs was contained 18/18 on held-out workloads -- and every
    band it produced was **95.7x wide**, with the measurements sitting at a median position of 0.12
    inside it. The reason is that the slowest class on this machine (convolution, 2.67 MACs/cycle)
    and the fastest (a resident matmul, 94.1) differ by 35x, so a bound covering both is 35x looser
    than either needs. Two bands that wide essentially never fail to overlap, and a band that never
    separates eliminates nothing.

    Splitting on the compute opcode is not a free parameter: it is read off the program. What it must
    not become is a per-WORKLOAD rate, which would be fitting the answer.
    """
    if not isinstance(buffer, Mapping):
        return None
    opcodes = {str(row.get("opcode") or "") for row in (buffer.get("commands") or [])
               if isinstance(row, Mapping)}
    for candidate in COMPUTE_CLASSES:
        if candidate in opcodes:
            return candidate
    return None


def cost_model_artifact(target: str) -> Path | None:
    """A target's calibrated coefficients, resolved BY NAME rather than hardcoded per target.

    Adding a second target is then a calibration file, not a code edit -- which is the seam this
    module is required to go through.
    """
    from merlin.common.paths import merlin_dir  # noqa: PLC0415

    if not target:
        return None
    candidate = merlin_dir() / "python" / "merlin" / "cost_model" / f"{target}_cost_coeffs.json"
    return candidate if candidate.is_file() else None


def _serial_ceiling(target: str, buffer: Mapping[str, Any]) -> dict[str, Any]:
    """The per-command model's serial sum, plus its own measured error, as an upper bound."""
    events = command_events(buffer)
    if events is None:
        return {"status": UNAVAILABLE,
                "reason": ("the buffer declares a command outside the calibrated vocabulary, so its "
                           "event histogram is incomplete and pricing it would understate this arm")}
    artifact = cost_model_artifact(target)
    if artifact is None:
        return {"status": UNAVAILABLE, "reason": f"no calibrated cost model for target {target!r}"}
    try:
        from merlin.cost_model.linear import LinearCostModel  # noqa: PLC0415
        model = LinearCostModel.load(artifact)
        cycles, spread = model.predict_with_band(events)
    except Exception as exc:  # noqa: BLE001 - an uncalibrated target screens nothing, and says so
        return {"status": UNAVAILABLE,
                "reason": f"the calibrated model for {target!r} did not load: {type(exc).__name__}"}
    # The BAND IS ADDED, not subtracted. A ceiling that quoted the fit's central value would be beaten
    # by any program inside the model's own measured error, which is not a ceiling.
    return {"status": DERIVED, "cycles": float(cycles) + float(spread), "central": float(cycles),
            "model_band": float(spread), "events": events,
            "basis": "per-command coefficients summed serially, plus the model's measured error",
            "licence": "an upper bound: a serial sum credits no overlap this machine achieves"}


def _structural_floor(buffer: Mapping[str, Any], peak_macs_per_cycle: float | None) -> dict[str, Any]:
    """Cycles the arm cannot beat, from the work its own command buffer prices."""
    if not peak_macs_per_cycle:
        return {"status": UNAVAILABLE, "reason": "no derived structural peak for this target"}
    from merlin.perf.work_volume import work_from_command_buffer  # noqa: PLC0415

    work = work_from_command_buffer(buffer)
    macs = int(getattr(work, "known_macs", 0) or 0)
    if not macs:
        return {"status": UNAVAILABLE, "reason": "the buffer prices no work"}
    # A LOWER BOUND OVER PARTIAL WORK IS STILL A LOWER BOUND. An unrecognised opcode means some
    # commands went uncounted, so the true demand is at least this -- which is the direction a floor
    # may err in. It is reported, because a floor built from half the program is much weaker.
    return {"status": DERIVED, "cycles": macs / float(peak_macs_per_cycle), "macs": macs,
            "counts_every_command": not bool(getattr(work, "is_lower_bound", False)),
            "basis": "priced MAC demand over the target's derived structural peak",
            "licence": "a floor the arm cannot beat; never an estimate of what it will cost"}


def _empirical_ceiling(buffer: Mapping[str, Any],
                       slowest_macs_per_cycle: float | None) -> dict[str, Any]:
    """The same work over the slowest rate anything on this machine has been MEASURED at.

    The rate is a parameter, never derived here, and it must come from measured baselines. It is an
    EMPIRICAL bound: a program slower than anything yet observed would sit above it, which is exactly
    why the containment rate has to be measured on held-out workloads rather than asserted.
    """
    if not slowest_macs_per_cycle or slowest_macs_per_cycle <= 0:
        return {"status": UNAVAILABLE,
                "reason": "no measured slowest rate was supplied, and one is not invented here"}
    from merlin.perf.work_volume import work_from_command_buffer  # noqa: PLC0415

    work = work_from_command_buffer(buffer)
    macs = int(getattr(work, "known_macs", 0) or 0)
    if not macs:
        return {"status": UNAVAILABLE, "reason": "the buffer prices no work"}
    if getattr(work, "is_lower_bound", False):
        # A CEILING BUILT ON PARTIAL WORK IS NOT A CEILING. The floor may err downward on uncounted
        # commands; the ceiling may not, because uncounted work makes the true cost larger while this
        # estimate stays the same. The asymmetry is the reason the two ends check different things.
        return {"status": UNAVAILABLE,
                "reason": ("some commands have no work-counting rule, so the priced work is a lower "
                           "bound and dividing it by a rate cannot bound the cost from above")}
    return {"status": DERIVED, "cycles": macs / float(slowest_macs_per_cycle), "macs": macs,
            "slowest_macs_per_cycle": float(slowest_macs_per_cycle),
            "basis": "priced MAC demand over the slowest measured rate on this machine",
            "licence": "an EMPIRICAL ceiling: a program slower than anything measured would exceed it"}


def band(buffer: object, *, target: str, peak_macs_per_cycle: float | None,
         slowest_macs_per_cycle: float | None = None) -> dict[str, Any]:
    """The interval a program's cycle count must lie in, or a refusal naming what is missing.

    Both ends carry their own provenance and either may be absent. An absent end is reported as
    ``None`` with a reason and never substituted with a number -- a band with an invented end is not
    a weaker band, it is a false one.
    """
    if not isinstance(buffer, Mapping):
        return {"status": UNAVAILABLE, "reason": "the command buffer is not a mapping",
                "lower": None, "upper": None}
    floor = _structural_floor(buffer, peak_macs_per_cycle)
    ceiling = _empirical_ceiling(buffer, slowest_macs_per_cycle)
    lower = floor["cycles"] if floor["status"] == DERIVED else None
    upper = ceiling["cycles"] if ceiling["status"] == DERIVED else None
    out: dict[str, Any] = {"lower": lower, "upper": upper, "floor": floor, "ceiling": ceiling,
                           "target": target,
                           # Reported, never used as a bound -- see the refutation in the module
                           # docstring. Kept so the histogram stays visible as a diagnostic.
                           "command_histogram": command_events(buffer)}
    if lower is None or upper is None:
        out["status"] = UNAVAILABLE
        out["reason"] = "; ".join(
            part["reason"] for part in (floor, ceiling) if part["status"] != DERIVED)
        return out
    if upper < lower:
        # THE ONE INCONSISTENCY WORTH REFUSING OVER. A ceiling below a floor means one of the two is
        # wrong -- an uncalibrated event, a mis-derived peak -- and the interval it describes is
        # empty. Returning it would let `compare` declare two empty intervals disjoint.
        out["status"] = UNAVAILABLE
        out["reason"] = (f"the serial ceiling {upper:.1f} is below the structural floor {lower:.1f}, "
                         f"so one of the two bounds is wrong and the interval is empty")
        return out
    out["status"] = DERIVED
    out["width"] = upper - lower
    out["licence"] = "may ELIMINATE a candidate; may never certify one"
    return out


def compare(baseline: object, candidate: object) -> dict[str, Any]:
    """Is the candidate's band wholly on one side of the baseline's?

    ELIMINATE only when the intervals do not overlap. Anything else is UNKNOWN -- including the case
    where they merely touch, and including the case where one is much wider than the other. Two
    programs whose bands overlap have not been shown to differ, and reporting the more likely of two
    overlapping intervals would be the point estimate this module exists to avoid.
    """
    if not isinstance(baseline, Mapping) or not isinstance(candidate, Mapping):
        return {"verdict": UNKNOWN, "reason": "a band is not a mapping"}
    if baseline.get("status") != DERIVED or candidate.get("status") != DERIVED:
        return {"verdict": UNKNOWN,
                "reason": "; ".join(str(side.get("reason") or "a band could not be derived")
                                    for side in (baseline, candidate)
                                    if side.get("status") != DERIVED)}
    b_lo, b_hi = float(baseline["lower"]), float(baseline["upper"])
    c_lo, c_hi = float(candidate["lower"]), float(candidate["upper"])
    if c_hi < b_lo:
        return {"verdict": ELIMINATE, "faster": "candidate",
                "reason": (f"the candidate's whole band [{c_lo:.1f}, {c_hi:.1f}] lies below the "
                           f"baseline's [{b_lo:.1f}, {b_hi:.1f}]"),
                "separation_cycles": b_lo - c_hi}
    if b_hi < c_lo:
        return {"verdict": ELIMINATE, "faster": "baseline",
                "reason": (f"the candidate's whole band [{c_lo:.1f}, {c_hi:.1f}] lies above the "
                           f"baseline's [{b_lo:.1f}, {b_hi:.1f}]"),
                "separation_cycles": c_lo - b_hi}
    return {"verdict": UNKNOWN,
            "reason": (f"the bands overlap ([{b_lo:.1f}, {b_hi:.1f}] against [{c_lo:.1f}, {c_hi:.1f}]), "
                       f"so neither program has been shown to be faster")}


def contains(one_band: object, measured_cycles: float) -> bool | None:
    """Did a MEASURED cycle count fall inside the band? ``None`` when the band was not derived.

    This is the acceptance question. A band is only worth exposing if measurements land inside it, and
    the way to find out is to replay it over the members that HAVE been measured -- see
    ``validate_composed_bands.py``. Kept here so the containment test and the band share one
    definition of the interval.
    """
    if not isinstance(one_band, Mapping) or one_band.get("status") != DERIVED:
        return None
    return float(one_band["lower"]) <= float(measured_cycles) <= float(one_band["upper"])
