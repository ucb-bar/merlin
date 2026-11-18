"""Recover the timing evidence that functional runs already produce -- and refuse to over-claim it.

Every capsule this tree grades executes on a real oracle. Those runs cost hours of simulator time and
they emit a cycle count, and until now the cycle count was thrown away the moment the functional
verdict was written. This module recovers it, from artifacts already on disk as well as from live
runs, and hands it to the performance layer under three rules that are enforced here rather than
remembered.

WHY A HARVESTED LATENCY IS NOT A LATENCY
----------------------------------------
A number read out of a real program is **contended**. The unit was not alone: other engines were
moving data, the sequencer was arbitrating, and whatever the number is, the isolated cost of the
operation is no larger. So a harvested value is an **upper bound with a validity domain naming what
else was active**, and this module refuses to spell it any other way:

* its provenance kind is forced to :data:`HARVEST_KIND` (``trace_derived``) -- weaker than
  ``measured`` and weaker than ``calibrated``, so :func:`merlin.analysis.perf.contract.terms.combine_kinds` drags every
  composite built on it down to the same confidence;
* the occurrence **spread** travels with it (a value observed once at 1090 and once at 8889 is not a
  latency, it is a distribution, and the corpus contains exactly that case);
* :func:`promote` -- the only door from harvested to calibrated -- always raises. Calibration needs a
  dedicated experiment in which the thing being priced is the only thing running. There is no code
  path that turns a trace observation into a constant.

WHAT MAY CONTRIBUTE
-------------------
Only substrates the target's :class:`~merlin.analysis.perf.measurement.authority.MeasurementAuthority` declares
citable at the tier the number was reached at. A target that declares nothing contributes nothing:
an undeclared authority is a policy statement, not a licence to fall back to whichever substrate
happened to emit a ``cycles`` field. Measured on the corpus on disk this is not a formality -- on one
target it drops 929 functional-tier observations and keeps 360 RTL ones, and the functional tier is
the one that overcounts by ~3x.

AND WHAT IS NEVER FILLED IN
---------------------------
Harvest never turns an UNKNOWN into a number. Where a series is empty, or a pairing cannot be
established, or a program leaves the law's validated regime, the output is an UNKNOWN term with the
reason and a recorded refusal -- never a zero, never a silently dropped input.
:data:`merlin.analysis.perf.contract.terms.UNKNOWN` refuses ``__bool__`` and ``__float__``, and nothing here works
around that.

THE RULE REGISTRY
-----------------
The second half of this module reads ``src/merlin/resources/contracts/perf_rules/*.yaml``: *detected trait ->
required model term -> optimization family -> experiment family*, as data. The registry is data
because a rule expressed as code is a rule that gets a target baked into it; YAML also keeps the
name gate's scan roots clean.

A rule emits an experiment only when the evidence can actually carry it: **at least two observation
points per fitted parameter, and at least as many distinct x-levels as parameters**. That single
gate is what makes the registry's output honest rather than aspirational -- a rate fitted through one
point cannot separate a per-unit cost from a fixed overhead, and the rules whose axis the corpus
cannot observe come back as refusals naming what to buy.

Nothing here names a target, a unit, an opcode or a geometry. Bucket-to-family pairings are derived
from the evidence's own support sets and regression behaviour, and refuse when they are not unique.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any

from merlin.analysis.perf.contract.harvest import FIT_FORMS, AxisEvidence, Fit
from merlin.analysis.perf.contract.profile import TargetProfile
from merlin.analysis.perf.contract.record import (
    SUITE_SCHEMA,
    SuiteSchema,
    compose_unit_busy,
    derive_delay_mnemonic,
    derive_unit_roles,
    fill_cycles,
)
from merlin.analysis.perf.contract.terms import UNKNOWN, Bounds, PerformanceTerm, Provenance, Validity
from merlin.analysis.perf.measurement.authority import MeasurementAuthority, authority_for, citable
from merlin.analysis.perf.measurement.observations import TIMING_OBSERVATIONS_KEY as _TIMING_OBSERVATIONS_KEY
from merlin.common.paths import artifacts_dir, contract_dir, runs_dir
from merlin.common.record_formats import load_yaml

# ---- harvest -----------------------------------------------------------------------------------


def axes_from_suite(
    suite: Mapping[str, Any], *, schema: SuiteSchema = SUITE_SCHEMA, movement_policy: Mapping[str, Any] | None = None
) -> tuple[dict[str, AxisEvidence], list[Refusal], dict[str, Any]]:
    """Build every observation axis a measured cycle suite can carry.

    Returns ``(axes, refusals, derivation)``. The derivation records WHICH bucket was paired with
    what and why, so a reader never has to reverse-engineer the attribution from the numbers.
    """
    from merlin.analysis.perf.contract.harvest import AxisEvidence, Point, Refusal, _pair_compute, _pair_movement

    kernels: Mapping[str, Any] = suite.get(schema.kernels_key) or {}
    axes: dict[str, AxisEvidence] = {}
    refusals: list[Refusal] = []
    derivation: dict[str, Any] = {}
    if not kernels:
        return axes, [Refusal("suite", "carries no kernels")], derivation
    buckets = sorted(
        {k for e in kernels.values() for k in (e.get(schema.activity_key) or {}) if k not in schema.non_unit_keys}
    )
    derivation["activity_buckets"] = buckets

    # --- movement -----------------------------------------------------------------------------
    mover, why, scores = _pair_movement(kernels, buckets, schema, movement_policy or {})
    derivation["movement"] = {"bucket": mover, "why": why, "beat_regression_r2": scores}
    beats = {
        n: float(e[schema.activity_key][schema.read_beats_key] + e[schema.activity_key][schema.write_beats_key])
        for n, e in kernels.items()
    }
    if mover is None:
        refusals.append(Refusal("movement bucket", why))
    else:
        axes["movement_beat_count"] = AxisEvidence(
            axis="movement_beat_count",
            x_name="beats",
            y_name=f"{mover} busy",
            y_unit="cycles",
            points=tuple(Point(beats[n], float(kernels[n][schema.activity_key][mover]), n) for n in sorted(kernels)),
            note=why,
        )

    # --- moved bytes per beat ------------------------------------------------------------------
    foot_pts = [
        Point(beats[n], float(kernels[n][schema.footprint_key]), n)
        for n in sorted(kernels)
        if kernels[n].get(schema.footprint_key) is not None
    ]
    missing = [n for n in sorted(kernels) if kernels[n].get(schema.footprint_key) is None]
    axes["moved_byte_footprint"] = AxisEvidence(
        axis="moved_byte_footprint",
        x_name="beats",
        y_name="operand footprint",
        y_unit="bytes",
        points=tuple(foot_pts),
        excluded=tuple(Refusal(n, f"carries no {schema.footprint_key!r}") for n in missing),
        note="the width of one beat is the ratio between the bytes a program touches and the beats "
        "the instrument counted for it",
    )

    # --- per-compute-unit fill and issue packing ----------------------------------------------
    streams = [e[schema.op_stream_key] for e in kernels.values() if e.get(schema.op_stream_key)]
    if not streams:
        refusals.append(Refusal("op streams", "the suite carries none, so no unit's issue structure is observable"))
        return axes, refusals, derivation
    try:
        delay_mnemonic = derive_delay_mnemonic(streams)
    except ValueError as exc:
        refusals.append(Refusal("delay marker", str(exc)))
        return axes, refusals, derivation
    derivation["delay_marker"] = delay_mnemonic
    compute_buckets = [b for b in buckets if b != mover]
    paired, pair_refusals = _pair_compute(kernels, compute_buckets, schema, delay_mnemonic)
    refusals.extend(pair_refusals)
    derivation["compute_pairings"] = dict(sorted(paired.items()))

    group_pts: list[Point] = []
    group_excl: list[Refusal] = []
    packed_pts: list[Point] = []
    packed_excl: list[Refusal] = []
    for bucket, family in sorted(paired.items()):
        try:
            roles = derive_unit_roles(streams, family, delay_mnemonic)
        except ValueError as exc:
            refusals.append(Refusal(f"{bucket}/{family} roles", str(exc)))
            continue
        derivation.setdefault("unit_roles", {})[bucket] = {
            "family": roles.family,
            "compute": roles.compute,
            "drain": roles.drain,
            "longest_scheduled_delay": roles.compute_delay,
        }
        for name in sorted(kernels):
            stream = kernels[name].get(schema.op_stream_key) or []
            busy = kernels[name][schema.activity_key].get(bucket)
            if busy is None:
                continue
            # fill=0 makes the composition return exactly the delays the PROGRAM scheduled, so the
            # measured busy minus that is what the unit's own pipeline contributed.
            composed = compose_unit_busy(stream, roles, 0, delay_mnemonic)
            if composed.groups == 0:
                continue
            label = f"{name}:{bucket}"
            if composed.cycles is None:
                group_excl.append(Refusal(label, composed.reason))
                packed_pts.append(Point(float(composed.computes) / max(composed.groups, 1), float(busy), label))
                continue
            group_pts.append(Point(float(composed.groups), float(busy) - float(composed.cycles), label))
            packed_excl.append(
                Refusal(
                    label,
                    "one compute op per drained result: this program says nothing about the cost of packing several",
                )
            )
    axes["compute_group_count"] = AxisEvidence(
        axis="compute_group_count",
        x_name="drained result groups",
        y_name="measured busy less the delays the program scheduled",
        y_unit="cycles",
        points=tuple(group_pts),
        excluded=tuple(group_excl),
        note="what remains after the program's own scheduled delays are removed is the unit's "
        "pipeline contribution per drained result",
    )
    axes["back_to_back_compute"] = AxisEvidence(
        axis="back_to_back_compute",
        x_name="compute ops per drained result",
        y_name="measured busy",
        y_unit="cycles",
        points=tuple(packed_pts),
        excluded=tuple(packed_excl),
        note="only programs that accumulate several compute ops into one drain say anything about "
        "the initiation interval",
    )

    # --- axes the instrument cannot carry at all ------------------------------------------------
    zero_work = [
        n
        for n in sorted(kernels)
        if not any(str(op[0]) in paired.values() for op in (kernels[n].get(schema.op_stream_key) or []))
        and not (kernels[n].get(schema.op_stream_key) or [])
    ]
    axes["zero_work_program"] = AxisEvidence(
        axis="zero_work_program",
        x_name="issued engine ops",
        y_name="total cycles",
        y_unit="cycles",
        points=tuple(Point(0.0, float(kernels[n][schema.activity_key][schema.total_key]), n) for n in zero_work),
        note="the run's fixed cost is separable only by a program that does no engine work; fitting "
        "an intercept through programs that DO work confounds startup with everything else "
        "that is constant across them",
    )
    axes["working_set_bytes"] = AxisEvidence(
        axis="working_set_bytes",
        x_name="resident bytes",
        y_name="total cycles",
        y_unit="cycles",
        note="a capacity shows itself as the knee where the cost per byte changes; this instrument "
        "reports no residency and no spill signal, so the knee is not observable",
    )
    axes["algorithmic_bytes"] = AxisEvidence(
        axis="algorithmic_bytes",
        x_name="bytes the algorithm needs",
        y_name="bytes moved",
        y_unit="bytes",
        note="amplification is measured bytes-moved over bytes-needed; this artifact carries no "
        "shape or dtype field, so the denominator is not derivable from it and hand-entered "
        "operand sizes would make the ratio an assumption wearing a measurement's clothes",
    )
    axes["concurrent_unit_busy"] = AxisEvidence(
        axis="concurrent_unit_busy",
        x_name="summed per-unit busy",
        y_name="observed total",
        y_unit="cycles",
        note="the activity buckets PARTITION the cycle count, so they report zero overlap whether "
        "or not overlap exists; an overlap coefficient needs an instrument that can report two "
        "units busy in the same cycle",
    )
    return axes, refusals, derivation


def detected_traits(axes: Mapping[str, AxisEvidence], derivation: Mapping[str, Any]) -> dict[str, tuple[bool, str]]:
    """Traits the EVIDENCE establishes, which static facts may have left unestablished.

    A profile trait is derived from what a target IS; these are derived from what a run DID. They are
    strictly weaker and they never overturn a refutation -- they answer exactly the question the
    profile's own ``missing`` list asks, for the cases where an engine that no static fact names is
    plainly there in the measurement.
    """
    from merlin.analysis.perf.contract.harvest import AxisEvidence

    out: dict[str, tuple[bool, str]] = {}
    mover = (derivation.get("movement") or {}).get("bucket")
    if mover:
        out["explicit_dma"] = (
            True,
            (
                derivation["movement"]["why"] + " -- an engine whose cycles track a beat counter is a data "
                "movement engine, observed rather than declared"
            ),
        )
    pairings = derivation.get("compute_pairings") or {}
    if pairings:
        out["structural_pipeline_depth"] = (
            True,
            f"{len(pairings)} activity bucket(s) pair uniquely with an op-stream family, so a "
            "per-unit pipeline contribution is separable from the program's own schedule",
        )
    if mover and pairings:
        out["multiple_engine_groups"] = (
            True,
            f"a movement engine ({mover}) and {len(pairings)} compute unit(s) carry work in the same corpus",
        )
    axis = axes.get("concurrent_unit_busy")
    if axis is not None and not axis.points:
        out["explicit_completion"] = (
            False,
            "the only activity instrument available partitions the cycle count, so no "
            "per-engine completion is observable from it",
        )
    return out


_TRAIT_STATES = {"satisfied": True, "refuted": False, "unestablished": None}


@dataclass(frozen=True)
class Rule:
    """Detected trait -> required model term -> optimization family -> experiment family."""

    id: str
    constant: str
    term: str
    axis: str
    fit_form: str
    recover: dict[str, Any]
    optimization_family: str
    experiment_family: str
    when: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    cross_check: str = ""
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "constant": self.constant,
            "term": self.term,
            "axis": self.axis,
            "fit_form": self.fit_form,
            "recover": dict(self.recover),
            "when": dict(self.when),
            "optimization_family": self.optimization_family,
            "experiment_family": self.experiment_family,
            "rationale": self.rationale,
            "cross_check": self.cross_check,
            "source": self.source,
        }

    @property
    def n_parameters(self) -> int:
        return len(FIT_FORMS.get(self.fit_form, ()))


@dataclass(frozen=True)
class Experiment:
    """A rule that the evidence can actually carry, with the point budget it demands."""

    rule: Rule
    axis: AxisEvidence
    points_required: int
    levels_required: int

    @property
    def id(self) -> str:
        return self.rule.id

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule": self.rule.id,
            "constant": self.rule.constant,
            "term": self.rule.term,
            "axis": self.rule.axis,
            "fit_form": self.rule.fit_form,
            "optimization_family": self.rule.optimization_family,
            "experiment_family": self.rule.experiment_family,
            "points_required": self.points_required,
            "levels_required": self.levels_required,
            "points_available": len(self.axis.points),
            "levels_available": self.axis.distinct_x,
            "cross_check": self.rule.cross_check,
        }


@dataclass(frozen=True)
class Deferral:
    """A rule that applies and cannot be run, with what would make it runnable. Computed, not chosen."""

    rule: Rule
    reason: str
    missing: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule": self.rule.id,
            "constant": self.rule.constant,
            "term": self.rule.term,
            "axis": self.rule.axis,
            "reason": self.reason,
            "missing": list(self.missing),
            "experiment_family": self.rule.experiment_family,
        }


@dataclass(frozen=True)
class Recovery:
    """What running one emitted experiment recovered, and how it compares to a declared constant."""

    experiment: str
    constant: str
    value: Any
    unit: str
    fit: Fit
    note: str = ""
    cross_check_value: Any = None
    within_tolerance: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment": self.experiment,
            "constant": self.constant,
            "value": (None if self.value is UNKNOWN else self.value),
            "unit": self.unit,
            "fit": self.fit.to_dict(),
            "note": self.note,
            "cross_check_value": self.cross_check_value,
            "within_tolerance": self.within_tolerance,
        }


def registry_dir() -> Path:
    """The selected contract's rule registry (:func:`merlin.common.paths.contract_dir`), never the checkout's."""
    return contract_dir() / "perf_rules"


@dataclass(frozen=True)
class RuleRegistry:
    """The loaded rules plus the axis and family vocabularies they are written against."""

    rules: tuple[Rule, ...]
    axes: dict[str, Any] = field(default_factory=dict)
    optimization_families: dict[str, Any] = field(default_factory=dict)
    experiment_families: dict[str, Any] = field(default_factory=dict)
    movement_policy: dict[str, Any] = field(default_factory=dict)
    tolerances: dict[str, Any] = field(default_factory=dict)

    def rule(self, rule_id: str) -> Rule:
        for r in self.rules:
            if r.id == rule_id:
                return r
        raise KeyError(f"no rule {rule_id!r}; the registry carries {[r.id for r in self.rules]}")

    def validate(self) -> tuple[str, ...]:
        """Every way a rule is inconsistent with the vocabularies. Empty means the data is sound."""
        problems: list[str] = []
        seen: set[str] = set()
        for r in self.rules:
            if r.id in seen:
                problems.append(f"duplicate rule id {r.id!r}")
            seen.add(r.id)
            if r.axis not in self.axes:
                problems.append(f"{r.id}: axis {r.axis!r} is not in the axis vocabulary")
            if r.fit_form not in FIT_FORMS:
                problems.append(f"{r.id}: fit form {r.fit_form!r} is not implemented ({sorted(FIT_FORMS)})")
            if r.optimization_family not in self.optimization_families:
                problems.append(f"{r.id}: optimization family {r.optimization_family!r} is not in the vocabulary")
            if r.experiment_family not in self.experiment_families:
                problems.append(f"{r.id}: experiment family {r.experiment_family!r} is not in the vocabulary")
            src = str(r.recover.get("from") or "")
            if src not in FIT_FORMS.get(r.fit_form, ()) and src != "law_inverse":
                problems.append(
                    f"{r.id}: recovers from {src!r}, which is neither a parameter of {r.fit_form!r} nor 'law_inverse'"
                )
            for trait, want in (r.when.get("traits") or {}).items():
                for state in [want] if isinstance(want, str) else list(want):
                    if state not in _TRAIT_STATES:
                        problems.append(
                            f"{r.id}: trait state {state!r} for {trait!r} is not one of {sorted(_TRAIT_STATES)}"
                        )
            if not r.rationale.strip():
                problems.append(f"{r.id}: carries no rationale; a rule nobody can read is a rule nobody can refute")
        return tuple(problems)


def _as_rule(raw: Mapping[str, Any], source: str) -> Rule:
    return Rule(
        id=str(raw["id"]),
        constant=str(raw["constant"]),
        term=str(raw["term"]),
        axis=str(raw["axis"]),
        fit_form=str(raw["fit_form"]),
        recover=dict(raw.get("recover") or {}),
        optimization_family=str(raw["optimization_family"]),
        experiment_family=str(raw["experiment_family"]),
        when=dict(raw.get("when") or {}),
        rationale=str(raw.get("rationale") or ""),
        cross_check=str(raw.get("cross_check") or ""),
        source=source,
    )


def load_registry(directory: "str | Path | None" = None) -> RuleRegistry:
    """Load ``src/merlin/resources/contracts/perf_rules/*.yaml``. Data only: the directory holds no code."""
    root = Path(directory) if directory is not None else registry_dir()
    if not root.is_dir():
        # An absent registry is not an empty one: every rule would silently stop firing.
        raise FileNotFoundError(f"no perf-rule registry at {root}")
    rules: list[Rule] = []
    axes: dict[str, Any] = {}
    opt: dict[str, Any] = {}
    exp: dict[str, Any] = {}
    movement: dict[str, Any] = {}
    tolerances: dict[str, Any] = {}
    for path in sorted(root.glob("*.yaml")):
        doc = load_yaml(path) or {}
        axes.update(doc.get("axes") or {})
        opt.update(doc.get("optimization_families") or {})
        exp.update(doc.get("experiment_families") or {})
        movement.update(doc.get("movement_pairing") or {})
        tolerances.update(doc.get("tolerances") or {})
        for raw in doc.get("rules") or []:
            rules.append(_as_rule(raw, path.name))
    return RuleRegistry(
        rules=tuple(sorted(rules, key=lambda r: r.id)),
        axes=axes,
        optimization_families=opt,
        experiment_families=exp,
        movement_policy=movement,
        tolerances=tolerances,
    )


def _trait_state(profile: TargetProfile | None, detected: Mapping[str, tuple[bool, str]], name: str) -> tuple[Any, str]:
    """The trait's answer and where it came from. A REFUTATION from static facts always wins.

    Evidence from a run can settle what static facts left open; it cannot overturn a fact that says
    the machine does not have the thing. That asymmetry is the whole reason the two are merged here
    rather than concatenated.
    """
    static: Any = None
    why = "no profile"
    if profile is not None and name in profile.traits:
        static = profile.traits[name].satisfied
        why = f"profile: {profile.traits[name].evidence}"
    if static is False:
        return False, why
    if static is True:
        return True, why
    if name in detected:
        value, evidence = detected[name]
        return value, f"harvest: {evidence}"
    return static, why


def emit_experiments(
    registry: RuleRegistry,
    *,
    axes: Mapping[str, AxisEvidence],
    profile: TargetProfile | None = None,
    detected: Mapping[str, tuple[bool, str]] | None = None,
) -> tuple[tuple[Experiment, ...], tuple[Deferral, ...]]:
    """Apply every rule; emit the ones the evidence can carry and defer the rest with a reason.

    Two gates, in order. The rule's ``when`` block decides whether the rule is ABOUT this target at
    all; the point budget decides whether it can be RUN. Both are data-driven, so the emitted set is
    computed rather than curated -- which is the property that makes it checkable.
    """
    from merlin.analysis.perf.contract.harvest import AxisEvidence

    det = dict(detected or {})
    experiments: list[Experiment] = []
    deferrals: list[Deferral] = []
    for rule in registry.rules:
        skip: str | None = None
        missing: list[str] = []
        for trait, want in sorted((rule.when.get("traits") or {}).items()):
            allowed = {_TRAIT_STATES[s] for s in ([want] if isinstance(want, str) else list(want))}
            state, why = _trait_state(profile, det, trait)
            if state not in allowed:
                skip = (
                    f"trait {trait!r} is "
                    f"{ {True: 'satisfied', False: 'refuted', None: 'unestablished'}[state] }, "
                    f"and this rule applies when it is {want} ({why})"
                )
                missing.append(f"settle trait {trait!r}")
                break
        kinds = rule.when.get("datapath_kind")
        if skip is None and kinds:
            have = profile.archetype.datapath_kind if profile is not None else None
            if have not in list(kinds):
                skip = f"the datapath kind is {have!r} and this rule applies to {list(kinds)}"
                missing.append("a datapath whose fill law this rule states")
        if skip is not None:
            deferrals.append(Deferral(rule, skip, tuple(missing)))
            continue
        axis = axes.get(rule.axis)
        if axis is None:
            deferrals.append(
                Deferral(
                    rule,
                    f"axis {rule.axis!r} is not carried by any harvested evidence",
                    (f"an instrument that reports {rule.axis!r}",),
                )
            )
            continue
        k = rule.n_parameters
        need_points, need_levels = 2 * k, k
        if len(axis.points) < need_points or axis.distinct_x < need_levels:
            deferrals.append(
                Deferral(
                    rule,
                    (
                        f"{rule.fit_form} fits {k} parameter(s) and needs >={need_points} points at "
                        f">={need_levels} distinct {axis.x_name} level(s); the evidence carries "
                        f"{len(axis.points)} at {axis.distinct_x}. " + (axis.note or "")
                    ),
                    tuple(
                        [f"{need_points - len(axis.points)} more observation(s) on {rule.axis!r}"]
                        if len(axis.points) < need_points
                        else []
                    )
                    + tuple(
                        [f"{need_levels - axis.distinct_x} more distinct {axis.x_name} level(s)"]
                        if axis.distinct_x < need_levels
                        else []
                    ),
                )
            )
            continue
        experiments.append(Experiment(rule=rule, axis=axis, points_required=need_points, levels_required=need_levels))
    return tuple(experiments), tuple(deferrals)


def run_experiment(
    experiment: Experiment, *, cross_check: Mapping[str, Any] | None = None, tolerance: float = 0.0
) -> Recovery:
    """Fit the experiment's axis and read the constant out of the fitted parameters.

    ``law_inverse`` recovers a structural dimension by evaluating the named fill law forwards until
    it reproduces the fitted value, so the law is stated in exactly one place.
    """
    from merlin.analysis.perf.contract.harvest import fit_points, invert_fill_law

    rule = experiment.rule
    fit = fit_points(experiment.axis, rule.fit_form)
    unit = str(rule.recover.get("unit") or experiment.axis.y_unit)
    if not fit.ok:
        return Recovery(experiment=rule.id, constant=rule.constant, value=UNKNOWN, unit=unit, fit=fit, note=fit.refusal)
    source = str(rule.recover.get("from") or "")
    note = ""
    if source == "law_inverse":
        inner = str(rule.recover.get("of") or "")
        law = str(rule.recover.get("law") or "")
        raw = fit.parameters.get(inner)
        if raw is None:
            return Recovery(
                experiment=rule.id,
                constant=rule.constant,
                value=UNKNOWN,
                unit=unit,
                fit=fit,
                note=f"the fit carries no parameter {inner!r}",
            )
        dim, why = invert_fill_law(law, raw)
        value: Any = UNKNOWN if dim is None else float(dim)
        note = why
    else:
        raw = fit.parameters.get(source)
        value = UNKNOWN if raw is None else float(raw)
        note = (
            f"{rule.constant} = the {source} of a fit of form {rule.fit_form!r} over "
            f"{fit.n_points} point(s) at {fit.distinct_x} distinct level(s)"
        )
    expected = (cross_check or {}).get(rule.cross_check) if rule.cross_check else None
    within: bool | None = None
    if expected is not None and value is not UNKNOWN:
        within = (
            abs(float(value) - float(expected)) <= float(tolerance) * abs(float(expected))
            if float(expected)
            else abs(float(value)) <= float(tolerance)
        )
    return Recovery(
        experiment=rule.id,
        constant=rule.constant,
        value=value,
        unit=unit,
        fit=fit,
        note=note,
        cross_check_value=expected,
        within_tolerance=within,
    )


def emit_harvest(
    *,
    target: str,
    suite_path: "str | Path | None" = None,
    roots: "Sequence[str | Path] | None" = None,
    version: int = 1,
    registry: RuleRegistry | None = None,
    profile: TargetProfile | None = None,
    schema: SuiteSchema = SUITE_SCHEMA,
) -> Path:
    """Retro-mine, derive the axes, emit the experiment set, run it, and write the product."""
    from merlin.analysis.perf.contract.harvest import AxisEvidence, Refusal, harvest_op_stream, retro_mine
    from merlin.common.artifacts import new_product

    reg = registry if registry is not None else load_registry()
    harvest = retro_mine(roots, target=target)
    axes: dict[str, AxisEvidence] = {}
    refusals: list[Refusal] = list(harvest.refusals)
    derivation: dict[str, Any] = {}
    cross: dict[str, Any] = {}
    suite: dict[str, Any] = {}
    if suite_path is not None:
        suite = json.loads(Path(suite_path).read_text(encoding="utf-8"))
        cross = dict(suite.get(schema.meta_key) or {})
        axes, axis_refusals, derivation = axes_from_suite(suite, schema=schema, movement_policy=reg.movement_policy)
        refusals.extend(axis_refusals)
        stream_harvest = harvest_op_stream(
            suite, target=target, authority=harvest.authority, schema=schema, submission=str(suite_path)
        )
        harvest.observations = harvest.observations + stream_harvest.observations
        refusals.extend(stream_harvest.refusals)
    harvest.refusals = tuple(refusals)
    prof = profile
    if prof is None:
        try:
            from merlin.analysis.perf.contract.profile import derive_profile

            prof = derive_profile(target)
        except Exception as exc:  # noqa: BLE001 -- a real answer
            harvest.refusals = harvest.refusals + (Refusal("target profile", f"{type(exc).__name__}: {exc}"),)
    detected = detected_traits(axes, derivation)
    experiments, deferrals = emit_experiments(reg, axes=axes, profile=prof, detected=detected)
    tol = float(reg.tolerances.get("relative", 0.0))
    recoveries = [run_experiment(e, cross_check=cross, tolerance=tol) for e in experiments]

    pd = new_product(
        "perf-harvest",
        version=version,
        target=target,
        sources=[str(s) for s in (harvest.roots or ())] + ([str(suite_path)] if suite_path else []),
        notes="timing observations recovered from runs executed for a functional "
        "verdict, plus the experiment set the rule registry emits for this "
        "target. Every harvested term is trace_derived and contended; none of "
        "them may become a calibrated constant",
    )
    pd.add_artifact("harvest.json").write_text(
        json.dumps(harvest.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    pd.add_artifact("experiments.json").write_text(
        json.dumps(
            {
                "target": target,
                "detected_traits": {k: {"value": v[0], "evidence": v[1]} for k, v in sorted(detected.items())},
                "derivation": derivation,
                "axes": {k: v.to_dict() for k, v in sorted(axes.items())},
                "emitted": [e.to_dict() for e in experiments],
                "deferred": [d.to_dict() for d in deferrals],
                "recoveries": [r.to_dict() for r in recoveries],
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    pd.write_manifest()
    return pd.path


def main(argv: "Sequence[str] | None" = None) -> int:
    """Harvest one target's already-executed runs and emit its experiment set."""
    import argparse

    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--target", required=True, help="target the observations are about")
    ap.add_argument(
        "--root", action="append", default=[], help="a run root to mine (default: this target's roots under out/)"
    )
    ap.add_argument("--suite", default=None, help="a measured cycle suite to derive the axes from")
    ap.add_argument("--version", type=int, default=1)
    args = ap.parse_args(list(argv) if argv is not None else None)
    out = emit_harvest(target=args.target, suite_path=args.suite, roots=args.root or None, version=args.version)
    print(out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
