"""Select declared compiler settings from exact completed cycle observations.

Existing compilation and evaluation owners keep their resource, source and
numerical checks. This module copies no compiler tree, starts no evaluator and
learns no shape cutoff. Use shape_policy for separately validated range rules.
"""

from dataclasses import asdict, dataclass
from hashlib import sha256
from math import isfinite

from merlin.compile.tuning_knobs import KnobSettings

from .evaluated_exploration import CompletedObservation, ExplorationIdentity
from .pass_exploration import EmittedProgram, MappingProposalKey, ObservedMetrics, rank_performance_variants


@dataclass(frozen=True)
class KnobTrial:
    settings: KnobSettings
    emission: EmittedProgram | None = None
    observation: CompletedObservation | None = None
    error: str | None = None

    def __post_init__(self):
        if (
            type(self.settings) is not KnobSettings
            or self.emission is not None
            and type(self.emission) is not EmittedProgram
            or self.observation is not None
            and type(self.observation) is not CompletedObservation
            or self.observation is not None
            and self.emission is None
            or self.error is not None
            and (type(self.error) is not str or not self.error)
        ):
            raise ValueError("knob trial needs immutable settings and explicit emitted/observed results")
        self.settings.__post_init__()
        if self.emission is not None:
            self.emission.__post_init__()
        if self.observation is not None:
            self.observation.__post_init__()
            if self.observation.program_sha256 != self.program_sha256:
                raise ValueError("knob observation differs from its actual emitted bytes")
        if self.error is not None:
            object.__setattr__(self, "error", self.error[:512])

    @property
    def program_sha256(self):
        return None if self.emission is None else sha256(self.emission.data).hexdigest()


def _checked(trials, identity):
    if (
        type(identity) is not ExplorationIdentity
        or type(trials) is not tuple
        or not 1 <= len(trials) <= 256
        or any(type(trial) is not KnobTrial for trial in trials)
        or len({trial.settings.identity for trial in trials}) != len(trials)
    ):
        raise ValueError("selection requires one exact domain and bounded distinct settings")
    identity.__post_init__()
    for trial in trials:
        trial.__post_init__()
        if trial.emission is not None and trial.emission.format != identity.artifact_format:
            raise ValueError("knob emission format differs from the selected observation domain")
        if trial.observation is not None and trial.observation.identity != identity:
            raise ValueError("knob observation source/provider/ROI differs from the selected exact domain")


def unique_emissions(trials, *, identity):
    """Return first actual unique emissions for the ordinary evaluator owner.

    Baseline-first input keeps the incumbent as a duplicate's representative.
    A caller can feed these into existing evaluate_emitted using its own actual
    receipts. Failure/missing emissions are retained in trials, but not submitted.
    Dedup never carries evidence between different source/provider/ROI domains.
    """
    _checked(trials, identity)
    seen, unique = set(), []
    for trial in trials:
        digest = trial.program_sha256
        if digest is not None and trial.error is None and digest not in seen:
            seen.add(digest)
            unique.append(trial)
    return tuple(unique)


@dataclass(frozen=True)
class KnobSelection:
    domain: MappingProposalKey
    identity: ExplorationIdentity
    selected: KnobSettings
    baseline: KnobSettings
    status: str
    ranking: dict
    outcomes: tuple[dict, ...]

    def record(self):
        return dict(
            schema="merlin.knob_selection.v1",
            mapping_key=self.domain.identity,
            mapping_domain=asdict(self.domain),
            evaluation_identity=self.identity.digest(),
            evaluation_domain=asdict(self.identity),
            selected_settings_sha256=self.selected.identity,
            selected_settings=self.selected.values,
            baseline_settings_sha256=self.baseline.identity,
            status=self.status,
            ranking=self.ranking,
            outcomes=self.outcomes,
            reuse="settings proposal only; current source requires ordinary revalidation",
            performance_qualification=False,
        )


def select_settings(domain, *, identity, trials, minimum_relative_improvement=0.0):
    """Retain baseline unless a passing complete same-domain program is faster.

    Exact-byte duplicates share their one completed observation. Contradictory
    eligible cycles or numerical outcomes remain unranked. Errors, UNKNOWN,
    partial regions and static metrics never outrank a supported baseline.
    An unranked/failed baseline is retained without a performance claim.
    The returned geometry key is not a policy for neighboring/unseen shapes.
    A caller may require a minimum relative improvement to avoid changing
    settings for differences below its selected decision tolerance. This is
    a promotion policy, not a claim about the timing provider's accuracy.
    """
    if (type(minimum_relative_improvement) not in (int, float)
            or not isfinite(minimum_relative_improvement)
            or not 0 <= minimum_relative_improvement < 1):
        raise ValueError("tuning improvement tolerance must be finite and in [0, 1)")
    if type(domain) is not MappingProposalKey:
        raise ValueError("settings selection needs an explicit semantic/shape/hardware/numerical domain")
    domain.__post_init__()
    _checked(trials, identity)
    if not trials[0].settings.baseline or any(trial.settings.baseline for trial in trials[1:]):
        raise ValueError("settings selection needs exactly one baseline, first")
    if domain.library_sha256 != identity.library_sha256:
        raise ValueError("mapping settings and observations select different compiler libraries")
    programs = {}
    for trial in trials:
        if trial.program_sha256 is not None:
            programs.setdefault(trial.program_sha256, []).append(trial)
    observations, conflicts = {}, set()
    for digest, members in programs.items():
        observed = [member.observation for member in members if member.error is None and member.observation is not None]
        eligible = [observation for observation in observed if observation.eligible]
        if (
            len({observation.cycles for observation in eligible}) > 1
            or eligible
            and any(observation.numerical_status == "fail" for observation in observed)
        ):
            conflicts.add(digest)
        elif eligible:
            observations[digest] = eligible[0]
    metrics = tuple(
        (digest, ObservedMetrics(identity.basis, identity.digest(), (("cycles", observed.cycles),)))
        for digest, observed in observations.items()
    )
    ranking = (
        rank_performance_variants(metrics)
        if metrics
        else dict(ranked=(), decision_metric="complete_same_roi_cycles", performance_qualification=False)
    )
    baseline = trials[0]
    baseline_observed = observations.get(baseline.program_sha256) if baseline.error is None else None
    selected, status = baseline.settings, "baseline_unranked"
    ranking["minimum_relative_improvement"] = minimum_relative_improvement
    if baseline_observed is not None:
        status = "baseline_retained"
        best_cycles = ranking["ranked"][0][1][0] if ranking["ranked"] else baseline_observed.cycles
        if (best_cycles < baseline_observed.cycles and
                (baseline_observed.cycles-best_cycles)/baseline_observed.cycles >= minimum_relative_improvement):
            digest = ranking["ranked"][0][0]
            winner = next(member for member in programs[digest] if member.error is None)
            selected, status = winner.settings, "cycle_improvement"
    outcomes = []
    for trial in trials:
        digest, observed = trial.program_sha256, trial.observation
        effective = observations.get(digest)
        representative = programs[digest][0].settings.identity if digest is not None else None
        outcome = (
            "failed"
            if trial.error
            else "emission_missing"
            if digest is None
            else "conflicting_feedback"
            if digest in conflicts
            else "cycle_ranked"
            if effective is not None
            else "pending"
            if observed is None
            else "unranked"
        )
        outcomes.append(
            dict(
                settings_sha256=trial.settings.identity,
                changes=trial.settings.changes,
                baseline=trial.settings.baseline,
                program_sha256=digest,
                duplicate_of=None if representative == trial.settings.identity else representative,
                status=outcome,
                error=trial.error,
                numerical_status=None if observed is None else observed.numerical_status,
                timing_status=None if observed is None else observed.timing_status,
                complete_roi=None if observed is None else observed.complete_roi,
                cycles=None if observed is None else observed.cycles,
                feedback_sha256=None if observed is None else observed.feedback_sha256,
                ranked_cycles=None if effective is None or trial.error else effective.cycles,
            )
        )
    return KnobSelection(domain, identity, selected, baseline.settings, status, ranking, tuple(outcomes))
