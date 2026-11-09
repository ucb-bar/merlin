"""Trusted component CCA selection and complete, answer-free structural reports.

The provider owns lifting the actual emitted baseline/candidate artifacts. This
module owns their exact input bindings and delegates every populated gap to the
core comparator. It never executes a compiler, calibrates cost or admits launch.
"""

from __future__ import annotations

import math
import types
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Protocol, Union, get_args, get_origin, get_type_hints

from merlin.benchharness import hash_tree
from merlin.kernels.cca import CCA
from merlin.kernels.cca_compare import compare, uncomparable_axes
from merlin.kernels.cca_contract import FACET_CLASSES

from . import corpus as C
from .broker_evidence import _is_sha256
from .component_baseline import verify_baseline_admission
from .component_runtime import require_independent_runtime
from .contracts import StageGateError, document_sha256, sha256_file

ACTION = "component-cca-feedback"
TIER = "structural_component_cca"
LEGACY_REPORT_SCHEMA = "merlin.component_cca_report.v1"
REPORT_SCHEMA = "merlin.component_cca_report.v2"
_PROVENANCE = {"level", "source_sha256", "producer_sha256", "private_metadata_sha256"}


def _artifact(path, digest):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or not _is_sha256(digest) or sha256_file(path) != digest:
        raise StageGateError("component CCA artifact/source identity changed")
    return path.resolve()


def _atom(value):
    # Facet atoms describe compiler choices, not source text or private paths.
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 256
        or not all(character.isalnum() or character in "._-+=,:<>[]()|" for character in value)
    ):
        raise StageGateError("component CCA has a non-structural string")
    return value


def _value(value):
    if value is None or type(value) in (bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if isinstance(value, str):
        return _atom(value)
    if type(value) in (tuple, list) and len(value) <= 128:
        return [_value(item) for item in value]
    raise StageGateError("component CCA facet value is unsupported")


def _typed(value, annotation):
    origin, args = get_origin(annotation), get_args(annotation)
    if origin in (Union, types.UnionType):
        for option in args:
            try:
                return _typed(value, option)
            except StageGateError:
                pass
        raise StageGateError("component CCA value contradicts its field type")
    if annotation is type(None) and value is None:
        return None
    if annotation is str and isinstance(value, str):
        return _atom(value)
    if annotation in (int, bool) and type(value) is annotation:
        return value
    if annotation is float and type(value) in (int, float):
        try:
            if math.isfinite(value):
                return float(value)
        except OverflowError:
            pass
    if annotation is tuple or origin is tuple:
        if type(value) not in (tuple, list) or len(value) > 128:
            raise StageGateError("component CCA tuple is malformed")
        if not args:
            return tuple(_value(item) for item in value)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_typed(item, args[0]) for item in value)
        if len(value) == len(args):
            return tuple(_typed(item, kind) for item, kind in zip(value, args, strict=True))
    raise StageGateError("component CCA value contradicts its field type")


def _decode(document):
    if not isinstance(document, Mapping) or set(document) != {field.name for field in fields(CCA)}:
        raise StageGateError("component CCA does not preserve the complete schema")
    provenance = document["provenance"]
    if not isinstance(provenance, Mapping) or set(provenance) != _PROVENANCE:
        raise StageGateError("component CCA provenance must be a digest projection")
    _atom(provenance["level"])
    if not all(_is_sha256(provenance[key]) for key in _PROVENANCE - {"level"}):
        raise StageGateError("component CCA provenance digest is invalid")
    _atom(document["op"])
    backend = document["backend"]
    if type(backend) is not list or not backend:
        raise StageGateError("component CCA backend identity is invalid")
    for name in backend:
        _atom(name)
    if len(set(backend)) != len(backend):
        raise StageGateError("component CCA backend identity is invalid")
    facets = {}
    for name, cls in FACET_CLASSES.items():
        raw = document[name]
        if raw is None:
            if name == "compute":
                raise StageGateError("component CCA has no compute facet")
            facets[name] = None
        else:
            if not isinstance(raw, Mapping) or set(raw) != {field.name for field in fields(cls)}:
                raise StageGateError("component CCA facet schema is incomplete")
            hints = get_type_hints(cls)
            facets[name] = cls(**{key: _typed(value, hints[key]) for key, value in raw.items()})
    result = CCA(document["op"], list(backend), **facets, scope=document["scope"], provenance=dict(provenance))
    if result.scope_problems():
        raise StageGateError("component CCA scope is inconsistent")
    return result


@dataclass(frozen=True)
class ComponentCorrespondence:
    """Provider-bound use/effect/ownership facts for the exact emitted program.

    Atoms are reviewed structural vocabulary or opaque identities; source text,
    raw baseline programs and private absolute paths never enter the projection.
    """

    users: tuple[str, ...] | None = None
    effects: tuple[str, ...] | None = None
    representations: tuple[str, ...] | None = None
    lifetimes: tuple[tuple[str, int, int], ...] | None = None
    resource_pressure: tuple[tuple[str, float], ...] | None = None
    emitted_choices: tuple[str, ...] | None = None
    refusal_reasons: tuple[str, ...] | None = None
    edit_owners: tuple[tuple[str, str], ...] | None = None


def _correspondence(value):
    if value is None:
        value = ComponentCorrespondence()
    if type(value) is ComponentCorrespondence:
        raw = {field.name: getattr(value, field.name) for field in fields(ComponentCorrespondence)}
    elif isinstance(value, Mapping):
        raw = dict(value)
    else:
        raise StageGateError("component correspondence requires typed provider facts")
    hints = get_type_hints(ComponentCorrespondence)
    if set(raw) != set(hints):
        raise StageGateError("component correspondence facet schema is incomplete")
    return {name: _value(_typed(raw[name], annotation)) for name, annotation in hints.items()}


@dataclass(frozen=True)
class ComponentCCAObservation:
    """Host-private typed observation of one exact generated member artifact."""

    cca: CCA
    artifact: Path
    artifact_sha256: str
    compiler_sha256: str
    member_sha256: str
    corpus_sha256: str
    target_sha256: str
    correspondence: ComponentCorrespondence | None = None


class ComponentCCAEvaluation(Protocol):
    def __call__(
        self,
        *,
        baseline: Path,
        candidate: Path,
        corpus: C.FrozenPerformanceCorpus,
        target_descriptor: Path,
        timeout_s: float,
    ) -> Mapping[tuple[str, str], tuple[ComponentCCAObservation, ComponentCCAObservation]]: ...


@dataclass(frozen=True)
class ComponentCCAProvider:
    """Independent observation and the fresh seed, revalidated around each use."""

    evaluate: ComponentCCAEvaluation
    implementation: Path
    implementation_sha256: str
    baseline: Path
    baseline_sha256: str
    baseline_admission: object = None
    independent_runtime: object = None

    def validate(self):
        from .component_workflow import _callable_source

        verify_baseline_admission(self.baseline_admission, baseline=self.baseline,
                                  corpus=None, target_descriptor=None)
        runtime = require_independent_runtime(self.independent_runtime, required_roles=("cca_provider",),
                                              target_descriptor=self.baseline_admission.target_descriptor)
        verify_binding = getattr(runtime.qualification, "verify_component_binding", None)
        if not callable(verify_binding):
            raise StageGateError("component CCA lacks independent exact compiler/cohort observation qualification")
        verify_binding(self.baseline_admission)
        if self.evaluate is not runtime.services.cca_provider:
            raise StageGateError("component CCA differs from independently evaluated observation support")
        owner = _artifact(self.implementation, self.implementation_sha256)
        if _callable_source(self.evaluate)[0] != owner:
            raise StageGateError("component CCA callback differs from its pinned owner")
        baseline = Path(self.baseline)
        if (
            baseline.is_symlink()
            or not baseline.is_dir()
            or not _is_sha256(self.baseline_sha256)
            or hash_tree(baseline)["sha256"] != self.baseline_sha256
        ):
            raise StageGateError("component CCA baseline compiler changed")
        return baseline.resolve()

    def configuration_sha256(self, target_sha256):
        return document_sha256({"baseline_sha256": self.baseline_sha256, "target_sha256": target_sha256,
                                "baseline_admission_sha256": self.baseline_admission.sha256,
                                "independent_runtime_sha256": self.independent_runtime.sha256})


def _projection(observation, *, compiler_sha256, member, corpus, target_sha256, provider_sha256):
    if type(observation) is not ComponentCCAObservation or type(observation.cca) is not CCA:
        raise StageGateError("component CCA provider needs typed observations")
    if (
        observation.compiler_sha256 != compiler_sha256
        or observation.member_sha256 != member.source_sha256
        or observation.corpus_sha256 != corpus.capsules_sha256
        or observation.target_sha256 != target_sha256
    ):
        raise StageGateError("component CCA observation is bound to different inputs")
    _artifact(observation.artifact, observation.artifact_sha256)
    raw = observation.cca.to_dict()
    metadata = raw["provenance"]
    if not isinstance(metadata, Mapping) or "level" not in metadata:
        raise StageGateError("component CCA observation lacks its lifting provenance")
    raw["provenance"] = {
        "level": _atom(metadata["level"]),
        "source_sha256": observation.artifact_sha256,
        "producer_sha256": provider_sha256,
        "private_metadata_sha256": document_sha256(metadata),
    }
    # This canonical projection contains facet values and opaque identities only.
    cca = _decode(raw)
    return cca.to_dict()


def complete_report(baseline, candidate, baseline_correspondence=None, candidate_correspondence=None):
    """All reflected facets, including unknown on both sides, with core gaps."""
    left, right = _decode(baseline), _decode(candidate)
    if left.scope != right.scope or left.op != right.op or left.backend != right.backend:
        raise StageGateError("component CCA scope/operation/backend comparison differs")
    gaps = compare(left, right, evidence=[left.provenance["source_sha256"]])
    divergent = {gap.axis for gap in gaps}
    facets, common_missing, reflected = {}, [], []
    for name, cls in FACET_CLASSES.items():
        a, b = getattr(left, name), getattr(right, name)
        axes = []
        for field in fields(cls):
            axis = f"{name}.{field.name}"
            reflected.append(axis)
            av, bv = getattr(a, field.name, None), getattr(b, field.name, None)
            missing = [arm for arm, value in (("baseline", av), ("candidate", bv)) if value is None]
            if len(missing) == 2:
                common_missing.append(axis)
            if not missing and av != bv and axis not in divergent:
                raise StageGateError("core CCA comparison omitted a populated facet divergence")
            axes.append(
                {
                    "axis": axis,
                    "status": "UNKNOWN" if missing else "DIVERGENCE" if axis in divergent else "SAME",
                    "missing": missing,
                }
            )
        facets[name] = {
            "status": "UNKNOWN"
            if all(row["status"] == "UNKNOWN" for row in axes)
            else "PARTIAL"
            if any(row["status"] == "UNKNOWN" for row in axes)
            else "COMPARABLE",
            "axes": axes,
        }
    if divergent - set(reflected):
        raise StageGateError("core CCA comparison has an unreflected axis")
    correspondence = {}
    ca, cb = _correspondence(baseline_correspondence), _correspondence(candidate_correspondence)
    for axis in ca:
        missing = [arm for arm, value in (("baseline", ca[axis]), ("candidate", cb[axis])) if value is None]
        correspondence[axis] = {"status": "UNKNOWN" if missing else "SAME" if ca[axis] == cb[axis] else "DIVERGENCE",
                                "missing": missing, "baseline": ca[axis], "candidate": cb[axis]}
    return {
        "schema": REPORT_SCHEMA,
        "correspondence": correspondence,
        "scope": left.scope,
        "scope_basis": "exact_generated_component",
        "reflected_axes": reflected,
        "facets": facets,
        "divergences": [
            {
                "axis": gap.axis,
                "baseline": _value(gap.expert),
                "candidate": _value(gap.ours),
                "backend": _atom(gap.backend),
                "evidence": list(gap.evidence),
            }
            for gap in gaps
        ],
        "uncomparable_axes": [
            {"axis": axis, "missing": "baseline" if side == "expert" else "candidate"}
            for axis, side in uncomparable_axes(left, right)
        ],
        "common_missing_axes": common_missing,
        "application_coverage": {
            "status": "UNKNOWN",
            "reason": "generated components do not establish application coverage",
        },
        "cycle_cost": {"status": "UNKNOWN", "reason": "CCA structure is not calibrated or measured cycle evidence"},
    }


def evaluate(provider, *, candidate, candidate_sha256, corpus, target_descriptor, timeout_s):
    baseline = provider.validate()
    candidate = Path(candidate).resolve()
    if baseline.is_relative_to(candidate) or candidate.is_relative_to(baseline):
        raise StageGateError("component CCA compiler arms overlap")
    target_sha = sha256_file(target_descriptor)
    raw = provider.evaluate(
        baseline=baseline, candidate=candidate, corpus=corpus, target_descriptor=target_descriptor, timeout_s=timeout_s
    )
    expected = {(member.family, member.capsule): member for member in corpus.capsules}
    if not isinstance(raw, Mapping) or set(raw) != set(expected):
        raise StageGateError("component CCA does not cover every exact generated member")
    provider.validate()
    evidence = []
    for identity, member in sorted(expected.items()):
        pair = raw[identity]
        if type(pair) is not tuple or len(pair) != 2:
            raise StageGateError("component CCA needs exactly two compiler arms")
        projections = [
            _projection(
                observation,
                compiler_sha256=compiler,
                member=member,
                corpus=corpus,
                target_sha256=target_sha,
                provider_sha256=provider.implementation_sha256,
            )
            for observation, compiler in zip(pair, (provider.baseline_sha256, candidate_sha256), strict=True)
        ]
        evidence.append(
            {
                "family": identity[0],
                "capsule": identity[1],
                "member_sha256": member.source_sha256,
                "corpus_sha256": corpus.capsules_sha256,
                "manifest_sha256": corpus.manifest_sha256,
                "target_sha256": target_sha,
                "baseline_compiler_sha256": provider.baseline_sha256,
                "baseline_admission_sha256": provider.baseline_admission.sha256,
                "independent_runtime_sha256": provider.independent_runtime.sha256,
                "candidate_compiler_sha256": candidate_sha256,
                "baseline_cca": projections[0],
                "candidate_cca": projections[1],
                "baseline_correspondence": _correspondence(pair[0].correspondence),
                "candidate_correspondence": _correspondence(pair[1].correspondence),
                "report": complete_report(*projections, pair[0].correspondence, pair[1].correspondence),
            }
        )
    return evidence


def validate_evidence(document):
    """Replay the complete report, rather than trusting serialized gap claims."""
    evidence = document["evidence"]
    if type(evidence) is not list or not evidence:
        raise StageGateError("component CCA evidence is empty")
    seen, configurations = set(), set()
    for row in evidence:
        expected = {
            "family",
            "capsule",
            "member_sha256",
            "corpus_sha256",
            "manifest_sha256",
            "target_sha256",
            "baseline_compiler_sha256",
            "candidate_compiler_sha256",
            "baseline_cca",
            "candidate_cca",
            "baseline_correspondence",
            "candidate_correspondence",
            "report",
        }
        legacy = (isinstance(row, Mapping) and isinstance(row.get("report"), Mapping)
                  and row["report"].get("schema") == LEGACY_REPORT_SCHEMA)
        if legacy:
            expected -= {"baseline_correspondence", "candidate_correspondence"}
        for key in ("baseline_admission_sha256", "independent_runtime_sha256"):
            if isinstance(row, Mapping) and key in row:
                expected.add(key)
        if not isinstance(row, Mapping) or set(row) != expected:
            raise StageGateError("component CCA member schema is invalid")
        identity = row["family"], row["capsule"]
        if any(not isinstance(value, str) or not value for value in identity) or identity in seen:
            raise StageGateError("component CCA member identity is invalid")
        seen.add(identity)
        if any(not _is_sha256(row[key]) for key in expected if key.endswith("sha256")):
            raise StageGateError("component CCA member digest is invalid")
        if row["candidate_compiler_sha256"] != document["candidate_sha256"]:
            raise StageGateError("component CCA compiler differs from feedback envelope")
        if row["corpus_sha256"] != document["corpus_sha256"] or row["manifest_sha256"] != document["manifest_sha256"]:
            raise StageGateError("component CCA corpus differs from feedback envelope")
        configuration = {"baseline_sha256": row["baseline_compiler_sha256"], "target_sha256": row["target_sha256"]}
        if "baseline_admission_sha256" in row:
            configuration["baseline_admission_sha256"] = row["baseline_admission_sha256"]
        if "independent_runtime_sha256" in row:
            configuration["independent_runtime_sha256"] = row["independent_runtime_sha256"]
        configurations.add(document_sha256(configuration))
        for arm in ("baseline_cca", "candidate_cca"):
            cca = _decode(row[arm])
            if cca.provenance["producer_sha256"] != document["provider_sha256"]:
                raise StageGateError("component CCA producer differs from feedback envelope")
        replay = complete_report(row["baseline_cca"], row["candidate_cca"],
                                 row.get("baseline_correspondence"), row.get("candidate_correspondence"))
        if legacy:
            replay.pop("correspondence")
            replay["schema"] = LEGACY_REPORT_SCHEMA
        if document_sha256(replay) != document_sha256(row["report"]):
            raise StageGateError("component CCA report omits or changes reflected evidence")
    if configurations != {document["configuration_sha256"]}:
        raise StageGateError("component CCA configuration differs from feedback envelope")
