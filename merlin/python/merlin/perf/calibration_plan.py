"""Pure, fail-closed planning for RTL-derived roofline calibration sweeps.

This module decides *what* must be measured; it never invokes a compiler, runner, or simulator.
Every workload coordinate comes from a provenance-bearing fact supplied by the caller.  In
particular, there are no fallback tile dimensions, transfer sizes, capacities, bandwidths, or ISA
encodings here.  A missing or underived input leaves the affected sweep ``UNKNOWN``.  A present but
invalid input (including too few independent points) is ``REFUSED``.

The accepted mapping is deliberately small and structural.  Facts may be direct dotted keys, nested
mappings, or entries below a target fact bundle's ``fields`` mapping::

    {
      "fields": {
        "dma.read.sizes_bytes": {
          "value": [64, 256, 1024, 4096],
          "derived": True,
          "source": "facts.memory_port.read_legal_sizes",
        },
        ...
      }
    }

Callers with already-normalised resource traits may instead supply :class:`DerivedFact` instances.
The semantic keys are exported as :data:`REQUIRED_FACT_KEYS`; adapters can therefore bind whatever
paths their extractor emits without this planner guessing a target's structure.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

__all__ = [
    "CalibrationPlan",
    "CalibrationPoint",
    "DerivedFact",
    "Disposition",
    "FactEvidence",
    "FitSpecification",
    "MIN_POINTS_PER_PARAMETER",
    "REQUIRED_CAPABILITY_KEYS",
    "REQUIRED_FACT_KEYS",
    "SweepIssue",
    "SweepSpec",
    "build_calibration_plan",
    "build_calibration_plan_from_rtl",
    "derive_resource_traits",
    "plan_calibration",
]


MIN_POINTS_PER_PARAMETER = 2

_DMA_DIRECTIONS = ("read", "write", "copy")
_DMA_PROTOCOL_KEY = "dma.measurement_protocols"
_COMPUTE_SHAPE_KEY = "compute.tile_shape"
_COMPUTE_MULTIPLES_KEY = "compute.tile_multiples"
_COMPUTE_EMITTER_KEY = "compute.workload_emitter"

REQUIRED_FACT_KEYS = (
    *(f"dma.{direction}.sizes_bytes" for direction in _DMA_DIRECTIONS),
    _DMA_PROTOCOL_KEY,
    _COMPUTE_SHAPE_KEY,
    _COMPUTE_MULTIPLES_KEY,
)

REQUIRED_CAPABILITY_KEYS = (
    "dma.directions",
    *(f"dma.{direction}.sizes_bytes" for direction in _DMA_DIRECTIONS),
    _DMA_PROTOCOL_KEY,
    _COMPUTE_EMITTER_KEY,
    _COMPUTE_MULTIPLES_KEY,
)


class Disposition(str, Enum):
    """Whether a requested calibration sweep may be scheduled."""

    READY = "READY"
    UNKNOWN = "UNKNOWN"
    REFUSED = "REFUSED"


@dataclass(frozen=True)
class DerivedFact:
    """One caller-supplied fact and the exact evidence that grounds it.

    ``derived_from_rtl`` has no default.  Hardware coordinates require that standing.  A narrow
    ``derived_from_tool`` standing is also represented for execution conditions that cannot be RTL
    facts (for example, whether a harness can establish a fresh-process or predecessor-run protocol);
    only consumers that explicitly allow tool evidence admit it.  An unstated declaration is never
    promoted to either.  A protocol label does not claim that cache contents were observed.
    """

    value: Any
    path: str
    provenance: tuple[str, ...]
    derived_from_rtl: bool
    derived_from_tool: bool = False


@dataclass(frozen=True)
class FactEvidence:
    """The fact references attached to an admitted or refused sweep."""

    semantic: str
    path: str
    provenance: tuple[str, ...]
    standing: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "semantic": self.semantic,
            "path": self.path,
            "provenance": list(self.provenance),
            "standing": self.standing,
        }


@dataclass(frozen=True)
class SweepIssue:
    """A machine-readable reason that a sweep is not READY."""

    code: str
    reason: str
    fact_paths: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "reason": self.reason, "fact_paths": list(self.fact_paths)}


@dataclass(frozen=True)
class FitSpecification:
    """The parameters a sweep is intended to identify.

    ``required_points`` enforces the project invariant literally: every fitted parameter receives at
    least two observations.  The planner currently fits a rate and a fixed term, so each ready sweep
    necessarily contains at least four distinct independent-variable values.
    """

    response: str
    independent_variable: str
    parameters: tuple[str, ...]
    points_per_parameter: int

    @property
    def required_points(self) -> int:
        return len(self.parameters) * self.points_per_parameter

    def to_dict(self) -> dict[str, Any]:
        return {
            "response": self.response,
            "independent_variable": self.independent_variable,
            "parameters": list(self.parameters),
            "points_per_parameter": self.points_per_parameter,
            "required_points": self.required_points,
        }


@dataclass(frozen=True)
class CalibrationPoint:
    """One abstract workload point.  Execution details belong to a later emitter."""

    coordinates: tuple[tuple[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {name: list(value) if isinstance(value, tuple) else value
                for name, value in self.coordinates}


@dataclass(frozen=True)
class SweepSpec:
    """A complete sweep request, including holes rather than omitting them."""

    sweep_id: str
    mechanism: str
    condition: str
    objective: str
    fit: FitSpecification
    points: tuple[CalibrationPoint, ...]
    facts: tuple[FactEvidence, ...]
    disposition: Disposition
    issues: tuple[SweepIssue, ...] = ()

    @property
    def ready(self) -> bool:
        return self.disposition is Disposition.READY

    def to_dict(self) -> dict[str, Any]:
        return {
            "sweep_id": self.sweep_id,
            "mechanism": self.mechanism,
            "condition": self.condition,
            "objective": self.objective,
            "fit": self.fit.to_dict(),
            "points": [point.to_dict() for point in self.points],
            "facts": [fact.to_dict() for fact in self.facts],
            "disposition": self.disposition.value,
            "issues": [issue.to_dict() for issue in self.issues],
        }


@dataclass(frozen=True)
class CalibrationPlan:
    """All required roofline sweeps and their fail-closed admission decisions."""

    sweeps: tuple[SweepSpec, ...]

    @property
    def ready(self) -> bool:
        return bool(self.sweeps) and all(sweep.ready for sweep in self.sweeps)

    @property
    def status(self) -> str:
        return "READY" if self.ready else "INCOMPLETE"

    @property
    def unknown(self) -> tuple[SweepSpec, ...]:
        return tuple(sweep for sweep in self.sweeps
                     if sweep.disposition is Disposition.UNKNOWN)

    @property
    def refusals(self) -> tuple[SweepSpec, ...]:
        return tuple(sweep for sweep in self.sweeps
                     if sweep.disposition is Disposition.REFUSED)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "ready_sweeps": sum(sweep.ready for sweep in self.sweeps),
            "required_sweeps": len(self.sweeps),
            "sweeps": [sweep.to_dict() for sweep in self.sweeps],
        }


@dataclass(frozen=True)
class _FactResult:
    semantic: str
    fact: DerivedFact | None
    issue: SweepIssue | None

    @property
    def evidence(self) -> FactEvidence | None:
        if self.fact is None:
            return None
        standing = ("rtl_derived" if self.fact.derived_from_rtl else
                    "tool_derived" if self.fact.derived_from_tool else "unestablished")
        return FactEvidence(self.semantic, self.fact.path, self.fact.provenance, standing)


def _nested_get(root: Mapping[str, Any], semantic: str) -> tuple[Any, str] | None:
    """Resolve an exact semantic path without fuzzy/name-based inference."""
    if semantic in root:
        return root[semantic], semantic
    current: Any = root
    walked: list[str] = []
    for component in semantic.split("."):
        if not isinstance(current, Mapping) or component not in current:
            return None
        current = current[component]
        walked.append(component)
    return current, ".".join(walked)


def _provenance(node: Mapping[str, Any]) -> tuple[str, ...]:
    out: list[str] = []
    for key in ("provenance", "source", "evidence"):
        value = node.get(key)
        if isinstance(value, str) and value.strip():
            out.append(value.strip())
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            out.extend(str(item).strip() for item in value if str(item).strip())
        elif isinstance(value, Mapping):
            out.extend(f"{name}={value[name]}" for name in sorted(value))
    return tuple(dict.fromkeys(out))


def _normalise_fact(semantic: str, node: Any, actual_path: str, *, allow_tool: bool) -> _FactResult:
    if isinstance(node, DerivedFact):
        fact = node
    elif isinstance(node, Mapping) and "value" in node:
        marker = node.get("derived_from_rtl", node.get("derived"))
        fact = DerivedFact(
            value=node.get("value"),
            path=str(node.get("path") or actual_path),
            provenance=_provenance(node),
            derived_from_rtl=marker is True,
            derived_from_tool=node.get("derived_from_tool") is True,
        )
    else:
        return _FactResult(
            semantic, None,
            SweepIssue(
                "UNPROVEN_FACT",
                f"{semantic} is present without a value/derived/provenance fact record",
                (actual_path,),
            ),
        )

    missing: list[str] = []
    if not fact.path.strip():
        missing.append("fact path")
    if not fact.provenance:
        missing.append("provenance")
    if fact.derived_from_rtl is not True and not (allow_tool and fact.derived_from_tool is True):
        allowed = "RTL- or tool-derived" if allow_tool else "RTL-derived"
        missing.append(f"positive {allowed} standing")
    if fact.value is None:
        missing.append("value")
    if missing:
        return _FactResult(
            semantic, fact,
            SweepIssue(
                "UNKNOWN_FACT",
                f"{semantic} lacks {', '.join(missing)}; no calibration coordinate is guessed",
                (fact.path or actual_path,),
            ),
        )
    return _FactResult(semantic, fact, None)


def _fact(source: Mapping[str, Any], semantic: str, *, allow_tool: bool = False) -> _FactResult:
    found = _nested_get(source, semantic)
    prefix = ""
    if found is None and isinstance(source.get("fields"), Mapping):
        found = _nested_get(source["fields"], semantic)
        prefix = "fields."
    if found is None:
        return _FactResult(
            semantic, None,
            SweepIssue(
                "MISSING_FACT",
                f"{semantic} is absent; the affected sweep remains UNKNOWN",
                (semantic,),
            ),
        )
    node, actual_path = found
    return _normalise_fact(semantic, node, prefix + actual_path, allow_tool=allow_tool)


def _positive_distinct_ints(result: _FactResult) -> tuple[tuple[int, ...] | None, SweepIssue | None]:
    if result.issue is not None or result.fact is None:
        return None, result.issue
    value = result.fact.value
    path = result.fact.path
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None, SweepIssue(
            "INVALID_FACT", f"{result.semantic} must be a sequence of positive integers", (path,))
    values = tuple(value)
    if (not values or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0
                          for item in values)):
        return None, SweepIssue(
            "INVALID_FACT", f"{result.semantic} must contain only positive integers", (path,))
    if len(set(values)) != len(values):
        return None, SweepIssue(
            "INVALID_FACT", f"{result.semantic} repeats an independent-variable value", (path,))
    return tuple(sorted(values)), None


def _protocols(result: _FactResult) -> tuple[tuple[str, ...] | None, SweepIssue | None]:
    if result.issue is not None or result.fact is None:
        return None, result.issue
    value = result.fact.value
    path = result.fact.path
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None, SweepIssue(
            "INVALID_FACT", f"{result.semantic} must be a sequence of measured protocol names", (path,))
    protocols = tuple(value)
    if (not protocols
            or any(not isinstance(protocol, str) or not protocol.strip()
                   for protocol in protocols)):
        return None, SweepIssue(
            "INVALID_FACT", f"{result.semantic} contains an invalid protocol name", (path,))
    if len(set(protocols)) != len(protocols):
        return None, SweepIssue(
            "INVALID_FACT", f"{result.semantic} repeats a protocol", (path,))
    return protocols, None


def _shape(result: _FactResult) -> tuple[tuple[int, ...] | None, SweepIssue | None]:
    if result.issue is not None or result.fact is None:
        return None, result.issue
    assert result.fact is not None
    value = result.fact.value
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None, SweepIssue(
            "INVALID_FACT", f"{result.semantic} must be a sequence of positive dimensions",
            (result.fact.path,),
        )
    raw = tuple(value)
    if (not raw or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0
                       for item in raw)):
        return None, SweepIssue(
            "INVALID_FACT", f"{result.semantic} must contain only positive dimensions",
            (result.fact.path,),
        )
    return raw, None


def _disposition(issues: Sequence[SweepIssue]) -> Disposition:
    if not issues:
        return Disposition.READY
    unknown_codes = {"MISSING_FACT", "UNKNOWN_FACT", "UNPROVEN_FACT"}
    return (Disposition.UNKNOWN if any(issue.code in unknown_codes for issue in issues)
            else Disposition.REFUSED)


def _dma_sweeps(source: Mapping[str, Any]) -> tuple[SweepSpec, ...]:
    protocol_result = _fact(source, _DMA_PROTOCOL_KEY, allow_tool=True)
    protocols, protocol_issue = _protocols(protocol_result)
    fit = FitSpecification(
        response="cycles",
        independent_variable="transfer_bytes",
        parameters=("fixed_cycles", "cycles_per_byte"),
        points_per_parameter=MIN_POINTS_PER_PARAMETER,
    )
    sweeps: list[SweepSpec] = []
    for direction in _DMA_DIRECTIONS:
        # Legal/executable descriptor sizes are a compiler/harness capability, not a property that
        # can be reconstructed from a memory's capacity or row width.  Tool evidence is admitted
        # here only when it is explicit and provenance-bearing; bare lists remain UNKNOWN.
        size_result = _fact(source, f"dma.{direction}.sizes_bytes", allow_tool=True)
        sizes, size_issue = _positive_distinct_ints(size_result)
        count_issue = None
        if sizes is not None and len(sizes) < fit.required_points:
            assert size_result.fact is not None
            count_issue = SweepIssue(
                "INSUFFICIENT_FIT_POINTS",
                f"{len(sizes)} distinct sizes cannot fit {len(fit.parameters)} parameters at "
                f"{fit.points_per_parameter} points per parameter; {fit.required_points} required",
                (size_result.fact.path,),
            )
        for protocol in protocols or (None,):
            issues = tuple(issue for issue in (protocol_issue, size_issue, count_issue)
                           if issue is not None)
            facts = tuple(evidence for evidence in
                          (size_result.evidence, protocol_result.evidence) if evidence is not None)
            points = () if issues else tuple(
                CalibrationPoint((("transfer_bytes", size),
                                  ("measurement_protocol", protocol)))
                for size in sizes or ())
            sweeps.append(SweepSpec(
                sweep_id=f"dma.{direction}.{protocol or 'UNKNOWN'}",
                mechanism=f"dma_{direction}",
                condition=protocol or "UNKNOWN",
                objective="roofline_rate_and_fixed_term_separation",
                fit=fit,
                points=points,
                facts=facts,
                disposition=_disposition(issues),
                issues=issues,
            ))
    return tuple(sweeps)


def _compute_sweep(source: Mapping[str, Any]) -> SweepSpec:
    shape_result = _fact(source, _COMPUTE_SHAPE_KEY)
    # The array shape is hardware geometry; which multiples the emitter can legally materialise is
    # an execution-tool fact.  Keeping those standings separate prevents an unbounded synthetic
    # ladder from masquerading as an RTL capability.
    multiples_result = _fact(source, _COMPUTE_MULTIPLES_KEY, allow_tool=True)
    shape, shape_issue = _shape(shape_result)
    multiples, multiples_issue = _positive_distinct_ints(multiples_result)
    fit = FitSpecification(
        response="cycles",
        independent_variable="tile_multiple",
        parameters=("fixed_cycles", "cycles_per_tile_multiple"),
        points_per_parameter=MIN_POINTS_PER_PARAMETER,
    )
    count_issue = None
    if multiples is not None and len(multiples) < fit.required_points:
        assert multiples_result.fact is not None
        count_issue = SweepIssue(
            "INSUFFICIENT_FIT_POINTS",
            f"{len(multiples)} distinct tile multiples cannot fit {len(fit.parameters)} parameters "
            f"at {fit.points_per_parameter} points per parameter; {fit.required_points} required",
            (multiples_result.fact.path,),
        )
    issues = tuple(issue for issue in (shape_issue, multiples_issue, count_issue)
                   if issue is not None)
    facts = tuple(evidence for evidence in
                  (shape_result.evidence, multiples_result.evidence) if evidence is not None)
    points = () if issues else tuple(
        CalibrationPoint((("tile_multiple", multiple), ("tile_shape", shape)))
        for multiple in multiples or ())
    return SweepSpec(
        sweep_id="compute.saturation",
        mechanism="compute",
        condition="derived_tile_multiples",
        objective="roofline_saturation_and_fixed_term_separation",
        fit=fit,
        points=points,
        facts=facts,
        disposition=_disposition(issues),
        issues=issues,
    )


def _unknown_derived_fact(path: str, reason: str) -> DerivedFact:
    """An adapter hole that the core planner will preserve as UNKNOWN."""
    return DerivedFact(
        value=None,
        path=path,
        provenance=(reason,),
        derived_from_rtl=False,
        derived_from_tool=False,
    )


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _raw_provenance(root: Mapping[str, Any], row: Mapping[str, Any]) -> tuple[str, ...]:
    provenance = list(_provenance(row))
    generator = root.get("generator")
    if isinstance(generator, str) and generator.strip():
        provenance.append(generator.strip())
    elif isinstance(generator, Mapping):
        provenance.extend(f"generator.{key}={generator[key]}" for key in sorted(generator))
    source = (root.get("facts") or {}).get("source") if isinstance(root.get("facts"), Mapping) else None
    if isinstance(source, str) and source.strip():
        provenance.append(source.strip())
    return tuple(dict.fromkeys(provenance))


def _compute_array(rtl_facts: Mapping[str, Any]) -> tuple[tuple[int, int], str,
                                                            tuple[str, ...]] | None:
    """Select one structurally unambiguous compute array and return its derived dimensions."""
    body = rtl_facts.get("facts")
    arrays = body.get("arrays") if isinstance(body, Mapping) else None
    if (not isinstance(arrays, Sequence) or isinstance(arrays, (str, bytes)) or not arrays
            or any(not isinstance(array, Mapping) for array in arrays)):
        return None
    selected = [index for index, array in enumerate(arrays) if array.get("primary") is True]
    if not selected and len(arrays) == 1:
        selected = [0]
    if len(selected) != 1:
        return None
    index = selected[0]
    array = arrays[index]
    rows, cols = array.get("rows"), array.get("cols")
    provenance = _raw_provenance(rtl_facts, array)
    if not _positive_int(rows) or not _positive_int(cols) or not provenance:
        return None
    return (rows, cols), f"facts.arrays[{index}]", provenance


def _capability(capabilities: Mapping[str, Any], semantic: str) -> _FactResult:
    """Read an explicit harness/tool fact; declarations without tool provenance remain UNKNOWN."""
    return _fact(capabilities, semantic, allow_tool=True)


def derive_resource_traits(
    rtl_facts: Mapping[str, Any],
    harness_capabilities: Mapping[str, Any],
) -> dict[str, DerivedFact]:
    """Adapt raw RTL facts plus explicit harness capabilities to the planner's semantic facts.

    Compute-array shape is derived from raw RTL records.  Legal DMA descriptor sizes, executable
    measurement protocols, and compute multiples are tool capabilities: storage capacity/row width
    cannot prove them.  Those facts must carry provenance and ``derived_from_tool=True``.  Missing
    capability records create UNKNOWN facts rather than optimistic defaults.
    """
    if not isinstance(rtl_facts, Mapping):
        raise TypeError("raw RTL facts must be a mapping")
    if not isinstance(harness_capabilities, Mapping):
        raise TypeError("harness/tool capabilities must be a mapping")

    traits: dict[str, DerivedFact] = {}
    directions_result = _capability(harness_capabilities, "dma.directions")
    directions: tuple[str, ...] | None = None
    if directions_result.issue is None and directions_result.fact is not None:
        value = directions_result.fact.value
        if (isinstance(value, Sequence) and not isinstance(value, (str, bytes))
                and all(isinstance(item, str) for item in value)):
            directions = tuple(value)

    for direction in _DMA_DIRECTIONS:
        semantic = f"dma.{direction}.sizes_bytes"
        if directions is None:
            traits[semantic] = _unknown_derived_fact(
                "capabilities.dma.directions",
                "no provenance-bearing tool capability establishes executable DMA directions",
            )
        elif direction not in directions:
            traits[semantic] = _unknown_derived_fact(
                "capabilities.dma.directions",
                f"the tool-derived capability does not establish the {direction} direction",
            )
        else:
            sizes_result = _capability(harness_capabilities, semantic)
            if sizes_result.issue is not None or sizes_result.fact is None:
                traits[semantic] = _unknown_derived_fact(
                    f"capabilities.{semantic}",
                    f"no provenance-bearing tool capability establishes legal {direction} sizes",
                )
                continue
            sizes = sizes_result.fact
            traits[semantic] = DerivedFact(
                value=sizes.value,
                path=sizes.path,
                provenance=sizes.provenance,
                derived_from_rtl=False,
                derived_from_tool=True,
            )

    protocols_result = _capability(harness_capabilities, _DMA_PROTOCOL_KEY)
    if protocols_result.issue is not None or protocols_result.fact is None:
        traits[_DMA_PROTOCOL_KEY] = _unknown_derived_fact(
            "capabilities.dma.measurement_protocols",
            "no provenance-bearing tool capability establishes a measurement protocol",
        )
    else:
        protocols = protocols_result.fact
        traits[_DMA_PROTOCOL_KEY] = DerivedFact(
            value=protocols.value,
            path=protocols.path,
            provenance=protocols.provenance,
            derived_from_rtl=False,
            derived_from_tool=True,
        )

    compute_result = _capability(harness_capabilities, _COMPUTE_EMITTER_KEY)
    multiples_result = _capability(harness_capabilities, _COMPUTE_MULTIPLES_KEY)
    array = _compute_array(rtl_facts)
    compute_available = (compute_result.issue is None and compute_result.fact is not None
                         and compute_result.fact.value is True)
    if not compute_available:
        reason = "no provenance-bearing tool capability establishes a compute calibration workload"
        traits[_COMPUTE_SHAPE_KEY] = _unknown_derived_fact(
            "capabilities.compute.workload_emitter", reason)
        traits[_COMPUTE_MULTIPLES_KEY] = _unknown_derived_fact(
            "capabilities.compute.workload_emitter", reason)
    elif array is None:
        reason = "raw RTL facts do not identify exactly one provenance-bearing compute array"
        traits[_COMPUTE_SHAPE_KEY] = _unknown_derived_fact("facts.arrays", reason)
        traits[_COMPUTE_MULTIPLES_KEY] = _unknown_derived_fact("facts.arrays", reason)
    elif multiples_result.issue is not None or multiples_result.fact is None:
        shape, path, array_provenance = array
        assert compute_result.fact is not None
        traits[_COMPUTE_SHAPE_KEY] = DerivedFact(
            value=shape,
            path=f"{path}.{{rows,cols}}",
            provenance=array_provenance + compute_result.fact.provenance,
            derived_from_rtl=True,
        )
        traits[_COMPUTE_MULTIPLES_KEY] = _unknown_derived_fact(
            "capabilities.compute.tile_multiples",
            "no provenance-bearing tool capability establishes executable compute multiples",
        )
    else:
        shape, path, array_provenance = array
        assert compute_result.fact is not None
        provenance = array_provenance + compute_result.fact.provenance
        traits[_COMPUTE_SHAPE_KEY] = DerivedFact(
            value=shape,
            path=f"{path}.{{rows,cols}}",
            provenance=provenance,
            derived_from_rtl=True,
        )
        traits[_COMPUTE_MULTIPLES_KEY] = DerivedFact(
            value=multiples_result.fact.value,
            path=multiples_result.fact.path,
            provenance=multiples_result.fact.provenance,
            derived_from_rtl=False,
            derived_from_tool=True,
        )
    return traits


def build_calibration_plan_from_rtl(
    rtl_facts: Mapping[str, Any],
    harness_capabilities: Mapping[str, Any],
) -> CalibrationPlan:
    """Adapt raw evidence and build a calibration plan without performing any I/O."""
    return build_calibration_plan(derive_resource_traits(rtl_facts, harness_capabilities))


def build_calibration_plan(
    target_facts_or_resource_traits: Mapping[str, Any],
) -> CalibrationPlan:
    """Build the complete calibration plan from explicit RTL facts or derived resource traits.

    No I/O occurs.  One read/write/copy DMA sweep is returned for each tool-derived measurement
    protocol, plus compute saturation over derived tile multiples.  When protocols are unavailable,
    one explicit UNKNOWN sweep per direction preserves the hole instead of letting omission look like
    non-applicability.
    """
    if not isinstance(target_facts_or_resource_traits, Mapping):
        raise TypeError("calibration facts/resource traits must be a mapping")
    return CalibrationPlan(
        sweeps=_dma_sweeps(target_facts_or_resource_traits)
        + (_compute_sweep(target_facts_or_resource_traits),)
    )


def plan_calibration(target_facts_or_resource_traits: Mapping[str, Any]) -> CalibrationPlan:
    """Alias with verb-first naming for callers that treat planning as the primary operation."""
    return build_calibration_plan(target_facts_or_resource_traits)
