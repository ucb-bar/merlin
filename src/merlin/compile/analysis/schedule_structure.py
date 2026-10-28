"""Structural schedules from explicit effects, counted IR, and decision IR.

This analysis never assigns meaning to an opcode or an address. The selected
adapter supplies normalized accesses, roles and completion events. A repeated
issued subsequence is a finite observation, not a timing or legality proof.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any


def _identity(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _version(record: Mapping) -> None:
    if type(record.get("version", 1)) is not int or record.get("version", 1) != 1:
        raise ValueError("unsupported structural schedule version")


@dataclass(frozen=True)
class ScheduleAccess:
    """A caller-selected canonical backing and coordinate unit; never a bank guess."""

    space: str
    buffer: str
    offset: int | str | None
    extent: int | None
    mode: str
    unit: str
    operand: str | None = None

    def __post_init__(self):
        if (any(type(value) is not str or not value for value in (self.space, self.buffer, self.unit))
                or self.mode not in ("read", "write", "read_write")
                or (self.offset is not None and type(self.offset) not in (int, str))
                or (self.extent is not None and (type(self.extent) is not int or self.extent < 1))):
            raise ValueError("schedule accesses need explicit backing, units and bounded extents")

    @classmethod
    def from_record(cls, record):
        return cls(**dict(record))

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class ScheduleStage:
    key: str
    index: int
    identity: str
    roles: tuple[str, ...] = ()
    accesses: tuple[ScheduleAccess, ...] = ()
    configuration: Mapping[str, Any] = field(default_factory=dict)
    produces: tuple[str, ...] = ()
    waits_for: tuple[str, ...] = ()
    parameters: Mapping[str, Any] = field(default_factory=dict)
    kind: str = "command"

    @classmethod
    def from_record(cls, record):
        row = dict(record)
        row["accesses"] = tuple(ScheduleAccess.from_record(item) for item in row.get("accesses", ()))
        for key in ("roles", "produces", "waits_for"):
            row[key] = tuple(row.get(key, ()))
        return cls(**row)

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class CountedLoop:
    """An observed affine recurrence, or an explicit loop from counted payload IR."""

    key: str
    start: int
    extent: int
    body: tuple[str, ...]
    strides: Mapping[str, int] = field(default_factory=dict)
    status: str = "inferred"
    variable: str | None = None
    iter_args: tuple[tuple[str, str], ...] = ()
    yields: tuple[str, ...] = ()
    results: tuple[str, ...] = ()

    @classmethod
    def from_record(cls, record):
        row = dict(record)
        for key in ("body", "yields", "results"):
            row[key] = tuple(row.get(key, ()))
        row["iter_args"] = tuple(tuple(item) for item in row.get("iter_args", ()))
        return cls(**row)


@dataclass(frozen=True)
class ScheduleDependency:
    source: str
    target: str
    kind: str
    status: str
    resource: str | None = None
    completion: str = "unknown"


@dataclass(frozen=True)
class StorageLifetime:
    space: str
    buffer: str
    unit: str
    first_issue: int
    last_issue: int
    offset_min: int | None
    offset_end: int | None
    reads: int
    writes: int
    status: str = "observed"
    completion: str = "unknown"


@dataclass(frozen=True)
class ScheduleModel:
    """Reference-independent structure; counts cannot become latency estimates."""

    source_kind: str
    stages: tuple[ScheduleStage, ...] = ()
    loops: tuple[CountedLoop, ...] = ()
    dependencies: tuple[ScheduleDependency, ...] = ()
    storage: tuple[StorageLifetime, ...] = ()
    configuration_scopes: tuple[Mapping[str, Any], ...] = ()
    decisions: tuple[Mapping[str, Any], ...] = ()
    role_counts: Mapping[str, int] = field(default_factory=dict)
    command_count: int = 0
    complete: bool = False
    truncated: bool = False
    unknowns: tuple[str, ...] = ()
    evidence_id: str | None = None
    version: int = 1
    event_counts: Mapping[str, int] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)

    @property
    def identity(self):
        return _identity(self.to_dict())

    @classmethod
    def from_record(cls, record):
        _version(record)
        row = dict(record)
        row["stages"] = tuple(ScheduleStage.from_record(item) for item in row.get("stages", ()))
        row["loops"] = tuple(CountedLoop.from_record(item) for item in row.get("loops", ()))
        row["dependencies"] = tuple(ScheduleDependency(**item) for item in row.get("dependencies", ()))
        row["storage"] = tuple(StorageLifetime(**item) for item in row.get("storage", ()))
        for key in ("configuration_scopes", "decisions", "unknowns"):
            row[key] = tuple(row.get(key, ()))
        return cls(**row)


def _stage(record: Mapping, position: int) -> ScheduleStage:
    if not isinstance(record, Mapping):
        raise ValueError("normalized commands must be mappings")
    index = record.get("index", position)
    identity = record.get("identity", record.get("class"))
    if type(index) is not int or index < 0 or type(identity) is not str or not identity:
        raise ValueError("normalized commands need an ordered nonnegative index and identity")
    kind = record.get("kind", "command")
    if kind not in ("command", "fence"):
        raise ValueError("normalized issue events must distinguish commands from scalar fences")
    roles, produces, waits = record.get("roles", ()), record.get("produces", ()), record.get("waits_for", ())
    if isinstance(produces, str):
        produces = (produces,)
    if any(isinstance(values, (str, bytes)) or not isinstance(values, (list, tuple))
           or any(type(value) is not str or not value for value in values) for values in (roles, produces, waits)):
        raise ValueError("command roles and event references must be explicit sequences")
    for name in ("configuration", "parameters"):
        if not isinstance(record.get(name, {}), Mapping):
            raise ValueError("configuration and parameters must be mappings")
    result = ScheduleStage(
        key=f"issue:{index}", index=index, identity=identity, roles=tuple(roles),
        accesses=tuple(ScheduleAccess.from_record(item) for item in record.get("accesses", ())),
        configuration=dict(record.get("configuration", {})), produces=tuple(produces), waits_for=tuple(waits),
        parameters=dict(record.get("parameters", {})),
        kind=kind,
    )
    _identity(result.to_dict())
    return result


def _signature(stage):
    return _identity(dict(identity=stage.identity, kind=stage.kind, roles=stage.roles,
                          configuration=stage.configuration,
                          produces=stage.produces, waits_for=stage.waits_for, parameters=stage.parameters,
                          accesses=[{key: value for key, value in access.to_dict().items() if key != "offset"}
                                    for access in stage.accesses]))


def _compact(stages, *, max_period, max_steps):
    signatures = [_signature(stage) for stage in stages]
    loops, retained = [], []
    start = steps = 0
    truncated = False
    while start < len(stages):
        best = None
        for period in range(1, min(max_period, (len(stages) - start) // 2) + 1):
            steps += 1
            if steps > max_steps:
                truncated = True
                break
            if signatures[start:start + period] != signatures[start + period:start + 2 * period]:
                continue
            strides = {}
            valid = True
            for offset in range(period):
                for access_index, (a, b) in enumerate(zip(stages[start + offset].accesses,
                                                         stages[start + period + offset].accesses)):
                    if type(a.offset) is int and type(b.offset) is int:
                        strides[f"{offset}:{access_index}"] = b.offset - a.offset
                    elif a.offset != b.offset:
                        valid = False
            if not valid:
                continue
            repeats = 2
            while start + (repeats + 1) * period <= len(stages):
                steps += period
                if steps > max_steps:
                    truncated = True
                    break
                at = start + repeats * period
                if signatures[start:start + period] != signatures[at:at + period]:
                    break
                if any(
                    b.offset != a.offset + repeats * strides[f"{offset}:{access_index}"]
                    if type(a.offset) is int and type(b.offset) is int else a.offset != b.offset
                    for offset in range(period)
                    for access_index, (a, b) in enumerate(zip(stages[start + offset].accesses,
                                                             stages[at + offset].accesses))
                ):
                    break
                repeats += 1
            score = repeats * period - period
            if best is None or score > best[0]:
                best = score, period, repeats, strides
        if best is None:
            retained.append(stages[start])
            start += 1
        else:
            _, period, repeats, strides = best
            body = stages[start:start + period]
            retained.extend(body)
            loops.append(CountedLoop(f"recurrence:{start}", stages[start].index, repeats,
                                     tuple(stage.key for stage in body), strides))
            start += period * repeats
        if truncated:
            retained.extend(stages[start:])
            break
    return tuple(retained), tuple(loops), truncated


def _effects(stages):
    lifetimes, config, scopes, dependencies, events, frontiers = {}, {}, [], [], {}, {}
    unknowns = set()
    for position, stage in enumerate(stages):
        if position:
            dependencies.append(ScheduleDependency(stages[position - 1].key, stage.key, "issue_order", "observed"))
        for event in stage.waits_for:
            if event in events:
                dependencies.append(ScheduleDependency(events[event], stage.key, "event_wait", "observed", event,
                                                       "declared_completion"))
            else:
                unknowns.add(f"event_without_producer:{event}")
        for event in stage.produces:
            if event in events:
                unknowns.add(f"repeated_event_identity:{event}")
            events[event] = stage.key
        for key, value in stage.configuration.items():
            if key in config:
                first, old = config[key]
                scopes.append(dict(key=key, value=old, first_issue=first, last_issue=stage.index - 1,
                                   status="observed", completion="unknown"))
            config[key] = stage.index, value
        for access in stage.accesses:
            key = access.space, access.buffer, access.unit
            bounds = type(access.offset) is int and type(access.extent) is int
            frontier = frontiers.setdefault(key, [])
            retained = []
            for previous, producer in frontier:
                previous_bounds = type(previous.offset) is int and type(previous.extent) is int
                overlap = (max(previous.offset, access.offset) < min(previous.offset + previous.extent,
                                                                     access.offset + access.extent)
                           if bounds and previous_bounds else None)
                if overlap is None:
                    unknowns.add("storage_alias_or_extent_unresolved")
                if overlap is not False and producer != stage.key:
                    previous_writes = previous.mode in ("write", "read_write")
                    writes = access.mode in ("write", "read_write")
                    reads = access.mode in ("read", "read_write")
                    kinds = []
                    if previous_writes and reads:
                        kinds.append("storage_RAW")
                    if previous_writes and writes:
                        kinds.append("storage_WAW")
                    if previous.mode in ("read", "read_write") and writes:
                        kinds.append("storage_WAR")
                    for kind in kinds:
                        dependencies.append(ScheduleDependency(producer, stage.key, kind, "inferred",
                                                               ":".join(key)))
                # A covering write supersedes this issue witness; partial writes
                # retain the old envelope conservatively.
                covered = (bounds and previous_bounds and access.mode in ("write", "read_write")
                           and access.offset <= previous.offset
                           and access.offset + access.extent >= previous.offset + previous.extent)
                if not covered:
                    retained.append((previous, producer))
            retained.append((access, stage.key))
            if len(retained) > 256:
                unknowns.add("storage_dependency_frontier_limit_reached")
                retained = retained[-256:]
            frontiers[key] = retained
            if key not in lifetimes:
                lifetimes[key] = dict(first_issue=stage.index, last_issue=stage.index,
                                      offset_min=access.offset if bounds else None,
                                      offset_end=access.offset + access.extent if bounds else None,
                                      reads=0, writes=0)
            row = lifetimes[key]
            row["last_issue"] = stage.index
            if bounds and row["offset_min"] is not None:
                row["offset_min"] = min(row["offset_min"], access.offset)
                row["offset_end"] = max(row["offset_end"], access.offset + access.extent)
            else:
                row["offset_min"] = row["offset_end"] = None
            row["reads"] += access.mode in ("read", "read_write")
            row["writes"] += access.mode in ("write", "read_write")
    if stages:
        scopes.extend(dict(key=key, value=value, first_issue=first, last_issue=stages[-1].index,
                           status="observed", completion="unknown") for key, (first, value) in config.items())
    return (tuple(StorageLifetime(*key, **value) for key, value in sorted(lifetimes.items())),
            tuple(scopes), tuple(dependencies), tuple(sorted(unknowns)))


def analyze_normalized_commands(
    commands: Iterable[Mapping], *, evidence_id=None, complete=False, source_kind="normalized_commands",
    max_commands=100_000, max_period=64, max_compaction_steps=1_000_000,
) -> ScheduleModel:
    """Inspect a bounded normalized stream; unknown effects stay unknown.

    ``complete`` attests the selected input boundary, not hardware retirement or
    completion. Effects are supplied by the selected fact-bound projector. The
    observed buffer envelope is an issue lifetime, never a safe allocation size.
    """
    if any(type(v) is not int or v < 1 for v in (max_commands, max_period, max_compaction_steps)):
        raise ValueError("schedule analysis requires finite positive bounds")
    stages, truncated = [], False
    for position, command in enumerate(commands):
        if position == max_commands:
            truncated = True
            break
        stage = _stage(command, position)
        if stages and stage.index <= stages[-1].index:
            raise ValueError("normalized command indices must be strictly increasing")
        stages.append(stage)
    storage, scopes, dependencies, unknowns = _effects(stages)
    compact, loops, compaction_truncated = _compact(stages, max_period=max_period, max_steps=max_compaction_steps)
    unknowns = set(unknowns)
    unknowns.add("hardware_completion_and_timing_not_established")
    if any(not stage.roles for stage in stages):
        unknowns.add("instruction_roles_unresolved")
    if any(not stage.accesses for stage in stages):
        unknowns.add("memory_effects_not_complete")
    if compaction_truncated:
        unknowns.add("recurrence_search_budget_exhausted")
    if truncated:
        unknowns.add("command_limit_reached")
    if loops:
        unknowns.add("observed_recurrence_is_not_generalized_loop_proof")
    return ScheduleModel(source_kind=source_kind, stages=compact, loops=loops, dependencies=dependencies,
                         storage=storage, configuration_scopes=scopes,
                         role_counts=dict(Counter(role for stage in stages for role in stage.roles)),
                         command_count=len(stages), complete=bool(complete and not truncated), truncated=truncated,
                         unknowns=tuple(sorted(unknowns)), evidence_id=evidence_id,
                         event_counts=dict(Counter(stage.kind for stage in stages)))


def analyze_ir_schedule(module, *, evidence_id=None) -> ScheduleModel:
    """Read the real scheduling dialect as *decisions*, without asserting realization."""
    from io import StringIO

    from xdsl.printer import Printer

    decisions = []
    for operation in module.walk():
        if operation.name.partition(".")[0] != "schedule":
            continue
        def text(value):
            stream = StringIO()
            Printer(stream=stream, print_generic_format=True).print_attribute(value)
            return stream.getvalue()
        properties = {key: text(value) for key, value in operation.properties.items()}
        attributes = {key: text(value) for key, value in operation.attributes.items()}
        decisions.append(dict(operation=operation.name, properties=properties, attributes=attributes,
                              status="observed", realized=False))
    return ScheduleModel("schedule_dialect", decisions=tuple(decisions), complete=True, evidence_id=evidence_id,
                         unknowns=("decision_realization_requires_executable_consumer",))


def analyze_counted_kernel(kernel, *, effect_projector: Callable | None = None, evidence_id=None) -> ScheduleModel:
    """Read existing counted payload IR without expanding its dynamic domain.

    Staging effects and token recurrences remain declared payload facts. A
    selected projector may add external accesses and roles, which this analysis
    cannot reconstruct from instruction names alone.
    """
    from merlin.sched.ir import Call, Kernel, Loop, check_structure

    if type(kernel) is not Kernel or check_structure(kernel):
        raise ValueError("counted schedule analysis needs a structurally valid kernel")
    stages, loops, dependencies = [], [], []
    count, role_counts = 0, Counter()

    def visit(body, path, multiplier):
        nonlocal count
        keys = []
        for index, node in enumerate(body):
            key = "/".join((*path, str(index)))
            keys.append(key)
            if type(node) is Loop:
                children = visit(node.body, (*path, str(index)), multiplier * node.extent)
                loops.append(CountedLoop(key, 0, node.extent, tuple(children), status="observed", variable=node.var,
                                         iter_args=node.iter_args, yields=node.yields, results=node.results))
            elif type(node) is Call:
                projected = dict(effect_projector(node) if effect_projector else {})
                projected.update(index=len(stages), identity=node.instr,
                                 produces=() if node.produces is None else (node.produces,), waits_for=node.awaits,
                                 configuration=dict(node.sets))
                if "accesses" not in projected:
                    projected["accesses"] = [dict(space=stage.memory, buffer=stage.memory,
                                                  offset=stage.row, extent=stage.rows * stage.depth,
                                                  mode="write" if stage.writes else "read", unit="row")
                                                 for stage in node.stages]
                stage = _stage(projected, len(stages))
                stages.append(ScheduleStage(key, stage.index, stage.identity, stage.roles, stage.accesses,
                                             stage.configuration, stage.produces, stage.waits_for, stage.parameters))
                count += multiplier
                for role in stage.roles:
                    role_counts[role] += multiplier
            else:
                raise ValueError("unsupported counted schedule node")
        return keys

    visit(kernel.body, ("payload",), 1)
    storage, scopes, dependencies, unknowns = _effects(stages)
    return ScheduleModel("counted_payload_ir", stages=tuple(stages), loops=tuple(loops), dependencies=dependencies,
                         storage=storage, configuration_scopes=scopes, command_count=count, complete=True,
                         role_counts=dict(role_counts), event_counts={"command": count},
                         evidence_id=evidence_id, unknowns=tuple(sorted({*unknowns,
                             "payload_effects_are_declarations_not_executed_trace",
                             "storage_lifetimes_are_static_payload_not_dynamic_domain",
                             "hardware_completion_and_timing_not_established"})))
