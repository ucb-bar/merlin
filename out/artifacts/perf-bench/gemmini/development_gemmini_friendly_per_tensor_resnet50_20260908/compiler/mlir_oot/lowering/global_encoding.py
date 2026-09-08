"""Target-neutral global tensor-encoding selection.

This module deliberately knows nothing about Gemmini, ResNet, or an MLIR dialect.  A target
adapter supplies the exact operation alternatives it can implement.  The solver then chooses one
alternative per operation and one storage encoding per tensor boundary while accounting for all
producers, consumers, ABI endpoints, fanout, lifetimes, adapters, and an optional arena limit.

The objective is *structural materialisation bytes*.  It is not a latency predictor: operation
alternatives state the derived buffers they would materialise, while encoding mismatches pay one
read plus one write.  This is useful before expensive simulation because it cannot manufacture a
cycle speedup from an unmeasured bandwidth constant.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from math import prod
from typing import Iterable, Mapping, Sequence


DTYPE_BYTES = {
    "i1": 1, "i8": 1, "i16": 2, "i32": 4, "i64": 8,
    "f16": 2, "bf16": 2, "f32": 4, "f64": 8,
}


@dataclass(frozen=True)
class Encoding:
    """A physical axis order and innermost-dimension alignment."""

    name: str
    axes: tuple[str, ...]
    alignment: int = 1


@dataclass(frozen=True)
class TensorBoundary:
    """One logical tensor value shared by operation endpoints."""

    name: str
    extents: Mapping[str, int]
    dtype: str
    logical_encoding: str
    producer: int | None
    consumers: tuple[int, ...] = ()
    external_input: bool = False
    external_output: bool = False


@dataclass(frozen=True)
class OperationAlternative:
    """One exact implementation of an operation under fixed endpoint encodings."""

    name: str
    input_encodings: Mapping[str, str]
    output_encodings: Mapping[str, str]
    materialized_bytes: int = 0
    exact: bool = True
    refusal: str = ""
    capability: str = ""


@dataclass(frozen=True)
class EncodingOperation:
    index: int
    name: str
    kind: str
    alternatives: tuple[OperationAlternative, ...]


@dataclass(frozen=True)
class EncodingProblem:
    tensors: tuple[TensorBoundary, ...]
    operations: tuple[EncodingOperation, ...]
    encodings: Mapping[str, Encoding]
    arena_capacity_bytes: int | None = None
    exhaustive_combination_limit: int = 4096


@dataclass
class _Evaluated:
    choices: tuple[OperationAlternative, ...]
    storage: dict[str, str]
    transforms: list[dict]
    operation_materialized_bytes: int
    transform_bytes: int
    encoded_boundary_bytes: int
    logical_boundary_bytes: int
    peak_live_bytes: int
    peak_live_at: int
    live_intervals: list[dict]
    over_capacity: bool

    @property
    def objective(self) -> tuple[int, int, int]:
        return (
            self.operation_materialized_bytes + self.transform_bytes,
            self.peak_live_bytes,
            len(self.transforms),
        )


def _extent_tuple(tensor: TensorBoundary, encoding: Encoding) -> tuple[int, ...]:
    if set(encoding.axes) != set(tensor.extents):
        raise ValueError(
            f"encoding {encoding.name!r} axes {encoding.axes} do not cover tensor "
            f"{tensor.name!r} axes {tuple(tensor.extents)}"
        )
    return tuple(int(tensor.extents[axis]) for axis in encoding.axes)


def storage_bytes(tensor: TensorBoundary, encoding: Encoding) -> int:
    """Physical bytes, including only explicit innermost alignment padding."""
    shape = _extent_tuple(tensor, encoding)
    if tensor.dtype not in DTYPE_BYTES:
        raise ValueError(f"unknown dtype {tensor.dtype!r} for {tensor.name!r}")
    if not shape:
        return DTYPE_BYTES[tensor.dtype]
    aligned_last = -(-shape[-1] // encoding.alignment) * encoding.alignment
    return prod(shape[:-1]) * aligned_last * DTYPE_BYTES[tensor.dtype]


def _requirements(
    tensor: TensorBoundary,
    choices_by_index: Mapping[int, OperationAlternative],
) -> list[tuple[str, str, int | str]]:
    endpoints: list[tuple[str, str, int | str]] = []
    if tensor.external_input:
        endpoints.append(("abi_input", tensor.logical_encoding, "input"))
    if tensor.producer is not None:
        alt = choices_by_index[tensor.producer]
        endpoints.append(("producer", alt.output_encodings[tensor.name], tensor.producer))
    for consumer in tensor.consumers:
        alt = choices_by_index[consumer]
        endpoints.append(("consumer", alt.input_encodings[tensor.name], consumer))
    if tensor.external_output:
        endpoints.append(("abi_output", tensor.logical_encoding, "output"))
    return endpoints


def _adapter_bytes(tensor: TensorBoundary, source: Encoding, target: Encoding) -> int:
    if source.name == target.name:
        return 0
    return storage_bytes(tensor, source) + storage_bytes(tensor, target)


def _evaluate(problem: EncodingProblem, choices: Sequence[OperationAlternative]) -> _Evaluated:
    choices_by_index = {op.index: choice for op, choice in zip(problem.operations, choices)}
    storage: dict[str, str] = {}
    transforms: list[dict] = []
    logical_bytes = 0
    encoded_bytes = 0

    for tensor in problem.tensors:
        logical = problem.encodings[tensor.logical_encoding]
        logical_bytes += storage_bytes(tensor, logical)
        endpoints = _requirements(tensor, choices_by_index)
        candidates = {tensor.logical_encoding, *(encoding for _, encoding, _ in endpoints)}
        ranked: list[tuple[int, int, str]] = []
        for candidate_name in sorted(candidates):
            candidate = problem.encodings[candidate_name]
            conversion = sum(
                _adapter_bytes(tensor, candidate, problem.encodings[required])
                for _, required, _ in endpoints
            )
            ranked.append((conversion, storage_bytes(tensor, candidate), candidate_name))
        _, physical_bytes, selected = min(ranked)
        storage[tensor.name] = selected
        encoded_bytes += physical_bytes
        for endpoint, required, owner in endpoints:
            if selected == required:
                continue
            source_name, target_name = (
                (required, selected) if endpoint in ("abi_input", "producer")
                else (selected, required)
            )
            source, target = problem.encodings[source_name], problem.encodings[target_name]
            transforms.append({
                "tensor": tensor.name,
                "endpoint": endpoint,
                "owner": owner,
                "from": source_name,
                "to": target_name,
                "bytes": _adapter_bytes(tensor, source, target),
                "reason": "physical encoding required by exact endpoint",
            })

    last_operation = max((op.index for op in problem.operations), default=0)
    intervals: list[dict] = []
    events: dict[int, int] = {}
    tensor_by_name = {tensor.name: tensor for tensor in problem.tensors}
    for name, encoding_name in storage.items():
        tensor = tensor_by_name[name]
        start = 0 if tensor.external_input or tensor.producer is None else tensor.producer
        end_candidates = list(tensor.consumers)
        if tensor.external_output:
            end_candidates.append(last_operation + 1)
        end = max(end_candidates, default=start)
        size = storage_bytes(tensor, problem.encodings[encoding_name])
        intervals.append({
            "tensor": name, "first_endpoint": start, "last_endpoint": end,
            "encoding": encoding_name, "bytes": size,
        })
        events[start] = events.get(start, 0) + size
        events[end + 1] = events.get(end + 1, 0) - size
    live = peak = peak_at = 0
    for point in sorted(events):
        live += events[point]
        if live > peak:
            peak, peak_at = live, point

    operation_bytes = sum(choice.materialized_bytes for choice in choices)
    transform_bytes = sum(item["bytes"] for item in transforms)
    return _Evaluated(
        tuple(choices), storage, transforms, operation_bytes, transform_bytes,
        encoded_bytes, logical_bytes, peak, peak_at, intervals,
        problem.arena_capacity_bytes is not None and peak > problem.arena_capacity_bytes,
    )


def _choice_space(problem: EncodingProblem) -> Iterable[tuple[OperationAlternative, ...]]:
    exact = [tuple(alt for alt in op.alternatives if alt.exact) for op in problem.operations]
    if any(not alternatives for alternatives in exact):
        return ()
    combinations = prod(len(alternatives) for alternatives in exact)
    if combinations <= problem.exhaustive_combination_limit:
        return product(*exact)

    # Large graphs normally have one dominant exact alternative per op.  Start there and perform
    # deterministic coordinate descent.  This evaluates every alternative against the complete
    # fanout/lifetime state, rather than making a producer-local decision.
    current = tuple(min(alternatives, key=lambda alt: (alt.materialized_bytes, alt.name))
                    for alternatives in exact)
    while True:
        incumbent = _evaluate(problem, current)
        changed = False
        for position, alternatives in enumerate(exact):
            candidates = []
            for alternative in alternatives:
                trial = list(current)
                trial[position] = alternative
                evaluated = _evaluate(problem, trial)
                penalty = (1 if evaluated.over_capacity else 0, *evaluated.objective,
                           tuple(choice.name for choice in evaluated.choices))
                candidates.append((penalty, tuple(trial)))
            best = min(candidates, key=lambda item: item[0])[1]
            if best != current:
                current, changed = best, True
        if not changed:
            return (current,)


def solve(problem: EncodingProblem) -> dict:
    """Choose exact operation alternatives and tensor encodings, or fail closed."""
    operation_indices = [op.index for op in problem.operations]
    if len(operation_indices) != len(set(operation_indices)):
        raise ValueError("operation indices must be unique")
    known_indices = set(operation_indices)
    known_tensors = {tensor.name for tensor in problem.tensors}
    for tensor in problem.tensors:
        if tensor.logical_encoding not in problem.encodings:
            raise ValueError(f"unknown logical encoding {tensor.logical_encoding!r}")
        if tensor.producer is not None and tensor.producer not in known_indices:
            raise ValueError(f"unknown producer {tensor.producer} for {tensor.name}")
        if any(consumer not in known_indices for consumer in tensor.consumers):
            raise ValueError(f"unknown consumer for {tensor.name}")
    for op in problem.operations:
        for alt in op.alternatives:
            if not set(alt.input_encodings).issubset(known_tensors):
                raise ValueError(f"{op.name}/{alt.name} names unknown input")
            if not set(alt.output_encodings).issubset(known_tensors):
                raise ValueError(f"{op.name}/{alt.name} names unknown output")
            for encoding in (*alt.input_encodings.values(), *alt.output_encodings.values()):
                if encoding not in problem.encodings:
                    raise ValueError(f"{op.name}/{alt.name} names unknown encoding {encoding}")

    evaluated = [_evaluate(problem, choices) for choices in _choice_space(problem)]
    feasible = [entry for entry in evaluated if not entry.over_capacity]
    refusals = [
        {"operation_index": op.index, "operation": op.name, "alternative": alt.name,
         "capability": alt.capability, "reason": alt.refusal or "not_exact"}
        for op in problem.operations for alt in op.alternatives if not alt.exact
    ]
    if not feasible:
        best = min(evaluated, key=lambda entry: entry.objective) if evaluated else None
        return {
            "schema": "target_neutral_global_encoding_v1",
            "status": "refused",
            "reason": ("arena_capacity_exceeded" if best is not None
                       else "operation_has_no_exact_alternative"),
            "arena_capacity_bytes": problem.arena_capacity_bytes,
            "minimum_peak_live_bytes": None if best is None else best.peak_live_bytes,
            "refusals": refusals,
        }

    best = min(
        feasible,
        key=lambda entry: (*entry.objective, tuple(choice.name for choice in entry.choices)),
    )
    baseline_transforms = 0
    baseline_transform_bytes = 0
    choices_by_index = {op.index: choice for op, choice in zip(problem.operations, best.choices)}
    for tensor in problem.tensors:
        logical = problem.encodings[tensor.logical_encoding]
        for _, required, _ in _requirements(tensor, choices_by_index):
            if required != tensor.logical_encoding:
                baseline_transforms += 1
                baseline_transform_bytes += _adapter_bytes(
                    tensor, logical, problem.encodings[required])

    return {
        "schema": "target_neutral_global_encoding_v1",
        "status": "selected",
        "objective": {
            "name": "structural_materialization_bytes",
            "operation_materialized_bytes": best.operation_materialized_bytes,
            "layout_transform_bytes": best.transform_bytes,
            "total_bytes": best.operation_materialized_bytes + best.transform_bytes,
            "cycle_prediction_claimed": False,
        },
        "operations": [
            {"index": op.index, "name": op.name, "kind": op.kind,
             "selected": choice.name, "capability": choice.capability,
             "materialized_bytes": choice.materialized_bytes,
             "selection_reason": "minimum exact global structural-materialization objective",
             "considered_alternatives": [
                 {"name": alternative.name, "capability": alternative.capability,
                  "exact": alternative.exact,
                  "materialized_bytes": alternative.materialized_bytes,
                  "refusal": alternative.refusal}
                 for alternative in op.alternatives
             ]}
            for op, choice in zip(problem.operations, best.choices)
        ],
        "boundaries": [
            {"tensor": tensor.name, "dtype": tensor.dtype,
             "logical_encoding": tensor.logical_encoding,
             "selected_encoding": best.storage[tensor.name],
             "logical_bytes": storage_bytes(
                 tensor, problem.encodings[tensor.logical_encoding]),
             "encoded_bytes": storage_bytes(
                 tensor, problem.encodings[best.storage[tensor.name]]),
             "producer": tensor.producer, "consumers": list(tensor.consumers),
             "fanout": len(tensor.consumers),
             "external_input": tensor.external_input,
             "external_output": tensor.external_output}
            for tensor in problem.tensors
        ],
        "logical_boundary_bytes": best.logical_boundary_bytes,
        "encoded_boundary_bytes": best.encoded_boundary_bytes,
        "layout_transforms": best.transforms,
        "layout_transforms_inserted": len(best.transforms),
        "layout_transforms_eliminated": baseline_transforms - len(best.transforms),
        "layout_transform_bytes_eliminated": baseline_transform_bytes - best.transform_bytes,
        "capacity": {
            "arena_capacity_bytes": problem.arena_capacity_bytes,
            "peak_live_encoded_bytes": best.peak_live_bytes,
            "peak_live_at_operation": best.peak_live_at,
            "fits": True,
        },
        "lifetimes": best.live_intervals,
        "refusals": refusals,
    }
