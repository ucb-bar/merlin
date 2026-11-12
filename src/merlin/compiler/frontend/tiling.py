"""Select explicit ordered native request partitions through the chosen backend.

Geometry and numerical admission stay with the selected OOT implementation.
The shared transformations preserve logical source boundaries and reduction order;
these records do not qualify execution or hardware support.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

from merlin.compiler.semantic.model import KernelRequest


@dataclass(frozen=True)
class OrderedNativeRequest:
    """Selected request and the exact two source boundaries needed for replay."""

    source: KernelRequest
    k_source: KernelRequest
    request: KernelRequest
    mn_record: bytes | None
    k_record: bytes | None


def prepare_ordered_request(args, binding, source_request, precision_plan=None):
    """Keep optional outer-M/N selection before ordered reduction partitioning."""
    if precision_plan is not None and any(
        getattr(args, name, None) is not None
        for name in ("ordered_k_tile", "ordered_m_tile", "ordered_n_tile", "outer_tile_index")
    ):
        raise ValueError("source-bound precision tiles cannot also use ordered matrix partitioning")
    outer, mn_record = prepare_ordered_mn(args, binding, source_request)
    selected, k_record = prepare_ordered_k(args, binding, outer)
    return OrderedNativeRequest(source_request, outer, selected, mn_record, k_record)


def prepare_ordered_k(args: argparse.Namespace, binding, source_request):
    """Ask the selected target to qualify exact ordered-K partitioning."""
    from merlin.compiler.semantic.target_binding import NativeCompilationError
    from merlin.compiler.transforms.contraction_k import OrderedKTiling

    tile_k = getattr(args, "ordered_k_tile", None)
    if tile_k is None:
        return source_request, None
    prepare = getattr(binding, "prepare_ordered_k", None) if binding is not None else None
    if not callable(prepare):
        raise NativeCompilationError("unsupported_semantics", "selected target has no qualified ordered-K tiling")
    prepared = prepare(source_request, tile_k)
    if (
        not isinstance(prepared, OrderedKTiling)
        or prepared.request.target_identity != source_request.target_identity
        or prepared.request.lowering_policy != source_request.lowering_policy
        or prepared.request.source_identity != source_request.source_identity
    ):
        raise ValueError("target returned an invalid ordered-K partition")
    record = {
        **prepared.record(),
        "source_request_digest": source_request.digest(),
        "tile_k": tile_k,
    }
    return prepared.request, (json.dumps(record, sort_keys=True, indent=2) + "\n").encode()


def prepare_ordered_mn(args: argparse.Namespace, binding, source_request):
    """Select one checked outer output tile from the selected target's geometry."""
    from merlin.compiler.semantic.target_binding import NativeCompilationError
    from merlin.compiler.transforms.contraction_mn import OrderedMNTiling

    tile_m = getattr(args, "ordered_m_tile", None)
    tile_n = getattr(args, "ordered_n_tile", None)
    index = getattr(args, "outer_tile_index", None)
    if tile_m is None and tile_n is None and index is None:
        return source_request, None
    if tile_m is None or tile_n is None or index is None:
        raise ValueError("ordered M/N tiling needs both geometry dimensions and an output tile index")
    prepare = getattr(binding, "prepare_ordered_mn", None) if binding is not None else None
    if not callable(prepare):
        raise NativeCompilationError("unsupported_semantics", "selected target has no qualified M/N tiling")
    prepared = prepare(source_request, tile_m, tile_n)
    if (
        not isinstance(prepared, OrderedMNTiling)
        or prepared.source.digest() != source_request.digest()
        or type(index) is not int
        or not 0 <= index < len(prepared.tiles)
    ):
        raise ValueError("target returned an invalid ordered M/N partition or tile index")
    selected = prepared.tiles[index].request
    if (
        selected.target_identity != source_request.target_identity
        or selected.lowering_policy != source_request.lowering_policy
        or selected.source_identity != source_request.source_identity
    ):
        raise ValueError("outer output tile changed source identity or compilation policy")
    record = {
        **prepared.record(),
        "selected_tile_index": index,
        "selected_request_digest": selected.digest(),
    }
    return selected, (json.dumps(record, sort_keys=True, indent=2) + "\n").encode()
