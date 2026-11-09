"""Independently verify shared packing storage and every original island budget.

One import/check serves the common compiled entry. Shared roots count once in
its union and once in each participating island; capacity cannot transfer across
islands. Reference values, target execution and total host RAM remain separate.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from merlin.llvmlower.borrowed_buffer_calls import validate_calls

from .packing_ir import _require
from .packing_ir import verify_packing_storage as verify_packing_storage
from .packing_types import PackingAllocationEffects, PackingStorageRequest
from .packing_types import PackingStorageUse as PackingStorageUse


def implementation_paths():
    from merlin.common import digest
    from merlin.llvmlower import arena_bind
    from merlin.perf import host_cfg_index
    from merlin.targetgen.oot_starterkit import llvm_context
    from merlin.xdsl_dialects.lowering import integer_constant_eval

    from . import host_call_alignment, packing_ir, packing_types

    return tuple(
        Path(p).resolve()
        for p in (
            __file__,
            packing_ir.__file__,
            packing_types.__file__,
            host_call_alignment.__file__,
            digest.__file__,
            host_cfg_index.__file__,
            arena_bind.__file__,
            integer_constant_eval.__file__,
            llvm_context.__file__,
        )
    )


def verify_packing_storage_batch(
    module,
    *,
    borrowed_calls,
    requests,
    allocation_effects,
    entry,
    max_operations=100000,
    max_alias_values=4096,
):
    """Check one actual entry once, retaining each island's independent budget.

    Common roots count once in the union and once in every participating island.
    Projection cannot transfer spare budget between islands. Source-domain
    coverage remains the caller's separate original-source obligation.
    """
    _require(
        type(max_operations) is int
        and max_operations > 0
        and type(requests) is tuple
        and 0 < len(requests) <= max_operations
        and all(type(request) is PackingStorageRequest for request in requests),
        "an immutable bounded original packing-request roster is required",
    )
    for request in requests:
        request.__post_init__()
    union_uses = tuple(dict.fromkeys(use for request in requests for use in request.uses))
    combined = verify_packing_storage(
        module,
        borrowed_calls=borrowed_calls,
        uses=union_uses,
        allocation_effects=allocation_effects,
        max_payload_bytes=sum(request.max_payload_bytes for request in requests),
        entry=entry,
        max_operations=max_operations,
        max_alias_values=max_alias_values,
    )
    roots = {row["allocation_operation"]: row for row in combined["buffers"]}
    results = []
    for request in requests:
        selected = {(use.wrapper_symbol, use.argument) for use in request.uses}
        observed = [row for row in combined["uses"] if (row["symbol"], row["argument"]) in selected]
        _require(
            {(row["symbol"], row["argument"]) for row in observed} == selected,
            "actual island packing coverage differs from original uses",
        )
        payloads = {}
        for use in observed:
            root = use["allocation_operation"]
            if root not in payloads:
                payloads[root] = use["logical_payload_bytes"]
            else:
                payloads[root] = max(payloads[root], use["logical_payload_bytes"])
        buffers = [
            {
                **roots[root],
                "logical_payload_bytes": payload,
                "alignment_overhead_bytes": roots[root]["allocator_requested_bytes"] - payload,
            }
            for root, payload in payloads.items()
        ]
        payload = sum(payloads.values())
        requested = sum(row["allocator_requested_bytes"] for row in buffers)
        _require(payload <= request.max_payload_bytes, "actual island packing payload exceeds original budget")
        results.append(
            dict(
                logical_payload_bytes=payload,
                allocator_requested_bytes=requested,
                alignment_overhead_bytes=requested - payload,
                max_payload_bytes=request.max_payload_bytes,
                buffers=buffers,
                uses=observed,
            )
        )
    return dict(
        schema="merlin.final_packing_storage_batch.v1",
        entry=entry,
        requests=[asdict(request) for request in requests],
        combined=combined,
        results=results,
    )


def inspect_packing_storage_batch(
    host_artifacts,
    *,
    borrowed_calls,
    requests,
    allocation_effects,
    entry,
    diagnostics,
    max_operations=100000,
    max_alias_values=4096,
    max_ir_bytes=64 << 20,
):
    """Import and analyze the common actual entry once for all original islands."""
    from .host_call_alignment import import_host_entry, revalidate_host_entry

    _require(diagnostics is not None, "final packing proof requires retained import products")
    _require(type(allocation_effects) is PackingAllocationEffects, "original allocator effects are unavailable")
    validate_calls(borrowed_calls)
    _require(
        type(max_operations) is int
        and max_operations > 0
        and type(requests) is tuple
        and 0 < len(requests) <= max_operations
        and all(type(request) is PackingStorageRequest for request in requests),
        "an immutable bounded original packing-request roster is required",
    )
    for request in requests:
        request.__post_init__()
    uses = tuple(dict.fromkeys(use for request in requests for use in request.uses))

    def original_record():
        return json.dumps(
            dict(
                calls=[asdict(call) for call in borrowed_calls],
                requests=[asdict(request) for request in requests],
                effects=asdict(allocation_effects),
            ),
            sort_keys=True,
        )

    originals = original_record()
    owners = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in implementation_paths()}
    module, observation = import_host_entry(
        host_artifacts,
        diagnostics=diagnostics,
        entry=entry,
        include_functions=tuple(
            dict.fromkeys(
                (
                    *(use.wrapper_symbol for use in uses),
                    allocation_effects.allocator_symbol,
                    allocation_effects.deallocator_symbol,
                )
            )
        ),
        max_ir_bytes=max_ir_bytes,
    )
    batch = verify_packing_storage_batch(
        module,
        borrowed_calls=borrowed_calls,
        requests=requests,
        allocation_effects=allocation_effects,
        entry=entry,
        max_operations=max_operations,
        max_alias_values=max_alias_values,
    )
    result = batch
    revalidate_host_entry(observation)
    _require(originals == original_record(), "original packing contracts changed during inspection")
    _require(
        owners == {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in implementation_paths()},
        "packing checker implementation changed during inspection",
    )
    result["observation"] = observation
    pins = {**owners, **observation["inputs"], **observation["tools"], **observation["products"]}
    payload = (json.dumps(result, sort_keys=True, indent=2) + "\n").encode()
    digest = hashlib.sha256(payload).hexdigest()
    report = Path(diagnostics) / ("packing-storage-" + digest + ".json")
    if report.exists():
        _require(not report.is_symlink() and report.read_bytes() == payload, "retained packing report changed")
    else:
        report.write_bytes(payload)
    result.update(pins={**pins, str(report.resolve()): digest}, report=dict(path=str(report.resolve()), sha256=digest))
    return result
