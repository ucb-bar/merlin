"""Format-only layout of the original data-bound logical pointer renderer.

The bound runtime is a distinct authority, not an out-of-tree plugin wrapper.
Only the exact core type and its freshly reselected contract/facts are accepted.
No target hook, arithmetic, execution, memory transport or qualification follows.
"""

from __future__ import annotations

import hashlib
import json
from math import prod
from pathlib import Path


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _source(path):
    path = Path(path)
    if (not path.is_absolute() or path.resolve() != path or path.is_symlink()
            or not path.is_file() or not 0 < path.stat().st_size <= 16 << 20):
        raise ValueError("bound caller-layout source is absent, indirect or exceeds its budget")
    raw = path.read_bytes()
    return hashlib.sha256(raw).hexdigest()


def _binding(backend, *, target, facts):
    from merlin.common import quant_formats
    from merlin.runtime import direct_kernel_harness, harness_render
    from merlin.runtime.backends import base, chipyard_rocc_bound
    from merlin.targetgen import target_registry

    if type(backend) is not chipyard_rocc_bound.BoundRoCCBackend or backend.target != target:
        raise ValueError("caller layout needs the exact selected core data-bound runtime")
    # Resolve again from the original selection. Never unwrap an arbitrary
    # object or accept a saved record as a live backend/contract selector.
    current = base.get_backend(target)
    if type(current) is not type(backend) or current._selection_json != backend._selection_json:
        raise ValueError("bound caller-layout runtime differs from its current original selection")
    data = json.loads(backend._selection_json)
    selected = target_registry.resolve(target)
    contract = selected.load_contract()
    if data != {"contract": contract, "facts": facts}:
        raise ValueError("bound caller-layout contract or facts differ from the original runtime")
    harness_render.validate_contract(contract)
    contract_path = selected.capability_contract_path
    modules = dict(bound_runtime=chipyard_rocc_bound, registry=base, target_selection=target_registry,
                   logical_renderer=harness_render, pointer_renderer=direct_kernel_harness,
                   scalar_formats=quant_formats)
    return {
        "schema": "merlin.bound_logical_caller_layout_identity.v1",
        "target": target,
        "runtime_selection_sha256": hashlib.sha256(backend._selection_json.encode()).hexdigest(),
        "selected_contract_file_sha256": _source(contract_path),
        "selected_contract_sha256": hashlib.sha256(_canonical(contract)).hexdigest(),
        "selected_facts_sha256": hashlib.sha256(_canonical(facts)).hexdigest(),
        "sources": {**{role: _source(module.__file__) for role, module in modules.items()},
                    "layout_inspector": _source(__file__)},
    }, contract


def inspect(backend, cb, *, target, facts):
    """Describe the same bounded dense allocations as its original renderer.

    This supplies no OOT hook delegation. The existing neutral renderer rejects
    padded/strided tensors; explicit transformed-storage maps are also refused.
    No input values are materialized, no executable is created and no model runs.
    """
    from merlin.runtime import harness_render

    before, contract = _binding(backend, target=target, facts=facts)
    abi = harness_render.validate_contract(contract).pointer_abi
    if abi.byte_order != "little":
        raise ValueError("bound coherent caller layout has no selected big-endian decoder")
    if cb.get("target") != target or "storage_encodings" in cb.get("params", {}):
        raise ValueError("bound logical caller layout has another target or unsupported transformed storage")
    normalized, _mutable = harness_render._roster(cb)
    budget = (contract.get("logical_harness") or {}).get("storage_budget_bytes")
    if budget is None:
        # Read the original renderer's actual default, never a target-specific
        # or separately maintained allocation ceiling.
        budget = harness_render.render_harness.__kwdefaults__["max_storage_bytes"]
    if type(budget) is not int or not 0 < budget < 2**64:
        raise ValueError("bound caller-layout allocation budget is unsupported")
    rows, allocated = [], 0
    for ordinal, argument in enumerate(normalized["kernel_abi"]["args"]):
        name = argument["tensor"]
        spec = normalized["tensors"][name]
        shape, dtype, size = harness_render._tensor(spec)
        if not shape:
            raise ValueError("coherent caller layout needs positive-rank outputs and arguments")
        if dtype.element_bits // 8 > abi.tensor_alignment:
            raise ValueError("bound caller layout violates the original renderer's scalar alignment")
        allocated += size
        if allocated > budget:
            raise ValueError("bound caller layout exceeds its original aggregate storage budget")
        # The original source-pinned pointer renderer declares and passes this
        # ordinal symbol (including streamed sidecars). Actual ELF size/address
        # checks remain mandatory; no logical tensor name selects an address.
        rows.append({"tensor": name, "symbol": f"tensor_{ordinal}",
                     "dtype": spec["dtype"], "logical_shape": list(shape),
                     "physical_extents": list(shape),
                     "logical_strides_elements": [prod(shape[axis + 1:]) for axis in range(len(shape))],
                     "storage_elements": prod(shape), "offset_elements": 0})
    after, _ = _binding(backend, target=target, facts=facts)
    if before != after:
        raise ValueError("bound caller-layout selection or original source changed during inspection")
    return {
        "layout": {"schema": "caller_storage_layout_v2", "policy": {"mode": "logical_dense_row_major_v2"},
                   "tensors": rows},
        "binding": before,
        "provider_sha256": hashlib.sha256(_canonical(before)).hexdigest(),
    }
