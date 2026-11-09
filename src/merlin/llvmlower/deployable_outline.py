"""Closed static tensor dispatch chains for a deployable compute-only executor.

The existing outliner and typed dispatch DAG remain authoritative. This adapter
admits a strict dense tensor/view subset, preserves all arithmetic in isolated
kernel functions, and reserves unique storage for the entire chain lifetime.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from typing import Any

from merlin.common.jsonio import canonical_sha256
from merlin.llvmlower.deployable_ir import checked_generic_text
from merlin.xdsl_dialects.lowering.dispatch_program import (
    build_dispatch_program,
    prune_dead_nodes,
    verify_program,
)
from merlin.xdsl_dialects.lowering.outline import outline_dispatches

# Standard FP8 storage is byte-sized. This metadata does not provide a numeric
# encoder or an executable kernel: an explicitly selected implementation still
# binds the exact source format/semantics and its own execution qualification.
_BYTES = {"i8": 1, "i16": 2, "i32": 4, "i64": 8, "f32": 4, "f64": 8, "f8E4M3FN": 1, "f8E5M2": 1}


@dataclass(frozen=True)
class ChainBuffer:
    id: str
    shape: tuple[int, ...]
    dtype: str
    bytes: int
    arg_index: int | None
    workspace_offset: int | None


@dataclass
class DeployableChain:
    outlined: Any
    program: Any
    buffers: list[ChainBuffer]
    dispatches: list[dict]
    trailing_views: list[dict]
    workspace_bytes: int
    alignment_bytes: int
    worker_count: int
    source_ir_sha256: str
    data_preparation: dict | None = None
    frozen_inputs: dict | None = None

    def to_dict(self) -> dict:
        result = {
            "schema": "merlin.deployable_outlined_chain.v1",
            "source_ir_sha256": self.source_ir_sha256,
            "program": self.program.to_dict(),
            "buffers": [asdict(buffer) for buffer in self.buffers],
            "dispatches": self.dispatches,
            "trailing_views": self.trailing_views,
            "workspace_bytes": self.workspace_bytes,
            "alignment_bytes": self.alignment_bytes,
            "worker_count": self.worker_count,
            "storage_policy": "unique intermediate storage retained until entire chain quiesces",
            "arithmetic_policy": "every dispatch executes on one explicitly assigned compute worker",
            "controller_arithmetic": False,
            "kernel_parallelism": "serial dependency chain; no nested OpenMP",
            "full_model_admission": False,
        }
        if self.data_preparation is not None:
            result["data_preparation"] = self.data_preparation
        if self.frozen_inputs is not None:
            result["frozen_inputs"] = self.frozen_inputs
        return result


def static_tensor_contract(t):
    from xdsl.dialects.builtin import TensorType

    if not isinstance(t, TensorType) or not 1 <= t.get_num_dims() <= 8:
        raise ValueError("deployable chains require ranked tensor arguments/results of rank 1..8")
    shape, dtype = tuple(t.get_shape()), str(t.element_type)
    if dtype not in _BYTES or any(type(d) is not int or d <= 0 for d in shape):
        raise ValueError("deployable chains require supported static positive tensor shapes and dtypes")
    elements = math.prod(shape)
    if elements > (1 << 63) - 1 or elements * _BYTES[dtype] > (1 << 63) - 1:
        raise ValueError("tensor size exceeds the checked LP64 descriptor ABI")
    return shape, dtype, elements * _BYTES[dtype]


def _array(attribute):
    return tuple(int(value) for value in attribute.get_values())


def _view(op, node, buffers) -> dict:
    if len(node.inputs) != 1 or len(node.outputs) != 1 or op.regions:
        raise ValueError(f"unsupported deployable driver/view ABI: {op.name}")
    src, out = buffers[node.inputs[0]], buffers[node.outputs[0]]
    if src.dtype != out.dtype:
        raise ValueError("a metadata view cannot change element dtype")
    if op.name in ("tensor.collapse_shape", "tensor.expand_shape", "tensor.cast"):
        if src.bytes != out.bytes or (op.name == "tensor.cast" and src.shape != out.shape):
            raise ValueError("only exact static contiguous shape/cast aliases are supported")
        return {"kind": "alias", "input": src.id, "output": out.id, "source_op": op.name}
    if op.name != "tensor.extract_slice":
        raise ValueError(f"unsupported deployable driver operation {op.name}; arithmetic never runs on controller")
    if _array(op.properties["operandSegmentSizes"]) != (1, 0, 0, 0):
        raise ValueError("dynamic slice operands are unsupported")
    offsets, sizes, strides = (
        _array(op.properties[name]) for name in ("static_offsets", "static_sizes", "static_strides")
    )
    rank = len(src.shape)
    if any(len(values) != rank for values in (offsets, sizes, strides)):
        raise ValueError("slice metadata must explicitly cover every source dimension")
    if any(
        o < 0 or n <= 0 or s <= 0 or o + (n - 1) * s >= extent
        for o, n, s, extent in zip(offsets, sizes, strides, src.shape, strict=True)
    ):
        raise ValueError("slice coordinates exceed the source tensor")
    # Rank reduction may discard only unit extents. Flattened lexicographic
    # packing is unchanged by which indistinguishable unit dimension is dropped.
    from itertools import combinations

    kept = any(
        tuple(sizes[i] for i in indices) == out.shape and all(sizes[i] == 1 for i in range(rank) if i not in indices)
        for indices in combinations(range(rank), len(out.shape))
    )
    if not kept or math.prod(sizes) * _BYTES[src.dtype] != out.bytes:
        raise ValueError("slice rank reduction or result shape is not exact")
    return {
        "kind": "pack_slice",
        "input": src.id,
        "output": out.id,
        "source_op": op.name,
        "offsets": offsets,
        "sizes": sizes,
        "strides": strides,
        "source_shape": src.shape,
        "element_bytes": _BYTES[src.dtype],
    }


def plan_deployable_chain(
    module,
    *,
    worker_slots: tuple[int, ...],
    worker_count: int,
    workspace_budget_bytes: int,
    alignment_bytes: int,
    entry: str = "forward",
    data_preparation: bool = False,
    packing_tile_shape: tuple[int, int] | None = None,
    packing_tile_budget_bytes: int | None = None,
) -> DeployableChain:
    """Admit a closed chain without changing the caller's module or arithmetic.

    Live scalar arithmetic, control flow, dynamic/unknown views, plain scalar
    ABIs and external implementations refuse. Every intermediate gets its own
    storage; alias lifetimes cannot be shortened by a speculative memory plan.
    """
    if type(worker_count) is not int or not 1 <= worker_count <= 8:
        raise ValueError("explicit bounded compute-worker count required")
    if (
        type(workspace_budget_bytes) is not int
        or workspace_budget_bytes < 1
        or type(alignment_bytes) is not int
        or alignment_bytes < 8
        or alignment_bytes & (alignment_bytes - 1)
    ):
        raise ValueError("positive workspace budget and power-of-two alignment >=8 required")
    if type(data_preparation) is not bool:
        raise ValueError("explicit boolean data preparation selection required")
    packing_policy = None
    if packing_tile_shape is not None or packing_tile_budget_bytes is not None:
        from merlin.llvmlower.deployable_packing import tile_policy

        packing_policy = tile_policy(packing_tile_shape, packing_tile_budget_bytes)
        if not data_preparation:
            raise ValueError("blocked byte packing requires explicit data_preparation=True")
    module.verify()
    source_hash = hashlib.sha256(checked_generic_text(module).encode()).hexdigest()
    preparation = None
    if data_preparation:
        from merlin.llvmlower.deployable_packing import canonicalize_constant_splats

        module = module.clone()
        count = canonicalize_constant_splats(module)
        module.verify()
        preparation = {"constant_splats_canonicalized": count, "controller_permutations": []}
        if packing_policy is not None:
            preparation["controller_packing_policy"] = packing_policy
    entries = [op for op in module.body.block.ops if op.name == "func.func" and op.sym_name.data == entry]
    if len(entries) != 1:
        raise ValueError("one explicit source entry required")
    if set(entries[0].attributes) - {"llvm.emit_c_interface"} or set(entries[0].properties) - {
        "sym_name",
        "function_type",
        "sym_visibility",
    }:
        raise ValueError("function-level contracts cannot be redistributed across outlined kernel arities")
    outlined = outline_dispatches(module.clone(), forward=entry)
    full = build_dispatch_program(outlined, entry=entry)
    program = prune_dead_nodes(full)
    if verify_program(program):
        raise ValueError("outlined dispatch program is not a closed DAG")
    driver = next(op for op in outlined.module.body.block.ops if op.name == "func.func" and op.sym_name.data == entry)
    operations = [op for op in driver.body.block.ops if op.name != "func.return"]
    by_node = {id(node): op for node, op in zip(full.nodes, operations, strict=True)}
    source_types = {}
    for buffer in full.buffers.values():
        if buffer.kind == "arg":
            source_types[buffer.id] = driver.body.block.args[buffer.arg_index].type
    for node in full.nodes:
        for bid, result in zip(node.outputs, by_node[id(node)].results, strict=True):
            source_types[bid] = result.type
    live = set(program.buffers) | {b.id for b in full.buffers.values() if b.kind == "arg"}
    specs = {}
    for bid, buffer in full.buffers.items():
        if bid not in live:
            continue
        shape, dtype, size = static_tensor_contract(source_types[bid])
        specs[bid] = ChainBuffer(bid, shape, dtype, size, buffer.arg_index, None)
    packing = {}
    if data_preparation:
        from merlin.llvmlower.deployable_packing import select_packing_policy, static_permutation

        for node in program.nodes:
            if node.kind == "dispatch":
                function = next(
                    op
                    for op in outlined.module.body.block.ops
                    if op.name == "func.func" and op.sym_name.data == node.op
                )
                view = static_permutation(function, node, specs)
                if view is not None:
                    view = select_packing_policy(view, packing_policy)
                    node.kind = "view"
                    packing[id(node)] = view
                    preparation["controller_permutations"].append(view)
    program = prune_dead_nodes(full)
    if verify_program(program):
        raise ValueError("outlined dispatch program is not a closed DAG")
    if (
        not isinstance(worker_slots, tuple)
        or len(worker_slots) != program.n_dispatches
        or not worker_slots
        or any(type(slot) is not int or not 0 <= slot < worker_count for slot in worker_slots)
    ):
        raise ValueError("one explicit legal compute-worker assignment required per dispatch")
    views, dispatches, aliases = [], [], set()
    for node in program.nodes:
        if node.kind == "view":
            view = packing.get(id(node))
            if view is None:
                view = _view(by_node[id(node)], node, specs)
            if view["kind"] == "alias":
                aliases.add(view["output"])
            views.append(view)
            continue
        function = next(
            op for op in outlined.module.body.block.ops if op.name == "func.func" and op.sym_name.data == node.op
        )
        for t in (*function.function_type.inputs.data, *function.function_type.outputs.data):
            static_tensor_contract(t)
        params = set(function.body.block.args)
        aliases_of_inputs = set(params)
        for op in function.walk():
            if op.name not in ("func.func", "func.return") and not op.name.startswith(
                ("linalg.", "tensor.", "arith.", "math.")
            ):
                raise ValueError(f"unsupported closed tensor kernel operation {op.name}")
            if any(value in aliases_of_inputs for value in getattr(op, "outputs", ())):
                raise ValueError("kernel parameter used as a writable output; immutable input ABI unproved")
            if op.name in ("tensor.insert", "tensor.insert_slice"):
                raise ValueError(f"unsupported kernel memory/effect operation {op.name}")
            if not op.name.startswith("linalg.") and any(value in aliases_of_inputs for value in op.operands):
                from xdsl.dialects.builtin import TensorType

                aliases_of_inputs.update(value for value in op.results if isinstance(value.type, TensorType))
        dispatches.append(
            {
                "symbol": node.op,
                "inputs": node.inputs,
                "outputs": node.outputs,
                "worker_slot": worker_slots[len(dispatches)],
                "views_before": views,
                "provenance": node.prov,
            }
        )
        views = []
    offset = 0
    buffers = []
    for spec in specs.values():
        if spec.arg_index is None and spec.id not in aliases:
            offset = (offset + alignment_bytes - 1) // alignment_bytes * alignment_bytes
            spec = ChainBuffer(spec.id, spec.shape, spec.dtype, spec.bytes, None, offset)
            offset += spec.bytes
        buffers.append(spec)
    size = (offset + alignment_bytes - 1) // alignment_bytes * alignment_bytes
    if size > workspace_budget_bytes:
        raise ValueError(f"outlined chain needs {size} unique-storage bytes, budget is {workspace_budget_bytes}")
    return DeployableChain(
        outlined,
        program,
        buffers,
        dispatches,
        views,
        size,
        alignment_bytes,
        worker_count,
        source_hash,
        preparation,
    )


def canonical_kernel(module, symbol: str):
    """Canonicalize only function/SSA spelling; preserve every semantic/unknown attribute."""
    from .kernel_backend import extract_kernel

    kernel = extract_kernel(module, symbol, entry="merlin_canonical_kernel")
    for op in kernel.walk():
        for value in op.results:
            value.name_hint = None
        for region in op.regions:
            for block in region.blocks:
                for value in block.args:
                    value.name_hint = None
    return kernel


def kernel_code_key(kernel, lowering_identity: dict) -> str:
    """Exact canonical IR plus actual lowering identity, never a shape/signature key."""
    return canonical_sha256({"ir": checked_generic_text(kernel), "lowering_identity": lowering_identity})
