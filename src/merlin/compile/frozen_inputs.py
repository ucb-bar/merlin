"""Explicit byte-sealed immutable arguments for a deployable outlined chain.

This opt-in capability removes selected arguments from the generated caller ABI.
It evaluates only admitted wrapping integer kernels, using the selected native
host compiler and an independent typed array interpreter. Floating arithmetic,
unknown attributes/effects and unsupported integer bodies remain runtime work.
Original source/kernel identities and logical argument ordinals remain recorded.
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from merlin.common.digest import sha256_file


@dataclass(frozen=True)
class FrozenInput:
    """One dense little-endian source argument, explicitly immutable per build."""

    arg_index: int
    dtype: str
    shape: tuple[int, ...]
    file: Path
    sha256: str


@dataclass(frozen=True)
class FrozenInputs:
    """Source-bound permission and conservative offline/storage budgets."""

    source_ir_sha256: str
    inputs: tuple[FrozenInput, ...]
    offline_budget_bytes: int
    published_budget_bytes: int


_DTYPES = {"i8": "<i1", "i16": "<i2", "i32": "<i4", "i64": "<i8"}
_PROVENANCE = {
    "prov.region_id",
    "prov.family",
    "prov._pattern_hint",
    "prov.op",
    "prov.source_node_ids",
    "prov.origin_node_ids",
    "prov.trace_role",
    "prov.aten",
    "prov.orig_dtype",
    "prov.module",
    "prov.fqn",
    "prov.weights_file",
    "prov.level",
    "prov.quantization",
}
_PROPERTIES = {
    "builtin.module": set(),
    "func.func": {"sym_name", "function_type", "sym_visibility"},
    "func.return": set(),
    "tensor.empty": set(),
    "arith.constant": {"value"},
    "arith.extsi": set(),
    "arith.addi": {"overflowFlags"},
    "arith.muli": {"overflowFlags"},
    "linalg.fill": {"operandSegmentSizes"},
    "linalg.yield": set(),
    "linalg.generic": {"indexing_maps", "iterator_types", "operandSegmentSizes"},
    "linalg.reduce": {"dimensions"},
}


def _dtype(typ):
    from xdsl.dialects.builtin import IntegerType, NoneAttr, TensorType

    if isinstance(typ, TensorType):
        if not isinstance(typ.encoding, NoneAttr):
            raise ValueError("encoded frozen tensor is unsupported")
        typ = typ.element_type
    if not isinstance(typ, IntegerType) or str(typ) not in _DTYPES:
        raise ValueError("offline arithmetic admits signless i8/i16/i32/i64 only")
    return _DTYPES[str(typ)]


def _admit(kernel):
    """Deny unknown contracts, not just unknown operation spellings."""
    from xdsl.dialects.builtin import IntegerAttr, TensorType

    for op in kernel.walk():
        if op.name not in _PROPERTIES or set(op.properties) - _PROPERTIES.get(op.name, set()):
            raise ValueError("unsupported offline operation/property: " + op.name)
        allowed = _PROVENANCE | ({"llvm.emit_c_interface"} if op.name == "func.func" else set())
        if set(op.attributes) - allowed:
            raise ValueError("unsupported offline attribute: " + op.name)
        if (
            op.name in {"arith.addi", "arith.muli"}
            and str(op.properties.get("overflowFlags")) != "#arith.overflow<none>"
        ):
            raise ValueError("offline arithmetic requires explicit wrapping overflow flags")
        for value in (*op.operands, *op.results):
            _dtype(value.type)
            if isinstance(value.type, TensorType) and any(d <= 0 for d in value.type.get_shape()):
                raise ValueError("offline tensor shape must be static and positive")
        if op.name == "arith.constant" and not isinstance(op.properties["value"], IntegerAttr):
            raise ValueError("offline constants must be scalar integer literals")
    function = next(op for op in kernel.body.block.ops if op.name == "func.func")
    if len(function.body.blocks) != 1:
        raise ValueError("offline kernel requires one straight-line block")
    return function


def _scalar(block, values, *, stop=None):
    import numpy as np

    env = dict(zip(block.args, values, strict=True))
    for op in block.ops:
        if op is stop:
            break
        if op.name == "linalg.yield":
            if len(op.operands) != 1:
                raise ValueError("one offline yield required")
            return env[op.operands[0]]
        if op.name == "arith.constant":
            value = np.asarray(op.properties["value"].value.data, dtype=_dtype(op.results[0].type))
        elif op.name == "arith.extsi":
            if env[op.operands[0]] is None:
                raise ValueError("offline kernel reads uninitialized output")
            value = np.asarray(env[op.operands[0]]).astype(_dtype(op.results[0].type))
        elif op.name in {"arith.addi", "arith.muli"}:
            a, b = (env[v] for v in op.operands)
            if a is None or b is None:
                raise ValueError("offline kernel reads uninitialized output")
            fn = np.add if op.name == "arith.addi" else np.multiply
            value = fn(a, b, dtype=_dtype(op.results[0].type))
        else:
            raise ValueError("unsupported offline scalar body")
        env[op.results[0]] = value
    return env


def _projected_dimensions(op):
    from xdsl.ir.affine import AffineDimExpr

    maps = [a.data for a in op.indexing_maps]
    kinds = [a.data.value for a in op.iterator_types]
    if any(k not in {"parallel", "reduction"} for k in kinds) or len(op.outputs) != 1:
        raise ValueError("unsupported offline generic iterators/results")
    dims, extents = [], [None] * len(kinds)
    for amap, operand in zip(maps, op.operands, strict=True):
        if (
            amap.num_symbols
            or amap.num_dims != len(kinds)
            or any(not isinstance(e, AffineDimExpr) for e in amap.results)
        ):
            raise ValueError("offline maps require dimension projections/permutations")
        positions = [e.position for e in amap.results]
        if len(set(positions)) != len(positions):
            raise ValueError("offline map repeats a dimension")
        for pos, size in zip(positions, operand.type.get_shape(), strict=True):
            if extents[pos] not in (None, size):
                raise ValueError("offline map dimensions disagree")
            extents[pos] = size
        dims.append(positions)
    if any(n is None for n in extents):
        raise ValueError("offline generic has unbounded iteration dimension")
    return dims, extents, kinds


def _generic(op, operands):
    import numpy as np

    dims, extents, kinds = _projected_dimensions(op)
    projected = []
    for array, positions in zip(operands, dims, strict=True):
        if array is None:
            projected.append(None)
            continue
        order = sorted(range(len(positions)), key=positions.__getitem__)
        shape = [1] * len(extents)
        for pos in positions:
            shape[pos] = extents[pos]
        projected.append(np.transpose(array, order).reshape(shape))
    block = op.body.block
    reductions = tuple(i for i, kind in enumerate(kinds) if kind == "reduction")
    parallel = [i for i, kind in enumerate(kinds) if kind == "parallel"]
    if reductions:
        if dims[-1] != parallel:
            raise ValueError("offline reduction output must retain parallel dimensions in order")
        add = block.last_op.operands[0].owner
        acc = block.args[-1]
        if add.name != "arith.addi" or acc not in add.operands or sum(1 for _ in acc.uses) != 1:
            raise ValueError("offline reduction must add one independent term to its accumulator")
        term = next((v for v in add.operands if v is not acc), None)
        if term is None or operands[-1] is None:
            raise ValueError("offline reduction requires initialized accumulator")
        env = _scalar(block, projected, stop=add)
        value = np.broadcast_to(env[term], extents).sum(axis=reductions, dtype=_dtype(op.results[0].type))
        return np.add(value, operands[-1], dtype=_dtype(op.results[0].type))
    value = _scalar(block, projected)
    if value is None or sorted(dims[-1]) != list(range(len(extents))):
        raise ValueError("offline parallel output must cover the entire domain")
    value = np.broadcast_to(value, extents)
    return np.ascontiguousarray(np.transpose(value, dims[-1]))


def _reference(function, inputs):
    """Independent wrapping integer interpreter for the admitted closed subset."""
    import numpy as np

    env = dict(zip(function.body.block.args, inputs, strict=True))
    for op in function.body.block.ops:
        values = [env[v] for v in op.operands]
        if op.name == "func.return":
            if any(v is None for v in values):
                raise ValueError("offline result is uninitialized")
            return [np.ascontiguousarray(v) for v in values]
        if op.name == "tensor.empty":
            result = None
        elif op.name == "arith.constant":
            result = np.asarray(op.properties["value"].value.data, dtype=_dtype(op.results[0].type))
        elif op.name == "linalg.fill":
            block = op.regions[0].block
            if (
                len(block.ops) != 1
                or block.last_op.name != "linalg.yield"
                or tuple(block.last_op.operands) != (block.args[0],)
            ):
                raise ValueError("offline fill requires identity scalar body")
            result = np.full(op.results[0].type.get_shape(), values[0], dtype=_dtype(op.results[0].type))
        elif op.name == "linalg.generic":
            result = _generic(op, values)
        elif op.name == "linalg.reduce":
            block = op.regions[0].block
            ops = list(block.ops)
            if (
                len(values) != 2
                or len(ops) != 2
                or ops[0].name != "arith.addi"
                or set(ops[0].operands) != set(block.args)
                or tuple(ops[1].operands) != (ops[0].results[0],)
            ):
                raise ValueError("offline reduce requires exact wrapping addition")
            axes = tuple(op.dimensions.get_values())
            result = np.add(
                np.sum(values[0], axis=axes, dtype=_dtype(op.results[0].type)),
                values[1],
                dtype=_dtype(op.results[0].type),
            )
        else:
            raise ValueError("unsupported offline tensor body")
        if len(op.results) != 1:
            raise ValueError("one offline operation result required")
        env[op.results[0]] = result
    raise ValueError("offline function has no return")


def _evaluate(kernel, arrays, work, budget):
    import shutil

    import numpy as np
    from xdsl.dialects.builtin import TensorType

    from merlin.llvmlower import codegen, toolchain
    from merlin.llvmlower.abi import HostModel, PrivateHostImagePolicy
    from merlin.llvmlower.deployable_ir import checked_generic_text
    from merlin.llvmlower.lower import lower_model

    function = _admit(kernel)
    # Conservative bound includes independent reference intermediates, guarded
    # compiler inputs/outputs and domain-sized scalar temporaries. Native compiler
    # process memory and code generation time are separately unbounded by this.
    estimate = sum(a.nbytes for a in arrays)
    domain = max([math.prod(a.shape) for a in arrays] + [1])
    for op in function.walk():
        if op.name == "linalg.generic":
            _, extents, _ = _projected_dimensions(op)
            domain = max(domain, math.prod(extents))
    for op in function.walk():
        for value in op.results:
            if isinstance(value.type, TensorType):
                estimate += math.prod(value.type.get_shape()) * np.dtype(_dtype(value.type)).itemsize
            elif op.name.startswith("arith."):
                estimate += domain * np.dtype(_dtype(value.type)).itemsize
    estimate = estimate * 4 + 128 * (len(arrays) + len(function.function_type.outputs))
    if estimate > budget:
        raise ValueError(f"offline intermediate budget exceeded: bound {estimate}, budget {budget}")
    expected = _reference(function, arrays)
    result = lower_model(checked_generic_text(kernel), work, textual=True, targets=("host",))
    model = HostModel.load(
        str(result.host_so), name="merlin_canonical_kernel", image_policy=PrivateHostImagePolicy(work.resolve())
    )
    if sha256_file(result.host_so) != model.image_sha256:
        raise ValueError("retained canonical host image differs from the actually loaded private image")
    storage, buffers, args = [], [], []
    for array in [*arrays, *expected]:
        raw = np.full(array.nbytes + 128, 0xA7, dtype=np.uint8)
        view = raw[64:-64].view(array.dtype).reshape(array.shape)
        if len(buffers) < len(arrays):
            view[...] = array
        else:
            raw[64:-64] = 0x5B
        storage.append(raw)
        buffers.append(view)
        args.append((view.ctypes.data, view.shape))
    snapshots = [a.tobytes() for a in buffers[: len(arrays)]]
    model(args)
    for index, (raw, view) in enumerate(zip(storage, buffers, strict=True)):
        if not np.all(raw[:64] == 0xA7) or not np.all(raw[-64:] == 0xA7):
            raise ValueError("offline native evaluation corrupted guards")
        wanted = snapshots[index] if index < len(arrays) else expected[index - len(arrays)].tobytes()
        if view.tobytes() != wanted:
            raise ValueError("offline native evaluation differs from complete wrapping reference/input")
    artifacts = [
        work / name
        for name in (
            "model.mlir",
            "run_lowering.py",
            "lowering_recipe.json",
            "model.upstream.mlir",
            "model.ll",
            "model_host.o",
            "mlir_runtime_host.o",
            "model_host.so",
        )
    ]
    cc = shutil.which("cc")
    if cc is None:
        raise ValueError("actual host runtime compiler is unavailable")
    tools = [toolchain.clang(), Path(cc).resolve(), Path(codegen.__file__), codegen.mlir_runtime_c()]
    selected_llc = toolchain.host_llc()
    if selected_llc is not None:
        tools.append(selected_llc)
    return expected, {
        "compiler": str(toolchain.clang()),
        "compiler_sha256": sha256_file(toolchain.clang()),
        "host_image_sha256": model.image_sha256,
        "retained_host_image": str(result.host_so),
        "host_image_policy": {
            "kind": "PrivateHostImagePolicy",
            "directory": str(work.resolve()),
            "loader_mode": "RTLD_LOCAL",
            "scalar_return": None,
        },
        "entry": "_mlir_ciface_merlin_canonical_kernel",
        "input_sha256": {str(path): sha256_file(path) for path in [*artifacts, *tools]},
        "inputs": [
            {
                "shape": list(a.shape),
                "dtype": a.dtype.str,
                "bytes": a.nbytes,
                "sha256": hashlib.sha256(a.tobytes()).hexdigest(),
            }
            for a in arrays
        ],
        "accounted_intermediate_bound_bytes": estimate,
        "outputs": [{"bytes": a.nbytes, "sha256": hashlib.sha256(a.tobytes()).hexdigest()} for a in expected],
    }


def _view(view, value):
    import numpy as np

    if view["kind"] == "alias":
        return value  # The caller reshapes to the independently checked result contract.
    if view["kind"] == "pack_permutation":
        flat = value.reshape(-1)
        indices = np.zeros(view["output_shape"], dtype=np.int64)
        for axis, stride in enumerate(view["source_strides"]):
            shape = [1] * len(view["output_shape"])
            shape[axis] = view["output_shape"][axis]
            indices += np.arange(shape[axis], dtype=np.int64).reshape(shape) * stride
        return np.ascontiguousarray(flat[indices])
    slices = tuple(
        slice(o, o + n * s, s) for o, n, s in zip(view["offsets"], view["sizes"], view["strides"], strict=True)
    )
    return np.ascontiguousarray(value[slices])


def specialize(plan, permission: FrozenInputs, output: Path):
    """Return a new executable plan and sealed live-blob bytes; never mutate source."""
    import numpy as np

    from merlin.llvmlower.deployable_ir import checked_generic_text
    from merlin.llvmlower.deployable_outline import canonical_kernel

    if not isinstance(permission, FrozenInputs) or permission.source_ir_sha256 != plan.source_ir_sha256:
        raise ValueError("frozen inputs require the exact original source IR identity")
    if (
        not isinstance(permission.inputs, tuple)
        or not permission.inputs
        or any(
            type(b) is not int or b <= 0 for b in (permission.offline_budget_bytes, permission.published_budget_bytes)
        )
    ):
        raise ValueError("explicit frozen inputs and positive offline/published budgets required")
    specs = {b.id: b for b in plan.buffers}
    args = {b.arg_index: b for b in plan.buffers if b.arg_index is not None}
    entry = next(
        op
        for op in plan.outlined.module.body.block.ops
        if op.name == "func.func" and op.sym_name.data == plan.program.entry
    )
    known, pins, roots, rows, retained, pending, offline_views = {}, {}, {}, [], [], [], []
    start = time.monotonic()
    for item in permission.inputs:
        if (
            not isinstance(item, FrozenInput)
            or type(item.arg_index) is not int
            or item.arg_index not in args
            or item.arg_index in roots
        ):
            raise ValueError("unique existing source argument ordinals required")
        spec = args[item.arg_index]
        _dtype(entry.body.block.args[item.arg_index].type)
        if item.dtype not in _DTYPES or item.shape != spec.shape or item.dtype != spec.dtype:
            raise ValueError("frozen input type/shape differs from source ABI")
        if (
            not isinstance(item.file, Path)
            or not item.file.is_absolute()
            or not item.file.is_file()
            or sha256_file(item.file) != item.sha256
            or item.file.stat().st_size != spec.bytes
        ):
            raise ValueError("frozen input bytes do not match sealed source argument")
        if sum(v.nbytes for v in known.values()) + spec.bytes > permission.offline_budget_bytes:
            raise ValueError("frozen root storage exceeds offline budget")
        raw = item.file.read_bytes()
        if len(raw) != spec.bytes or hashlib.sha256(raw).hexdigest() != item.sha256:
            raise ValueError("frozen input changed while reading sealed bytes")
        known[spec.id] = np.frombuffer(raw, dtype=_DTYPES[spec.dtype]).reshape(spec.shape)
        roots[item.arg_index] = {**asdict(item), "file": str(item.file), "buffer": spec.id, "bytes": spec.bytes}
        pins[str(item.file)] = item.sha256
    output.mkdir(parents=True, exist_ok=False)

    def visit_views(views):
        for view in views:
            if view["input"] in known:
                spec = specs[view["output"]]
                bound = sum(a.nbytes for a in known.values()) + spec.bytes * 12
                if bound > permission.offline_budget_bytes:
                    raise ValueError("offline view intermediate budget exceeded")
                known[spec.id] = _view(view, known[view["input"]]).reshape(spec.shape)
                offline_views.append(
                    {**view, "bytes": spec.bytes, "sha256": hashlib.sha256(known[spec.id].tobytes()).hexdigest()}
                )
            else:
                pending.append(view)

    for index, step in enumerate(plan.dispatches):
        visit_views(step["views_before"])
        row = {"source_dispatch_index": index, "symbol": step["symbol"], "folded": False}
        if step["inputs"] and all(b in known for b in step["inputs"]):
            kernel = canonical_kernel(plan.outlined.module, step["symbol"])
            try:
                _admit(kernel)
            except ValueError as exc:
                row["reason"] = str(exc)
            else:
                live = sum(a.nbytes for a in known.values())
                values, proof = _evaluate(
                    kernel,
                    [known[b] for b in step["inputs"]],
                    output / f"kernel_{index}",
                    permission.offline_budget_bytes - live,
                )
                for bid, value in zip(step["outputs"], values, strict=True):
                    if value.shape != specs[bid].shape or value.dtype != np.dtype(_DTYPES[specs[bid].dtype]):
                        raise ValueError("offline output ABI differs from original source")
                    known[bid] = value
                row.update(
                    folded=True,
                    canonical_ir_sha256=hashlib.sha256(checked_generic_text(kernel).encode()).hexdigest(),
                    proof=proof,
                )
        if not row["folded"]:
            row["runtime_dispatch_index"] = len(retained)
            retained.append({**step, "source_dispatch_index": index, "views_before": list(pending)})
            pending.clear()
        rows.append(row)
    visit_views(plan.trailing_views)
    if not retained:
        raise ValueError("a deployable frozen chain must retain at least one runtime arithmetic dispatch")
    needed = set(plan.program.results)
    for step in retained:
        needed.update(step["inputs"] + step["outputs"])
        for view in step["views_before"]:
            needed.update((view["input"], view["output"]))
    for view in pending:
        needed.update((view["input"], view["output"]))
    external = [b for b in args.values() if b.arg_index not in roots]
    needed.update(b.id for b in external)
    blob, entries, offset, buffers = bytearray(), {}, 0, []
    aliases = {v["output"] for s in retained for v in s["views_before"] if v["kind"] == "alias"} | {
        v["output"] for v in pending if v["kind"] == "alias"
    }
    for spec in plan.buffers:
        if spec.id not in needed:
            continue
        if spec.id in known:
            padding = (-len(blob)) % plan.alignment_bytes
            if len(blob) + padding + spec.bytes > permission.published_budget_bytes:
                raise ValueError("published frozen blob exceeds explicit storage budget")
            blob.extend(bytes(padding))
            data = known[spec.id].tobytes()
            if len(data) != spec.bytes:
                raise ValueError("frozen blob buffer differs from source byte extent")
            entries[spec.id] = {"offset": len(blob), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            blob.extend(data)
            spec = replace(spec, workspace_offset=None)
        elif spec.arg_index is None and spec.id not in aliases:
            offset = (offset + plan.alignment_bytes - 1) // plan.alignment_bytes * plan.alignment_bytes
            spec = replace(spec, workspace_offset=offset)
            offset += spec.bytes
        else:
            spec = replace(spec, workspace_offset=None)
        buffers.append(spec)
    size = (offset + plan.alignment_bytes - 1) // plan.alignment_bytes * plan.alignment_bytes
    if not blob:
        raise ValueError("no live readonly data survives frozen specialization")
    for path, digest in pins.items():
        if sha256_file(path) != digest:
            raise ValueError("frozen input changed during offline evaluation")
    record = {
        "schema": "merlin.frozen_inputs.v1",
        "source_ir_sha256": plan.source_ir_sha256,
        "roots": list(roots.values()),
        "external_input_source_indices": [b.arg_index for b in external],
        "readonly_buffers": entries,
        "published_blob_bytes": len(blob),
        "published_blob_sha256": hashlib.sha256(blob).hexdigest(),
        "offline_budget_bytes": permission.offline_budget_bytes,
        "published_budget_bytes": permission.published_budget_bytes,
        "dispatch_mapping": rows,
        "offline_views": offline_views,
        "offline_host_seconds": time.monotonic() - start,
        "input_sha256": pins,
        "original_program_preserved": True,
        "runtime_arithmetic_on_controller": False,
        "compiler_process_memory_bounded": False,
        "full_model_admission": False,
    }
    return replace(
        plan,
        buffers=buffers,
        dispatches=retained,
        trailing_views=list(pending),
        workspace_bytes=size,
        frozen_inputs=record,
    ), bytes(blob)
