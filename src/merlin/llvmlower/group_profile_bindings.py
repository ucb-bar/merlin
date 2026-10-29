"""Observe routed calls through structurally checked borrowed writer wrappers.

This establishes a profiling boundary and pointer-call prototype only. It grants
no writer effect, numerical, ownership, timing or device execution authority.
Unknown transformations retain the existing profiling diagnostic.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from xdsl.dialects import bufferization, func, memref
from xdsl.dialects.builtin import MemRefType, NoneAttr, StringAttr, TensorType, UnitAttr

from merlin.frontends.linalg_mlir import layout_to_buffer_type, parse_mlir_text
from merlin.runtime.whole_model_readback import ReadbackError


def _functions(module):
    functions = [op for op in module.body.block.ops if isinstance(op, func.FuncOp)]
    result = {op.sym_name.data: op for op in functions}
    if len(result) != len(functions):
        raise ReadbackError("profile source contains duplicate function symbols")
    return result


def _borrowed(wrapper, declaration, functions):
    """Recognize the existing fresh-writer wrapper shape, without claiming effects."""
    if (
        len(wrapper.body.blocks) != 1
        or wrapper.function_type != declaration.function_type
        or wrapper.sym_visibility != StringAttr("private")
    ):
        raise ReadbackError("profile requires a private, signature-preserving writer wrapper")
    block = wrapper.body.block
    operations = list(block.ops)
    types = tuple(declaration.function_type.inputs)
    if (
        not types
        or len(types) > 64
        or len(declaration.function_type.outputs) != 1
        or any(
            not isinstance(t, TensorType) or t.encoding != NoneAttr() or any(d <= 0 for d in t.get_shape())
            for t in types
        )
    ):
        raise ReadbackError("profile writer needs complete static tensor arguments")
    attrs = declaration.properties.get("arg_attrs")
    if (
        attrs is None
        or len(attrs) != len(types)
        or wrapper.properties.get("arg_attrs") != attrs
        or any(set(a.data) != {"bufferization.access"} for a in attrs)
    ):
        raise ReadbackError("profile writer argument access declarations changed")
    accesses = [getattr(a.data.get("bufferization.access"), "data", None) for a in attrs]
    if (
        any(a not in ("read", "write") for a in accesses)
        or len(operations) < len(types) + 3
        or any(op.regions for op in operations)
    ):
        raise ReadbackError("profile writer has unknown body or access structure")
    arguments = []
    for index, (typ, access, op) in enumerate(zip(types, accesses, operations)):
        if access == "read":
            if (
                type(op) not in (bufferization.ToBufferOp, layout_to_buffer_type())
                or tuple(op.operands) != (block.args[index],)
                or op.properties.get("read_only") != UnitAttr()
            ):
                raise ReadbackError("profile writer does not retain its original read-only input")
        elif type(op) is not memref.AllocOp or op.operands:
            raise ReadbackError("profile writer does not allocate its declared fresh output")
        if tuple(value.type for value in op.results) != (MemRefType(typ.element_type, typ.get_shape()),):
            raise ReadbackError("profile writer buffer types differ from the source")
        arguments.append(op.results[0])
    # Additional private byte scratch is passed as ordinary descriptor pointers.
    extra, (call, result, returned) = operations[len(types) : -3], operations[-3:]
    for op in extra:
        if (
            type(op) is not memref.AllocOp
            or op.operands
            or len(op.results) != 1
            or not isinstance(op.results[0].type, MemRefType)
            or str(op.results[0].type.element_type) != "i8"
            or len(op.results[0].type.get_shape()) != 1
            or op.results[0].type.get_shape()[0] <= 0
        ):
            raise ReadbackError("profile writer has unknown private workspace structure")
        arguments.append(op.results[0])
    if (
        type(call) is not func.CallOp
        or call.attributes
        or set(call.properties) != {"callee"}
        or call.results
        or tuple(call.arguments) != tuple(arguments)
        or type(result) is not bufferization.ToTensorOp
        or len(result.operands) != 1
        or type(returned) is not func.ReturnOp
        or tuple(returned.arguments) != tuple(result.results)
        or result.operands[0] not in arguments[: len(types)]
        or accesses[arguments.index(result.operands[0])] != "write"
    ):
        raise ReadbackError("profile writer has no single direct borrowed full-result boundary")
    borrowed = call.callee.root_reference.data
    external = functions.get(borrowed)
    if (
        external is None
        or external.body.blocks
        or tuple(external.function_type.inputs) != tuple(a.type for a in arguments)
        or external.function_type.outputs
        or external.attributes.get("llvm.emit_c_interface") != UnitAttr()
        or set(external.attributes) != {"llvm.emit_c_interface"}
        or set(external.properties) - {"sym_name", "sym_visibility", "function_type", "arg_attrs"}
        or not borrowed.isascii()
        or not borrowed.isidentifier()
    ):
        raise ReadbackError("profile borrowed writer has no declared void ranked C interface")
    return dict(
        borrowed_symbol=borrowed, buffers=len(arguments), argument_ranks=[len(a.type.get_shape()) for a in arguments]
    )


def derive_bindings(original, selected, routed):
    """Check all original function bodies, allowing only routed callee retargeting.

    No attribute or supplied report alone selects a source-to-writer mapping.
    Original control flow, types, operands and operations must remain identical.
    New private wrapper implementations are read separately for their real ABI.
    """
    original.verify()
    selected.verify()
    before, after = _functions(original), _functions(selected)
    original_other = [op for op in original.body.block.ops if not isinstance(op, func.FuncOp)]
    selected_other = [op for op in selected.body.block.ops if not isinstance(op, func.FuncOp)]
    if (
        original.attributes != selected.attributes
        or original.properties != selected.properties
        or len(original_other) != len(selected_other)
        or any(not a.is_structurally_equivalent(b) for a, b in zip(original_other, selected_other, strict=True))
    ):
        raise ReadbackError("profile transform changed module-level declarations or attributes")
    symbols = {row["symbol"] for row in routed if row.get("symbol")}
    replacements = {}
    for name, source in before.items():
        if not source.body.blocks:
            continue
        target = after.get(name)
        if target is None:
            raise ReadbackError("profile transform removed an original source function")
        restored = target.clone()
        source_ops, target_ops = list(source.walk()), list(restored.walk())
        if len(source_ops) != len(target_ops):
            raise ReadbackError("profile transform changed source operation/control-flow structure")
        for old, new in zip(source_ops, target_ops, strict=True):
            if type(old) is func.CallOp and old.callee.root_reference.data in symbols:
                if type(new) is not func.CallOp:
                    raise ReadbackError("profile transform removed a routed call")
                origin, wrapper = old.callee.root_reference.data, new.callee.root_reference.data
                if origin in replacements and replacements[origin] != wrapper:
                    raise ReadbackError("profile routed symbol has ambiguous writer replacements")
                replacements[origin] = wrapper
                new.properties["callee"] = old.callee
        if not source.is_structurally_equivalent(restored):
            raise ReadbackError("profile transform changed original source beyond routed call retargeting")
    result = {}
    for origin, wrapper in replacements.items():
        if origin == wrapper:
            continue
        declaration = before.get(origin)
        function = after.get(wrapper)
        if declaration is None or declaration.body.blocks or function is None or wrapper in before:
            raise ReadbackError("profile route has no original declaration and fresh wrapper")
        binding = _borrowed(function, declaration, after)
        borrowed = binding["borrowed_symbol"]
        uses = [op for op in selected.walk() if type(op) is func.CallOp and op.callee.root_reference.data == borrowed]
        if len(uses) != 1 or uses[0].parent_op() is not function:
            raise ReadbackError("profile borrowed entry is shared or called outside its source wrapper")
        if any(b["borrowed_symbol"] == borrowed for b in result.values()):
            raise ReadbackError("profile borrowed entry cannot identify a unique source route")
        result[origin] = dict(binding, wrapper_symbol=wrapper)
    allowed = {b[key] for b in result.values() for key in ("wrapper_symbol", "borrowed_symbol")}
    if set(after) - set(before) - allowed:
        raise ReadbackError("profile transform added an unrecognized function")
    for name, declaration in before.items():
        if not declaration.body.blocks and name not in result:
            if name not in after or not declaration.is_structurally_equivalent(after[name]):
                raise ReadbackError("profile transform changed an unrelated original declaration")
    return result


def read_bindings(receipt, prepared, sidecar):
    """Reopen original source, current prepared IR and immutable routing identity."""
    receipt = Path(receipt)
    if not receipt.is_file():
        return {}
    receipt_bytes = receipt.read_bytes()
    record = json.loads(receipt_bytes)
    snapshot = record.get("source_snapshot")
    if snapshot is None:
        raise ReadbackError("profile transformed route has no retained original source; rebuild")
    original, selected, routing = Path(snapshot).read_bytes(), Path(prepared).read_bytes(), Path(sidecar).read_bytes()
    if (
        hashlib.sha256(original).hexdigest() != record["source_sha256"]
        or hashlib.sha256(routing).hexdigest() != record["routing_sidecar_sha256"]
    ):
        raise ReadbackError("profile original source or routing identity changed; rebuild")
    result = derive_bindings(
        parse_mlir_text(original.decode()), parse_mlir_text(selected.decode()), json.loads(routing).get("routed") or []
    )
    if (
        receipt.read_bytes() != receipt_bytes
        or Path(snapshot).read_bytes() != original
        or Path(prepared).read_bytes() != selected
        or Path(sidecar).read_bytes() != routing
    ):
        raise ReadbackError("profile source or routing changed during binding; rebuild")
    return result


def prepare_profile(work, prepared, directory, routing, arguments, undefined, *, transformed):
    """Generate the existing profiler under the actual source/object boundary.

    The runtime driver supplies the observed undefined symbols and retains its
    own process, compilation and link authority. Returned flags select no target.
    """
    from .device_offload import SIDECAR_NAME

    rows = list(routing.get("routed") or [])
    missing = any(
        row.get("symbol") not in undefined and f"_mlir_ciface_{row.get('symbol')}" not in undefined for row in rows
    )
    bindings = (
        read_bindings(work / "device_host_abi" / "post_offload_transform.json", prepared, work / SIDECAR_NAME)
        if missing and transformed
        else {}
    )
    return emit_profile(directory, routing, arguments, undefined, bindings=bindings,
                        source=prepared, model_object=work / "model.o", source_kind="prepared_mlir")


def emit_profile(directory, routing, arguments, undefined, *, bindings, source, model_object,
                 source_kind, observations=()):
    """Emit the existing counter wrappers from an observed source/object ABI.

    Completed-object packaging may observe already compiled pointer bridges.
    This supplies no effects or implementation permission, and labels its
    source independently from the ordinary prepared-MLIR observation.
    """
    from merlin.common.digest import sha256_file
    from merlin.runtime.whole_model_readback import callees, group_names, render_group_profile, wrap_flags

    rows = list(routing.get("routed") or [])
    selected = callees(rows, arguments.get("signatures") or {}, undefined, arguments.get("call_buffers"), bindings)
    # A completed object proves its referenced callees and their ABI, not the
    # dynamic order of source routes. Mixed producer routes may share a callee
    # while their sidecar roster has a different order from actual calls.
    names = group_names(rows) if source_kind == "prepared_mlir" else []
    (directory / "group_profile_bindings.json").write_text(
        json.dumps(
            dict(
                schema="merlin.group_profile_bindings.v1",
                bindings=bindings,
                source_observation=dict(kind=source_kind, path=str(Path(source).resolve()),
                                        sha256=sha256_file(source)),
                **({"prepared_sha256": sha256_file(source)} if source_kind == "prepared_mlir" else {}),
                model_object_sha256=sha256_file(model_object),
                undefined_symbols=sorted(undefined),
                observations=list(observations),
                labels="source_route_order" if names else "observed_callee_only",
                boundary="inclusive selected external ABI call; host gaps reported separately",
                scope="source/ABI observation plus actual undefined symbols; no numerical or timing authority",
            ),
            indent=2,
        )
        + "\n"
    )
    source = directory / "merlin_group_profile.c"
    source.write_text(render_group_profile(selected, names), encoding="utf-8")
    return source, wrap_flags(selected)
