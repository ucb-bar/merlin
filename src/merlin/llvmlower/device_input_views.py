"""Expose only an original logical shim's dense, nonzero-offset read ABI.

Fixed row strides remain required. Upstream bufferization therefore retains
copies for noncontiguous inputs. Output layouts, alias guards, call effects and
the original numerical/target obligations are unchanged.
"""

from __future__ import annotations

import io
import json
from math import prod
from pathlib import Path

from merlin.common.digest import sha256_file


def _dense_tensor(value, active=None):
    """Recognize original dense tensors and exact contiguous static view chains.

    Unknown calls, permutations and strided slices are barriers. This proves
    only the source preparation shape; it does not grant memory effects.
    """
    from math import prod

    from xdsl.dialects import arith, builtin, func, tensor
    from xdsl.ir import Block

    typ = value.type
    active = set() if active is None else active
    if (value in active or len(active) > 64 or not isinstance(typ, builtin.TensorType)
            or typ.encoding != builtin.NoneAttr() or any(n <= 0 for n in typ.get_shape())):
        return False
    active = active | {value}
    owner = value.owner
    if isinstance(owner, Block):
        function = owner.parent_op()
        if not isinstance(function, func.FuncOp) or owner is not function.body.block:
            return False
        attrs = function.arg_attrs
        return attrs is None or "bufferization.buffer_layout" not in attrs.data[owner.args.index(value)].data
    if isinstance(owner, arith.ConstantOp | tensor.EmptyOp):
        return True
    if isinstance(owner, tensor.CollapseShapeOp | tensor.ExpandShapeOp | tensor.CastOp):
        return len(owner.operands) == 1 and _dense_tensor(owner.operands[0], active)
    if isinstance(owner, tensor.ExtractSliceOp):
        source = owner.source.type
        if not _dense_tensor(owner.source, active) or len(owner.operands) != 1:
            return False
        offsets = tuple(owner.static_offsets.get_values())
        sizes = tuple(owner.static_sizes.get_values())
        strides = tuple(owner.static_strides.get_values())
        shape = source.get_shape()
        if not len(offsets) == len(sizes) == len(strides) == len(shape):
            return False
        for i, (offset, size, stride, extent) in enumerate(zip(offsets, sizes, strides, shape, strict=True)):
            if (offset < 0 or size <= 0 or stride <= 0 or offset + (size - 1) * stride >= extent
                    or size > 1 and prod(shape[i + 1:]) * stride != prod(sizes[i + 1:])):
                return False
        return prod(sizes) == prod(typ.get_shape())
    # Tensor-only destination-style operations inherit their fresh dense init.
    # Shared/unknown destinations and opaque producers keep the old route.
    if owner.name.startswith("linalg.") and hasattr(owner, "outputs"):
        results = tuple(owner.results)
        outputs = tuple(owner.outputs)
        if value in results and len(results) == len(outputs):
            init = outputs[results.index(value)]
            return isinstance(init.owner, tensor.EmptyOp) and init.type == typ
    return False


def validate_selection(enabled, *, backend, device, unsupported=()):
    """Keep the opt-in in the ordinary scalar logical route; infer no composition."""
    if type(enabled) is not bool:
        raise ValueError("dense input views require an explicit Boolean choice")
    if enabled and (
        backend != "scalar" or device is None or getattr(device, "granularity", None) != "group"
        or any(unsupported)
        or any(getattr(device, name, None) for name in (
            "connected_spatial", "catalog_manifest",
            "catalog_builder", "post_offload_transform", "prepared_transform",
        ))
    ):
        raise ValueError("dense input views require the ordinary scalar logical group route")
    sibling = getattr(device, "sibling_source", None)
    if enabled and sibling is not None:
        from .head_source_route import HeadBatchSourceRoute
        from .sibling_source_route import SiblingSourceRoute

        if type(sibling) not in (SiblingSourceRoute, HeadBatchSourceRoute):
            raise ValueError("dense sibling inputs require original typed source custody")
        sibling.record()


def rewrite_readonly_dense_offsets(module, selected):
    """Bind exact original declaration positions; accept no writable argument.

    ``selected`` comes from the original logical shim binder below, not a
    candidate declaration. This structural helper grants no external ABI.
    """
    from xdsl.dialects import builtin, func

    module.verify()
    functions = {}
    for op in module.body.block.ops:
        if isinstance(op, func.FuncOp):
            if op.sym_name.data in functions:
                raise ValueError("dense input views require unambiguous original symbols")
            functions[op.sym_name.data] = op
    plans = []
    for symbol, indices in selected.items():
        declaration = functions.get(symbol)
        if (declaration is None or declaration.body.blocks or declaration.arg_attrs is None
                or type(indices) is not tuple or len(set(indices)) != len(indices)):
            raise ValueError("dense input views require exact original declarations and positions")
        attrs = list(declaration.arg_attrs)
        types = tuple(declaration.function_type.inputs)
        if len(attrs) != len(types):
            raise ValueError("dense input views lost the original positional access contract")
        for index in indices:
            if type(index) is not int or not 0 <= index < len(types):
                raise ValueError("dense input view position is outside its original declaration")
            typ = types[index]
            if (attrs[index].data != {"bufferization.access": builtin.StringAttr("read")}
                    or not isinstance(typ, builtin.TensorType) or typ.encoding != builtin.NoneAttr()
                    or len(typ.get_shape()) not in (2, 3) or any(n <= 0 for n in typ.get_shape())):
                raise ValueError("dense input views require static unencoded readonly matrix or batched tensors")
            shape = typ.get_shape()
            attrs[index] = builtin.DictionaryAttr({
                **attrs[index].data,
                "bufferization.buffer_layout": builtin.StridedLayoutAttr(
                    [prod(shape[axis + 1:]) for axis in range(len(shape))], None),
            })
        plans.append((declaration, attrs))
    for declaration, attrs in plans:
        declaration.properties["arg_attrs"] = builtin.ArrayAttr(attrs)
    module.verify()
    return {symbol: list(indices) for symbol, indices in selected.items()}


def prepare(source, sidecar, built, work, compilation, *, sibling_source=None):
    """Join actual compiled shim custody before selecting any declaration view.

    Re-emission must reproduce the complete logical C shim byte-for-byte from
    the actual emitted interfaces and original host-call buffers. Unsupported
    symbols/ranks retain their identity-layout route. There is no guessed
    support from kernel names, target labels, tensor contents or execution.
    """
    from xdsl.printer import Printer

    from merlin.common import invocation_record as I
    from merlin.frontends.linalg_mlir import parse_mlir_text

    from . import device_shim as S
    from . import whole_program as W
    from .device_build import DeviceBuild

    if type(built) is not DeviceBuild or not built.ok or built.shim_unit is None or built.shim_invocation is None:
        raise ValueError("dense input views require actual original logical device objects")
    source, sidecar, work = Path(source), Path(sidecar), Path(work)
    unit = built.shim_unit
    invocation = built.shim_invocation
    invocation_path = Path(invocation["path"])
    if sha256_file(invocation_path) != invocation["sha256"]:
        raise ValueError("dense input view shim invocation changed")
    observed = I.verify(invocation_path)
    shim = built.shim_object.with_suffix(".c")
    if (observed.get("returncode") != 0 or shim.read_text() != unit.text
            or not any(Path(pin["path"]) == shim and pin["sha256"] == sha256_file(shim)
                       for pin in observed["inputs"])
            or not any(Path(pin["path"]) == built.shim_object
                       and pin["sha256"] == sha256_file(built.shim_object) for pin in observed["outputs"])):
        raise ValueError("dense input views lost actual compiled logical shim custody")
    data = json.loads(sidecar.read_bytes())
    if data.get("device") != built.device or data.get("granularity") != "group":
        raise ValueError("dense input views require the original complete group call sidecar")
    groups, selected, paths = {}, {}, [source, sidecar, shim, built.shim_object, invocation_path, Path(S.__file__),
                                     Path(W.__file__), Path(__file__)]
    for symbol in built.kernels:
        interface = built.shim_object.parent / f"{symbol}.iface.mlir"
        entry = data.get("entries", {}).get(symbol)
        buffers, reason = W.source_bound_group_buffers(entry, interface.read_text(), target=built.device)
        call = data.get("call_buffers", {}).get(symbol)
        access = data.get("arg_access", {}).get(symbol)
        if buffers is None or call is None or access is None or len(call) != len(access):
            raise ValueError("dense input views lost original logical interface: " + reason)
        groups[symbol] = (buffers, call)
        by_role = {buf.role: buf for buf in buffers}
        positions = tuple(i for i, (buf, mode) in enumerate(zip(call, access, strict=True))
                          if mode == "read" and len(buf["shape"]) in (2, 3)
                          and by_role[buf["role"]].kind != "output")
        if positions:
            selected[symbol] = positions
        paths.append(interface)
    reopened = S.emit_logical_translation_unit(built.device, groups, kernel_symbol_for=built.kernels.get,
                                              descriptor_results=bool(unit.result_identities))
    if sibling_source is not None:
        from .sibling_source_route import with_borrowed_bridges

        original_selection, sibling_host = sibling_source
        reopened = with_borrowed_bridges(reopened, original_selection, sibling_host)
    if reopened != unit or reopened.skipped:
        raise ValueError("dense input views differ from the actual original logical shim emitter")
    pins = [(path.resolve(), sha256_file(path)) for path in paths]
    module = parse_mlir_text(source.read_text())
    declarations = {op.sym_name.data: op for op in module.body.block.ops if op.name == "func.func"}
    sibling_arguments, retained_sibling_arguments = {}, {}
    if sibling_source is not None:
        from . import sibling_input_views

        sibling_arguments, retained_sibling_arguments, component_symbols = sibling_input_views.bind(
            module, sibling_host, original_selection, groups)
        groups = {symbol: value for symbol, value in groups.items() if symbol not in component_symbols}
        selected = {symbol: value for symbol, value in selected.items() if symbol not in component_symbols}
        from merlin.frontends import linalg_mlir

        for owner in (sibling_input_views, linalg_mlir):
            helper = Path(owner.__file__).resolve()
            paths.append(helper)
            pins.append((helper, sha256_file(helper)))
    from xdsl.dialects import func

    calls = {}
    for operation in module.walk():
        if isinstance(operation, func.CallOp):
            calls.setdefault(operation.callee.root_reference.data, []).append(operation)
    for symbol, (_, call) in groups.items():
        declaration = declarations.get(symbol)
        if declaration is None or len(declaration.function_type.inputs) != len(call):
            raise ValueError("dense input view source declaration differs from the actual shim")
        for typ, buf in zip(declaration.function_type.inputs, call, strict=True):
            if (not hasattr(typ, "get_shape") or tuple(typ.get_shape()) != tuple(buf["shape"])
                    or S._elem_bytes(str(typ.element_type)) != S._elem_bytes(buf["dtype"])):
                raise ValueError("dense input view source type differs from the actual shim")
    selected = {
        symbol: tuple(index for index in indices if calls.get(symbol) and all(
            index < len(call.arguments) and _dense_tensor(call.arguments[index]) for call in calls[symbol]
        )) for symbol, indices in selected.items()
    }
    selected = {symbol: indices for symbol, indices in selected.items() if indices}
    report = rewrite_readonly_dense_offsets(module, selected)
    stream = io.StringIO()
    Printer(stream=stream, print_generic_format=True).print_op(module)
    work.mkdir(parents=True, exist_ok=False)
    destination = work / "model.mlir"
    destination.write_text(stream.getvalue())
    document = {
        "schema": "merlin.device_readonly_dense_offsets.v1",
        "original_source_sha256": sha256_file(source), "selected_source_sha256": sha256_file(destination),
        "arguments": report, "shim_source_sha256": sha256_file(shim),
        **({"sibling_arguments": sibling_arguments, "retained_sibling_arguments": retained_sibling_arguments}
           if sibling_source is not None else {}),
        "layout": "fixed dense row strides; dynamic nonnegative element offset; original output and alias guards",
        "fallback": "noncontiguous views retain upstream layout-conversion copies",
        "numerical_qualification": False, "performance_cycles": None,
    }
    receipt = work / "selection.json"
    receipt.write_text(json.dumps(document, indent=2) + "\n")
    for path, digest in pins:
        if sha256_file(path) != digest:
            raise ValueError("dense input view source or original shim changed during binding")
    # Preserve roles already selected by normal source/offload preparation.
    for index, path in enumerate((*paths, destination, receipt)):
        original = [pin for pin in compilation.record.get("preparation", {}).values()
                    if pin["path"] == str(path.resolve())]
        if original:
            if len(original) != 1 or original[0]["sha256"] != sha256_file(path):
                raise ValueError("dense input views conflict with original preparation custody")
            continue
        compilation.bind_preparation(f"dense_input_views_{index}", path)
    return destination
