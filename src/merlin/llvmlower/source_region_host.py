"""Prepare original tensor SSA around a separately qualified shared device writer.

These are source/ABI transforms. The caller retains original placement, effects,
object, numerical and target-execution obligations; no helper is qualified here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import prod


@dataclass(frozen=True)
class SiblingSourceHost:
    source: object
    module: object
    core: object
    writer: object
    writer_rewrite: tuple
    receipt: dict


def outline_source_siblings(module):
    """Propose the exact original sibling source core without implementation effects.

    The result keeps original host epilogues and observers. No executable writer,
    target implementation or borrowed-buffer premise is selected by this reader.
    """
    from xdsl.dialects import func

    from merlin.compile.scheduling.dense_source import read_zero_seeded_integer_matmul
    from merlin.llvmlower.source_region import outline_sibling_contractions
    from merlin.transforms.tensor_views import canonicalize_tensor_views
    from merlin.xdsl_dialects._common import text

    module.verify()
    before = text(module, generic=True)
    normalized = module.clone()
    view_report = canonicalize_tensor_views(normalized)
    functions = [op for op in normalized.body.block.ops if isinstance(op, func.FuncOp)]
    if len(functions) != 1 or len(functions[0].body.blocks) != 1 or functions[0].sym_name.data != "forward":
        raise ValueError("shared source owner requires one original closed forward block")
    roots = []
    for op in functions[0].body.block.ops:
        # The ordinary reader grants exact signed zero-seeded tensor semantics.
        # A non-contraction remains host source; failed contraction proofs must
        # not be silently treated as another admitted integer implementation.
        if op.name not in {"linalg.generic", "linalg.matmul"} or not op.results:
            continue
        inputs = tuple(getattr(op, "inputs", ()))
        if len(inputs) != 2 or any(str(v.type.element_type) != "i8" for v in inputs):
            continue
        if len(op.results) != 1 or str(op.results[0].type.element_type) != "i32":
            continue
        read_zero_seeded_integer_matmul(op)
        roots.append(op)
    outlined = outline_sibling_contractions(normalized, tuple(roots))
    if text(module, generic=True) != before:
        raise ValueError("sibling source reader changed the original module")
    return outlined, tuple(view_report)


def prepare_sibling_source_host(module, *, effects):
    """Outline every exact sibling signed-i8/i32 source root, keeping all observers.

    Original immutable view CSE runs before the ordinary sibling proof. Source
    quantization frontiers stay available for subsequent output-path decisions. Every remaining tensor computation stays in the source owner.
    No root, epilogue or input is reconstructed from shapes or workload names.
    """
    outlined, view_report = outline_source_siblings(module)
    return bind_sibling_source_outline(module, outlined, effects=effects, view_report=view_report)


def bind_sibling_source_outline(module, outlined, *, effects, view_report=(), symbol="source_shared_core"):
    """Bind one checked outline to the existing explicit borrowed writer ABI."""
    from merlin.llvmlower.fresh_tensor_writer import rewrite_fresh_tensor_writers
    from merlin.xdsl_dialects._common import text

    before = text(module, generic=True)
    host, core = outlined.module, outlined.core
    writer = _prepare_sibling_writer(host, effects=effects, symbol=symbol)
    rewritten = rewrite_fresh_tensor_writers(host, (writer,))
    host.verify()
    if text(module, generic=True) != before:
        raise ValueError("shared source owner changed the original module")
    return SiblingSourceHost(
        module,
        host,
        core,
        writer,
        tuple(rewritten),
        {
            "schema": "merlin.original_sibling_source_host.v1",
            "view_cse": [asdict(witness) for witness in view_report],
            "outline": outlined.receipt,
            "source_unchanged": True,
            "original_epilogues_and_all_live_outputs_preserved": True,
            "physical_effects_and_target_execution_established": False,
        },
    )


def _prepare_sibling_writer(host, *, effects, symbol):
    """Prepare a declaration on an owned graph for one checked source core.

    A batch owner can then check and rewrite all borrowed writers together.
    This helper grants no target effects or source custody on its own.
    """
    from xdsl.dialects import func, tensor
    from xdsl.dialects.builtin import ArrayAttr, DictionaryAttr, StringAttr, UnitAttr
    from xdsl.ir import Region

    from merlin.llvmlower.borrowed_effects import BorrowedBufferEffects
    from merlin.llvmlower.fresh_tensor_writer import FreshTensorWriterContract

    if type(effects) is not BorrowedBufferEffects:
        raise ValueError("original independently selected borrowed effects required")
    effects.__post_init__()
    callee = next(
        op for op in host.body.block.ops if isinstance(op, func.FuncOp) and op.sym_name.data == symbol
    )
    calls = [
        op
        for op in host.walk()
        if isinstance(op, func.CallOp) and op.callee.root_reference.data == callee.sym_name.data
    ]
    if len(calls) != 1:
        raise ValueError("shared source has no unique original core call")
    call = calls[0]
    ins, outs = tuple(callee.function_type.inputs), tuple(callee.function_type.outputs)
    empty = [tensor.EmptyOp([], typ) for typ in outs]
    for op in empty:
        call.parent.insert_op_before(op, call)
    replacement = func.CallOp(callee.sym_name.data, [*call.arguments, *(op.tensor for op in empty)], outs)
    call.parent.insert_op_before(replacement, call)
    for old, new in zip(call.results, replacement.results, strict=True):
        old.replace_all_uses_with(new)
    call.parent.erase_op(call)
    types = (*ins, *outs)
    attrs = ArrayAttr(
        [
            DictionaryAttr({"bufferization.access": StringAttr("read" if i < len(ins) else "write")})
            for i in range(len(types))
        ]
    )
    declaration = func.FuncOp(callee.sym_name.data, (types, outs), Region(), visibility="private", arg_attrs=attrs)
    declaration.attributes["llvm.emit_c_interface"] = UnitAttr()
    host.body.block.insert_op_before(declaration, callee)
    host.body.block.erase_op(callee)
    result_args = tuple(range(len(ins), len(types)))
    return FreshTensorWriterContract(
        callee.sym_name.data,
        result_args[0],
        result_args,
        symbol + "_borrowed",
        64,
        result_arguments=result_args,
        borrowed_effects=effects,
    )


def tensor_abi(module):
    """Read complete static public tensor types in original input/result order."""
    from xdsl.dialects import func
    from xdsl.dialects.builtin import TensorType

    fs = [op for op in module.body.block.ops if isinstance(op, func.FuncOp) and op.sym_name.data == "forward"]
    if len(fs) != 1:
        raise ValueError("source owner requires a unique forward")
    f = fs[0]
    ins, outs = tuple(f.function_type.inputs), tuple(f.function_type.outputs)
    if (
        not ins
        or not outs
        or any(
            not isinstance(t, TensorType) or not t.get_shape() or any(n < 1 for n in t.get_shape())
            for t in (*ins, *outs)
        )
    ):
        raise ValueError("source owner requires complete nonempty static tensor inputs/results")
    return tuple(
        {"name": "arg" + str(i), "shape": list(t.get_shape()), "dtype": str(t.element_type), "role": "input"}
        for i, t in enumerate(ins)
    ), tuple(
        {"name": "output" + str(i), "shape": list(t.get_shape()), "dtype": str(t.element_type)}
        for i, t in enumerate(outs)
    )


def emit_ranked_source_entry(module, *, entry_symbol="kernel_entry", owner_symbol="forward"):
    """Transport flat original buffers into the selected 64-bit ranked ABI.

    No tensor elements are read or written here. The original host owns every
    temporary; all argument/result descriptors retain their source types/order.
    """
    from merlin.llvmlower.spatial_scratch_abi import _transport_preamble
    from merlin.runtime.host_arithmetic import _identifier

    _identifier(entry_symbol)
    _identifier(owner_symbol)
    ins, outs = tensor_abi(module)
    buffers = (*ins, *outs)
    if any(row["dtype"] not in {"i8", "i16", "i32", "i64", "f32", "f64"} for row in buffers):
        raise ValueError("ranked source transport requires an explicit whole-byte software format")
    preamble = _transport_preamble([(r["name"], tuple(r["shape"]), r["dtype"]) for r in buffers])
    lines = [
        preamble,
        "extern void _mlir_ciface_"
        + owner_symbol
        + "("
        + ",".join("M" + str(len(r["shape"])) + "*" for r in buffers)
        + ");",
        "void " + entry_symbol + "(" + ",".join("void*p" + str(i) for i in range(len(buffers))) + ") {",
        f"uintptr_t address[{len(buffers)}],bytes[{len(buffers)}];",
    ]
    for i, row in enumerate(buffers):
        shape = row["shape"]
        strides = [prod(shape[j + 1 :]) for j in range(len(shape))]
        lines += [
            f"M{len(shape)} d{i}={{p{i},p{i},0,{{"
            + ",".join(map(str, shape))
            + "},{"
            + ",".join(map(str, strides))
            + "}};",
            (
                f"if(!spatial_span(p{i},0,{int(row['dtype'][1:]) // 8},UINT64_C({prod(shape)}),"
                f"&address[{i}],&bytes[{i}])) __builtin_trap();"
            ),
        ]
    lines += [
        (
            f"for(unsigned i=0;i<{len(buffers)};++i)for(unsigned j=i+1;j<{len(buffers)};++j)"
            "if(!spatial_disjoint(address[i],bytes[i],address[j],bytes[j]))__builtin_trap();"
        ),
        "_mlir_ciface_" + owner_symbol + "(" + ",".join("&d" + str(i) for i in range(len(buffers))) + ");",
        "}",
    ]
    return "\n".join(lines) + "\n"
