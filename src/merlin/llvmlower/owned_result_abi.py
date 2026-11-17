"""Explicit host ABI for result descriptors produced by actual execution.

The MLIR result types fix ranks/storage, never invocation extents. Upstream
ownership transfers returned allocations to the caller and clones borrowed
results. The generated C bridge delegates aggregate layout to the C compiler.
"""

from dataclasses import dataclass
from io import StringIO

from xdsl.dialects import builtin, func
from xdsl.printer import Printer

from merlin.frontends.linalg_mlir import parse_mlir_text

from .runtime_invocation import SCALAR_C

PROFILE = "merlin.execution-owned-results.v1"
STORAGE = {
    "i1",
    "i8",
    "i16",
    "i32",
    "i64",
    "f16",
    "bf16",
    "f32",
    "f64",
    "f8E4M3FN",
    "f8E4M3FNUZ",
    "f8E5M2",
    "f8E5M2FNUZ",
}


@dataclass(frozen=True)
class TensorSlot:
    dtype: str
    shape: tuple[int, ...]


@dataclass(frozen=True)
class ScalarSlot:
    dtype: str


def prepare(source, *, entry="forward"):
    """Read original types and add only the actual C-interface attribute."""
    if type(source) is not str or not source or len(source.encode()) > 16 << 20:
        raise ValueError("owned results require bounded original source")
    if type(entry) is not str or not entry.isascii() or not entry.isidentifier():
        raise ValueError("owned result entry must be a portable C identifier")
    module = parse_mlir_text(source)
    module.verify()
    functions = [op for op in module.body.block.ops if isinstance(op, func.FuncOp)]
    if any(op.sym_name.data.startswith(("merlin_owned_", "_mlir_memref_to_llvm_")) for op in functions):
        raise ValueError("source collides with the selected owned-result runtime")
    entries = [op for op in functions if op.sym_name.data == entry]
    if len(entries) != 1 or not entries[0].body.blocks:
        raise ValueError("owned results require one defined original entry")
    function = entries[0]

    def tensor(typ):
        if not isinstance(typ, builtin.TensorType) or not isinstance(typ.encoding, builtin.NoneAttr):
            raise ValueError("owned result tensors require a ranked unencoded type")
        shape = tuple(typ.get_shape())
        if len(shape) > 16 or any(size < 0 and size != builtin.DYNAMIC_INDEX for size in shape):
            raise ValueError("owned result rank/shape is unsupported")
        dtype = str(typ.element_type)
        if dtype not in STORAGE:
            raise ValueError("owned result storage is unsupported: " + dtype)
        return TensorSlot(dtype, tuple(-1 if size == builtin.DYNAMIC_INDEX else size for size in shape))

    inputs = tuple(
        tensor(t) if isinstance(t, builtin.TensorType) else ScalarSlot(str(t)) for t in function.function_type.inputs
    )
    if any(isinstance(slot, ScalarSlot) and slot.dtype not in SCALAR_C for slot in inputs):
        raise ValueError("owned result scalar input needs the supported native by-value ABI")
    outputs = tuple(tensor(t) for t in function.function_type.outputs)
    if not outputs or len(inputs) > 64 or len(outputs) > 64:
        raise ValueError("owned results require 1..64 tensor results and at most 64 arguments")
    if function.arg_attrs is not None:
        for slot, attrs in zip(inputs, function.arg_attrs, strict=True):
            writable = attrs.data.get("bufferization.writable")
            if (
                isinstance(slot, TensorSlot)
                and writable is not None
                and (not isinstance(writable, builtin.IntegerAttr) or writable.value.data != 0)
            ):
                raise ValueError("owned result inputs are borrowed immutable tensor values")
    function.attributes["llvm.emit_c_interface"] = builtin.UnitAttr()
    stream = StringIO()
    Printer(stream=stream, print_generic_format=True, print_debuginfo=True).print_op(module)
    return stream.getvalue(), inputs, outputs


def pipeline(*, features=frozenset(), index_bits=64):
    """Retain the normal ownership gates; select upstream returned descriptors."""
    from .impr_features import apply_pipeline, normalize
    from .pipeline import _INDEX_CONVERSION_PASSES, _UPSTREAM_PASSES, _splice

    if type(index_bits) is not int or index_bits != 64:
        raise ValueError("owned result host transport currently requires observed 64-bit indices")
    passes = apply_pipeline(list(_UPSTREAM_PASSES), normalize(features))
    outs = [p for p in passes if p.startswith("buffer-results-to-out-params")]
    if len(outs) != 1 or passes.count("__DEALLOC__") != 1:
        raise ValueError("owned results require the ordinary result and ownership seams")
    passes.remove(outs[0])
    passes.insert(passes.index("__DEALLOC__") + 1, "convert-bufferization-to-memref")
    from .owned_result_allocations import MARKER

    passes.insert(passes.index("convert-scf-to-cf"), MARKER)
    for name in _INDEX_CONVERSION_PASSES:
        if passes.count(name) != 1:
            raise ValueError("owned result index conversion is missing or preconfigured")
        options = "use-generic-functions " if name == "finalize-memref-to-llvm" else ""
        passes[passes.index(name)] = f"{name}{{{options}index-bitwidth={index_bits}}}"
    selected = _splice(["scf-forall-to-for", *passes])
    if "ownership-based-buffer-deallocation" not in selected:
        raise ValueError("owned result allocation requires active ownership/deallocation checks")
    return selected


def bridge(inputs, outputs, *, entry="forward"):
    """Use the official first-result-pointer C wrapper; never guess tuple offsets."""
    if type(entry) is not str or not entry.isascii() or not entry.isidentifier():
        raise ValueError("owned result bridge needs a portable entry symbol")
    ranks = sorted({len(s.shape) for s in (*inputs, *outputs) if isinstance(s, TensorSlot)})
    lines = [
        "#include <stdint.h>",
        "#include <stddef.h>",
        "#include <string.h>",
        '_Static_assert(sizeof(void *) == 8 && sizeof(intptr_t) == 8, "selected host ABI");',
    ]
    for rank in ranks:
        arrays = f"intptr_t sizes[{rank}], strides[{rank}];" if rank else ""
        lines.append(f"typedef struct {{void *allocated, *aligned; intptr_t offset; {arrays}}} D{rank};")
    if len(outputs) == 1:
        lines.append(f"typedef D{len(outputs[0].shape)} Results;")
    else:
        lines.append(
            "typedef struct {" + "".join(f"D{len(s.shape)} r{i};" for i, s in enumerate(outputs)) + "} Results;"
        )
    types = [f"D{len(s.shape)} *" if isinstance(s, TensorSlot) else SCALAR_C[s.dtype] for s in inputs]
    arguments = [
        f"(D{len(s.shape)} *)a[{i}]" if isinstance(s, TensorSlot) else f"*({SCALAR_C[s.dtype]} *)a[{i}]"
        for i, s in enumerate(inputs)
    ]
    lines.append(f"extern void _mlir_ciface_{entry}(Results *" + "".join("," + t for t in types) + ");")
    lines.append("void merlin_owned_result_invoke(void **a, void **out) { Results r; memset(&r,0,sizeof(r));")
    lines.append(f"_mlir_ciface_{entry}(&r" + "".join("," + a for a in arguments) + ");")
    for i, slot in enumerate(outputs):
        result = "r" if len(outputs) == 1 else f"r.r{i}"
        lines.append(f"memcpy(out[{i}], &{result}, sizeof(D{len(slot.shape)}));")
    lines.append("}")
    lines.append("size_t merlin_owned_result_descriptor_bytes(size_t rank) { switch(rank) {")
    lines.extend(f"case {rank}: return sizeof(D{rank});" for rank in ranks)
    lines.extend(("default: return 0;}}", ""))
    from .owned_result_c import checked_bridge

    return "\n".join(lines) + checked_bridge(inputs, outputs)
