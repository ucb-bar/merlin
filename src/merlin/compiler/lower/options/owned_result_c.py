"""Source-typed checked C invocation using the ordinary aggregate bridge.

MLIR CRunnerUtils.h at LLVM a47bddcc defines allocated/aligned, element
offset, extents and element strides. The existing bridge delegates aggregate
layout to the selected C compiler. This adds no physical target mapping.
"""

from merlin.compiler.lower.options.owned_result_abi import TensorSlot
from merlin.compiler.lower.options.runtime_invocation import SCALAR_C


def checked_bridge(inputs, outputs):
    """Append one-call validation; callers own invocation begin/end and copies."""
    storage = {
        "i1": "uint8_t",
        "i4": "uint8_t",
        "i8": "uint8_t",
        "i16": "uint16_t",
        "i32": "uint32_t",
        "i64": "uint64_t",
        "bf16": "uint16_t",
        "f16": "uint16_t",
        "f32": "float",
        "f64": "double",
        "f8E4M3FN": "uint8_t",
        "f8E4M3FNUZ": "uint8_t",
        "f8E5M2": "uint8_t",
        "f8E5M2FNUZ": "uint8_t",
    }
    if not 1 <= len(outputs) <= 64 or len(inputs) > 64:
        raise ValueError("checked C results require the bounded source signature")
    if any(not isinstance(slot, TensorSlot) or slot.dtype not in storage or len(slot.shape) > 16 for slot in outputs):
        raise ValueError("checked C results require supported source tensor types")
    if any(not isinstance(slot, TensorSlot) and slot.dtype not in SCALAR_C for slot in inputs):
        raise ValueError("checked C results require supported by-value inputs")
    lines = [
        '#include "owned_result_views.h"',
        "int merlin_owned_result_invoke_checked(void **a, MerlinOwnedResultView *views, size_t budget) {",
        "if (!views) return MERLIN_OWNED_RESULT_TYPE;",
    ]
    for i, slot in enumerate(outputs):
        rank = len(slot.shape)
        lines.append(f"D{rank} r{i}; memset(&r{i},0,sizeof(r{i}));")
        if rank:
            lines.append(f"const intptr_t shape{i}[] = {{" + ",".join(map(str, slot.shape)) + "};")
    lines.append("void *raw[] = {" + ",".join(f"&r{i}" for i in range(len(outputs))) + "};")
    lines.append("merlin_owned_result_invoke(a,raw);")
    lines.append("const MerlinOwnedResultDescriptor descriptors[] = {")
    for i, slot in enumerate(outputs):
        rank, typ = len(slot.shape), storage[slot.dtype]
        arrays = f"r{i}.sizes,r{i}.strides,shape{i}" if rank else "NULL,NULL,NULL"
        lines.append(f"{{r{i}.allocated,r{i}.aligned,r{i}.offset,{rank},sizeof({typ}),_Alignof({typ}),{arrays}}},")
    lines.extend(("};", f"return merlin_owned_result_validate_all(descriptors,{len(outputs)},views,budget);", "}"))
    return "\n".join(lines) + "\n"
