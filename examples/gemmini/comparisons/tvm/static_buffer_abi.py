"""Opt-in specialization of generic TIR offsets for the static raw-pointer ABI.

The exporter allocates compact standalone tensors, so their implicit buffer
offsets are zero. Use TVM's public specialization API to bind those symbols;
do not erase offsets, rewrite loads, or relax the exporter's ABI checks.
"""


def specialize_zero_offsets(mod, tvm):
    changes = []
    names = [(value, gv.name_hint) for gv, value in mod.functions.items() if isinstance(value, tvm.tir.PrimFunc)]

    @tvm.tir.transform.prim_func_pass(opt_level=0, name="SpecializeStaticBufferOffsets")
    def specialize(function, module, context):
        del context
        mapping, records = {}, []
        for parameter, buffer in function.buffer_map.items():
            offset = buffer.elem_offset
            if isinstance(offset, tvm.tir.IntImm) and int(offset) == 0 and not buffer.strides:
                continue
            if buffer.strides or not isinstance(offset, tvm.tir.Var):
                raise ValueError("Static ABI specialization admits only symbolic offsets of compact buffers")
            if buffer.scope() != "global" or buffer.axis_separators or any(not isinstance(dim, tvm.tir.IntImm) or int(dim) <= 0 for dim in buffer.shape):
                raise ValueError("Static ABI specialization requires ordinary positive static global buffers")
            mapping[parameter] = tvm.tir.decl_buffer(
                buffer.shape, buffer.dtype, name=buffer.name, scope=buffer.scope(),
                data_alignment=buffer.data_alignment, offset_factor=buffer.offset_factor, elem_offset=0)
            records.append({"buffer": buffer.name, "original_offset": str(offset), "specialized_offset": 0,
                            "shape": [int(dim) for dim in buffer.shape], "dtype": buffer.dtype})
        if not mapping:
            return function
        name = next((name for value, name in names if value.same_as(function)),
                    str(function.attrs["global_symbol"]) if function.attrs and "global_symbol" in function.attrs else "unnamed")
        result = function.specialize(mapping)
        if any(buffer.strides or not isinstance(buffer.elem_offset, tvm.tir.IntImm) or int(buffer.elem_offset) != 0 for buffer in result.buffer_map.values()):
            raise ValueError("Public specialization did not produce a compact zero-offset ABI")
        changes.append({"function": name, "buffers": records})
        return result

    return specialize(mod), changes
