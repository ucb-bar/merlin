"""Restate original source-proved sibling readouts in ordinary COMMIT commands."""


def source_readout_commands(group, readouts, tensors):
    """Return checked member attributes and explicitly ordered immutable inputs.

    The source owner established the theorem and binds these records in its
    integrity snapshot. Reconstructing its command-buffer vocabulary here keeps
    source proof coefficients, folded bias inputs and output types in agreement.
    This grants no target effects or numerical execution qualification.
    """
    from merlin.xdsl_dialects.lowering import group_output_program as GOP

    if readouts is None or readouts == []:
        return [{"epilogue": [], "output_dtype": "i32"} for _ in group.widths]
    if type(readouts) is not list or len(readouts) != len(group.widths):
        raise ValueError("sibling readouts must cover every original member in order")
    result = []
    for member, (row, columns) in enumerate(zip(readouts, group.widths, strict=True)):
        if type(row) is not dict or type(row.get("member")) is not int or row["member"] != member:
            raise ValueError("sibling readout changed original member order")
        dtype = row.get("output_dtype")
        if dtype == "i32":
            attributes = {"epilogue": [], "output_dtype": "i32"}
            if row.get("absorbed") is not False or row.get("bias_argument") is not None or row.get("bias_values") != []:
                raise ValueError("raw sibling readout has an unproved transformed exit")
        elif dtype == "i8":
            decision = row.get("output_fusion") or {}
            stages, proof = decision.get("stages", []), decision.get("proof", {})
            if (
                row.get("absorbed") is not True
                or decision.get("schema") != "output_fusion_v1"
                or decision.get("cut") != "requant"
                or "residual" in stages
            ):
                raise ValueError("sibling i8 readout needs its complete original requant proof")
            offsets = GOP._integer_offsets(proof, columns)
            attributes = dict(
                GOP._readout(proof, stages, columns),
                output_dtype="i8",
                epilogue=(["bias_add"] if any(offsets) else [])
                + ["acc_scale"]
                + (["relu"] if "relu" in stages else []),
            )
            if row.get("bias_values") != offsets:
                raise ValueError("sibling readout bias differs from its original proof")
            bias_argument = row.get("bias_argument")
            if any(offsets):
                if type(bias_argument) is not int or bias_argument != len(tensors):
                    raise ValueError("sibling folded bias must retain its original input position")
                name = "bias_" + str(member)
                tensors[name] = {"shape": [columns], "dtype": "i32", "role": "bias"}
            elif bias_argument is not None:
                raise ValueError("zero sibling offsets must not invent a bias input")
        else:
            raise ValueError("sibling readout has no supported source result type")
        if row.get("attributes") != attributes:
            raise ValueError("sibling readout attributes differ from its original proof")
        if dtype == "i8" and any(offsets):
            attributes = {**attributes, "bias": name}
        result.append(attributes)
    return result


def require_writer_interface(module, writer, buffers):
    """Bind every original writer pointer position to its real logical type."""
    from xdsl.dialects import func
    from xdsl.dialects.builtin import TensorType

    declarations = [
        op for op in module.body.block.ops if isinstance(op, func.FuncOp) and op.sym_name.data == writer.symbol
    ]
    if len(declarations) != 1:
        raise ValueError("sibling route lost its original borrowed writer declaration")
    types = tuple(declarations[0].function_type.inputs)
    if len(types) != len(buffers):
        raise ValueError("sibling logical interface changed original writer arity")
    outputs = []
    for index, (typ, buffer) in enumerate(zip(types, buffers, strict=True)):
        if (
            not isinstance(typ, TensorType)
            or tuple(typ.get_shape()) != tuple(buffer.shape)
            or str(typ.element_type) != buffer.dtype
        ):
            raise ValueError("sibling logical interface changed original writer pointer type or order")
        if buffer.role.startswith("out_"):
            outputs.append(index)
    if tuple(outputs) != writer.result_arguments:
        raise ValueError("sibling logical interface changed original writer result positions")
