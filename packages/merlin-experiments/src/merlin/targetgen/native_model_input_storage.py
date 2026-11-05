"""Original frozen entry storage; this is grader code, not a compiler route."""


def bind_entry_storage(capture, packed, manifest, *, storage, error_type, decode):
    from merlin.targetgen.capture_source import capture_tensor_source

    if storage not in ("inline", "streamed"):
        raise error_type("select inline or streamed exact logical input storage")
    source, report = capture_tensor_source(capture, packed, weight_manifest=manifest, streamed=storage == "streamed")
    inputs = {}
    for row in packed.const:
        if row.role != "argument":
            raise error_type("carried model state needs an explicit session execution contract")
        if not row.weight:
            raise error_type(f"{row.tensor}: no frozen source key in weight manifest")
        if storage == "streamed":
            from merlin.runtime.tensor_payload import TensorPayload

            inputs[row.tensor] = TensorPayload(tuple(row.shape), row.dtype, source(row.weight))
            inputs[row.tensor].bind(shape=row.shape, dtype=row.dtype, byte_order="little")
        else:
            inputs[row.tensor] = decode(source(row.weight), tensor=row.tensor, dtype=row.dtype, shape=row.shape)
    return inputs, report
