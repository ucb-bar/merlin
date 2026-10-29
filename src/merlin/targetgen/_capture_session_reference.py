"""Independent pre-quantization trajectory for an audited FP32 stage."""


def freeze_fp32_session_reference(module, inputs, session):
    """Freeze the staged program's trajectory, not its later integerized output.

    The caller must first complete the frontend FP32 precision audit. This is
    a stage-local numerical reference over the declared inputs and recurrence;
    it does not establish a complete application's FP32 accuracy.
    """
    if session is None:
        return None
    import torch
    from m2m.capture.bundle import capture_session_trajectory

    streams = []
    for stream in session.get("streams") or ():
        value = stream.get("values")
        value = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        index = int(stream["input_index"])
        if not 0 <= index < len(inputs):
            raise ValueError("session stream references an unknown staged input")
        # Match the executable input ABI, preserving nonfloating leaves exactly.
        value = value.detach().clone()
        if value.is_floating_point():
            value = value.to(dtype=inputs[index].dtype, device=inputs[index].device)
        elif value.dtype != inputs[index].dtype:
            raise ValueError("session stream changes a nonfloating input dtype")
        streams.append({**stream, "values": value})
    session = {**session, "streams": streams}
    quality = dict(session.get("quality") or {})
    quality.update(
        reference="eager_fp32",
        reference_values=capture_session_trajectory(module, tuple(inputs), session).copy(),
    )
    return {**session, "quality": quality}


def pre_quantization_session_reference(module, inputs, session, *, include_diagnostic=False):
    """The source program's eager trajectory, captured before any quantization.

    A prequantized (recipe int8) paper-ready stage cannot let the bundle writer
    derive its quality reference: by then only the quantized program exists. This
    executes the untransformed loader program on its own declared inputs and
    recurrence -- the computation Model2MLIR's stage-wise PT2E path records as the
    ``eager_fp32`` reference -- so the reference is generated independently of the
    quantized program. Every RNG is restored afterwards, so the later capture sees
    the same random state it would have without this run. A session that already
    carries reference values is returned unchanged. Non-paper sessions require
    explicit ``include_diagnostic``; this observes quality without granting
    paper readiness. That path refuses non-FP32 source storage or outputs.
    """
    if type(include_diagnostic) is not bool:
        raise ValueError("diagnostic session reference requires an explicit Boolean")
    if session is None or (session.get("paper_ready") is not True and not include_diagnostic):
        return session
    quality = dict(session.get("quality") or {})
    if quality.get("reference_values") is not None:
        return session
    import random

    import numpy as np
    import torch
    from m2m.capture.bundle import capture_session_trajectory

    if include_diagnostic:
        leaves, _ = torch.utils._pytree.tree_flatten(inputs)
        tensors = [*module.parameters(), *module.buffers(), *leaves]
        if any(isinstance(value, torch.Tensor) and
               (value.is_complex() or value.is_floating_point() and value.dtype != torch.float32)
               for value in tensors):
            return {**session, "quality": {**quality, "reference": "unavailable",
                    "reference_unavailable_reason": "source storage is not uniformly FP32"}}
    states = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    try:
        module.eval()
        values = capture_session_trajectory(module, tuple(inputs), session).copy()
    finally:
        random.setstate(states[0])
        np.random.set_state(states[1])
        torch.set_rng_state(states[2])
    if include_diagnostic and (values.dtype != np.float32 or not np.isfinite(values).all()):
        return {**session, "quality": {**quality, "reference": "unavailable",
                "reference_unavailable_reason": "source trajectory is not finite FP32"}}
    quality["reference"] = "eager_fp32"
    quality["reference_values"] = values
    return {**session, "quality": quality}
