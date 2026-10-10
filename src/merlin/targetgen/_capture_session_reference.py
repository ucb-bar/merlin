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


def pre_quantization_session_reference(module, inputs, session):
    """The source program's eager trajectory, captured before any quantization.

    A prequantized (recipe int8) paper-ready stage cannot let the bundle writer
    derive its quality reference: by then only the quantized program exists. This
    executes the untransformed loader program on its own declared inputs and
    recurrence -- the computation Model2MLIR's stage-wise PT2E path records as the
    ``eager_fp32`` reference -- so the reference is generated independently of the
    quantized program. Every RNG is restored afterwards, so the later capture sees
    the same random state it would have without this run. A session that already
    carries reference values, or is not paper-ready, is returned unchanged.
    """
    if session is None or session.get("paper_ready") is not True:
        return session
    quality = dict(session.get("quality") or {})
    if quality.get("reference_values") is not None:
        return session
    import random

    import numpy as np
    import torch
    from m2m.capture.bundle import capture_session_trajectory

    states = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    try:
        module.eval()
        values = capture_session_trajectory(module, tuple(inputs), session).copy()
    finally:
        random.setstate(states[0])
        np.random.set_state(states[1])
        torch.set_rng_state(states[2])
    quality.setdefault("reference", "eager_fp32")
    quality["reference_values"] = values
    return {**session, "quality": quality}
