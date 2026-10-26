"""Use a functional export boundary for the prescribed TinyLlama numerical source.

Transformers' mutable Python cache can retain FakeTensors after export. Export
it once, then use its ordinary functional FX graph for subsequent capture.
Original pretrained modules supply the same weights and class/shape inventory.
No TVM or framework functions are replaced.
"""

import selected_tinyllama_loader as numerical
import torch

_calibration_source = None


def functional_boundary(source, inputs):
    exported = torch.export.export(source, inputs)
    loaded = exported.module()
    functional = torch.fx.GraphModule(loaded, loaded.graph)
    for name, original in source.named_modules():
        if not isinstance(original, torch.nn.Linear):
            continue
        captured = functional.get_submodule(name)
        for field in ("weight", "bias"):
            before, after = getattr(captured, field, None), getattr(original, field)
            if before is None and after is None:
                continue
            if not isinstance(before, torch.Tensor) or not isinstance(after, torch.Tensor):
                raise ValueError("Export changed original linear state")
            if before.shape != after.shape or before.dtype != after.dtype or not torch.equal(before, after):
                raise ValueError("Export changed original pretrained state")
        # Restore only the verified linear leaf for the stock recipe inventory.
        # Retain exported containers and lifted constants; do not reattach the
        # mutable Hugging Face model/cache object to the functional graph.
        functional.set_submodule(name, original)
    return functional.eval()


def get_model_and_inputs():
    global _calibration_source
    source, inputs = numerical.get_model_and_inputs()
    if len(inputs) == 3:
        # Freeze real decode contexts before tracing mutates private cache state.
        # The later recipe passes replay those exact file-backed tensors.
        for arguments in numerical.get_calibration_inputs(source, inputs):
            del arguments
    _calibration_source = source
    return functional_boundary(source, inputs), inputs


def get_calibration_inputs(model, inputs):
    if _calibration_source is None:
        raise ValueError("Numerical source must be constructed before calibration")
    return numerical.get_calibration_inputs(_calibration_source, inputs)
