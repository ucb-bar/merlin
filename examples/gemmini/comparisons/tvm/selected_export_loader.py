"""Load a frozen prepared ExportedProgram through the ordinary capture API."""

import os
from pathlib import Path

import torch
import torch.ao.quantization.fx._decomposed  # noqa: F401 -- register standard serialized Q/DQ operators


def get_model_and_inputs():
    source = Path(os.environ["SELECTED_PREPARED_PROGRAM"])
    exported = torch.export.load(source)
    loaded = exported.module()
    # A fresh public FX GraphModule permits the capture worker's ordinary eval()
    # lifecycle; ExportedProgram.module() intentionally forbids train/eval toggles.
    model = torch.fx.GraphModule(loaded, loaded.graph).eval()
    return model, exported.example_inputs[0]


def get_verification_artifacts(model, inputs):
    return [str(Path(os.environ["SELECTED_PREPARED_PROGRAM"]).resolve())]
