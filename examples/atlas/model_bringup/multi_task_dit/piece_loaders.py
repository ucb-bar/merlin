"""Actual policy subregions, random DiT weights, no accuracy claim.

Set MTDIT_REGION to vision, text, block, or dit. Inputs are recorded
by prehooks during the seed-0 policy's first inference step, and the genuine
module is returned without replacing parameters or arithmetic.
"""

import importlib.util
import os
from pathlib import Path

import torch


class _InputsCollected(Exception):
    pass


def _source_loader():
    path = Path(__file__).with_name("loader.py")
    spec = importlib.util.spec_from_file_location("_mtdit_piece_source", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def select_region(region):
    """Return the genuine submodule and actual first-step positional inputs."""
    if region not in {"vision", "text", "block", "dit"}:
        raise ValueError(f"Unknown MTDIT_CAPTURE_REGION {region!r}")
    source = _source_loader()
    model, inputs = source.get_model_and_inputs()
    policy = model.policy
    modules = {
        "vision": policy.observation_encoder.vision_encoder,
        "text": policy.observation_encoder.text_encoder,
        "block": policy.noise_predictor.transformer_blocks[0],
        "dit": policy.noise_predictor,
    }
    recorded = {}

    def capture(name):
        def hook(module, args, kwargs):
            args = tuple(args)
            if name == "dit":
                args += (kwargs["conditioning_vec"],)
            recorded[name] = tuple(value.detach().clone() for value in args)
            if name == "block":
                raise _InputsCollected
        return hook

    handles = [module.register_forward_pre_hook(capture(name), with_kwargs=True)
               for name, module in modules.items()]
    try:
        with torch.no_grad():
            model(*inputs)
    except _InputsCollected:
        pass
    finally:
        for handle in handles:
            handle.remove()
    if set(recorded) != set(modules):
        raise RuntimeError(f"Incomplete first-step input capture: {sorted(recorded)}")
    return modules[region].eval().requires_grad_(False), recorded[region]


def get_model_and_inputs():
    """Worker loader ABI, selected with MTDIT_REGION (default block)."""
    return select_region(os.environ.get("MTDIT_REGION", os.environ.get("MTDIT_CAPTURE_REGION", "block")))


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("region", choices=("vision", "text", "block", "dit"))
    args = parser.parse_args()
    model, inputs = select_region(args.region)
    with torch.no_grad():
        output = model(*inputs)
        exported = torch.export.export(model, inputs)
        replay = exported.module()(*inputs)
    print(json.dumps({
        "region": args.region,
        "weight_scope": _source_loader().WEIGHT_SCOPE,
        "seed": 0, "dtype": "float32", "quantization": "none",
        "module": f"{type(model).__module__}.{type(model).__name__}",
        "inputs": [{"shape": list(x.shape), "dtype": str(x.dtype)} for x in inputs],
        "output_shape": list(output.shape),
        "torch_export": "pass",
        "replay_equal": bool(torch.equal(output, replay)),
        "replay_max_abs": float((output - replay).abs().max()),
        "block_identity": bool(torch.equal(output, inputs[0])) if args.region == "block" else None,
        "parameters": sum(p.numel() for p in model.parameters()),
    }, indent=2))
