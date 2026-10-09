"""Fixed private native execution of a freshly rederived original factory source.

The parent preflights all shapes/bytes/work and verifies every file/argv. This
worker returns complete typed framework outputs, not a numerical verdict.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def observe(loader, metadata, stimulus, *, version=1):
    import torch

    if type(version) is not int or version not in {1, 2}:
        raise ValueError("original reference observer needs its explicit storage version")
    spec = importlib.util.spec_from_file_location("selected_original_factory", loader)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model, examples = module.get_model_and_inputs()
    if len(examples) != len(metadata["inputs"]) or len(stimulus) != len(examples):
        raise ValueError("original reference observer lost complete original input slots")
    tensors = []
    for example, row, source in zip(examples, metadata["inputs"], stimulus, strict=True):
        if (
            set(source) != {"name", "dtype", "shape", "byteorder", "data_hex"}
            or {key: source[key] for key in ("name", "dtype", "shape")} != row
            or source["byteorder"] != sys.byteorder
            or list(example.shape) != row["shape"]
            or str(example.dtype) != "torch." + row["dtype"]
        ):
            raise ValueError("original reference observer changed original input storage/ABI")
        dtypes = {"float32", "int8"}
        if version == 2 and metadata["target"] in {"aten.relu.default", "aten.round.default", "aten.clamp.default"}:
            dtypes |= {"int16", "int32", "int64"}
        if row["dtype"] not in dtypes:
            raise ValueError("original reference observer does not implement this original binary dtype")
        encoded = source["data_hex"]
        if (
            type(encoded) is not str
            or len(encoded) != 2 * example.numel() * example.element_size()
            or any(character not in "0123456789abcdef" for character in encoded)
        ):
            raise ValueError("original reference observer received invalid storage before decoding")
        raw = bytearray.fromhex(encoded)
        if len(raw) != example.numel() * example.element_size():
            raise ValueError("original reference observer received incomplete original storage")
        tensors.append(torch.frombuffer(raw, dtype=example.dtype).clone().reshape(example.shape))
    output = model(*tensors)
    outputs = output if isinstance(output, tuple) else (output,)
    if len(outputs) != len(metadata["outputs"]):
        raise ValueError("original reference observer lost complete original return slots")
    rows = []
    for value, row in zip(outputs, metadata["outputs"], strict=True):
        if list(value.shape) != row["shape"] or str(value.dtype) != "torch." + row["dtype"]:
            raise ValueError("original reference observer changed original output storage/ABI")
        rows.append(
            {
                "name": row["name"],
                "dtype": row["dtype"],
                "shape": row["shape"],
                "byteorder": sys.byteorder,
                "data_hex": value.detach().contiguous().numpy().tobytes().hex(),
            }
        )
    return {
        "schema": "merlin.original_reference_native_output.v1",
        "outputs": rows,
        "runtime": {"torch_version": torch.__version__, "git_version": torch.version.git_version},
    }


if __name__ == "__main__":
    loader, metadata, stimulus, destination = map(Path, sys.argv[1:])
    output = observe(loader, json.loads(metadata.read_bytes()), json.loads(stimulus.read_bytes()))
    destination.write_text(json.dumps(output, sort_keys=True, allow_nan=False) + "\n")
    destination.chmod(0o600)
