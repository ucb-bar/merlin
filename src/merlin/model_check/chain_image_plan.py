"""Derive complete guest schedules from the selected models file and captures."""

from __future__ import annotations

import io
import math
from pathlib import Path

import numpy as np

from merlin.common.mlir_query import forward_signature
from merlin.llvmlower import c_runtime
from merlin.model_check import models
from merlin.model_check.chain import stream_input
from merlin.model_check.full_build_inputs import _bindings

from .chain_image_objects import completed_program, payload, pin, reopen


def _count(header: Path, name: str) -> int:
    rows = [line.split() for line in header.read_text().splitlines() if line.startswith("#define " + name + " ")]
    if len(rows) != 1 or len(rows[0]) != 3:
        raise ValueError("compiled argument header has no unique generated extent")
    value = int(rows[0][2])
    if value <= 0:
        raise ValueError("compiled argument header has an invalid extent")
    return value


def _storage(value, dtype):
    array = np.ascontiguousarray(value)
    if dtype == "bf16":
        return c_runtime._bf16_bits(array.astype(np.float32))
    if dtype not in c_runtime.NP_OF:
        raise ValueError("unsupported guest input storage dtype")
    return array.astype(c_runtime.NP_OF[dtype], copy=False)


def prepare(selection: dict, directory: Path) -> dict:
    """No graph lowering or model evaluation: inspect existing typed ABIs only."""
    model_file = reopen(selection["models"])
    loaded = models.load(model_file, source_bytes=payload(selection["models"]))
    model = loaded.model(selection["model"])
    with np.load(io.BytesIO(payload(selection["inputs"])), allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    if set(selection["programs"]) != {p.name for p in model.programs}:
        raise ValueError("chain selection must contain every declared program exactly once")
    programs, source_pins = [], [selection["models"], selection["inputs"]]
    for index, program in enumerate(model.programs):
        production = completed_program(selection["programs"][program.name])
        stage = Path(production["receipt"]["inputs"]["capture"])
        cgen = production["root"] / "cgen"
        header = cgen / "model_gen.h"
        n_args, n_outputs = _count(header, "MERLIN_N_ARGS"), _count(header, "MERLIN_N_OUTPUTS")
        if n_args > 4096 or program.steps <= 0:
            raise ValueError("compiled program exceeds the existing memref runtime bound")
        bindings = sorted(_bindings(stage, stage / "inputs.npz"), key=lambda row: int(row["key"][2:]))
        if [row["key"] for row in bindings] != [f"in{i}" for i in range(len(program.inputs))]:
            raise ValueError("models bindings do not cover the original runtime input ABI")
        _, outputs = forward_signature(stage / "model.mlir")
        if n_outputs != len(outputs) or any(row["argument"] >= n_args for row in bindings):
            raise ValueError("compiled tables differ from the original capture ABI")
        # Entry argument projection needs its own original-to-projected map.
        # Appended quantized weight arguments preserve original input ordinals.
        if any(production["root"].glob("*entry*projection*.json")):
            raise ValueError("projected entry arguments need an explicit original binding adapter")
        source_pins += [
            production["receipt_pin"],
            production["recipe_pin"],
            pin(header),
            pin(stage / "model.mlir"),
            pin(stage / "input_order.json"),
            pin(stage / "weights.safetensors.manifest.json"),
        ]
        programs.append(
            {
                "index": index,
                "name": program.name,
                "steps": program.steps,
                "production": production,
                "bindings": bindings,
                "n_args": n_args,
                "outputs": [(list(shape), dtype) for shape, dtype in outputs],
                "n_outputs": n_outputs,
            }
        )
    by_name = {row["name"]: row for row in programs}
    blob_path = directory / "inputs.bin"
    schedule, offsets = [], {}
    with blob_path.open("xb") as blob:
        for program, row in zip(model.programs, programs, strict=True):
            for step in range(program.steps):
                transfers = []
                for declared, arg in zip(program.inputs, row["bindings"], strict=True):
                    source = declared["from"]
                    if step > 0 and "carry" in declared:
                        origin, output = row, int(declared["carry"])
                    elif source == "output":
                        origin, output = by_name[declared["program"]], int(declared["output"])
                        if origin["index"] >= row["index"]:
                            raise ValueError("a first-step chain binding does not flow from an earlier program")
                    else:
                        origin = None
                    spec = (arg["shape"], arg["dtype"])
                    count = math.prod(arg["shape"])
                    nbytes = count * c_runtime.DT_BYTES[arg["dtype"]]
                    transfer = {
                        "argument": arg["argument"],
                        "shape": arg["shape"],
                        "dtype": arg["dtype"],
                        "bytes": nbytes,
                    }
                    if origin is not None:
                        if output < 0 or output >= len(origin["outputs"]) or origin["outputs"][output] != spec:
                            raise ValueError("carried/output binding changes shape or storage dtype")
                        transfer.update(source_program=origin["index"], source_output=output)
                    else:
                        if source == "input":
                            value = arrays[declared["key"]]
                        elif source == "stream":
                            value = stream_input(declared, arrays, step, program.steps)
                        elif source == "tokens":
                            n = int(model.chain["prefill_tokens"])
                            tokens = np.asarray(arrays["tokens"], np.int64)
                            value = tokens[:n][None] if declared["slice"] == "prefill" else tokens[n + step][None, None]
                        elif source == "constant":
                            value = np.asarray(declared["value"], dtype=declared["dtype"])
                        else:
                            raise ValueError("unsupported selected chain binding")
                        if list(value.shape) != arg["shape"]:
                            raise ValueError("selected chain input shape differs from its compiled ABI")
                        value = _storage(value, arg["dtype"])
                        encoded = value.tobytes()
                        if len(encoded) != nbytes:
                            raise ValueError("selected chain input has an inconsistent storage extent")
                        key = (arg["dtype"], tuple(arg["shape"]), encoded)
                        if key not in offsets:
                            blob.write(b"\0" * (-blob.tell() % 64))
                            offsets[key] = blob.tell()
                            blob.write(encoded)
                        transfer.update(source_program=-1, source_offset=offsets[key])
                    transfers.append(transfer)
                schedule.append({"program": row["index"], "step": step, "transfers": transfers})
        if blob.tell() == 0:
            blob.write(b"\0")
    return {
        "loaded": loaded,
        "model": model,
        "programs": programs,
        "schedule": schedule,
        "source_pins": source_pins,
        "input_blob": pin(blob_path),
        "float_policy": models.selected_float_policy(loaded),
        "host_math_policy": models.selected_host_math_policy(model, loaded.full),
    }


def public_plan(plan: dict) -> dict:
    """The exact generated schedule, without live Python model/build objects."""
    return {
        "programs": [
            {k: p[k] for k in ("index", "name", "steps", "bindings", "n_args", "outputs", "n_outputs")}
            for p in plan["programs"]
        ],
        "schedule": plan["schedule"],
        "source_pins": plan["source_pins"],
        "input_blob": plan["input_blob"],
        "float_policy": plan["float_policy"],
        "host_math_policy": plan["host_math_policy"],
    }
