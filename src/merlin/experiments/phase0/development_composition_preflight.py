"""Connect public source-plan checks to complete build storage before compilation.

This read-only preflight reuses the original mixed-program and pointer-layout
owners. It is not arithmetic, source-execution, hardware or timing admission.
All input buffers, temporaries and live-outs count toward the selected storage
limit; parameter bytes alone are not a complete linked-program budget.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import stat
from math import prod
from pathlib import Path

from merlin.capture.safetensors import load_header
from merlin.common import strict_json
from merlin.targetgen import bundle_pack
from merlin.targetgen.corpus_spec import dtype_info
from merlin.targetgen.oot_starterkit.plan import source_operation_inventory, validate_mixed_program_plan

SCHEMA = "merlin.phase0.public_composition_build_preflight.v1"


def _pin(path, limit):
    path = Path(path)
    if (
        not path.is_absolute()
        or ".." in path.parts
        or any(p.is_symlink() for p in (path, *path.parents))
        or not stat.S_ISREG(path.stat().st_mode)
        or path.stat().st_size > limit
    ):
        raise ValueError("preflight inputs require bounded direct original files")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            size += len(block)
            if size > limit:
                raise ValueError("preflight artifact exceeds the explicit original limit")
            digest.update(block)
    return {"path": str(path), "bytes": size, "sha256": digest.hexdigest()}


def _json(path, limit):
    return strict_json.loads(Path(path).read_bytes(), max_bytes=limit)


def _dtype(value):
    # Capture spells plain floating scalars float<N>; the original capsule
    # vocabulary uses f<N>. This preserves the declared width and signedness.
    if isinstance(value, str) and value.startswith("float") and value[5:].isdigit():
        value = "f" + value[5:]
    return dtype_info(value)[1]


def _parameters(manifest, inventory, weights, size):
    if not isinstance(manifest, dict) or set(manifest) != {str(i) for i in range(len(inventory["arguments"]))}:
        raise ValueError("capture manifest must preserve every exact source argument index")
    header, payload_at = load_header(weights)
    names = set(header) - {"__metadata__"}
    expected = set()
    parameters = []
    for index, source in enumerate(inventory["arguments"]):
        row = manifest[str(index)]
        if not isinstance(row, dict) or row.get("kind") not in {"param", "buffer", "input"}:
            raise ValueError("capture argument lacks its original ownership kind")
        if row["kind"] == "input":
            if row.get("weight") is not None or not isinstance(row.get("name"), str) or not row["name"]:
                raise ValueError("external input ownership cannot shadow a parameter")
            continue
        name = row.get("weight") or row.get("name")
        if not isinstance(name, str) or not name or name not in names:
            raise ValueError("complete captured parameter/buffer payload is unavailable")
        expected.add(name)
        item = header[name]
        if (
            not isinstance(item, dict)
            or set(item) != {"dtype", "shape", "data_offsets"}
            or not isinstance(row.get("shape"), list)
            or any(type(n) is not int or n < 1 for n in row["shape"])
            or row.get("shape") != source.get("shape")
            or item["shape"] != source.get("shape")
            or _dtype(row.get("dtype")) != source.get("dtype")
            or not isinstance(item["dtype"], str)
            or _dtype(item["dtype"].lower()) != source.get("dtype")
        ):
            raise ValueError("captured parameter dtype/shape differs from the original typed source")
        offsets = item["data_offsets"]
        shape = item["shape"]
        width = bundle_pack.element_bytes(source["dtype"])
        if (
            not isinstance(shape, list)
            or any(type(n) is not int or n < 1 for n in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(n) is not int for n in offsets)
            or not 0 <= offsets[0] <= offsets[1] <= size - payload_at
            or offsets[1] - offsets[0] != prod(shape) * width
        ):
            raise ValueError("complete captured parameter extent is malformed or truncated")
        parameters.append(
            {"arg_index": index, "name": name, "shape": shape, "dtype": source["dtype"], "data_offsets": offsets}
        )
    if names != expected:
        raise ValueError("captured parameter namespace has an unowned or missing tensor")
    intervals = sorted({tuple(row["data_offsets"]) for row in parameters})
    if len(intervals) != len(expected):
        raise ValueError("distinct captured parameters overlap or counterfeit immutable sharing")
    cursor = 0
    for begin, end in intervals:
        if begin != cursor:
            raise ValueError("captured parameter payload has a gap or overlap")
        cursor = end
    if cursor != size - payload_at:
        raise ValueError("captured parameter payload has unowned or missing bytes")
    return parameters, size - payload_at


def check(
    *,
    source,
    command_buffer,
    lowered,
    weight_manifest,
    weights,
    row_pitch_elements,
    max_storage_bytes,
    max_artifact_bytes,
):
    """Read existing artifacts only; failed preflight never drops required work.

    Callers supply the original selected row granule and admission ceilings.
    Success does not replace capture/reference/ISA/numerical/writer checks or
    prove that a device actually executed any declared task.
    """
    result = dict(
        schema=SCHEMA,
        status="refused",
        scope="public source/layout/complete-payload build preflight only",
        numerical_equivalence="UNVERIFIED",
        accelerator_execution="UNVERIFIED",
        carried_state="UNVERIFIED; original session contract still required",
        timing="UNKNOWN",
        formal_admission=False,
    )
    try:
        if any(
            type(value) is not int or value < 1 for value in (row_pitch_elements, max_storage_bytes, max_artifact_bytes)
        ):
            raise ValueError("preflight needs explicit positive integer granules and unchanged finite ceilings")
        paths = {
            "source": Path(source),
            "command_buffer": Path(command_buffer),
            "lowered": Path(lowered),
            "weight_manifest": Path(weight_manifest),
            "weights": Path(weights),
        }
        before = {name: _pin(path, max_artifact_bytes) for name, path in paths.items()}
        result["inputs"] = before
        text = paths["source"].read_bytes()
        cb = _json(paths["command_buffer"], max_artifact_bytes)
        llvm = paths["lowered"].read_text()
        checked = validate_mixed_program_plan(text, cb, llvm)
        result["original_public_plan"] = checked
        if not checked["ok"]:
            raise ValueError("original complete source/task/return preflight refused")
        inventory = source_operation_inventory(text)
        manifest = _json(paths["weight_manifest"], max_artifact_bytes)
        parameters, payload = _parameters(manifest, inventory, paths["weights"], before["weights"]["bytes"])
        layout = bundle_pack.plan(cb, row_pitch_elements=row_pitch_elements, weight_manifest=manifest)
        if layout.absent_indices:
            raise ValueError("compiled ABI omitted original captured source arguments")
        entry = cb["params"]["global_program_plan"]["entry_bindings"]
        for index, source_arg in enumerate(inventory["arguments"]):
            name = entry[index]
            if any(cb["tensors"][name].get(key) != source_arg.get(key) for key in ("shape", "dtype")):
                raise ValueError("compiled pointer ABI differs from exact source arguments")
        storage = layout.const_bytes + layout.mutable_bytes
        result.update(
            source_operations=inventory["source_op_count"],
            source_returns=inventory["returns"],
            complete_argument_count=len(layout.arguments),
            parameter_uses=parameters,
            unique_captured_parameters=len({row["name"] for row in parameters}),
            actual_captured_parameter_payload_bytes=payload,
            compiled_const_bytes=layout.const_bytes,
            compiled_mutable_bytes=layout.mutable_bytes,
            complete_compiled_storage_bytes=storage,
            max_storage_bytes=max_storage_bytes,
            row_pitch_elements=row_pitch_elements,
            declared_device_tasks=sum(row["kind"] == "device" for row in cb["params"]["global_program_plan"]["tasks"]),
            pointer_order=list(layout.abi_order),
            output_bindings=list(layout.output_bindings),
        )
        if storage > max_storage_bytes:
            raise ValueError(
                "complete compiled parameters, inputs, temporaries and outputs exceed unchanged storage ceiling"
            )
        after = {name: _pin(path, max_artifact_bytes) for name, path in paths.items()}
        if before != after:
            raise ValueError("original composition artifacts changed during preflight")
        result.update(status="public_composition_build_preflight_pass", findings=[])
    except Exception as error:
        result["findings"] = [str(error)]
        result["failure_type"] = type(error).__name__
    return result


def main(argv=None):
    """Run the same check under an original shared leaf; no build or simulation."""
    from merlin.common.cpu_pool import shared_compute_pool

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "command-buffer", "lowered", "weight-manifest", "weights"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("row-pitch-elements", "max-storage-bytes", "max-artifact-bytes"):
        parser.add_argument("--" + name, type=int, required=True)
    args = parser.parse_args(argv)
    with shared_compute_pool(timeout_s=60):
        result = check(**vars(args))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "public_composition_build_preflight_pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
