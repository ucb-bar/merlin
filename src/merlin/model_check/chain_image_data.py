"""Generate the guest's typed tables and optional all-output reference blob."""

from __future__ import annotations

import io
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from merlin.llvmlower.c_runtime import DT_BYTES, NP_OF

from .chain_image_objects import payload, pin
from .chain_image_plan import _storage
from .full_quantized import sealed_reference

TRANSPORTS = {"full_arrays_v1", "all_element_comparison_v1"}
KINDS = {"f32": 1, "f64": 2, "bf16": 3, "f16": 4, "i1": 5, "i8": 0, "i16": 0, "i32": 0, "i64": 0}


def references(selection, plan, directory):
    """Consume an explicitly admitted supplement; never derive new references."""
    compact = selection["transport"] == "all_element_comparison_v1"
    path = directory / "references.bin"
    source_pins, offsets, seal_binding = [], [], None
    with path.open("xb") as stream:
        if compact:
            manifest = json.loads(payload(selection["reference"]))
            if (
                manifest.get("schema") != "merlin.operator.full_declared_output_reference.v1"
                or manifest.get("status") != "complete"
                or manifest.get("model") != selection["model"]
                or manifest.get("selected_identity") != selection["reference_identity"]
                or manifest.get("input", {}).get("sha256") != selection["inputs"]["sha256"]
            ):
                raise ValueError("all-output reference does not match the selected model/input/identity")
            contract = manifest["implementation"]["numeric_contract"].get("host_math")
            policy = "native" if contract is None else contract["policy"]
            if policy != plan["host_math_policy"]:
                raise ValueError("all-output reference uses another host math policy")
            bundle = Path(selection["bundle"])
            frozen, seal_binding = sealed_reference(
                SimpleNamespace(root=bundle), plan["model"], Path(selection["models"]["path"]), manifest["input_id"]
            )
            identity = manifest["selected_identity"]
            if (
                seal_binding["manifest_sha256"] != identity["bundle_manifest_sha256"]
                or seal_binding["sums_sha256"] != identity["bundle_sums_sha256"]
                or seal_binding["models_sha256"] != identity["models_config_sha256"]
                or manifest.get("judged_crossjoin", {}).get("sealed_npz", {}).get("sha256") != seal_binding["sha256"]
                or manifest.get("judged_crossjoin", {}).get("bit_exact") is not True
            ):
                raise ValueError("supplement is not joined to the selected sealed quantized reference")
            source_pins += [pin(bundle / "MANIFEST.json"), pin(bundle / "SHA256SUMS"), pin(Path(seal_binding["path"]))]
            judged = {
                (b["program"], b["step"], b["output"]): value
                for b, value in zip(seal_binding["outputs"], frozen, strict=True)
            }
            source_pins.append(selection["reference"])
            original = {p["program"]: p for p in manifest["programs"]}
            if len(original) != len(plan["programs"]) or len(original) != len(manifest["programs"]):
                raise ValueError("reference has an incomplete/duplicate program roster")
            for program in plan["programs"]:
                selected = original[program["name"]]
                if (
                    selected["capture_program"]
                    != {k: v for k, v in pin(Path(selected["stage"]) / "model.mlir").items() if k != "bytes"}
                    or selected["capture_program"]["sha256"]
                    != pin(Path(program["production"]["receipt"]["inputs"]["capture"]) / "model.mlir")["sha256"]
                    or selected["steps_declared"] != program["steps"]
                    or len(selected["steps"]) != program["steps"]
                ):
                    raise ValueError("reference capture or complete step roster differs")
                for step, row in enumerate(selected["steps"]):
                    if row["step"] != step:
                        raise ValueError("reference step order is incomplete")
                    if len(row["outputs"]) != program["n_outputs"]:
                        raise ValueError("reference typed output roster differs")
                    source_pins.append(row["outputs_file"])
                    with np.load(io.BytesIO(payload(row["outputs_file"])), allow_pickle=False) as arrays:
                        if set(arrays.files) != {f"out{i}" for i in range(program["n_outputs"])}:
                            raise ValueError("reference output arity differs")
                        locations = []
                        for index, (shape, dtype) in enumerate(program["outputs"]):
                            value = arrays[f"out{index}"]
                            declared = row["outputs"][index]
                            if (
                                declared["output"] != index
                                or declared["shape"] != shape
                                or declared["element_type"] != dtype
                                or list(value.shape) != shape
                                or value.dtype.str != declared["dtype"]
                                or value.dtype != np.dtype(np.float32 if dtype == "bf16" else NP_OF[dtype])
                                or not np.isfinite(value).all()
                            ):
                                raise ValueError("reference output shape/dtype/finiteness differs")
                            import hashlib

                            if hashlib.sha256(value.tobytes()).hexdigest() != declared["sha256_of_array_bytes"]:
                                raise ValueError("reference typed array digest differs")
                            key = (program["name"], step, index)
                            if key in judged:
                                selected_value = judged.pop(key)
                                if (
                                    value.dtype != selected_value.dtype
                                    or value.shape != selected_value.shape
                                    or value.tobytes() != selected_value.tobytes()
                                ):
                                    raise ValueError("all-output supplement differs from its sealed judged output")
                            stream.write(b"\0" * (-stream.tell() % 64))
                            locations.append(stream.tell())
                            stream.write(_storage(value, dtype).tobytes())
                        offsets.append(locations)
            if judged:
                raise ValueError("supplement omitted a sealed judged output")
        else:
            offsets = [[0] * plan["programs"][s["program"]]["n_outputs"] for s in plan["schedule"]]
        if not stream.tell():
            stream.write(b"\0")
    return {"blob": pin(path), "offsets": offsets, "source_pins": source_pins, "seal_binding": seal_binding}


def _number(value):
    if type(value) is not int or value < 0 or value >= 1 << 63:
        raise ValueError("guest ABI extent is not a nonnegative signed 64-bit integer")
    return str(value) + "ULL"


def render(plan, namespaced, layout, reference, *, transport, counter_symbol, samples=4, group_profiles=None):
    """Only trusted numeric literals/issued symbol prefixes enter generated C."""
    if transport not in TRANSPORTS or type(samples) is not int or samples != 4:
        raise ValueError("select a supported bounded chain output transport")
    if not counter_symbol.isidentifier() or not counter_symbol.isascii():
        raise ValueError("counter needs an explicit C symbol")
    if group_profiles is None:
        group_profiles = [None] * len(plan["programs"])
    if len(group_profiles) != len(plan["programs"]):
        raise ValueError("group profiling requires one hook selection per program")
    rows = [
        f"#define CHAIN_PROGRAMS {len(plan['programs'])}",
        f"#define CHAIN_STEPS {len(plan['schedule'])}",
        f"#define CHAIN_MAX_ARGS {max(p['n_args'] for p in plan['programs'])}",
        f"#define CHAIN_COMPACT {int(transport == 'all_element_comparison_v1')}",
        f"#define CHAIN_SAMPLES {samples}",
        f"#define CHAIN_ATOL {float(plan['float_policy']['atol']).hex()}",
        f"#define CHAIN_RTOL {float(plan['float_policy']['rtol']).hex()}",
        f"#define CHAIN_INPUTS ((const unsigned char *)(uintptr_t){_number(layout['inputs'])})",
        f"#define CHAIN_REFERENCES ((const unsigned char *)(uintptr_t){_number(layout['references'])})",
        f"extern uint64_t {counter_symbol}(void);",
        f"#define chain_counter_now {counter_symbol}",
    ]
    programs = []
    for p, names, weights, profile in zip(plan["programs"], namespaced, layout["weights"], group_profiles, strict=True):
        i, prefix = p["index"], names["prefix"]
        sizes = names["table_sizes"]
        rows += [
            f"_Static_assert(sizeof(merlin_arg_t) * {p['n_args']} == {sizes['MERLIN_ARGS']}, "
            '"retained argument extent");',
            f'_Static_assert(sizeof(void *) * {p["n_args"]} == {sizes["MERLIN_INPUT_PTR"]}, "retained input extent");',
            f"_Static_assert(sizeof(void *) * {p['n_outputs']} == {sizes['MERLIN_OUTPUT_PTR']}, "
            '"retained output extent");',
        ]
        rows += [
            f"extern const merlin_arg_t {prefix}MERLIN_ARGS[];",
            f"extern void *const {prefix}MERLIN_INPUT_PTR[];",
            f"extern void *const {prefix}MERLIN_OUTPUT_PTR[];",
            f"extern void {prefix}merlin_invoke(void **);",
        ]
        reset, dump = "0", "0"
        if profile is not None:
            if set(profile) != {"reset", "dump"} or any(
                not isinstance(symbol, str) or not symbol.isascii() or not symbol.isidentifier()
                for symbol in profile.values()
            ):
                raise ValueError("group profiling requires explicit reset/dump C symbols")
            reset, dump = profile["reset"], profile["dump"]
            rows += [f"extern void {reset}(void);", f"extern void {dump}(uint64_t, uint64_t);"]
        specs = []
        for shape, dtype in p["outputs"]:
            if len(shape) > 8 or dtype not in KINDS or not math.prod(shape):
                raise ValueError("unsupported static guest output ABI")
            dims = ",".join(_number(d) for d in shape) or "0"
            specs.append(
                "{" + f"{len(shape)},{DT_BYTES[dtype]},{KINDS[dtype]},{{{dims}}},{_number(math.prod(shape))}" + "}"
            )
        rows.append(f"static const chain_spec specs_{i}[] = {{" + ",".join(specs) + "};")
        programs.append(
            "{" + f"{prefix}MERLIN_ARGS,{p['n_args']},{p['n_outputs']},{prefix}MERLIN_INPUT_PTR,"
            f"{prefix}MERLIN_OUTPUT_PTR,(const void *)(uintptr_t){_number(weights['begin'])},"
            f"{_number(weights['bytes'])},{prefix}merlin_invoke,specs_{i},{reset},{dump}" + "}"
        )
    rows.append("static const chain_program PROGRAMS[] = {" + ",".join(programs) + "};")
    steps = []
    for index, (step, refs) in enumerate(zip(plan["schedule"], reference["offsets"], strict=True)):
        values = [
            "{" + f"{t['argument']},{t['source_program']},{t.get('source_output', 0)},"
            f"{_number(t.get('source_offset', 0))},{_number(t['bytes'])}" + "}"
            for t in step["transfers"]
        ]
        rows += [
            f"static const chain_transfer transfers_{index}[] = {{" + ",".join(values) + "};",
            f"static const uint64_t refs_{index}[] = {{" + ",".join(_number(x) for x in refs) + "};",
        ]
        steps.append("{" + f"{step['program']},{step['step']},{len(values)},transfers_{index},refs_{index}" + "}")
    rows.append("static const chain_step STEPS[] = {" + ",".join(steps) + "};")
    return "\n".join(rows) + "\n"
