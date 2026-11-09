"""Observe every recurrent step through the existing complete-chain guest protocol.

An explicit independently selected output collection supplies comparison bytes,
not numerical-policy, placement, hardware or timing qualification.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np

from merlin.model_check import chain_image_readback
from merlin.model_check.chain_image_objects import payload, pin, reopen
from merlin.runtime.backends.base import get_backend
from merlin.runtime.out_bin import binary_console_diagnostics, parse_binary_console_details
from merlin.runtime.whole_model_readback import decode_outputs

from .model_observation import ObservationError, compare_outputs


def selected_steps(image, reference):
    """Close the complete ordered reference roster before any execution."""
    steps = reference.get("steps")
    schedule = image["plan"]["schedule"]
    if reference.get("schema") != "merlin.complete_chain_output_reference.v1" or not isinstance(steps, list):
        raise ObservationError("select an explicit complete-chain output reference")
    if len(steps) != len(schedule):
        raise ObservationError("reference omits or adds recurrent steps")
    selected = []
    for row, step in zip(steps, schedule, strict=True):
        program = image["plan"]["programs"][step["program"]]
        if row.get("program") != program["name"] or type(row.get("step")) is not int or row["step"] != step["step"]:
            raise ObservationError("reference recurrent steps are missing, duplicated or reordered")
        path = reopen(row["outputs_npz"])
        with np.load(path, allow_pickle=False) as arrays:
            if set(arrays.files) != {f"out{i}" for i in range(program["n_outputs"])}:
                raise ObservationError("reference array roster differs from the complete program ABI")
        selected.append((row, step, program))
    return selected


def compare_chain_outputs(image, reference, console, output):
    """Use the ordinary typed frame decoder and shared representation checker."""
    selected = selected_steps(image, reference)
    readback = chain_image_readback.decode(image, console)
    if readback["runtime_status"] != "pass":
        raise ObservationError("complete chain reported a runtime ABI mismatch")
    words, _metrics, frames = parse_binary_console_details(console)
    comparisons = []
    for ordinal, (row, step, program) in enumerate(selected):
        names = [f"p{step['program']}_s{step['step']}_out{i}" for i in range(program["n_outputs"])]
        own_words = {f"out{i}": words[name] for i, name in enumerate(names)}
        own_frames = {f"out{i}": frames[name] for i, name in enumerate(names)}
        arrays = decode_outputs(own_words, own_frames, program["outputs"])
        with np.load(reopen(row["outputs_npz"]), allow_pickle=False) as expected:
            result = compare_outputs(arrays, expected)
        comparisons.append({"program": program["name"], "step": row["step"], "outputs": result})
        np.savez(output / f"step-{ordinal:04d}.npz", **dict(zip(
            (f"out{i}" for i in range(len(arrays))), arrays, strict=True,
        )))
    return readback, comparisons


def observe(*, image_path, reference_path, reference_sha256, target, output, console=None, timeout_s=600):
    if type(timeout_s) is not int or not 1 <= timeout_s <= 3600 or not isinstance(target, str) or not target:
        raise ObservationError("select an explicit target and bounded execution timeout")
    image_pin = pin(Path(image_path).resolve(strict=True))
    reference_pin = pin(Path(reference_path).resolve(strict=True))
    if reference_pin["sha256"] != reference_sha256:
        raise ObservationError("selected complete-chain reference digest differs")
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise ObservationError("chain observation output must be a fresh directory")
    output.mkdir(parents=True, exist_ok=False)
    record = {
        "schema": "merlin.complete_chain_observation.v1", "status": "running",
        "image": image_pin, "reference": reference_pin, "target": target,
        "counter_kind": "spike_retired_instruction_proxy" if console is None else "raw_counter_ticks",
        "timing_qualified": False, "started_unix": time.time(),
        "scope": ("All declared recurrent outputs; step counters bracket forward invocations "
                  "and exclude transfers/readback."),
        "qualification": "development observation only; numerical-policy, source placement and hardware pending",
    }
    console_pin = None if console is None else pin(Path(console).resolve(strict=True))

    def save():
        (output / "observation.json").write_text(json.dumps(record, indent=2) + "\n")

    save()
    try:
        image = chain_image_readback.verify_image(image_pin)
        if image["transport"] != "full_arrays_v1":
            raise ObservationError("independent representation comparison requires complete raw arrays")
        reference = json.loads(payload(reference_pin))
        selected_steps(image, reference)
        record["boot_elf"] = image["boot_elf"]
        raw = (get_backend(target).run_declared_model_elf(
            reopen(image["boot_elf"]), timeout=timeout_s, capture_bytes=True,
        ) if console_pin is None else payload(console_pin))
        if type(raw) is not bytes:
            raise ObservationError("selected chain backend did not return binary readback")
        (output / "console.bin").write_bytes(raw)
        (output / "console.txt").write_bytes(binary_console_diagnostics(raw))
        readback, comparisons = compare_chain_outputs(image, reference, raw, output)
        record.update(readback=readback, comparison=comparisons)
        record["sum_forward_invocation_ticks"] = sum(row["cycles"] for row in readback["step_cycles"])
        if chain_image_readback.verify_image(image_pin) != image or json.loads(payload(reference_pin)) != reference:
            raise ObservationError("selected image or reference changed during observation")
        selected_steps(image, reference)
        if console_pin is not None:
            reopen(console_pin)
            record["console_source"] = console_pin
        if any(output["mismatches"] for row in comparisons for output in row["outputs"]):
            raise ObservationError("complete recurrent output representations differ")
        record["status"] = "exact_all_outputs_passed"
    except Exception as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        (output / "traceback.txt").write_text(traceback.format_exc())
    record["finished_unix"] = time.time()
    save()
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("image", "reference", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--reference-sha256", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--console", type=Path)
    parser.add_argument("--timeout-s", type=int, default=600)
    args = parser.parse_args(argv)
    record = observe(image_path=args.image, reference_path=args.reference, reference_sha256=args.reference_sha256,
                     target=args.target, output=args.output, console=args.console, timeout_s=args.timeout_s)
    print(json.dumps({key: record[key] for key in ("status", "counter_kind")}, indent=2))
    return 0 if record["status"] == "exact_all_outputs_passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
