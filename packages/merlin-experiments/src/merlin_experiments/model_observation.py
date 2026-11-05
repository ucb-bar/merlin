"""Observe a complete built model with an explicitly selected independent reference.

This development command preserves ordinary backend execution and full-readback
checks. It supplies no source, numerical-policy, hardware or timing qualification.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np

from merlin.common.digest import sha256_file


class ObservationError(ValueError):
    """A complete selected observation could not be established."""


def compare_outputs(values, reference):
    """Compare every element's representation, including signed zero and NaN bits."""
    expected = {f"out{i}" for i in range(len(values))}
    if not values or set(reference.files) != expected:
        raise ObservationError("reference must contain exactly out0 through the last output")
    comparisons = []
    for i, value in enumerate(values):
        name = f"out{i}"
        selected = reference[name]
        if value.shape != selected.shape or value.dtype != selected.dtype or value.dtype.hasobject:
            raise ObservationError(f"output shape/dtype differs from selected reference: {name}")
        actual_bytes = np.ascontiguousarray(value).view(np.uint8).reshape(value.size, value.dtype.itemsize)
        selected_bytes = np.ascontiguousarray(selected).view(np.uint8).reshape(value.size, value.dtype.itemsize)
        mismatches = int(np.count_nonzero(np.any(actual_bytes != selected_bytes, axis=1)))
        comparisons.append({"name": name, "elements": int(value.size), "mismatches": mismatches})
    return comparisons


def observe(
    *,
    receipt: Path,
    reference_npz: Path,
    reference_sha256: str,
    output: Path,
    console: Path | None = None,
    console_counter_kind: str = "raw_counter_ticks",
    timeout_s: int = 600,
):
    """Execute or rejudge one exact ELF; retain failed observations as well as successes.

    The caller must select the reference digest independently. Reference bytes
    are never inferred from a model name or substituted from an earlier run.
    This command currently measures Spike's instruction proxy only. Physical
    engines require their own selected measurement and provenance workflow.
    """
    from merlin.compile.baremetal_model import read_full_outputs
    from merlin.compile.model_execution_inputs import strict_tree_sha256
    from merlin.runtime.backends.base import get_backend
    from merlin.runtime.out_bin import binary_console_diagnostics
    from merlin.runtime.whole_model_readback import parse_group_profile

    if type(timeout_s) is not int or not 1 <= timeout_s <= 3600:
        raise ObservationError("timeout must be an integer in 1..3600 seconds")
    if console_counter_kind not in {"raw_counter_ticks", "spike_retired_instruction_proxy"}:
        raise ObservationError("unsupported retained-console counter annotation")
    if len(reference_sha256) != 64 or set(reference_sha256) - set("0123456789abcdef"):
        raise ObservationError("select a complete lowercase reference SHA-256")
    receipt, reference_npz = Path(receipt).resolve(strict=True), Path(reference_npz).resolve(strict=True)
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise ObservationError("observation output must be a fresh directory")
    build = json.loads(receipt.read_bytes())
    if (
        build.get("schema") != "merlin.baremetal-saved-model.v1"
        or build.get("status") != "compiled"
        or build.get("inputs", {}).get("readback") != "full"
    ):
        raise ObservationError("select a compiled ordinary model receipt with full readback")
    elf = Path(build["output"]["elf"]).resolve(strict=True)
    capture = Path(build["inputs"]["capture"]).resolve(strict=True)
    if output == capture or output.is_relative_to(capture):
        raise ObservationError("observation output cannot modify the selected capture")
    output.mkdir(parents=True, exist_ok=False)
    record = {
        "schema": "merlin.complete_model_observation.v1",
        "status": "running",
        "receipt": {"path": str(receipt), "sha256": sha256_file(receipt)},
        "elf": {"path": str(elf), "sha256": build["output"]["elf_sha256"]},
        "reference": {"path": str(reference_npz), "sha256": reference_sha256},
        "capture": {"path": str(capture), "sha256": build["inputs"]["capture_tree"]},
        "counter_kind": "spike_retired_instruction_proxy" if console is None else console_counter_kind,
        "counter_kind_source": "declared_model_spike_execution" if console is None else "original_caller_annotation",
        "timing_qualified": False,
        "qualification": (
            "development observation; independent reference selection grants no numerical-policy admission"
        ),
        "started_unix": time.time(),
    }
    console_path = None if console is None else Path(console).resolve(strict=True)

    def verify():
        for row in (record["receipt"], record["elf"], record["reference"]):
            if sha256_file(Path(row["path"])) != row["sha256"]:
                raise ObservationError("selected file changed: " + row["path"])
        if strict_tree_sha256(capture) != record["capture"]["sha256"]:
            raise ObservationError("selected capture changed")
        if console_path is not None and sha256_file(console_path) != record["console_source"]["sha256"]:
            raise ObservationError("selected retained console changed")

    def save():
        (output / "observation.json").write_text(json.dumps(record, indent=2) + "\n")

    save()
    try:
        if console_path is not None:
            record["console_source"] = {"path": str(console_path), "sha256": sha256_file(console_path)}
        verify()
        if console_path is None:
            raw = get_backend(build["inputs"]["target"]).run_declared_model_elf(
                elf, timeout=timeout_s, capture_bytes=True
            )
        else:
            raw = console_path.read_bytes()
        if not isinstance(raw, bytes):
            raise ObservationError("selected model backend did not return binary readback")
        (output / "console.bin").write_bytes(raw)
        text = binary_console_diagnostics(raw).decode("utf-8", errors="replace")
        (output / "console.txt").write_text(text)
        values, metrics = read_full_outputs(raw, capture, build["output"]["build_hash"])
        with np.load(reference_npz, allow_pickle=False) as reference:
            record["comparison"] = compare_outputs(values, reference)
        record["metrics"] = metrics
        np.savez(output / "outputs.npz", **{f"out{i}": value for i, value in enumerate(values)})
        if build["inputs"].get("group_profile"):
            profile = parse_group_profile(text)
            if profile is None:
                raise ObservationError("selected group profile is absent")
            record["group_profile"] = profile
            if profile["dropped"]:
                raise ObservationError("selected group profile dropped calls")
        verify()
        if any(row["mismatches"] for row in record["comparison"]):
            raise ObservationError("complete output representations differ from selected reference")
        record["status"] = "exact_all_outputs_passed"
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        (output / "traceback.txt").write_text(traceback.format_exc())
    record["finished_unix"] = time.time()
    save()
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(prog="merlin experiment model-observe", description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--reference-npz", type=Path, required=True)
    parser.add_argument("--reference-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--console", type=Path, help="rejudge these exact retained bytes without executing")
    parser.add_argument(
        "--console-counter-kind",
        default="raw_counter_ticks",
        choices=("raw_counter_ticks", "spike_retired_instruction_proxy"),
    )
    parser.add_argument("--timeout-s", type=int, default=600)
    args = parser.parse_args(argv)
    result = observe(**vars(args))
    print(
        json.dumps(
            {
                "status": result["status"],
                "observation": str(args.output / "observation.json"),
                "error": result.get("error"),
            },
            indent=2,
        )
    )
    return 0 if result["status"] == "exact_all_outputs_passed" else 1
