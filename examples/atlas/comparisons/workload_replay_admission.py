"""Fail-closed admission for trusted workload replay experiment artifacts."""

import json
import os
from pathlib import Path
import posixpath
import stat

import numpy as np

from capture_tinyllama_native import identity, safe_path, array_identity

from atlas_provider_files import PRODUCER_FILE, PRODUCER_FILES, PRODUCER_ID


def admit_executable(value):
    """Admit every symlink ancestor and destination while preserving venv entry.

    Each destination is screened lexically before lstat, readlink or opening any
    destination component. The returned entry path preserves Python venv lookup;
    its final physical binary identity is returned separately for provenance.
    """
    entry = Path(value).absolute()
    if ".." in entry.parts:
        raise ValueError("Parent traversal in executable argument")
    pending, current, links = list(entry.parts[1:]), Path(entry.anchor), 0
    while pending:
        part = pending.pop(0)
        candidate = current / part
        mode = os.lstat(candidate).st_mode
        if stat.S_ISLNK(mode):
            links += 1
            if links > 16:
                raise ValueError("Unbounded executable symlink chain")
            destination = Path(os.readlink(candidate))
            if ".." in destination.parts:
                raise ValueError("Parent traversal in executable symlink destination")
            expanded = destination if destination.is_absolute() else current / destination
            expanded = expanded.joinpath(*pending)
            if ".." in expanded.parts:
                raise ValueError("Parent traversal in expanded executable destination")
            expanded = Path(posixpath.normpath(str(expanded)))
            pending, current = list(expanded.parts[1:]), Path(expanded.anchor)
        else:
            if pending and not stat.S_ISDIR(mode):
                raise ValueError("Executable ancestor is not a directory")
            current = candidate
    if not stat.S_ISREG(os.lstat(current).st_mode) or not os.access(current, os.X_OK):
        raise ValueError("Executable target is not an executable regular file")
    return {"entry": str(entry), "physical_binary": identity(current), "symlinks": links}


def producer_identities(producer, expected_producer_sha256):
    producer = safe_path(producer)
    identities = {name: identity(producer.parent / name) for name in PRODUCER_FILES}
    if identities[PRODUCER_FILE]["sha256"] != expected_producer_sha256:
        raise ValueError("Producer executable differs from requested source epoch")
    return identities


def admit_child_result(model, code_identities, runtime_arrays, expected_shape, native_rows):
    """Authenticate child receipt, source inputs and complete physical result.

    A passed status alone is insufficient: require every nonempty execution run,
    both source bookends, all producer helper pins, artifact hashes, source-array
    identities and output physical layout before returning replacement words.
    """
    model = safe_path(model)
    if not isinstance(code_identities, dict) or set(code_identities) != set(PRODUCER_FILES):
        raise ValueError("Require the complete five-helper producer identity tuple")
    receipt_id = identity(model / "receipt.json")
    receipt = json.loads((model / "receipt.json").read_text())
    pins = {name: item["sha256"] for name, item in code_identities.items()}
    runs = receipt.get("runs")
    if (receipt.get("producer") != PRODUCER_ID
            or receipt.get("status") != "passed_model_only" or not isinstance(runs, list) or not runs
            or receipt.get("code_before") != pins or receipt.get("code_after") != pins
            or receipt.get("source_bookends", {}).get("unchanged") is not True):
        raise ValueError("Unadmitted child producer/status/source bookends")
    if any(not isinstance(run, dict) or not all(run.get(flag) is True for flag in (
            "all_output_bytes_exact", "input_and_guards_preserved", "drained_before_halt_observation"))
           or type(run.get("cycles")) is not int or run["cycles"] <= 0 for run in runs):
        raise ValueError("Child execution run lacks output/guard/drain/cycle gates")
    bookends = receipt["source_bookends"]
    if (not isinstance(receipt.get("source_inputs"), dict) or not receipt["source_inputs"]
            or not isinstance(receipt.get("model_binding_before"), dict) or not receipt["model_binding_before"]
            or bookends.get("input_before") != bookends.get("input_after")
            or receipt.get("source_inputs") != bookends.get("input_before")
            or receipt.get("model_binding_before") != bookends.get("model_after")):
        raise ValueError("Child source/model bookends disagree")
    for name, digest in pins.items():
        identity(model / "producer-sources" / name, digest)
    artifacts = receipt.get("artifacts", {})
    required = ("source.npz", "source-manifest.json", "encoded.npz", "program.bin", "program.S",
                "initial-dram.bin", "expected-bf16-tiles.bin", "actual-bf16-halves-0.bin", "result.npz")
    if any(name not in artifacts for name in required):
        raise ValueError("Child receipt omits a required artifact digest")
    artifact_ids = {name: identity(model / name, artifacts[name]) for name in required}
    with np.load(model / "source.npz", allow_pickle=False) as archive:
        if set(archive.files) != {"A", "B", "native_C"}:
            raise ValueError("Child source NPZ fields differ")
        arrays = {name: archive[name].copy() for name in archive.files}
    m, k, n = expected_shape
    if type(native_rows) is not int or not 0 < native_rows <= m:
        raise ValueError("Unadmitted native row count")
    shapes = {"A": (m, k), "B": (k, n), "native_C": (m, n)}
    for name, array in arrays.items():
        if array.dtype != np.float32 or array.shape != shapes[name] or not np.isfinite(array).all():
            raise ValueError("Child source dtype/shape/finite check differs")
        native = array if name == "B" else array[:native_rows]
        if array_identity(native) != runtime_arrays[name]:
            raise ValueError("Child source differs from actual runtime input/reference: " + name)
        if name != "B" and np.any(array[native_rows:].view(np.uint32)):
            raise ValueError("Child export padding is not positive zero")
    if receipt.get("shape") != [m, k, n]:
        raise ValueError("Child shape differs from current workload")
    with np.load(model / "result.npz", allow_pickle=False) as archive:
        if set(archive.files) != {"target_C", "actual_bits", "expected_bits"}:
            raise ValueError("Child result fields differ")
        words, expected, target = (archive[name].copy() for name in ("actual_bits", "expected_bits", "target_C"))
    if (words.dtype != np.uint16 or words.shape != (m, n) or expected.dtype != np.uint16
            or expected.shape != words.shape or not np.array_equal(words, expected)
            or target.dtype != np.float32 or target.shape != words.shape
            or not np.array_equal(target.view(np.uint32), words.astype(np.uint32) << 16)
            or not np.isfinite(target).all() or np.any(words[native_rows:])):
        raise ValueError("Child result target/oracle/BF16/tail contract differs")
    payload = b"".join(words[r:r + 32, c:c + 32].reshape(32, 2, 16).transpose(1, 0, 2)
                       .astype("<u2").tobytes()
                       for r in range(0, m, 32) for c in range(0, n, 32))
    if (payload != (model / "expected-bf16-tiles.bin").read_bytes()
            or payload != (model / "actual-bf16-halves-0.bin").read_bytes()):
        raise ValueError("Child physical output layout/payload differs from returned words")
    # Rehash all data just consumed to close the admission interval.
    identity(receipt_id["path"], receipt_id["sha256"])
    for item in artifact_ids.values():
        identity(item["path"], item["sha256"])
    return words, receipt, {"receipt": receipt_id, "artifacts": artifact_ids,
                            "code_identities": code_identities}
