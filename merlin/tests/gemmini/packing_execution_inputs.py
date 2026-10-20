"""Public synthetic inputs for the opt-in ordinary Gemmini packing experiment.

This validator constructs a trusted runtime materialization, not a capture,
numerical admission, simulator certificate or restored linked-ELF service.
"""

from __future__ import annotations

import hashlib
import json
import runpy
import struct
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from merlin.common.digest import sha256_file
from merlin.common.paths import repo_root
from merlin.common.strict_json import loads
from merlin.compile.materialized_model_inputs import MaterializedModelInputs
from merlin.compile.model_execution_inputs import strict_tree_sha256
from merlin.frontends.linalg_mlir import parse_mlir_text

SCHEMA = "merlin.test.gemmini_packing_pair.v1"
PUBLIC_COMPILER_SHA = "29213759089b71de26f8a5ab26ca6ef1b8071dc0bb690041eb7de5d0efb53f24"
SOURCE_SHA = "10973c303c8e47638ea1dbfd8d7510adf7e72e7b80255b8cfc37dae58c3b816d"
FILES = ("instruction_declaration", "quiet_contract", "facts", "board_catalog", "dts")
TREES = ("host_package", "public_compiler")


def canonical(value, *, directory=False):
    assert type(value) is str and Path(value).is_absolute(), "selection needs an absolute path"
    path = Path(value)
    assert path.resolve() == path and not any(p.is_symlink() for p in (path, *path.parents))
    assert path.is_dir() if directory else path.is_file()
    return path


def pin(path):
    return {"path": str(path), "sha256": sha256_file(path)}


def read_selection(path):
    """Reopen only explicit original data; no selected record grants authority."""
    path = canonical(str(path))
    document = loads(path.read_bytes(), max_bytes=64 << 10)
    assert type(document) is dict and set(document) == {"schema", "board", "trace_contract", *FILES, *TREES}
    assert document["schema"] == SCHEMA
    assert type(document["board"]) is str and 0 < len(document["board"]) <= 128
    for name in (*FILES, "trace_contract"):
        row = document[name]
        if name == "trace_contract" and row is None:
            continue
        assert type(row) is dict and set(row) == {"path", "sha256"}
        assert pin(canonical(row["path"])) == row, f"changed selected {name}"
    for name in TREES:
        row = document[name]
        assert type(row) is dict and set(row) == {"path", "sha256", "n_files"}
        assert type(row["n_files"]) is int and 0 < row["n_files"] <= 64
        if name == "public_compiler":
            assert row["n_files"] == 16 and row["sha256"] == PUBLIC_COMPILER_SHA
        assert {"path": row["path"], **strict_tree_sha256(canonical(row["path"], directory=True))} == row
    return document, pin(path)


def recheck_selection(document, selected_pin):
    assert pin(canonical(selected_pin["path"])) == selected_pin
    assert read_selection(selected_pin["path"])[0] == document


def original_case(monkeypatch):
    """Use the existing public ordered reference, without running its native tests."""
    directory = repo_root() / "merlin/tests/ir"
    monkeypatch.syspath_prepend(str(directory))
    paths = [
        directory / name
        for name in (
            "producer_packing_connected.mlir",
            "test_packed_producer_connected.py",
            "test_packed_producer_native.py",
            "test_packed_producer_destinations.py",
        )
    ]
    paths.extend(
        repo_root() / "merlin/tests/gemmini" / name
        for name in ("packing_execution_inputs.py", "test_public_packing_execution.py")
    )
    pins = [pin(path) for path in paths]
    assert pins[0]["sha256"] == SOURCE_SHA
    reference = SimpleNamespace(**runpy.run_path(str(paths[1])))
    source = paths[0].read_text()
    module = parse_mlir_text(source)
    module.verify()
    entry = module.body.block.first_op
    assert len(tuple(module.body.block.ops)) == 1 and entry.sym_name.data == "forward"
    assert tuple(map(str, entry.function_type.inputs)) == (
        "tensor<2x4xi8>",
        "tensor<4x12xi8>",
        "tensor<12x12xi8>",
        "tensor<2x12xi32>",
    )
    assert tuple(map(str, entry.function_type.outputs)) == ("tensor<2x12xi32>", "tensor<2x4x3xi8>", "tensor<2x12xi32>")
    assert pins == [pin(path) for path in paths]
    return source, reference, pins


def materialize(destination, *, epoch, state, source, reference, source_pins):
    """Bind complete source/ABI/reference artifacts after original validation."""
    assert type(epoch) is int and 0 <= epoch < 3
    assert type(state) is np.ndarray and state.dtype == np.int32 and state.shape == (2, 12)
    assert max(abs(int(value)) for value in state.flat) <= 1 << 20
    assert hashlib.sha256(source.encode()).hexdigest() == SOURCE_SHA
    assert source_pins == [pin(canonical(row["path"])) for row in source_pins]
    arrays = reference.inputs(epoch, state.copy())
    expected = reference.reference(*[value.copy() for value in arrays])
    assert [(value.shape, value.dtype) for value in arrays] == [
        ((2, 4), np.dtype("int8")),
        ((4, 12), np.dtype("int8")),
        ((12, 12), np.dtype("int8")),
        ((2, 12), np.dtype("int32")),
    ]
    assert [(value.shape, value.dtype) for value in expected] == [
        ((2, 12), np.dtype("int32")),
        ((2, 4, 3), np.dtype("int8")),
        ((2, 12), np.dtype("int32")),
    ]
    destination.mkdir()
    (destination / "model.mlir").write_text(source)
    (destination / "weights.safetensors").write_bytes(struct.pack("<Q", 8) + b"{}      ")
    names = ("x", "w1", "w2", "state")
    (destination / "weights.safetensors.manifest.json").write_text(
        json.dumps({str(i): {"kind": "input", "name": name} for i, name in enumerate(names)})
    )
    np.savez(destination / "inputs.npz", **{f"in{i}": value for i, value in enumerate(arrays)})
    (destination / "input_order.json").write_text(json.dumps(dict(zip(names, range(4), strict=True))))
    np.save(destination / "golden.npy", expected[0])
    output_names = ("next_state", "packed", "wide")
    np.savez(destination / "goldens.npz", **dict(zip(output_names, expected, strict=True)))
    (destination / "output_order.json").write_text(json.dumps(output_names))
    artifacts = {
        path.name: {"sha256": sha256_file(path), "size_bytes": path.stat().st_size} for path in destination.iterdir()
    }
    bound = MaterializedModelInputs.bind(
        destination,
        artifacts=artifacts,
        validator_identity={
            "worker_sha256": sha256_file(Path(__file__).resolve()),
            "source_and_reference": source_pins,
            "scope": "fixed public source SHA, verified typed ABI and independent ordered integer reference only",
        },
        source_identity={
            "content_sha256": SOURCE_SHA,
            "artifacts": source_pins,
            "role": "public synthetic source; no capture or target qualification",
        },
    )
    assert not (destination / "capture_receipt.json").exists()
    bound.verify(destination)
    assert source_pins == [pin(Path(row["path"])) for row in source_pins]
    return bound, arrays, expected


def actual_observation(directory, stage, *, before=()):
    from merlin.common import invocation_record as I

    records = []
    for path in (directory / "invocations").glob("*/invocation.json"):
        if path not in before:
            record = json.loads(path.read_text())
            if record.get("stage") == stage:
                records.append((path, I.verify(path)))
    assert len(records) == 1, f"expected exactly one actual {stage} invocation"
    return records[0]


def committed_custom_words(raw, *, match, mask):
    """Read bounded Spike commit words; disassembly alone is not retirement."""
    assert len(raw) <= 32 << 20
    result = []
    for line in raw.decode("utf-8", errors="strict").splitlines():
        fields = line.split()
        if len(fields) < 5 or fields[0] != "core" or not fields[1].endswith(":") or not fields[2].isdigit():
            continue
        assert fields[1][:-1].isdigit() and fields[3].startswith("0x")
        assert fields[4].startswith("(0x") and fields[4].endswith(")"), "unsupported Spike commit grammar"
        word = int(fields[4][1:-1], 16)
        assert 0 <= word < 1 << 32
        if word & mask == match:
            result.append({"hart": int(fields[1][:-1]), "pc": int(fields[3], 16), "word": word})
            assert len(result) <= 100_000
    assert result, "no selected custom instruction retirement observed"
    return result


def verify_event_bytes(elf, events):
    """Join every retired word to a unique actual executable section."""
    from merlin.targetgen.elf_lanes import executable_sections

    blob = canonical(str(elf)).read_bytes()
    assert blob[:4] == b"\x7fELF" and blob[5] == 1, "only little-endian ELF trace joins supported"
    sections = executable_sections(blob)
    assert events
    for event in events:
        matches = [
            (off + event["pc"] - address)
            for _, off, size, address in sections
            if address <= event["pc"] and event["pc"] + 4 <= address + size
        ]
        assert len(matches) == 1, "retired PC must belong to exactly one executable section"
        off = matches[0]
        assert int.from_bytes(blob[off : off + 4], "little") == event["word"], "retired word differs from actual ELF"


def recheck_artifacts(pins, *, invocations=(), decoder_receipts=()):
    """Callback-free final byte custody; this does not issue admission."""
    from merlin.common import invocation_record as I

    for row in pins:
        assert pin(canonical(row["path"])) == row, "retained artifact changed"
    for row in decoder_receipts:
        assert pin(canonical(row["path"])) == row, "native decoder receipt changed"
        receipt = json.loads(Path(row["path"]).read_bytes())
        for artifact in receipt["pins"]:
            expected = {key: artifact[key] for key in ("path", "sha256")}
            assert pin(canonical(artifact["path"])) == expected, "native decoder artifact changed"
            if Path(artifact["path"]).name == "invocation.json":
                I.verify(artifact["path"])
    for row in invocations:
        assert pin(canonical(row["path"])) == row, "original invocation receipt changed"
        I.verify(row["path"])
