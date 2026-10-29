"""Concrete decoder proofs, archive/C-generation correspondence, rejection and native tests."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import numpy as np
import pytest
from lean_qinner_decoder_cases import KINDS, SHAPES, array_from_words, native_compare, record

from merlin.verify.lean.qinner import prepare, run, write
from merlin.verify.lean.qinner_decoder import decode_records, payload


@pytest.fixture(scope="module")
def compiler():
    path = Path(os.environ.get("MERLIN_M2M_PYTHON", "/home/jeremy/Projects/model2MLIR/.venv/bin/python"))
    assert path.is_file(), "native compiler environment is required for decoder qualification"
    prepare()
    return path


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("shape", SHAPES)
def test_decode_exact_archive_cells(tmp_path, compiler, kind, shape):
    entry, expected = record(kind, shape)
    report = decode_records([entry], tmp_path)
    result = report["payloads"][0]
    assert result["checked"] and result["shape"] == list(shape)
    assert [int(x) for x in result["bits"]] == expected
    assert result["coordinates"] == [list(i) for i in np.ndindex(shape)]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("shape", [(2, 8), (0, 3)])
def test_native_archive_and_c_generation(tmp_path, compiler, kind, shape):
    assert native_compare(kind, shape, compiler, tmp_path)["all_output_bits_match"]


@pytest.mark.parametrize(
    "mutation",
    [
        "short",
        "long",
        "wrong_type",
        "wrong_shape",
        "unknown_type",
        "bf16",
        "encoded",
        "dynamic",
        "scalar_type",
        "byte_order",
        "layout",
        "bool_2",
        "bool_ff",
    ],
)
def test_decoder_rejects(tmp_path, compiler, mutation):
    entry, _ = record("i1" if mutation.startswith("bool") else "i32", (2, 8))
    request = {"byte_order": "little", "layout": "row_major", "payloads": [entry]}
    if mutation == "short":
        entry["hex"] = entry["hex"][:-2]
    elif mutation == "long":
        entry["hex"] += "00"
    elif mutation == "wrong_type":
        entry["type"]["element"] = {"kind": "float", "format": "f32"}
    elif mutation == "wrong_shape":
        entry["type"]["shape"] = [4, 4]
    elif mutation == "unknown_type":
        entry["type"]["element"] = {"kind": "int", "width": 24}
        entry["expected_type"] = copy.deepcopy(entry["type"])
    elif mutation == "bf16":
        entry["type"]["element"] = {"kind": "float", "format": "bf16"}
        entry["expected_type"] = copy.deepcopy(entry["type"])
    elif mutation == "encoded":
        entry["type"]["encoding"] = {"kind": "string", "value": "unknown_encoding"}
        entry["expected_type"] = copy.deepcopy(entry["type"])
    elif mutation == "dynamic":
        entry["type"]["shape"][0] = None
        entry["expected_type"] = copy.deepcopy(entry["type"])
    elif mutation == "scalar_type":
        entry["type"] = {"kind": "int", "width": 32}
        entry["expected_type"] = copy.deepcopy(entry["type"])
    elif mutation == "byte_order":
        request["byte_order"] = "big"
    elif mutation == "layout":
        request["layout"] = "column_major"
    else:
        entry["hex"] = ("02" if mutation == "bool_2" else "ff") + entry["hex"][2:]
    write(tmp_path / "bad.json", request)
    report = json.loads(
        run([str(prepare()), "decode", str((tmp_path / "bad.json").resolve())]).strip().splitlines()[-1]
    )
    assert not report["payloads"][0]["checked"]


@pytest.mark.parametrize("kind", ["i16", "i32", "i64", "f16", "f32", "f64"])
def test_big_endian_archives_are_not_repaired(tmp_path, kind):
    from merlin.llvmlower.qinner import QinnerArg, QinnerError, resolve

    array, _ = array_from_words(kind, (2, 8))
    swapped = array.astype(array.dtype.newbyteorder(">"))
    np.savez(tmp_path / "extra.npz", **{"qinner::values": swapped})
    with np.load(tmp_path / "extra.npz", allow_pickle=False) as archive:
        with pytest.raises(QinnerError):
            resolve(archive, [QinnerArg("values", (2, 8), kind)])
        with pytest.raises(ValueError, match="big-endian"):
            payload("values", archive["qinner::values"])


@pytest.mark.parametrize("byte", [2, 128, 255])
def test_invalid_bool_archive_is_rejected(tmp_path, byte):
    from merlin.llvmlower.qinner import QinnerArg, resolve

    invalid = np.array([byte, 0, 1], dtype=np.uint8).view(np.bool_)
    np.savez(tmp_path / "extra.npz", **{"qinner::values": invalid})
    with np.load(tmp_path / "extra.npz", allow_pickle=False) as archive:
        array = resolve(archive, [QinnerArg("values", (3,), "i1")])[0]
        assert array.tobytes()[0] == byte  # The production resolver does not canonicalize this data.
        with pytest.raises(ValueError, match="invalid bool"):
            payload("values", array)


def test_unsupported_host_is_explicit(monkeypatch):
    import merlin.verify.lean.qinner_decoder as decoder

    monkeypatch.setattr(decoder.sys, "byteorder", "big")
    with pytest.raises(ValueError, match="little-endian host"):
        payload("values", np.array([1], np.int8))


def test_decoder_receipt_replay_and_staleness(tmp_path, compiler):
    from merlin.verify.lean.qinner_decoder import replay_decode

    array, _ = array_from_words("f32", (2, 8))
    np.savez(tmp_path / "extra.npz", **{"qinner::values": array})
    entry, _ = record("f32", (2, 8))
    decode_records([entry], tmp_path / "decoder", archive_path=tmp_path / "extra.npz")
    receipt = tmp_path / "decoder/receipt.json"
    result = replay_decode(receipt, tmp_path / "replay")
    assert result["fresh"] and result["checked"] and result["matches_recorded"] and result["archive_bytes_match"]
    np.savez(tmp_path / "extra.npz", **{"qinner::values": np.zeros((2, 8), np.float32)})
    result = replay_decode(receipt, tmp_path / "changed")
    assert not result["fresh"] and not result["checked"]


@pytest.mark.slow
def test_retained_decoder_receipts_replay(tmp_path, compiler):
    from merlin.common.paths import artifacts_dir
    from merlin.verify.lean.qinner_decoder import replay_decode

    root = Path(
        os.environ.get(
            "MERLIN_QINNER_RECEIPT_ROOT", str(artifacts_dir() / "verification/merlin-passes/frontend/qinner-decoder")
        )
    )
    summary = json.loads((root / "summary.json").read_text())
    assert len(summary["native"]) == 16
    for index, entry in enumerate(summary["native"]):
        result = replay_decode(Path(entry["decoder_receipt"]), tmp_path / str(index))
        assert result["fresh"] and result["checked"] and result["matches_recorded"] and result["archive_bytes_match"]


def test_archive_bytes_cannot_be_replaced_in_a_decoder_receipt(tmp_path, compiler):
    array, _ = array_from_words("i32", (2, 8))
    np.savez(tmp_path / "extra.npz", **{"qinner::values": array})
    entry, _ = record("i32", (2, 8))
    raw = bytes.fromhex(entry["hex"])
    entry["hex"] = b"".join(raw[i : i + 4][::-1] for i in range(0, len(raw), 4)).hex()
    with pytest.raises(ValueError, match="disagree with the loaded archive"):
        decode_records([entry], tmp_path / "decoder", archive_path=tmp_path / "extra.npz")


def test_fortran_archive_decodes_row_major(tmp_path, compiler):
    from merlin.llvmlower.qinner import QinnerArg, resolve

    original, expected = array_from_words("i32", (2, 8))
    np.savez(tmp_path / "extra.npz", **{"qinner::values": np.asfortranarray(original)})
    with np.load(tmp_path / "extra.npz", allow_pickle=False) as archive:
        assert archive["qinner::values"].flags.f_contiguous
        resolved = resolve(archive, [QinnerArg("values", (2, 8), "i32")])[0]
        assert resolved.flags.c_contiguous
        entry = payload("values", resolved)
        entry["expected_type"] = copy.deepcopy(entry["type"])
    report = decode_records([entry], tmp_path / "decoder", archive_path=tmp_path / "extra.npz")
    assert [int(word) for word in report["payloads"][0]["bits"]] == expected


@pytest.mark.parametrize(
    "element", [{"kind": "float", "format": "bf16"}, {"kind": "int", "width": 24}, {"kind": "index"}]
)
def test_unsupported_empty_tensor_is_rejected(tmp_path, compiler, element):
    entry, _ = record("i32", (0, 3))
    entry["type"]["element"] = element
    entry["expected_type"] = copy.deepcopy(entry["type"])
    assert not decode_records([entry], tmp_path)["payloads"][0]["checked"]


def test_wrong_endianness_on_empty_tensor_is_rejected(tmp_path, compiler):
    entry, _ = record("f32", (0, 3))
    write(tmp_path / "empty-big.json", {"byte_order": "big", "layout": "row_major", "payloads": [entry]})
    response = json.loads(
        run([str(prepare()), "decode", str((tmp_path / "empty-big.json").resolve())]).strip().splitlines()[-1]
    )
    assert not response["payloads"][0]["checked"]
