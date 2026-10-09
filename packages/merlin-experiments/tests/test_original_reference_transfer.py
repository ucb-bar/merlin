"""Malformed complete native transfers refuse before raw payload decoding."""

import copy
import json
from types import SimpleNamespace

import pytest
from merlin_experiments.phase0 import original_reference_products as P

from merlin.targetgen.original_operator_reference import OriginalReferenceBudget


@pytest.fixture
def transfer(tmp_path):
    slot = {"name": "Y", "dtype": "int8", "shape": [1, 2]}
    contract = SimpleNamespace(
        verify=lambda: {"outputs": [slot]},
        budget=OriginalReferenceBudget(4, 4, 8, 512),
        output_byteorder="little",
    )
    frame = {
        "schema": "merlin.original_reference_native_output.v1",
        "outputs": [{**slot, "byteorder": "little", "data_hex": "fe03"}],
        "runtime": {"torch_version": "selected", "git_version": "selected"},
    }
    return contract, frame, tmp_path / "native.json"


def test_complete_bounded_native_transfer_preserves_signed_bytes(transfer):
    contract, frame, path = transfer
    path.write_text(json.dumps(frame))
    (actual,) = P.native_outputs(contract, path)
    assert actual.data == b"\xfe\x03" and actual.values() == (-2, 3)


@pytest.mark.parametrize(
    "defect", ["length", "whitespace", "invalid_hex", "slot", "shape", "byteorder", "missing", "extra", "runtime"]
)
def test_every_native_slot_is_checked_before_any_raw_decoding(transfer, monkeypatch, defect):
    contract, frame, path = transfer
    frame = copy.deepcopy(frame)
    row = frame["outputs"][0]
    if defect == "length":
        row["data_hex"] += "00"
    elif defect == "whitespace":
        row["data_hex"] = "fe 3"
    elif defect == "invalid_hex":
        row["data_hex"] = "fe0g"
    elif defect == "slot":
        row["name"] = "other"
    elif defect == "shape":
        row["shape"] = [True, 2]
    elif defect == "byteorder":
        row["byteorder"] = "big"
    elif defect == "missing":
        frame["outputs"] = []
    elif defect == "extra":
        frame["outputs"].append(dict(row))
    else:
        frame["runtime"]["answer"] = True
    path.write_text(json.dumps(frame))
    monkeypatch.setattr(P, "_decode_outputs", lambda *args: pytest.fail("malformed frame reached decoder"))
    with pytest.raises(ValueError):
        P.native_outputs(contract, path)


def test_oversized_native_frame_refuses_before_json_parsing(transfer, monkeypatch):
    contract, _, path = transfer
    path.write_bytes(b" " * 521)
    monkeypatch.setattr(P, "loads", lambda *args: pytest.fail("oversized native frame reached JSON parser"))
    with pytest.raises(ValueError, match="predecode transfer"):
        P.native_outputs(contract, path)
