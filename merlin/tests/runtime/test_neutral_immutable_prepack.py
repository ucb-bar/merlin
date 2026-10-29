"""Actual captured byte grants compose with the existing neutral pointer caller."""

import copy
import hashlib
import json
import struct

import pytest

from merlin.compile.scheduling.tensor_storage import column_panel_storage
from merlin.frontends.argument_identity import replay_argument_identity
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.perf.storage_encoding import GroupedAxesStorage
from merlin.runtime import harness_render as H
from merlin.runtime.captured_constants import verify_capture_constant
from merlin.runtime.prepack_authority import authorize_capture_prepack
from merlin.xdsl_dialects._common import text


@pytest.fixture
def prepared(tmp_path):
    sha = lambda raw: hashlib.sha256(raw).hexdigest()
    raw = bytes([1, 255, 2, 128, 3, 254, 4, 127])
    manifest = json.dumps({"0": {"kind": "param", "weight": "fixed", "shape": [2, 4], "dtype": "int8"}}).encode()
    header = json.dumps({"fixed": {"dtype": "I8", "shape": [2, 4], "data_offsets": [0, 8]}}).encode()
    blob = struct.pack("<Q", len(header)) + header + raw
    mp, bp = tmp_path / "weights.json", tmp_path / "weights.safetensors"
    mp.write_bytes(manifest)
    bp.write_bytes(blob)
    source = f'''builtin.module attributes {{prov.weights_file = "{bp}"}} {{
      func.func @forward(%w: tensor<2x4xi8>) -> tensor<2x4xi8> {{func.return %w : tensor<2x4xi8>}}
    }}'''
    normalized = text(parse_mlir_text(source))
    bridge = replay_argument_identity(raw_text=source, source_sha256=sha(source.encode()),
        normalized_sha256=sha(normalized.encode()), entry="forward", stages=(), source_pins={})
    constant = verify_capture_constant(manifest_path=mp, manifest_sha256=sha(manifest),
        safetensors_path=bp, safetensors_sha256=sha(blob), entry_argument_index=0,
        source_shape=[2, 4], source_dtype="i8", max_payload_bytes=8)
    storage = column_panel_storage(2, 4, 2, "i8")
    output = GroupedAxesStorage((2, 4), "i8", ((0,), (1,)), (2, 4), (4, 1), 8)
    cb = {"tensors": {"W": {"shape": [2, 2, 2], "dtype": "i8", "role": "input", "storage_encoding": storage.to_dict()},
                      "Y": {"shape": [2, 4], "dtype": "i8", "role": "output"}},
          "kernel_abi": {"kind": "whole_program", "args": [{"tensor": "W", "access": "read"},
                         {"tensor": "Y", "access": "write"}], "outputs": ["Y"]},
          "params": {"storage_encodings": {"W": storage.to_dict(), "Y": output.to_dict()},
                     "global_program_plan": {"source_sha256": bridge.normalized_sha256,
                         "entry_bindings": ["W"], "tasks": [{"writes": ["Y"]}]}}}
    grant = authorize_capture_prepack(constant=constant, bridge=bridge, raw_source_text=source,
        normalized_source_text=normalized, command_buffer=cb, tensor="W")
    contract = {"harness_abi": {"version": 2, "kind": "logical_pointer", "entry_symbol": "source_entry",
        "fence_symbol": "source_complete", "tensor_alignment": 8, "byte_order": "little",
        "main_convention": "void", "readback_transport": "out_b64_v1"}}
    return cb, {"W": [[1, -1, 2, -128], [3, -2, 4, 127]]}, {"W": grant}, contract


def test_exact_byte_preparation_changes_only_immutable_initializer(prepared):
    cb, inputs, grants, contract = prepared
    original = copy.deepcopy((cb, inputs, contract))
    source = H.render_harness(cb, target="neutral", inputs=inputs, contract=contract, prepack_authorizations=grants)
    assert "0x01,0xff,0x03,0xfe,0x02,0x80,0x04,0x7f" in source
    assert "source_entry(tensor_0, tensor_1);" in source
    assert source.index("source_complete();") < source.index("OUT_B64_BEGIN v1 Y 2 4")
    assert (cb, inputs, contract) == original
    assert H.render_harness(cb, target="neutral", inputs=inputs, contract=contract,
                            prepack_authorizations=grants, blobs={}) == source
    with pytest.raises(ValueError, match="streamed sidecars"):
        H.render_harness(cb, target="neutral", inputs=inputs, contract=contract,
                         prepack_authorizations=grants, blobs={"streamed": object()})


@pytest.mark.parametrize("change", ["missing_grant", "serialized_grant", "payload", "missing_input", "budget", "output"])
def test_missing_or_changed_source_authority_refuses_before_rendering(prepared, change):
    cb, inputs, grants, contract = prepared
    kwargs = {}
    if change == "missing_grant": grants = None
    elif change == "serialized_grant": grants = {"W": grants["W"].to_evidence()}
    elif change == "payload": inputs["W"][0][0] = 2
    elif change == "missing_input": inputs = {}
    elif change == "budget": kwargs["max_storage_bytes"] = 15
    else:
        # Bind the changed output to a fresh original input grant first; this
        # tests the neutral output restriction independently of stale-CB checks.
        cb = copy.deepcopy(cb)
        cb["params"]["storage_encodings"]["Y"] = column_panel_storage(2, 4, 2, "i8").to_dict()
        cb["tensors"]["Y"]["shape"] = [2, 2, 2]
        from dataclasses import replace
        from merlin.common.jsonio import canonical_sha256
        grants = {"W": replace(grants["W"], command_buffer_digest=canonical_sha256(cb))}
    with pytest.raises(ValueError):
        H.render_harness(cb, target="neutral", inputs=inputs, contract=contract, prepack_authorizations=grants, **kwargs)
