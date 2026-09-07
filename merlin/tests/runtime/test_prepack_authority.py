"""Prepacking authority is a host-owned capture binding, not a candidate weight tag."""
from copy import deepcopy
import hashlib
import json
import struct

import pytest

from merlin.frontends.argument_identity import replay_argument_identity
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.perf.storage_encoding import GroupedAxesStorage
from merlin.runtime.captured_constants import verify_capture_constant
from merlin.runtime.prepack_authority import authorize_capture_prepack
from merlin.xdsl_dialects._common import text as module_text


def authority_fixture(tmp_path):
    payload = bytes([253, 254, 255, 0, 1, 2])
    header = json.dumps({"fixed": {"dtype": "I8", "shape": [2, 3], "data_offsets": [0, 6]}}).encode()
    blob = tmp_path / "weights.safetensors"
    blob.write_bytes(struct.pack("<Q", len(header)) + header + payload)
    manifest = tmp_path / "weights.safetensors.manifest.json"
    manifest.write_text(json.dumps({"0": {"weight": "fixed", "kind": "param",
                                          "dtype": "int8", "shape": [2, 3]}}))
    constant = verify_capture_constant(manifest_path=manifest,
        manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        safetensors_path=blob, safetensors_sha256=hashlib.sha256(blob.read_bytes()).hexdigest(),
        entry_argument_index=0, source_shape=(2, 3), source_dtype="i8", max_payload_bytes=6)
    raw = f'''builtin.module attributes {{prov.weights_file = "{blob}"}} {{
      func.func @forward(%w: tensor<2x3xi8>) -> tensor<2x3xi8> {{
        func.return %w : tensor<2x3xi8>
      }}
    }}'''
    normalized = module_text(parse_mlir_text(raw))
    bridge = replay_argument_identity(raw_text=raw,
        source_sha256=hashlib.sha256(raw.encode()).hexdigest(),
        normalized_sha256=hashlib.sha256(normalized.encode()).hexdigest(), entry="forward",
        stages=(), source_pins={})
    encoding = GroupedAxesStorage((2, 3), "i8", ((1,), (0,)), (3, 2), (4, 1), 12)
    output = GroupedAxesStorage((2, 3), "i8", ((0,), (1,)), (2, 3), (3, 1), 6)
    cb = {"tensors": {"W": {"shape": [3, 2], "dtype": "i8", "role": "weight"},
                      "Y": {"shape": [2, 3], "dtype": "i8", "role": "output"}},
          "kernel_abi": {"kind": "whole_program", "args": [{"tensor": "W", "access": "read"},
                         {"tensor": "Y", "access": "write"}], "outputs": ["Y"]},
          "params": {"storage_encodings": {"W": encoding.to_dict(), "Y": output.to_dict()},
                     "global_program_plan": {"source_sha256": bridge.normalized_sha256,
                         "entry_bindings": ["W"], "tasks": [{"writes": ["Y"]}]}}}
    kwargs = dict(constant=constant, bridge=bridge, raw_source_text=raw,
                  normalized_source_text=normalized, command_buffer=cb, tensor="W")
    return kwargs, encoding, payload


def test_exact_capture_and_normalization_bind_one_payload_and_encoding(tmp_path):
    kwargs, encoding, payload = authority_fixture(tmp_path)
    grant = authorize_capture_prepack(**kwargs)
    grant.validate_binding(kwargs["command_buffer"], "W", encoding)
    grant.validate_payload(payload)
    assert grant.to_evidence()["prepack_authorized"] is True
    assert grant.to_evidence()["performance_promotion"] is False
    assert grant.to_evidence()["capture"]["prepack_authorized"] is False


def test_authority_rejects_changed_input_even_when_shape_and_role_are_unchanged(tmp_path):
    kwargs, encoding, payload = authority_fixture(tmp_path)
    grant = authorize_capture_prepack(**kwargs)
    with pytest.raises(ValueError, match="initializer"):
        grant.validate_payload(bytes(6))
    with pytest.raises(ValueError, match="initializer"):
        grant.validate_payload(list(payload))
    changed = deepcopy(kwargs["command_buffer"])
    changed["params"]["candidate_revision"] = "different"
    with pytest.raises(ValueError, match="command buffer"):
        grant.validate_binding(changed, "W", encoding)
    with pytest.raises(ValueError, match="command buffer"):
        grant.validate_binding(kwargs["command_buffer"], "Y", encoding)


@pytest.mark.parametrize("change", ["readwrite", "writes", "index", "source", "raw", "normalized",
                                    "serialized_constant", "serialized_bridge", "encoding", "dtype"])
def test_missing_or_contradictory_host_binding_refuses(tmp_path, change):
    kwargs, encoding, payload = authority_fixture(tmp_path)
    cb = kwargs["command_buffer"]
    if change == "readwrite": cb["kernel_abi"]["args"][0]["access"] = "readwrite"
    elif change == "writes": cb["params"]["global_program_plan"]["tasks"][0]["writes"].append("W")
    elif change == "index": cb["params"]["global_program_plan"]["entry_bindings"] = ["Y"]
    elif change == "source": cb["params"]["global_program_plan"]["source_sha256"] = "0" * 64
    elif change == "raw": kwargs["raw_source_text"] += "\n"
    elif change == "normalized": kwargs["normalized_source_text"] += "\n"
    elif change == "serialized_constant": kwargs["constant"] = kwargs["constant"].to_evidence()
    elif change == "serialized_bridge": kwargs["bridge"] = kwargs["bridge"].to_evidence()
    elif change == "encoding": cb["params"]["storage_encodings"]["W"]["logical_shape"] = [3, 2]
    elif change == "dtype": cb["tensors"]["W"]["dtype"] = "i16"
    with pytest.raises(ValueError):
        authorize_capture_prepack(**kwargs)
