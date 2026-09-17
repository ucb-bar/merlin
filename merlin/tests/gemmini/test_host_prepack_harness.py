"""Host-pinned format setup only: tiny captured bytes, no model execution."""
import copy
import hashlib
import json
import struct

import pytest

from merlin.frontends.argument_identity import replay_argument_identity
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.perf.storage_encoding import GroupedAxesStorage
from merlin.runtime.backends import base as bk
from merlin.runtime.captured_constants import verify_capture_constant
from merlin.runtime.prepack_authority import authorize_capture_prepack
from merlin.runtime.storage_binding import resolve_storage_bindings
from merlin.targetgen.contract import compile as compiler
from merlin.xdsl_dialects._common import text as module_text


def sha(data):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def authorized(tmp_path):
    manifest = json.dumps({"0": {"kind": "param", "weight": "W", "shape": [2, 2], "dtype": "int8"}}).encode()
    header = json.dumps({"W": {"dtype": "I8", "shape": [2, 2], "data_offsets": [0, 4]}}).encode()
    blob = struct.pack("<Q", len(header)) + header + b"\x01\xff\x02\x80"
    mp, bp = tmp_path / "weights.json", tmp_path / "weights.safetensors"
    mp.write_bytes(manifest)
    bp.write_bytes(blob)
    constant = verify_capture_constant(manifest_path=mp, manifest_sha256=sha(manifest),
        safetensors_path=bp, safetensors_sha256=sha(blob), entry_argument_index=0,
        source_shape=[2, 2], source_dtype="i8", max_payload_bytes=4)
    raw = f'''builtin.module attributes {{prov.weights_file = "{bp}"}} {{
      func.func @forward(%w: tensor<2x2xi8>) -> tensor<2x2xi8> {{
        func.return %w : tensor<2x2xi8>
      }}
    }}'''
    normalized = module_text(parse_mlir_text(raw))
    bridge = replay_argument_identity(raw_text=raw, source_sha256=sha(raw.encode()),
        normalized_sha256=sha(normalized.encode()), entry="forward", stages=(), source_pins={})
    enc = GroupedAxesStorage((2, 2), "i8", ((1,), (0,)), (2, 2), (3, 1), 6)
    out = GroupedAxesStorage((2, 2), "i8", ((0,), (1,)), (2, 2), (2, 1), 4)
    cb = {"abi_version": "0.1", "target": "gemmini", "commands": [],
        "tensors": {"W": {"shape": [2, 2], "dtype": "i8", "role": "input"},
                    "Y": {"shape": [2, 2], "dtype": "i8", "role": "output"}},
        "kernel_abi": {"kind": "whole_program", "args": [
            {"tensor": "W", "access": "read"}, {"tensor": "Y", "access": "write"}], "outputs": ["Y"]},
        "params": {"storage_encodings": {"W": enc.to_dict(), "Y": out.to_dict()},
                   "global_program_plan": {"source_sha256": sha(normalized.encode()),
                       "entry_bindings": ["W"], "tasks": [{"writes": ["Y"]}]}}}
    grant = authorize_capture_prepack(constant=constant, bridge=bridge, raw_source_text=raw,
        normalized_source_text=normalized, command_buffer=cb, tensor="W")
    return cb, {"W": [[1, -1], [2, -128]]}, {"W": grant}


def test_exact_supplied_initializer_is_checked_then_packed(authorized, monkeypatch):
    monkeypatch.setenv("MERLIN_CACHE_STATE", "warm")
    cb, inputs, grants = authorized
    binding = resolve_storage_bindings(cb, inputs, max_storage_bytes=100,
                                      prepack_authorizations=grants)["W"]
    assert binding.pack_words([1, -1, 2, -128]) == [1, 2, 0, -1, -128, 0]
    source = bk.get_backend("gemmini").render_harness(
        cb, target="gemmini", inputs=inputs, prepack_authorizations=grants)
    assert "{1,2,0,-1,-128,0}" in source
    assert source.count("gemmini_kernel((void*)T_W, (void*)T_Y);") == 2
    with pytest.raises(ValueError, match="actual initializer"):
        binding.pack_words([1, -1, 2, -127])
    with pytest.raises(ValueError, match="declared dtype width"):
        binding.pack_words([257, -1, 2, -128])
    with pytest.raises(ValueError, match="integer storage words"):
        binding.pack_words([True, -1, 2, -128])
    changed = copy.deepcopy(inputs)
    changed["W"][1][1] = -127
    with pytest.raises(ValueError, match="actual initializer"):
        bk.get_backend("gemmini").render_harness(
            cb, target="gemmini", inputs=changed, prepack_authorizations=grants)


def test_strict_warm_profile_preserves_authorized_prepack_and_emits_only_cycles(
        authorized, monkeypatch):
    from merlin.perf.execution_policy import WarmProfileContract

    monkeypatch.setenv("MERLIN_HW_COUNTERS", "1")
    cb, inputs, grants = authorized
    source = bk.get_backend("gemmini").render_harness(
        cb, target="gemmini", inputs=inputs, prepack_authorizations=grants,
        warm_profile=WarmProfileContract())
    call = "gemmini_kernel((void*)T_W, (void*)T_Y);"
    assert "{1,2,0,-1,-128,0}" in source
    assert source.count(call) == source.count("gemmini_fence();") == 2
    assert source.count("METRIC ") == 1
    assert "METRIC cycles " in source
    assert "cycle_window_gemmini_region" not in source
    assert "counter_configure" not in source


@pytest.mark.parametrize("change", ["missing", "substitute", "serialized", "access", "encoding", "cb"])
def test_grant_cannot_authorize_other_inputs_or_bindings(authorized, change):
    cb, inputs, grants = copy.deepcopy(authorized)
    if change == "missing":
        inputs = {}
        # Candidate recorded values do not satisfy explicitly supplied input authority.
        cb["canonical_inputs"] = {"W": {"values": [[1, -1], [2, -128]]}}
    elif change == "substitute": inputs["W"][0][0] = 0
    elif change == "serialized": grants["W"] = grants["W"].to_evidence()
    elif change == "access": cb["kernel_abi"]["args"][0]["access"] = "readwrite"
    elif change == "encoding": cb["params"]["storage_encodings"]["W"]["offset_elements"] = 1
    else: cb["commands"] = [{"different": True}]
    with pytest.raises(ValueError):
        binding = resolve_storage_bindings(cb, inputs, max_storage_bytes=100,
                                          prepack_authorizations=grants)["W"]
        binding.pack_words([int(value) for row in inputs["W"] for value in row])


def test_authorized_build_never_reuses_or_publishes_cached_elf(authorized, tmp_path, monkeypatch):
    from merlin.targetgen import build_cache
    cb, inputs, grants = authorized
    for name in ("build_identity", "reuse", "store"):
        monkeypatch.setattr(build_cache, name, lambda *a, **k: pytest.fail("authorized build reached cache"))
    monkeypatch.setattr(compiler, "llvm_mlir_to_object", lambda *a, **k: tmp_path / "kernel.o")
    seen = []
    def link(cb, obj, workdir, **kwargs):
        seen.append(kwargs)
        return tmp_path / "kernel.elf"
    monkeypatch.setattr(compiler, "link_elf", link)
    assert compiler.compile_lowered_to_elf(cb, "unused", tmp_path, target="gemmini",
        inputs=inputs, prepack_authorizations=grants) == tmp_path / "kernel.elf"
    assert seen == [{"target": "gemmini", "inputs": inputs, "prepack_authorizations": grants}]


def test_profiled_authorized_build_forwards_prepack_and_profile(
        authorized, tmp_path, monkeypatch):
    from merlin.perf.execution_policy import WarmProfileContract
    from merlin.targetgen import build_cache

    cb, inputs, grants = authorized
    for name in ("build_identity", "reuse", "store"):
        monkeypatch.setattr(
            build_cache, name,
            lambda *args, **kwargs: pytest.fail("profiled prepack build reached cache"))
    monkeypatch.setattr(
        compiler, "llvm_mlir_to_object",
        lambda *args, **kwargs: tmp_path / "kernel.o")
    seen = []

    def link(cb, obj, workdir, **kwargs):
        seen.append(kwargs)
        return tmp_path / "kernel.elf"

    monkeypatch.setattr(compiler, "link_elf", link)
    profile = WarmProfileContract()
    assert compiler.compile_lowered_to_elf(
        cb, "unused", tmp_path, target="gemmini", inputs=inputs,
        prepack_authorizations=grants, warm_profile=profile) == tmp_path / "kernel.elf"
    assert seen == [{
        "target": "gemmini", "inputs": inputs,
        "prepack_authorizations": grants, "warm_profile": profile,
    }]


def test_missing_explicit_input_refuses_before_compile_or_recorded_fallback(authorized, tmp_path, monkeypatch):
    cb, inputs, grants = authorized
    cb["canonical_inputs"] = {"W": {"values": inputs["W"]}}
    monkeypatch.setattr(compiler, "llvm_mlir_to_object", lambda *a, **k: pytest.fail("compiled without inputs"))
    for provided in (None, {}):
        with pytest.raises(ValueError, match="explicit logical inputs"):
            compiler.compile_lowered_to_elf(cb, "unused", tmp_path, target="gemmini",
                inputs=provided, prepack_authorizations=grants)


def test_backend_without_authorization_capability_refuses(authorized, tmp_path, monkeypatch):
    cb, inputs, grants = authorized
    monkeypatch.setattr(bk, "harness_build_recipe", lambda target: object())
    monkeypatch.setattr(bk, "harness_renderer", lambda target: lambda cb, *, target, inputs: "unused")
    with pytest.raises(NotImplementedError, match="cannot consume host prepack"):
        compiler.link_elf(cb, tmp_path / "object.o", tmp_path, target="target",
                          inputs=inputs, prepack_authorizations=grants)
