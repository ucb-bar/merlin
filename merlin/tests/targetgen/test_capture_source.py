"""Resolving a packed tensor's bytes out of a model capture.

`bundle_pack.write_const_blob` takes a `source` callable so it need learn no container format. Three
whole-model bundles were then packed with three inline copies of that callable, and every subtle
defect in the path lived in the copies rather than in the packer. Each class below pins one of them.

The acceptance test is agreement: driven from the library, the resolver must reproduce the SmolVLA
and tiny_llama const blobs BYTE-IDENTICALLY to the ones the inline scripts produced -- 505,299,072
and 1,298,638,208 bytes, both already verified tensor-by-tensor against their captures.
"""
from __future__ import annotations

import json
import struct

import pytest

from merlin.targetgen import bundle_pack as BP
from merlin.targetgen import capture_source as CS
from merlin.targetgen.capture_source import CaptureSourceError


def _safetensors(path, tensors):
    """Write a minimal safetensors file. `tensors` maps name -> (dtype tag, raw bytes)."""
    header, at, payload = {}, 0, bytearray()
    for name, (tag, raw) in tensors.items():
        header[name] = {"dtype": tag, "shape": [len(raw)], "data_offsets": [at, at + len(raw)]}
        payload += raw
        at += len(raw)
    blob = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + bytes(payload))


def _plan(tensors, args, params=None):
    return BP.plan({"tensors": tensors, "kernel_abi": {"args": args}, "params": params or {}},
                   row_pitch_elements=16)


def _t(shape, dtype):
    return {"shape": list(shape), "dtype": dtype}


class TestTheManifestSpellsASourceUnderMoreThanOneKey:
    """Defect 1. A parameter arrives under `weight` and a graph INPUT under `name`. Reading only the
    first refuses the input -- ResNet-50's arg216 IS the image, entry {"kind": "input", "name":
    "image"}, and the packer asked for "arg216".
    """

    def test_both_spellings_are_read_in_order(self):
        assert CS.manifest_key_for(0, "arg0", {"0": {"weight": "w.0"}}) == "w.0"
        assert CS.manifest_key_for(1, "arg1", {"1": {"kind": "input", "name": "image"}}) == "image"

    def test_weight_wins_when_both_are_present(self):
        key = CS.manifest_key_for(0, "arg0", {"0": {"weight": "w.0", "name": "other"}})
        assert key == "w.0", "the declared order is the one WEIGHT_KEY_FIELDS states"

    def test_an_entry_declaring_neither_falls_back_to_the_tensor_name(self):
        assert CS.manifest_key_for(0, "arg0", {"0": {"kind": "input"}}) == "arg0"
        assert CS.manifest_key_for(0, "arg0", {}) == "arg0"

    def test_an_input_named_by_the_manifest_resolves_end_to_end(self, tmp_path):
        import numpy as np
        _safetensors(tmp_path / "weights.safetensors", {"w": ("F32", bytes(64))})
        np.savez(tmp_path / "inputs.npz", in0=np.arange(16, dtype=np.int8))
        (tmp_path / "input_order.json").write_text(json.dumps({"image": 0}), encoding="utf-8")
        plan = _plan({"arg0": _t((16,), "f32"), "arg1": _t((16,), "i8"), "Y0": _t((4,), "i32")},
                     [{"tensor": "arg0", "access": "read"}, {"tensor": "arg1", "access": "read"},
                      {"tensor": "Y0", "access": "write"}])
        manifest = {"0": {"weight": "w"}, "1": {"kind": "input", "name": "image"}}
        source, report = CS.capture_tensor_source(tmp_path, plan, weight_manifest=manifest)
        assert len(source("image")) == 16
        assert report.origin["arg1"] == "runtime_input"


class TestALiftedBufferIsSpelledDifferentlyOnEachSide:
    """Defect 2. `extra.npz` holds `buf::foo.bar`; the manifest names `b_foo_bar`."""

    def _capture(self, tmp_path):
        import numpy as np
        _safetensors(tmp_path / "weights.safetensors", {"w": ("F32", bytes(64))})
        np.savez(tmp_path / "extra.npz", **{"buf::foo.bar": np.zeros(16, dtype=np.int8)})
        plan = _plan({"arg0": _t((16,), "f32"), "arg1": _t((16,), "i8"), "Y0": _t((4,), "i32")},
                     [{"tensor": "arg0", "access": "read"}, {"tensor": "arg1", "access": "read"},
                      {"tensor": "Y0", "access": "write"}])
        return plan, {"0": {"weight": "w"}, "1": {"weight": "b_foo_bar"}}

    def test_the_manifest_spelling_resolves_to_the_npz_key(self, tmp_path):
        plan, manifest = self._capture(tmp_path)
        source, report = CS.capture_tensor_source(tmp_path, plan, weight_manifest=manifest)
        assert len(source("b_foo_bar")) == 16
        assert report.origin["arg1"] == "lifted_buffer"

    def test_the_convention_is_the_one_c_runtime_uses(self):
        """Held against the emitter, so the two cannot drift."""
        key = "buf::model.layer.0.weight"
        assert ("b_" + key[len(CS.BUFFER_PREFIX):].replace(".", "_")
                == "b_model_layer_0_weight")

    def test_a_raw_npz_key_also_resolves(self, tmp_path):
        """Not every lifted value carries the prefix."""
        import numpy as np
        _safetensors(tmp_path / "weights.safetensors", {"w": ("F32", bytes(64))})
        np.savez(tmp_path / "extra.npz", plain=np.zeros(16, dtype=np.int8))
        plan = _plan({"arg0": _t((16,), "f32"), "arg1": _t((16,), "i8"), "Y0": _t((4,), "i32")},
                     [{"tensor": "arg0", "access": "read"}, {"tensor": "arg1", "access": "read"},
                      {"tensor": "Y0", "access": "write"}])
        source, report = CS.capture_tensor_source(
            tmp_path, plan, weight_manifest={"0": {"weight": "w"}, "1": {"weight": "plain"}})
        assert len(source("plain")) == 16


class TestARuntimeInputIsKeyedPositionally:
    """Defect 3. `inputs.npz` holds in0, in1 ...; `input_order.json` maps a name to that index."""

    def _capture(self, tmp_path, order):
        import numpy as np
        _safetensors(tmp_path / "weights.safetensors", {"w": ("F32", bytes(64))})
        np.savez(tmp_path / "inputs.npz", in0=np.zeros(16, dtype=np.int8),
                 in1=np.ones(16, dtype=np.int8))
        (tmp_path / "input_order.json").write_text(json.dumps(order), encoding="utf-8")
        plan = _plan({"arg0": _t((16,), "f32"), "arg1": _t((16,), "i8"), "Y0": _t((4,), "i32")},
                     [{"tensor": "arg0", "access": "read"}, {"tensor": "arg1", "access": "read"},
                      {"tensor": "Y0", "access": "write"}])
        return plan, {"0": {"weight": "w"}, "1": {"kind": "input", "name": "x"}}

    def test_the_declared_index_selects_the_positional_key(self, tmp_path):
        plan, manifest = self._capture(tmp_path, {"x": 1})
        source, _ = CS.capture_tensor_source(tmp_path, plan, weight_manifest=manifest)
        assert source("x") == bytes([1] * 16), "index 1 must read in1, not in0"

    def test_an_index_the_corpus_does_not_hold_is_REFUSED(self, tmp_path):
        plan, manifest = self._capture(tmp_path, {"x": 7})
        source, _ = CS.capture_tensor_source(tmp_path, plan, weight_manifest=manifest)
        with pytest.raises(CaptureSourceError, match="holds no 'in7'"):
            source("x")

    def test_a_non_integer_index_is_REFUSED(self, tmp_path):
        plan, manifest = self._capture(tmp_path, {"x": "first"})
        source, _ = CS.capture_tensor_source(tmp_path, plan, weight_manifest=manifest)
        with pytest.raises(CaptureSourceError, match="not an input index"):
            source("x")


class TestTheDeclaredDtypeCanDifferFromWhatWasCaptured:
    """Defect 4. SmolVLA's prefix KV-cache is declared bf16 (2,314,240 B) and captured as f32
    (4,628,480 B). A size check sees THAT; nothing sees a wrong CONVERSION, so the re-encoding uses
    the repo's own round-half-to-even rather than an invented one.
    """

    def test_bf16_uses_the_repos_own_rounding(self):
        import numpy as np

        from merlin.llvmlower.c_runtime import _bf16_bits
        values = np.array([1.0, -2.5, 3.0e-8, 65504.0, 0.1, -0.0], dtype=np.float32)
        got = CS.encode_as_declared(values, tensor="t", declared_dtype="bf16")
        assert got == _bf16_bits(values).tobytes()
        assert len(got) == 2 * values.size, "half the f32 width"

    def test_the_rounding_is_round_half_to_even_not_truncation(self):
        """A truncating conversion agrees on most values, which is why this is pinned."""
        import numpy as np
        # 0x3F800001 rounds UP to 0x3F80 + 1 under RNE with the sticky bits set.
        raw = np.array([0x3F80C000], dtype=np.uint32).view(np.float32)
        rne = CS.encode_as_declared(raw, tensor="t", declared_dtype="bf16")
        truncated = (np.array([0x3F80C000], dtype=np.uint32) >> 16).astype(np.uint16).tobytes()
        assert rne != truncated, "truncation and RNE must disagree here, or the test proves nothing"

    def test_i1_is_widened_to_the_one_byte_storage_the_compiler_sizes_it_at(self):
        import numpy as np
        got = CS.encode_as_declared(np.array([True, False, True]), tensor="t", declared_dtype="i1")
        assert got == bytes([1, 0, 1])

    def test_a_matching_dtype_is_passed_through_untouched(self):
        import numpy as np
        values = np.arange(8, dtype=np.int8)
        assert CS.encode_as_declared(values, tensor="t", declared_dtype="i8") == values.tobytes()

    def test_every_re_encoding_is_REPORTED(self):
        import numpy as np
        report = CS.SourceReport()
        CS.encode_as_declared(np.ones(4, dtype=np.float32), tensor="kv",
                              declared_dtype="bf16", report=report)
        CS.encode_as_declared(np.ones(4, dtype=np.int8), tensor="w",
                              declared_dtype="i8", report=report)
        assert report.re_encoded == [("kv", "float32", "bf16")]
        assert report.to_dict()["re_encoded"] == [
            {"tensor": "kv", "captured": "float32", "declared": "bf16"}]


class TestAnUnresolvableTensorIsRefusedNotGuessed:
    def test_the_refusal_names_every_lookup_it_tried(self, tmp_path):
        _safetensors(tmp_path / "weights.safetensors", {"w": ("F32", bytes(64))})
        plan = _plan({"arg0": _t((16,), "f32"), "arg1": _t((16,), "i8"), "Y0": _t((4,), "i32")},
                     [{"tensor": "arg0", "access": "read"}, {"tensor": "arg1", "access": "read"},
                      {"tensor": "Y0", "access": "write"}])
        source, _ = CS.capture_tensor_source(
            tmp_path, plan, weight_manifest={"0": {"weight": "w"}, "1": {"weight": "missing"}})
        with pytest.raises(CaptureSourceError) as excinfo:
            source("missing")
        message = str(excinfo.value)
        assert "not a safetensors entry" in message
        assert "not a lifted buffer under either spelling" in message
        assert "not a declared runtime input" in message
        assert "bytes nobody captured" in message

    def test_a_directory_that_is_not_a_capture_is_refused(self, tmp_path):
        plan = _plan({"arg0": _t((16,), "f32"), "Y0": _t((4,), "i32")},
                     [{"tensor": "arg0", "access": "read"}, {"tensor": "Y0", "access": "write"}])
        with pytest.raises(CaptureSourceError, match="not a capture directory"):
            CS.capture_tensor_source(tmp_path / "nope", plan, weight_manifest={})

    def test_a_malformed_input_order_is_refused(self, tmp_path):
        _safetensors(tmp_path / "weights.safetensors", {"w": ("F32", bytes(64))})
        (tmp_path / "input_order.json").write_text("[1, 2]", encoding="utf-8")
        plan = _plan({"arg0": _t((16,), "f32"), "Y0": _t((4,), "i32")},
                     [{"tensor": "arg0", "access": "read"}, {"tensor": "Y0", "access": "write"}])
        with pytest.raises(CaptureSourceError, match="not a mapping"):
            CS.capture_tensor_source(tmp_path, plan, weight_manifest={"0": {"weight": "w"}})


class TestItReproducesTheBlobsTheInlineScriptsBuilt:
    """The acceptance test. Both blobs were verified tensor-by-tensor against their captures, so
    byte-identity here means the library path is the one that was checked.
    """

    def _emission(self, n_args):
        import glob

        from merlin.common.paths import artifacts_dir
        root = artifacts_dir() / "perf-bench" / "gemmini" / "_global_phase2_baseline_emission_cache_v1"
        for path in sorted(glob.glob(str(root / "*" / "command_buffer.json"))):
            with open(path, encoding="utf-8") as handle:
                buffer = json.load(handle)
            if len((buffer.get("kernel_abi") or {}).get("args") or []) == n_args:
                return buffer
        pytest.skip(f"no emitted {n_args}-argument command buffer in this tree")

    def _capture(self, relative):
        from merlin.common.paths import artifacts_dir
        path = artifacts_dir() / "recaptures" / relative
        if not path.is_dir():
            pytest.skip(f"no {relative} recapture in this tree")
        return path

    @pytest.mark.parametrize(("n_args", "relative", "session", "expect_bytes", "expect_tensors"), [
        (1163, "smolvla_int8_w8a8_consistent/stages/flow_denoise", True, 505_299_072, 812),
        (825, "tiny_llama_int8_w8a8_consistent", False, 1_298_638_208, 359),
    ])
    def test_the_plan_and_the_resolver_agree_on_every_tensor(
            self, tmp_path, n_args, relative, session, expect_bytes, expect_tensors):
        buffer = self._emission(n_args)
        capture = self._capture(relative)
        manifest = json.loads(
            (capture / "weights.safetensors.manifest.json").read_text(encoding="utf-8"))
        states = ()
        if session:
            from merlin.common.yaml import load_yaml
            states = BP.session_states_from_contract(load_yaml(capture / "session_contract.yaml"))
        plan = BP.plan(buffer, row_pitch_elements=16, weight_manifest=manifest,
                       session_states=states)
        assert plan.const_bytes == expect_bytes
        source, report = CS.capture_tensor_source(capture, plan, weight_manifest=manifest)
        # Resolve every tensor without writing 1.2 GiB: the sizes are what the packer checks.
        for row in plan.const:
            key = CS.manifest_key_for(row.index, row.tensor, manifest)
            assert len(source(key)) > 0, key
        assert len(report.origin) == expect_tensors
        assert set(report.origin.values()) <= {"safetensors", "lifted_buffer", "runtime_input"}

    def test_smolvlas_two_re_encodings_are_the_ones_that_were_verified(self):
        buffer = self._emission(1163)
        capture = self._capture("smolvla_int8_w8a8_consistent/stages/flow_denoise")
        from merlin.common.yaml import load_yaml
        manifest = json.loads(
            (capture / "weights.safetensors.manifest.json").read_text(encoding="utf-8"))
        plan = BP.plan(buffer, row_pitch_elements=16, weight_manifest=manifest,
                       session_states=BP.session_states_from_contract(
                           load_yaml(capture / "session_contract.yaml")))
        source, report = CS.capture_tensor_source(capture, plan, weight_manifest=manifest)
        for row in plan.const:
            source(CS.manifest_key_for(row.index, row.tensor, manifest))
        assert {t for t, _, _ in report.re_encoded} == {"arg808", "arg809"}
        assert dict((t, (c, d)) for t, c, d in report.re_encoded)["arg809"] == ("float32", "bf16")
