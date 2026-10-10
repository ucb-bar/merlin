"""Trusted default-policy wiring with substituted builds/processes, not ISA proof."""

import copy
import json
from types import SimpleNamespace

import pytest

from merlin.common.paths import runtime_dir
from merlin.runtime.backends import base as backends
from merlin.targetgen import build_cache
from merlin.targetgen.contract import compile as compiler
from merlin.targetgen.contract import readback_policy as readback
from merlin.targetgen.contract.build_recipe import HarnessBuildRecipe


@pytest.mark.parametrize("backend", [None, SimpleNamespace(), SimpleNamespace(readback_policy=None)])
def test_absent_backend_default_keeps_legacy_selection(backend):
    assert readback.selected(None, backend=backend) is None


@pytest.mark.parametrize("value", [None, {}, "out_b64_v1", True])
def test_declared_default_requires_exact_policy(value):
    with pytest.raises(ValueError, match="trusted ReadbackPolicy"):
        readback.selected(None, backend=SimpleNamespace(readback_policy=lambda: value))


def test_explicit_policy_preserves_identity_and_never_calls_backend():
    def forbidden():
        pytest.fail("explicit selection invoked default")

    policy = readback.ReadbackPolicy(readback.FULL_VALUES_BIN)
    assert readback.selected(policy, backend=SimpleNamespace(readback_policy=forbidden)) is policy
    with pytest.raises(ValueError, match="trusted ReadbackPolicy"):
        readback.selected({}, backend=SimpleNamespace(readback_policy=forbidden))


@pytest.fixture
def ordinary(monkeypatch, tmp_path):
    policy = readback.ReadbackPolicy(readback.FULL_VALUES_B64)
    script = tmp_path / "layout.ld"
    script.write_text("SECTIONS {}\n")
    source = tmp_path / "selected-renderer.py"
    source.write_text("# owned diagnostic renderer selection\n")
    recipe = HarnessBuildRecipe(
        compiler=tmp_path / "unexecuted-compiler",
        include_roots=(),
        support_sources=(),
        link_script=script,
        load_address=0,
        cflags=("-march=rv64gc", "-mabi=lp64d"),
    )
    calls = []

    def render(cb, *, target, inputs, readback_policy):
        calls.append(("render", readback_policy))
        assert target == "synthetic" and inputs == {"X": [[1, 2]]}
        assert readback_policy is policy
        return '#include "out_b64.h"\nint main(void) { return 0; }\n'

    def substitute_build(argv, **kwargs):
        calls.append((kwargs["stage"], tuple(argv)))
        for path in kwargs["outputs"]:
            path.write_bytes(b"owned diagnostic product: " + kwargs["stage"].encode())
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    backend = SimpleNamespace(
        readback_policy=lambda: policy,
        render_harness=render,
        ORACLE={"spike": {"kind": "functional", "derived_from_rtl": False}},
        parse_output=backends.parse_console,
    )
    monkeypatch.setattr(backends, "get_backend", lambda _target: backend)
    monkeypatch.setattr(backends, "harness_build_recipe", lambda _target: recipe)
    monkeypatch.setattr(backends, "harness_renderer", lambda _target: render)
    monkeypatch.setattr(build_cache, "build_path", lambda _target: (source,))
    monkeypatch.setattr(compiler, "_observed_run", substitute_build)
    monkeypatch.setattr("merlin.targetgen.runtime_build.derived_link_script", lambda *_args: script)
    monkeypatch.setattr(compiler, "simulator_provenance", lambda *_args: None)
    cb = {
        "kernel_abi": {"kind": "whole_program", "outputs": ["Y"]},
        "tensors": {
            "X": {"shape": [1, 2], "dtype": "i8", "role": "input"},
            "Y": {"shape": [1, 2], "dtype": "i8", "role": "output"},
        },
    }
    return SimpleNamespace(policy=policy, recipe=recipe, backend=backend, cb=cb, calls=calls)


def test_default_link_stages_codec_and_binds_actual_diagnostic_products(ordinary, tmp_path):
    cb = copy.deepcopy(ordinary.cb)
    obj = tmp_path / "kernel.o"
    obj.write_bytes(b"owned object bytes")
    elf = compiler.link_elf(cb, obj, tmp_path, target="synthetic", inputs={"X": [[1, 2]]})
    receipt = readback.require_current_build_receipt(
        cb=cb, target="synthetic", workdir=tmp_path, elf_path=elf, policy=ordinary.policy
    )
    assert receipt["status"] == "completed"
    assert receipt["readback_policy"] == ordinary.policy.record()
    assert receipt["kernel_object_sha256"] == readback.file_sha256(obj)
    assert (tmp_path / "out_b64.h").read_bytes() == (runtime_dir() / "baremetal/out_b64.h").read_bytes()
    assert ordinary.calls[0] == ("render", ordinary.policy)
    assert cb == ordinary.cb


def test_default_policy_bypasses_legacy_cache_and_reaches_link(ordinary, monkeypatch, tmp_path):
    for name in ("build_identity", "reuse", "store"):
        monkeypatch.setattr(build_cache, name, lambda *a, **kw: pytest.fail("legacy cache reached"))
    monkeypatch.setattr(compiler, "llvm_mlir_to_object", lambda *a, **kw: tmp_path / "kernel.o")
    calls = []
    monkeypatch.setattr(compiler, "link_elf", lambda *a, **kw: calls.append(kw) or tmp_path / "model.elf")
    assert (
        compiler.compile_lowered_to_elf(ordinary.cb, "unused", tmp_path, target="synthetic") == tmp_path / "model.elf"
    )
    assert calls == [{"target": "synthetic", "inputs": None, "warm_profile": None, "readback_policy": ordinary.policy}]


def test_backend_without_default_keeps_legacy_link_without_receipt_or_codec(ordinary, monkeypatch, tmp_path):
    del ordinary.backend.readback_policy
    obj = tmp_path / "kernel.o"
    obj.write_bytes(b"owned object bytes")
    calls = []

    def legacy_render(cb, *, target, inputs):
        calls.append((target, inputs))
        return "int main(void) { return 0; }\n"

    monkeypatch.setattr(backends, "harness_renderer", lambda _target: legacy_render)
    compiler.link_elf(ordinary.cb, obj, tmp_path, target="synthetic", inputs={"X": [[1, 2]]})
    assert calls == [("synthetic", {"X": [[1, 2]]})]
    assert not (tmp_path / readback.BUILD_RECEIPT).exists()
    assert not (tmp_path / "out_b64.h").exists()


def test_backend_without_default_keeps_legacy_cache_path(ordinary, monkeypatch, tmp_path):
    del ordinary.backend.readback_policy
    events = []
    monkeypatch.setattr(build_cache, "build_identity", lambda **kwargs: events.append("key") or "owned-key")
    monkeypatch.setattr(build_cache, "reuse", lambda *args: events.append("reuse") or tmp_path / "existing.elf")
    monkeypatch.setattr(compiler, "llvm_mlir_to_object", lambda *a, **kw: pytest.fail("cached object rebuilt"))
    assert (
        compiler.compile_lowered_to_elf(ordinary.cb, "unused", tmp_path, target="synthetic")
        == tmp_path / "existing.elf"
    )
    assert events == ["key", "reuse"]


@pytest.mark.parametrize(
    "fault", [None, "implicit_abi", "missing", "extra", "shape", "digest", "receipt", "codec", "elf"]
)
def test_default_oracle_uses_complete_roster_and_postrun_build_recheck(ordinary, monkeypatch, tmp_path, fault):
    full = "OUT_B64_BEGIN v1 Y 1 2 1 s\nOUT_B64_CHUNK 00000000 0002 AQI=\nOUT_B64_END\nDONE\n"
    console = full
    if fault == "missing":
        console = "DONE\n"
    elif fault == "extra":
        console = full.replace("DONE\n", full.replace(" Y ", " Z "))
    elif fault == "shape":
        console = full.replace("Y 1 2", "Y 2 1")
    elif fault == "digest":
        console = "OUTSUM Y 1 2 0000000000000000\nDONE\n"
    cb = copy.deepcopy(ordinary.cb)
    if fault == "implicit_abi":
        cb.pop("kernel_abi")

    def substitute_lower(_cb, _text, work, **kwargs):
        assert kwargs["readback_policy"] is ordinary.policy
        obj = work / "kernel.o"
        obj.write_bytes(b"owned diagnostic kernel object")
        return compiler.link_elf(_cb, obj, work, target=kwargs["target"], inputs=kwargs["inputs"])

    def substitute_execution(elf, **kwargs):
        assert kwargs == {"simulator": "spike", "timeout": 600}
        if fault == "receipt":
            path = tmp_path / readback.BUILD_RECEIPT
            data = json.loads(path.read_text())
            data["status"] = "incomplete"
            path.write_text(json.dumps(data))
        elif fault == "codec":
            with (tmp_path / "out_b64.h").open("ab") as stream:
                stream.write(b"changed")
        elif fault == "elf":
            elf.write_bytes(b"changed")
        return console

    monkeypatch.setattr(compiler, "compile_lowered_to_elf", substitute_lower)
    ordinary.backend.run_elf = substitute_execution
    kwargs = dict(simulator="spike", target="synthetic", workdir=tmp_path, inputs={"X": [[1, 2]]})
    if fault in (None, "implicit_abi"):
        result = compiler.run_on_oracle(cb, "unused", **kwargs)
        assert result["outputs"] == {"Y": [[1, 2]]}
        assert result["readback_build"]["readback_policy"] == ordinary.policy.record()
    else:
        with pytest.raises(ValueError):
            compiler.run_on_oracle(cb, "unused", **kwargs)
    assert (tmp_path / "oracle_console.log").read_text() == console


@pytest.mark.parametrize(
    "fault", ["null", "empty", "partial", "list", "bool", "role", "layout", "dtype", "missing", "extra", "shape"]
)
def test_implicit_logical_roster_is_complete_and_never_repairs_explicit_abi(fault):
    cb = {
        "tensors": {
            "X": {"shape": [1, 2], "dtype": "i8", "role": "input"},
            "Y": {"shape": [1, 2], "dtype": "i8", "role": "output"},
            "Z": {"shape": [1, 2], "dtype": "i8", "role": "output"},
        }
    }
    console = "OUT_B64_BEGIN v1 Y 1 2 1 s\nOUT_B64_CHUNK 00000000 0002 AQI=\nOUT_B64_END\n"
    console += console.replace(" Y ", " Z ") + "DONE\n"
    if fault == "null":
        cb["kernel_abi"] = None
    elif fault == "empty":
        cb["kernel_abi"] = {}
    elif fault == "partial":
        cb["kernel_abi"] = {"kind": "whole_program"}
    elif fault == "list":
        cb["kernel_abi"] = ["not_an_abi"]
    elif fault == "bool":
        cb["kernel_abi"] = True
    elif fault == "role":
        cb["tensors"]["X"]["role"] = "intermediate"
    elif fault == "layout":
        cb["tensors"]["X"]["strides"] = [1, 2]
    elif fault == "dtype":
        cb["tensors"]["X"]["dtype"] = "unregistered"
    elif fault == "missing":
        console = console[console.index("OUT_B64_BEGIN v1 Z") :]
    elif fault == "extra":
        extra = console[: console.index("OUT_B64_BEGIN v1 Z")].replace(" Y ", " OTHER ")
        console = console.replace("DONE\n", extra + "DONE\n")
    elif fault == "shape":
        console = console.replace("Z 1 2", "Z 2 1")
    before = copy.deepcopy(cb)
    outputs, _metrics = backends.parse_console(console)
    with pytest.raises(ValueError):
        readback.require_full_value_roster(
            cb, console, outputs, policy=readback.ReadbackPolicy(readback.FULL_VALUES_B64)
        )
    assert cb == before


def test_implicit_logical_roster_keeps_every_ordered_output_and_source_bytes():
    from merlin.runtime.harness_render import logical_output_names

    cb = {
        "tensors": {
            "Z": {"shape": [1, 2], "dtype": "i8", "role": "output"},
            "X": {"shape": [1, 2], "dtype": "i8", "role": "input"},
            "Y": {"shape": [1, 2], "dtype": "i8", "role": "output"},
        }
    }
    before = copy.deepcopy(cb)
    assert logical_output_names(cb) == ("Z", "Y")
    console = "OUT_B64_BEGIN v1 Z 1 2 1 s\nOUT_B64_CHUNK 00000000 0002 AQI=\nOUT_B64_END\n"
    console += console.replace(" Z ", " Y ") + "DONE\n"
    outputs, _metrics = backends.parse_console(console)
    readback.require_full_value_roster(cb, console, outputs, policy=readback.ReadbackPolicy(readback.FULL_VALUES_B64))
    assert cb == before


@pytest.mark.parametrize("route", ["link", "compile"])
def test_explicit_service_does_not_resolve_backend_default(monkeypatch, tmp_path, route):
    # Existing typed service validation remains the authority; no backend lookup
    # can run first, even for a deliberately invalid service in this negative.
    monkeypatch.setattr(backends, "get_backend", lambda *_args: pytest.fail("service discovered a backend"))
    with pytest.raises(ValueError, match="build-only service"):
        if route == "link":
            compiler.link_elf({}, tmp_path / "kernel.o", tmp_path, target="synthetic", _build_service=object())
        else:
            compiler.compile_lowered_to_elf({}, "unused", tmp_path, target="synthetic", _build_service=object())
