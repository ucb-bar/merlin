"""Output-only ABI and ordinary build dispatch; no source-family qualification.

The private constant-output functions exercise the generic call/readback ABI.
They are not compiler seeds, independent semantic issuers or static proofs.
"""

import os
import shutil
from functools import partial
from pathlib import Path

import pytest

from merlin.common import invocation_record
from merlin.common.paths import runtime_dir
from merlin.runtime.direct_kernel_harness import DirectKernelAbi, render_direct_kernel
from merlin.runtime.out_b64 import OutB64Decoder
from merlin.targetgen.contract import compile as compiler
from merlin.targetgen.contract.build_recipe import HarnessBuildRecipe
from merlin.targetgen.contract.build_service import BuildOnlyService, file_digest
from merlin.targetgen.contract.compile_only import CompileOnlySourceAbi, CompileOnlyTensor, prepare_linkage
from merlin.targetgen.contract.linalg_iface import parse_linalg_mlir
from merlin.targetgen.contract.pointer_storage import OriginalPointerStorageContract, PointerStoragePolicy
from merlin.targetgen.contract.readback_policy import FULL_VALUES_B64, ReadbackPolicy


def _buffer():
    return {
        "abi_version": "0.1",
        "target": "zero_input_fixture",
        "commands": [],
        "tensors": {"Y": {"shape": [5], "dtype": "i16", "role": "output"}},
        "kernel_abi": {
            "kind": "whole_program",
            "args": [{"tensor": "Y", "access": "write"}],
            "outputs": ["Y"],
        },
        "params": {"global_program_plan": {"entry_bindings": []}},
    }


def _abi():
    return CompileOnlySourceAbi((), (CompileOnlyTensor("Y", (5,), "i16"),))


def _storage():
    return OriginalPointerStorageContract(
        _abi(),
        PointerStoragePolicy(
            "row_major_contiguous", "inputs_then_outputs", "disjoint", "static_original_tensor_type", "little", 8
        ),
    )


def _service(tmp_path, renderer):
    owner = Path(__file__).resolve()
    recipe = HarnessBuildRecipe(
        Path("/usr/bin/cc"),
        (),
        (),
        tmp_path / "unused.ld",
        0,
        ("-march=rv64gc", "-mabi=lp64d"),
    )
    return BuildOnlyService("zero_input_fixture", recipe, renderer, ((str(owner), file_digest(owner)),))


def test_original_zero_argument_tensor_signature_and_output_only_linkage():
    source = """module { func.func @forward() -> tensor<5xi16> {
      %value = arith.constant dense<[-32768, -7, 0, 15, 32767]> : tensor<5xi16>
      func.return %value : tensor<5xi16>
    } }"""
    parsed = parse_linalg_mlir(source)
    assert parsed["args"] == [] and parsed["results"] == [{"shape": [5], "dtype": "i16"}]
    cb = _buffer()
    linkage, binding = prepare_linkage(
        cb=cb,
        lowered_mlir="module { llvm.func @output_only(%out: !llvm.ptr) { llvm.return } }",
        entry_symbol="output_only",
        original_abi=_abi(),
    )
    assert binding["original_abi"]["inputs"] == []
    assert binding["bindings"] == [{"role": "output", "source": "Y", "emitted": "Y"}]
    assert linkage.pointer_arity == 1
    assert "output_only((void *)0)" in linkage.render(cb)
    slots = _storage().slots
    assert len(slots) == 1 and slots[0].role == "output" and slots[0].byte_extent == 10
    assert _storage().bind_candidate(cb) == binding


@pytest.mark.parametrize(
    "inputs,outputs",
    [
        (None, _abi().outputs),
        ([], _abi().outputs),
        ((), None),
        ((), []),
        ((), ()),
        ((), (CompileOnlyTensor("Y", (5,), "i16"),) * 2),
    ],
)
def test_missing_or_malformed_original_rosters_still_refuse(inputs, outputs):
    with pytest.raises(ValueError, match="immutable|repeats"):
        CompileOnlySourceAbi(inputs, outputs).record()


@pytest.mark.parametrize("defect", ["missing_output", "missing_pointer", "input_pointer", "shape", "dtype", "name"])
def test_output_only_candidate_must_match_every_original_slot(defect):
    cb = _buffer()
    if defect == "missing_output":
        cb["kernel_abi"]["outputs"] = []
    elif defect == "missing_pointer":
        cb["kernel_abi"]["args"] = []
    elif defect == "input_pointer":
        cb["kernel_abi"]["args"][0]["access"] = "readwrite"
    elif defect in {"shape", "dtype"}:
        cb["tensors"]["Y"][defect] = [4] if defect == "shape" else "i8"
    else:
        cb["kernel_abi"]["outputs"] = ["renamed"]
    with pytest.raises(ValueError):
        _abi().bind(cb)


def test_empty_inputs_with_selected_pure_service_executes_complete_native_outputs(tmp_path):
    native = os.environ.get("MERLIN_TEST_CLANG")
    if native is None:
        native = shutil.which("cc")
        if native is None:
            pytest.skip("requires a native C compiler")
    selected = Path(native).resolve(strict=True)
    assert selected.is_file(), "selected native compiler is not a file"
    renderer = partial(
        render_direct_kernel,
        abi=DirectKernelAbi("output_only", "complete", 8, "little", "void"),
        original_storage=_storage(),
    )
    service = _service(tmp_path, renderer)
    policy = ReadbackPolicy(FULL_VALUES_B64)
    harness = tmp_path / "harness.c"
    harness.write_text(service.render(_buffer(), target=service.target, inputs={}, readback_policy=policy))
    (tmp_path / "htif.h").write_text(
        "void console_init(void); void htif_puts(const char*); void htif_exit(int) __attribute__((noreturn));\n"
    )
    codec = tmp_path / "out_b64.h"
    codec.write_bytes((runtime_dir() / "baremetal/out_b64.h").read_bytes())
    kernel = tmp_path / "control.c"
    kernel.write_text(
        "#include <stdint.h>\n#include <stdio.h>\n#include <stdlib.h>\n"
        "static int completed; void console_init(void){}\n"
        "void htif_puts(const char*s){if(!completed)exit(4);fputs(s,stdout);}\n"
        "void htif_exit(int code){exit(code);}\n"
        "void output_only(void*p){int16_t*y=p;int16_t v[]={-32768,-7,0,15,32767};"
        "for(unsigned i=0;i<5;i++)y[i]=v[i];}\n"
        "void complete(void){completed=1;}\n"
        "int harness_main(void);int main(void){return harness_main();}\n"
    )
    obj, executable = tmp_path / "harness.o", tmp_path / "control.elf"
    environment = {"PATH": "/usr/bin:/bin", "LC_ALL": "C"}
    for command, inputs, outputs, stage in (
        (
            [str(selected), "-std=c11", "-O2", "-Dmain=harness_main", "-c", str(harness), "-o", str(obj)],
            (harness, codec, tmp_path / "htif.h"),
            (obj,),
            "zero_input_harness_object",
        ),
        (
            [str(selected), "-std=c11", "-O2", str(kernel), str(obj), "-o", str(executable)],
            (kernel, obj),
            (executable,),
            "zero_input_native_link",
        ),
        ([str(executable)], (executable,), (), "zero_input_native_execution"),
    ):
        observed = invocation_record.run(
            command,
            directory=tmp_path,
            stage=stage,
            inputs=inputs,
            outputs=outputs,
            dependencies=(Path(__file__).resolve(), selected),
            env=environment,
            capture_output=True,
            text=True,
            timeout=20,
        )
        observed.check_returncode()
    outputs, decoder = {}, OutB64Decoder()
    for line in observed.stdout.splitlines():
        decoder.consume(line.split(), outputs)
    decoder.require_closed()
    assert outputs == {"Y": [[-32768, -7, 0, 15, 32767]]}
    assert observed.stdout.splitlines().count("DONE") == 1
    assert observed.stdout.splitlines()[-1] == "DONE"
    for record in tmp_path.rglob("invocation.json"):
        invocation_record.require_environment(record, environment=environment)


@pytest.mark.parametrize("inputs", [None, [], (), False])
def test_pure_service_requires_an_explicit_input_map(tmp_path, inputs):
    def renderer(*_args, **_kwargs):
        pytest.fail("rendered absent or malformed inputs")

    with pytest.raises(ValueError, match="explicit logical inputs"):
        _service(tmp_path, renderer).render(_buffer(), target="zero_input_fixture", inputs=inputs)


def test_empty_map_cannot_omit_a_declared_read_operand(tmp_path):
    cb = _buffer()
    cb["tensors"]["X"] = {"shape": [5], "dtype": "i16", "role": "input"}
    cb["kernel_abi"]["args"].insert(0, {"tensor": "X", "access": "read"})
    renderer = partial(render_direct_kernel, abi=DirectKernelAbi("output_only", None, 8, "little", "void"))
    with pytest.raises(ValueError, match="omits or adds"):
        _service(tmp_path, renderer).render(
            cb, target="zero_input_fixture", inputs={}, readback_policy=ReadbackPolicy(FULL_VALUES_B64)
        )


@pytest.mark.parametrize("selected_inputs", [None, {}, {"actual": [7]}])
def test_ordinary_cache_and_link_preserve_explicit_input_identity(tmp_path, monkeypatch, selected_inputs):
    from merlin.runtime.backends import base
    from merlin.targetgen import build_cache

    recorded, seen = {"recorded": [99]}, []

    def fallback(_cb):
        assert selected_inputs is None
        seen.append("fallback")
        return recorded

    expected = recorded if selected_inputs is None else selected_inputs
    recipe = _service(tmp_path, lambda *_a, **_k: "").recipe
    monkeypatch.setattr(base, "harness_build_recipe", lambda _target: recipe)
    monkeypatch.setattr(compiler, "_recorded_operands", fallback)
    monkeypatch.setattr(compiler, "llvm_mlir_to_object", lambda *_a, **_k: tmp_path / "kernel.o")

    def identity(**kwargs):
        assert kwargs["inputs"] is expected
        seen.append("identity")
        return None

    def link(_cb, _obj, _work, **kwargs):
        assert kwargs["inputs"] is expected
        seen.append("link")
        return tmp_path / "kernel.elf"

    monkeypatch.setattr(build_cache, "build_identity", identity)
    monkeypatch.setattr(build_cache, "reuse", lambda *_args: None)
    monkeypatch.setattr(build_cache, "store", lambda *_args: None)
    monkeypatch.setattr(compiler, "link_elf", link)
    compiler.compile_lowered_to_elf(
        _buffer(), "diagnostic LLVM", tmp_path, target="zero_input_fixture", inputs=selected_inputs
    )
    assert seen == (["fallback"] if selected_inputs is None else []) + ["identity", "link"]


@pytest.mark.parametrize("selected_inputs", [None, {}, {"actual": [7]}])
@pytest.mark.parametrize("pure_service", [False, True])
def test_link_renderer_distinguishes_explicit_empty_and_absent_inputs(
    tmp_path,
    monkeypatch,
    selected_inputs,
    pure_service,
):
    from merlin.runtime.backends import base
    from merlin.targetgen import runtime_build

    absent, seen = object(), []
    recipe = _service(tmp_path, lambda *_a, **_k: "").recipe
    monkeypatch.setattr(base, "harness_build_recipe", lambda _target: recipe)
    monkeypatch.setattr(runtime_build, "derived_link_script", lambda *_args: tmp_path / "unused.ld")

    def fallback(_cb):
        assert selected_inputs is None
        return None

    def renderer(_cb, *, target=None, inputs=absent):
        assert inputs is (absent if selected_inputs is None else selected_inputs)
        seen.append(inputs)
        return "int main(void){return 0;}\n"

    class Rendered(Exception):
        pass

    def stop_before_compile(*_args, **_kwargs):
        raise Rendered

    monkeypatch.setattr(compiler, "_recorded_operands", fallback)
    monkeypatch.setattr(base, "harness_renderer", lambda _target: renderer)
    monkeypatch.setattr(compiler, "_observed_run", stop_before_compile)
    service_kwargs = {"_build_service": _service(tmp_path, renderer)} if pure_service else {}
    if pure_service and selected_inputs is None:
        with pytest.raises(TypeError, match="inputs"):
            compiler.link_elf(
                _buffer(), tmp_path / "kernel.o", tmp_path, target="zero_input_fixture", inputs=None, **service_kwargs
            )
        assert seen == []
        return
    with pytest.raises(Rendered):
        compiler.link_elf(
            _buffer(),
            tmp_path / "kernel.o",
            tmp_path,
            target="zero_input_fixture",
            inputs=selected_inputs,
            **service_kwargs,
        )
    assert len(seen) == 1


def test_explicit_zero_input_roster_requires_input_aware_renderer(tmp_path, monkeypatch):
    from merlin.runtime.backends import base

    monkeypatch.setattr(base, "harness_build_recipe", lambda _target: _service(tmp_path, None).recipe)

    def legacy_renderer(_cb, *, target):
        pytest.fail("renderer silently ignored the explicit zero-input declaration")

    monkeypatch.setattr(base, "harness_renderer", lambda _target: legacy_renderer)
    with pytest.raises(NotImplementedError, match="cannot take `inputs`"):
        compiler.link_elf(_buffer(), tmp_path / "kernel.o", tmp_path, target="zero_input_fixture", inputs={})
    assert not (tmp_path / "harness.c").exists()
