"""Ordinary DeviceBuild consumption of a neutral compact staged contraction."""

import copy
import ctypes
import hashlib
import inspect
import json
from pathlib import Path

import device_staged_dense_fixture as F
import numpy as np
import pytest
import test_target_staged_dense as original
from device_llvm_configuration_fixture import CLANG, TRANSLATOR, imported, run
from test_device_llvm_transform import _original_tools
from xdsl.dialects import llvm
from xdsl.dialects.builtin import StringAttr
from xdsl.parser import Parser

from merlin.common import invocation_record as I
from merlin.compile.scheduling import staged_dense as S
from merlin.llvmlower import device_build as DB
from merlin.llvmlower import device_object_optimization as optimization
from merlin.llvmlower import staged_dense_llvm as W
from merlin.llvmlower import target_staged_dense as T
from merlin.llvmlower.host_transform_chain import HostLLVMArtifact, HostLLVMTransformChain
from merlin.targetgen.oot_starterkit.llvm_context import make_llvm_context

pytest_plugins = ("test_device_build_parallel",)


def test_context_local_metadata_requires_complete_equal_representation():
    source = str(F.loop_module())
    assert source.startswith("builtin.module {")
    source = source.replace("builtin.module {", 'builtin.module attributes {trace = #neutral.trace<"original">} {', 1)
    left = Parser(make_llvm_context(), source).parse_module()
    right = Parser(make_llvm_context(), source).parse_module()
    owner = F.LoopOwner(left)
    assert owner.project(right) == F.expected_operations()
    right.attributes["trace"] = StringAttr("changed")
    assert owner.project(right) is None


def pin(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def setup(directory, mode=None, enabled=True):
    directory.mkdir()
    sources = [Path(module.__file__) for module in (F, original, W, T, S, I, DB, optimization)]
    sources.extend(Path(inspect.getsourcefile(obj)) for obj in (llvm.FuncOp, _original_tools, imported))
    sources.extend(Path(inspect.getsourcefile(fn)) for fn in T._functions())
    sources.extend((Path(__file__), CLANG(), TRANSLATOR()))
    sources = tuple(HostLLVMArtifact.capture(path) for path in dict.fromkeys(sources))
    runtime = directory / "runtime.c"
    runtime.write_text(F.RUNTIME)
    source, llvm_path = directory / "original.mlir", directory / "original.ll"
    source.write_text(str(F.loop_module()))
    run([TRANSLATOR(), "--mlir-to-llvmir", source, "-o", llvm_path])
    actual = imported(llvm_path, directory)
    owner = F.LoopOwner(actual, mode)
    selection = W.StagedDenseLLVMTransform(
        S.StagedDenseProblem(F.ROWS, 1, 1),
        original.contract(),
        T.StagedDenseTargetAdapter(owner.project, owner.emit, complete_effects=True),
        HostLLVMArtifact.capture(TRANSLATOR()),
        (HostLLVMArtifact.capture(runtime),),
        rewrites=(S.StagedDenseRewrite("live_rhs", S.eliminate_redundant_rhs_loads),) if enabled else (),
        implementation_sources=sources,
    )
    stage = selection.stage()
    return selection, HostLLVMTransformChain((stage.contract,), (stage,)), runtime, source.read_text()


def build_arm(build, output, selected, monkeypatch):
    original_builder = DB.build_device_objects

    def redirected(*args, **kwargs):
        kwargs["workdir"] = output
        return original_builder(*args, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(DB, "build_device_objects", redirected)
        return build(names=("a",), codegen_target="host", cflags=["-O0", "-fPIC"], llvm_transform_chain=selected)


def test_ordinary_device_object_consumes_compact_selected_loop(synthetic_build, tmp_path, monkeypatch):
    build, _, _, _, artifacts, _ = synthetic_build
    _original_tools(monkeypatch)
    records, owners, chains = [], [], []
    for arm, enabled in (("baseline", False), ("candidate", True)):
        selection, chain, runtime, source = setup(tmp_path / arm, enabled=enabled)
        owners.append(selection)
        chains.append(chain)
        artifacts["a"] = source
        frozen = optimization.freeze_transform_chain(chain)
        output = tmp_path / arm / "built"
        result = build_arm(build, output, chain, monkeypatch)
        assert result.ok, result.skipped
        custody = result.custody["a"]
        emission = custody["llvm_transform"]
        assert emission["selected_sha256"] != emission["source_sha256"] if enabled else True
        observed = json.loads(Path(custody["optimization"]["path"]).read_text())
        assert observed["inputs"][0]["path"] == emission["selected_path"]
        assert observed["inputs"][0]["sha256"] == emission["selected_sha256"]
        selected_path = Path(emission["selected_path"])
        typed = imported(selected_path, output)
        assert sum(isinstance(op, llvm.CondBrOp) for op in typed.walk()) == (2 if enabled else 1)
        library = output / "native.so"
        I.run(
            [str(CLANG()), "-shared", "-fPIC", str(result.objects[0]), str(runtime), "-o", str(library)],
            directory=output / "native-link",
            stage="neutral_native_link",
            inputs=(*result.objects, runtime),
            outputs=(library,),
            check=True,
            capture_output=True,
            timeout=60,
        )
        image_pin = pin(library)
        image = ctypes.CDLL(str(library))
        function = getattr(image, result.kernels["a"])
        function.argtypes = [ctypes.c_void_p] * 3
        function.restype = None
        state, expected_state, epochs, observations = 0, 0, [], []
        for epoch in range(3):
            a = np.array([91, *[((i * 7 + epoch + state) % 19) - 9 for i in range(F.ROWS)], 92], dtype=np.int8)
            b = np.array([93, epoch - 3, 94], dtype=np.int8)
            out = np.full(F.ROWS + 8, 0x12345678, dtype=np.int32)
            before_a, before_b, prior = a.copy(), b.copy(), state
            expected = np.array([int(a[i + 1]) * int(b[1]) for i in range(F.ROWS)], dtype=np.int32)
            function(a.ctypes.data + 1, b.ctypes.data + 1, out.ctypes.data + 16)
            assert out[4:-4].tobytes() == expected.tobytes()
            assert a.tobytes() == before_a.tobytes() and b.tobytes() == before_b.tobytes()
            assert np.all(out[:4] == 0x12345678) and np.all(out[-4:] == 0x12345678)
            state = sum(int(v) for v in out[4:-4]) + epoch
            expected_state = sum(int(v) for v in expected) + epoch
            assert state == expected_state
            arrays = output / f"epoch-{epoch}.npz"
            np.savez(
                arrays,
                A_before=before_a,
                A_after=a,
                B_before=before_b,
                B_after=b,
                output_and_guards=out,
                prior_state=np.array(prior),
                state=np.array(state),
                expected=expected,
            )
            epochs.append(pin(arrays))
            observations.append((out.tobytes().hex(), state))
        counts = tuple((ctypes.c_long * 2).in_dll(image, "counts"))
        assert counts == (9, 3 if enabled else 9)
        assert pin(library) == image_pin
        selection.revalidate()
        optimization.recheck_transform_chain(chain, frozen, emission)
        records.append(
            {
                "arm": arm,
                "custody": custody,
                "loaded_image": image_pin,
                "epochs": epochs,
                "observations": observations,
                "actual_operand_loads": counts,
                "remaining_source_loops": 1,
            }
        )
    assert records[0]["observations"] == records[1]["observations"]
    # All owner/recipe/invocation callbacks precede a collective BOTH-arm byte close.
    sources = [
        item.to_dict()
        for chain in chains
        for stage in chain.promised_stages
        for item in (*stage.sources, *stage.semantic_proofs)
    ]
    for owner in owners:
        owner.revalidate()
    for invocation in tmp_path.rglob("invocation.json"):
        I.verify(invocation)
    products = [pin(path) for path in sorted(tmp_path.rglob("*")) if path.is_file()]
    for item in (*products, *sources):
        assert pin(item["path"])["sha256"] == item["sha256"]
    (tmp_path / "pair.json").write_text(
        json.dumps(
            {
                "arms": records,
                "products": products,
                "original_sources": sources,
                "scope": "Neutral DeviceBuild LLVM/object/native consumption; "
                "no target effects, timing or hardware qualification",
            },
            indent=2,
        )
        + "\n"
    )


@pytest.mark.parametrize("mode", ["ignore", "wrong_pointer"])
def test_invalid_actual_emitter_never_reaches_accepted_object(synthetic_build, tmp_path, monkeypatch, mode):
    build, _, _, _, artifacts, _ = synthetic_build
    _original_tools(monkeypatch)
    _, chain, _, source = setup(tmp_path / "owner", mode=mode)
    artifacts["a"] = source
    result = build_arm(build, tmp_path / "built", chain, monkeypatch)
    assert not result.ok and not result.objects and result.shim_object is None


@pytest.mark.parametrize("damage", ["contract", "proof"])
def test_selected_owner_or_proof_cannot_drift(tmp_path, damage):
    selection, _, runtime, _ = setup(tmp_path / "owner")
    if damage == "contract":
        object.__setattr__(selection.contract, "rhs_immutable", False)
    else:
        runtime.write_text(runtime.read_text() + "\n/* drift */\n")
    with pytest.raises(ValueError):
        selection.revalidate()


def test_unknown_projection_retains_exact_original_llvm(tmp_path):
    selection, _, _, _ = setup(tmp_path / "owner", mode="unknown")
    source = tmp_path / "owner/original.ll"
    work = tmp_path / "transform"
    work.mkdir()
    output = selection.transform(source, work)
    assert output == source
    report = json.loads((work / "proposal.json").read_text())
    assert report["status"] == "fallback"


@pytest.mark.parametrize(
    "damage",
    [
        "parser_alias",
        "printer_alias",
        "context",
        "core_reader",
        "helper_code",
        "callback_defaults",
        "callback_owner",
        "parser_code",
        "generated_printer_code",
    ],
)
def test_first_original_tool_callback_cannot_replace_readers(tmp_path, monkeypatch, damage):
    real_run, invoked, replaced_calls = I.run, [], []

    def replacement(*args, **kwargs):
        replaced_calls.append(damage)
        raise AssertionError("changed implementation must not be invoked")

    def observed(argv, **kwargs):
        if not invoked:
            invoked.append(damage)
            if damage == "parser_alias":
                monkeypatch.setattr(W, "Parser", replacement)
            elif damage == "printer_alias":
                monkeypatch.setattr(W, "DenseFloatPrinter", replacement)
            elif damage == "context":
                monkeypatch.setattr(W, "make_llvm_context", replacement)
            elif damage == "core_reader":
                old_reader = T._functions
                monkeypatch.setattr(T, "_functions", lambda: old_reader())
            elif damage == "helper_code":
                monkeypatch.setattr(T._module_state, "__code__", T._module_state.__code__.replace(co_name="changed"))
            elif damage == "callback_defaults":
                monkeypatch.setattr(F.LoopOwner.project, "__defaults__", (None,))
            elif damage == "callback_owner":
                different_owner = copy.copy(selection.adapter.project.__self__)
                object.__setattr__(selection.adapter, "project", different_owner.project)
            elif damage == "parser_code":
                monkeypatch.setattr(
                    W.Parser.parse_module, "__code__", W.Parser.parse_module.__code__.replace(co_name="changed")
                )
            else:
                monkeypatch.setattr(
                    W.DenseFloatPrinter.__init__,
                    "__code__",
                    W.DenseFloatPrinter.__init__.__code__.replace(co_name="changed"),
                )
        return real_run(argv, **kwargs)

    monkeypatch.setattr(I, "run", observed)
    selection, _, _, _ = setup(tmp_path / "owner")
    assert not invoked
    work = tmp_path / "transform"
    work.mkdir()
    with pytest.raises(ValueError, match="staged LLVM.*changed|replan"):
        selection.transform(tmp_path / "owner/original.ll", work)
    assert invoked == [damage] and not replaced_calls
