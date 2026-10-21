"""Consumed compact LLVM, native rectangular tails and refused false emission."""

import ctypes
import json
from dataclasses import replace
from pathlib import Path

import device_compact_dense_fixture as F
import numpy as np
import pytest
from device_llvm_configuration_fixture import CLANG, TRANSLATOR, imported, run
from test_device_llvm_transform import _original_tools
from test_staged_dense_llvm_build import build_arm

from merlin.common import invocation_record as I
from merlin.compile.scheduling import compact_dense as D
from merlin.llvmlower import compact_dense_llvm as W
from merlin.llvmlower import counted_llvm_cfg as C
from merlin.llvmlower.host_transform_chain import HostLLVMArtifact, HostLLVMTransformChain
from merlin.transforms.dense_planning import dense_blocks

pytest_plugins = ("test_device_build_parallel",)


def setup(directory, *, mode=None, enabled=True, reuse_pe_weights=False, completion=True):
    directory.mkdir()
    problem, resources, storage = F.fixture()
    adapter = replace(F.ADAPTER, synchronous_completion=completion)
    block = dense_blocks(problem, replace(resources, max_m_blocks=1, max_k_blocks=1, max_n_blocks=1))[0]
    kernel = D.emit_compact_dense(problem, block, storage, F.ADAPTER, name="kernel", dtypes=("i8", "i8", "i32"))
    typed, source = directory / "source.mlir", directory / "source.ll"
    typed.write_text(str(F.module(kernel)))
    run([TRANSLATOR(), "--mlir-to-llvmir", typed, "-o", source])
    owner = F.Owner(imported(source, directory), kernel, mode)
    runtime = directory / "runtime.c"
    runtime.write_text(F.RUNTIME)
    dependencies = tuple(
        HostLLVMArtifact.capture(path)
        for path in dict.fromkeys([Path(module.__file__) for module in (F, C, W)] + [Path(__file__)])
    )
    selection = W.CompactDenseLLVMTransform(
        problem,
        resources,
        storage,
        adapter,
        owner.project,
        owner.emit,
        HostLLVMArtifact.capture(TRANSLATOR()),
        (HostLLVMArtifact.capture(runtime),),
        complete_effects=True,
        external_buffers_disjoint=True,
        implementation_sources=dependencies,
        enabled=enabled,
        reuse_pe_weights=reuse_pe_weights,
    )
    stage = selection.stage()
    return selection, HostLLVMTransformChain((stage.contract,), (stage,)), runtime, typed.read_text()


@pytest.mark.parametrize("pe", [False, True])
def test_ordinary_objects_consume_compact_blocking_and_retained_weights(synthetic_build, tmp_path, monkeypatch, pe):
    build, _, _, _, artifacts, _ = synthetic_build
    _original_tools(monkeypatch)
    observations, owners, counts = [], [], []
    for arm, enabled in (("baseline", False), ("candidate", True)):
        selection, chain, runtime, text = setup(tmp_path / arm, enabled=enabled, reuse_pe_weights=pe)
        owners.append(selection)
        artifacts["a"] = text
        output = tmp_path / arm / "built"
        result = build_arm(build, output, chain, monkeypatch)
        assert result.ok, result.skipped
        receipt = result.custody["a"]["llvm_transform"]
        consumed = json.loads(Path(result.custody["a"]["optimization"]["path"]).read_text())
        assert consumed["inputs"][0]["sha256"] == receipt["selected_sha256"]
        if enabled:
            assert receipt["selected_sha256"] != receipt["source_sha256"]
        selected = imported(Path(receipt["selected_path"]), output)
        actual = selection.project_module(selected)
        assert actual is not None
        selection._verify(actual, selection.storage)
        # pytest may reclaim a successful case's directory before the next
        # parameter, while dlopen still holds that pathname's old image.
        image = output / f"native-pe-{int(pe)}.so"
        I.run(
            [str(CLANG()), "-shared", "-fPIC", str(result.objects[0]), str(runtime), "-o", str(image)],
            directory=output / "native-link",
            stage="neutral_compact_native_link",
            inputs=(*result.objects, runtime),
            outputs=(image,),
            check=True,
            capture_output=True,
            timeout=60,
        )
        library = ctypes.CDLL(str(image))
        function = getattr(library, result.kernels["a"])
        function.argtypes, function.restype = [ctypes.c_void_p] * 3, None
        state, values = 0, []
        for epoch in range(3):
            a = (np.arange(35, dtype=np.int64).reshape(5, 7) * 3 + epoch + state) % 17 - 8
            b = (np.arange(63, dtype=np.int64).reshape(7, 9) * 5 + epoch) % 13 - 6
            left = np.array([91, *a.reshape(-1), 92], dtype=np.int8)
            right = np.array([93, *b.reshape(-1), 94], dtype=np.int8)
            out = np.full(53, 0x12345678, dtype=np.int32)
            before_left, before_right = left.copy(), right.copy()
            # Independent full reduction, not the blocked schedule or machine.
            expected = np.array(
                [sum(int(a[m, k]) * int(b[k, n]) for k in range(7)) for m in range(5) for n in range(9)], dtype=np.int32
            )
            function(left.ctypes.data + 1, right.ctypes.data + 1, out.ctypes.data + 16)
            np.testing.assert_array_equal(out[4:-4], expected)
            np.testing.assert_array_equal(left, before_left)
            np.testing.assert_array_equal(right, before_right)
            assert np.all(out[:4] == 0x12345678) and np.all(out[-4:] == 0x12345678)
            state = sum((i + 1) * int(v) for i, v in enumerate(out[4:-4])) + epoch
            values.append((out.tobytes(), state))
        assert len({state for _, state in values}) > 1
        observations.append(values)
        counts.append(tuple((ctypes.c_long * 7).in_dll(library, "counts")))
    assert observations[0] == observations[1]
    assert counts[0][:5] == (81, 81, 81, 81, 27)
    assert counts[1][0] < counts[0][0] and counts[1][1] < counts[0][1]
    assert counts[1][2:5] == counts[0][2:5]
    assert counts[0][5:] == (81, 0)
    assert counts[1][5] < 81 if pe else counts[1][5] == 81
    assert counts[1][6] > 0 if pe else counts[1][6] == 0
    for owner in owners:
        owner.revalidate()
    for receipt in tmp_path.rglob("invocation.json"):
        I.verify(receipt)


@pytest.mark.parametrize("mode", ["ignore", "wrong_pointer"])
def test_mismatching_emitter_refuses_before_object_acceptance(synthetic_build, tmp_path, monkeypatch, mode):
    build, _, _, _, artifacts, _ = synthetic_build
    _original_tools(monkeypatch)
    _, chain, _, text = setup(tmp_path / "owner", mode=mode)
    artifacts["a"] = text
    result = build_arm(build, tmp_path / "built", chain, monkeypatch)
    assert not result.ok and not result.objects and result.shim_object is None


@pytest.mark.parametrize("missing", ["unknown", "completion", "ownership", "effects"])
def test_unsupported_route_retains_exact_original_llvm(tmp_path, missing):
    options = {"mode": "unknown"} if missing == "unknown" else {"completion": False} if missing == "completion" else {}
    selection, _, _, _ = setup(tmp_path / "owner", **options)
    if missing in ("ownership", "effects"):
        selection = replace(
            selection, **{("external_buffers_disjoint" if missing == "ownership" else "complete_effects"): False}
        )
    source = tmp_path / "owner/source.ll"
    work = tmp_path / "check"
    work.mkdir()
    selected = selection.transform(source, work)
    assert selected.read_bytes() == source.read_bytes()


@pytest.mark.parametrize("mutation", ["policy", "storage", "proof", "callback", "reuse"])
def test_selected_premises_and_implementations_cannot_drift(tmp_path, mutation):
    selection, _, runtime, _ = setup(tmp_path / "owner")
    if mutation == "policy":
        object.__setattr__(selection.adapter, "numerical_policy_identity", "changed")
    elif mutation == "storage":
        object.__setattr__(selection.storage.lhs, "base", 1)
    elif mutation == "proof":
        runtime.write_text(runtime.read_text() + "\n/* changed */\n")
    elif mutation == "reuse":
        object.__setattr__(selection, "reuse_pe_weights", True)
    else:
        object.__setattr__(selection, "emit_module", F.module)
    with pytest.raises(ValueError, match="changed|drift"):
        selection.revalidate()


@pytest.mark.parametrize("damage", ["reader", "printer"])
def test_first_translation_cannot_replace_the_original_live_reader(tmp_path, monkeypatch, damage):
    original_run = I.run
    original_reader = C.project_counted_function
    changed = False

    def mutating(*args, **kwargs):
        nonlocal changed
        if not changed:
            changed = True
            if damage == "reader":
                monkeypatch.setattr(original_reader, "__code__", (lambda *args, **kwargs: ()).__code__)
            else:
                monkeypatch.setattr(W.DenseFloatPrinter, "__init__", lambda *args, **kwargs: None)
        return original_run(*args, **kwargs)

    monkeypatch.setattr(I, "run", mutating)
    selection, _, _, _ = setup(tmp_path / "owner")
    work = tmp_path / "check"
    work.mkdir()
    with pytest.raises(ValueError, match="implementation.*changed|constructor changed"):
        selection.transform(tmp_path / "owner/source.ll", work)
    assert changed


@pytest.mark.parametrize("damage", ["selection", "output_bytes"])
def test_invocation_verification_cannot_leave_late_mutation_unchecked(tmp_path, monkeypatch, damage):
    original_verify = I.verify
    state = {}

    def mutating(*args, **kwargs):
        result = original_verify(*args, **kwargs)
        if state.get("armed"):
            state["armed"] = False
            if damage == "selection":
                object.__setattr__(state["selection"].adapter, "configuration_identity", "changed")
            else:
                state["output"].write_text(state["output"].read_text() + "\n; late byte change\n")
        return result

    monkeypatch.setattr(I, "verify", mutating)
    selection, _, _, _ = setup(tmp_path / "owner", enabled=False)
    source = tmp_path / "owner/source.ll"
    work = tmp_path / "check"
    work.mkdir()
    output = selection.transform(source, work)
    state.update(armed=True, selection=selection, output=output)
    with pytest.raises(ValueError, match="changed|identity"):
        selection.verify_emission(source, output, work)
    assert state["armed"] is False
