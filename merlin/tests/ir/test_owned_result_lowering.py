"""Returned-descriptor lowering is independent of host linking and invocation."""

import subprocess

import pytest
from test_owned_result_execution import COMPACT, RESIZE

from merlin.common.paths import runtime_dir
from merlin.llvmlower import toolchain
from merlin.llvmlower.owned_result_lowering import lower_owned_results
from merlin.llvmlower.target_data_layout import of

pytestmark = pytest.mark.skipif(not toolchain.available(), reason="native MLIR unavailable")


@pytest.fixture(scope="module")
def lowered(tmp_path_factory):
    root = tmp_path_factory.mktemp("owned-result-lowering")
    return lower_owned_results(
        COMPACT, workdir=root / "lowered", data_layout=of(toolchain.clang(), ("-O2", "-fPIC")),
        index_bits=64, timeout=30,
    )


def test_source_types_and_complete_owned_pipeline_survive_without_a_host_library(lowered):
    lowered.verify()
    assert len(lowered.inputs) == 1 and len(lowered.outputs) == 3
    assert all(slot.shape == (-1,) for slot in (*lowered.inputs, *lowered.outputs))
    assert not (lowered.source.parent / "model.so").exists()
    assert not (lowered.source.parent / "model.o").exists()
    assert "merlin_owned_result_invoke_checked" in lowered.bridge.read_text()
    assert "_mlir_memref_to_llvm_alloc" in lowered.llvm.read_text()


@pytest.mark.parametrize("member", ["source", "prepared", "llvm", "bridge"])
def test_changed_lowering_members_refuse_before_any_linking(lowered, member):
    path = getattr(lowered, member)
    original = path.read_bytes()
    try:
        path.write_bytes(original + b"\n")
        with pytest.raises(ValueError, match="changed"):
            lowered.verify()
    finally:
        path.write_bytes(original)
    lowered.verify()


@pytest.mark.parametrize("layout,bits", [("", 64), ("e-p:32:32", 64), ("e-p:64:64:64:32", 64),
                                       ("e-p:64:64", 32), ("e-p:64:64", True)])
def test_incompatible_or_missing_layout_contracts_refuse_before_products(tmp_path, layout, bits):
    output = tmp_path / "invalid"
    with pytest.raises(ValueError):
        lower_owned_results(COMPACT, workdir=output, data_layout=layout, index_bits=bits)
    assert not output.exists()


def test_separate_lowering_proves_zero_trip_returned_generations(tmp_path):
    lowered = lower_owned_results(
        RESIZE, workdir=tmp_path / "resizing", data_layout=of(toolchain.clang(), ("-O2", "-fPIC")),
        index_bits=64, timeout=30,
    )
    lowered.verify()
    assert lowered.outputs[0].shape == (-1,)
    assert "_mlir_memref_to_llvm_alloc" in lowered.llvm.read_text()


def test_actual_cross_compiler_consumes_observed_returned_descriptor_abi(tmp_path):
    from merlin.runtime.backends import spike
    from merlin.runtime.backends.host_compiler_policy import selected_model_compiler_plan

    try:
        gcc = spike.gcc_path()
    except (FileNotFoundError, RuntimeError):
        pytest.skip("existing cross compiler unavailable")
    if not gcc.is_file():
        pytest.skip("existing cross compiler unavailable")
    plan = selected_model_compiler_plan(backend="scalar", cflags_override=None, vlen=None, features=frozenset())
    observation = plan["observation"]
    result = lower_owned_results(
        COMPACT, workdir=tmp_path / "cross", data_layout=observation["data_layout"],
        index_bits=observation["index_bits"], timeout=30,
    )
    objects = []
    model = tmp_path / "model.o"
    done = subprocess.run(
        [observation["compiler_resolved"], *observation["cross_flags"], "-c", str(result.llvm), "-o", str(model)],
        capture_output=True, text=True, timeout=30,
    )
    assert done.returncode == 0, done.stderr
    objects.append(model)
    bridge = tmp_path / "bridge.o"
    done = subprocess.run(
        [str(gcc), *plan["gcc_cflags"], "-std=c11", "-I", str(runtime_dir() / "abi"),
         "-c", str(result.bridge), "-o", str(bridge)],
        capture_output=True, text=True, timeout=30,
    )
    assert done.returncode == 0, done.stderr
    objects.append(bridge)
    assert all(path.read_bytes().startswith(b"\x7fELF") for path in objects)
    result.verify()
    # These are real cross objects, not a linked or executed target image.
    assert not (result.source.parent / "model.so").exists()
