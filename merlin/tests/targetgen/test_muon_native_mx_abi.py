"""Native mxfp8 GEMM selection, operand derivation, emission, and harness binding."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from merlin.common.paths import repo_root
from merlin.runtime.backends.base import get_backend
from merlin.targetgen.contract.interface_emit import parse_interface_mlir
from merlin.targetgen.mx_oracle import mx_matmul

MUON = get_backend("muon")
ABI = MUON.muon_mx_abi
SELECTION = MUON.muon_kernel_selection
CODEGEN = MUON.muon_codegen_mlir
HARNESS = MUON.muon_harness
CAPSULE = repo_root() / "merlin/contract/capsules/radiance/isa/R5_mx_tile_mxfp8/capsule.interface.mlir"
CONTRACT = repo_root() / "merlin/experiments/capsule_bench/targets/radiance/contracts/kernel_library_pr1_v1.yaml"


def _cb() -> dict:
    return parse_interface_mlir(CAPSULE.read_text(encoding="utf-8"))


def _actual() -> dict[str, list[float]]:
    # Non-symmetric, non-constant values spanning signs and binades; deterministic and golden-independent.
    return {
        "A0": [((row * 7 + col * 3) % 23 - 11) / 4.0 for row in range(16) for col in range(32)],
        "W": [((row * 5 + col * 11) % 29 - 14) / 8.0 for row in range(32) for col in range(16)],
    }


def _emit(cb: dict, actual: dict[str, list[float]]) -> str:
    return CODEGEN.emit_kernel_mlir(
        cb,
        selection_contract=SELECTION.load_selection_contract(CONTRACT),
        hardware_contract={"features": ["mx_mesh", "block_e8m0"]},
        actual_tensors=actual,
    )


def _run_derived_abi(cb: dict) -> np.ndarray:
    bundle = ABI.emitter_bundle(cb["params"]["mx_operand_abi"])
    result = mx_matmul(
        bundle["A_bytes"],
        bundle["B_bytes"],
        bundle["SA"],
        bundle["SB"],
        bundle["M"],
        bundle["N"],
        bundle["K"],
        fmt=bundle["fmt"],
    )
    if result is None:
        pytest.skip("validated MX numeric model is unavailable")
    return np.asarray(result)


def test_real_r5_semantics_select_and_emit_a_digest_bound_native_program():
    cb = _cb()
    source = _emit(cb, _actual())
    report = cb["params"]["kernel_family_selection"]
    assert report["request"] == {"op": "matmul", "dtype": "mxfp8", "shape": {"K": 32, "M": 16, "N": 16}}
    assert report["selected_family"] == ABI.NATIVE_FAMILY
    operand_abi = cb["params"]["mx_operand_abi"]
    assert operand_abi["schema"] == ABI.ABI_SCHEMA
    assert operand_abi["operands"]["lhs"]["source"] == "compiler_amax_quantized_actual_tensor"
    assert len(operand_abi["operands"]["lhs"]["codes"]) == 16 * 32
    assert len(operand_abi["operands"]["lhs"]["scale_codes"]) == 1
    assert "MX compiler-native kernel" in source and "mxgemm<CFG>" in source
    assert "MX reference kernel" not in source and "muon-reference MX placeholder" not in source
    ABI.bind_native_program(cb, source)
    assert HARNESS.program_from_cb(cb, source, model=None) is None


def test_perturbed_actual_input_changes_abi_program_and_numeric_result():
    actual = _actual()
    base_cb = _cb()
    base_source = _emit(base_cb, actual)
    base_output = _run_derived_abi(base_cb)

    changed = deepcopy(actual)
    changed["A0"][0] += 4.0
    changed_cb = _cb()
    changed_source = _emit(changed_cb, changed)
    changed_output = _run_derived_abi(changed_cb)

    assert base_cb["params"]["mx_operand_abi"]["abi_sha256"] != changed_cb["params"]["mx_operand_abi"]["abi_sha256"]
    assert base_source != changed_source
    assert not np.array_equal(base_output, changed_output), (
        "the strongest affordable MX numeric tier must observe an actual-input perturbation"
    )


def test_perturbed_expected_output_control_is_rejected_by_numeric_tier():
    cb = _cb()
    _emit(cb, _actual())
    observed = _run_derived_abi(cb)
    expected = observed.copy()
    expected[0, 0] += np.float32(1.0)
    assert not np.allclose(observed, expected, atol=0.03125, rtol=0.015625), (
        "a changed expected output must not pass the R5 tolerance policy"
    )


@pytest.mark.slow
def test_native_program_and_both_controls_execute_on_cyclotron(tmp_path):
    """Strongest affordable tier: run the selected program, not a harness reference substitute."""
    if not MUON.available("cyclotron"):
        pytest.skip("Cyclotron MX co-model is unavailable")
    actuals = [_actual(), deepcopy(_actual())]
    actuals[1]["A0"][0] += 4.0
    observed = []
    for index, actual in enumerate(actuals):
        cb = _cb()
        source = _emit(cb, actual)
        elf, toolchain = MUON.compile_for_oracle(source, tmp_path / str(index), target="radiance")
        console, cycles, _summary = MUON.run_elf(elf, simulator="cyclotron", timeout=240)
        outputs, _raw = MUON.parse_output(console, cycles)
        got = np.asarray(outputs["Y0"], dtype=np.float32)
        expected = _run_derived_abi(cb)
        assert "reference-kernel" not in toolchain
        assert np.array_equal(got, expected)
        observed.append(got)
    assert not np.array_equal(observed[0], observed[1])
    bad_expected = observed[0].copy()
    bad_expected[0, 0] += np.float32(1.0)
    assert not np.allclose(observed[0], bad_expected, atol=0.03125, rtol=0.015625)


def test_explicit_model_quantization_metadata_is_an_independent_source():
    actual_cb = _cb()
    _emit(actual_cb, _actual())
    derived = actual_cb["params"]["mx_operand_abi"]["operands"]
    metadata = {}
    for role in ("lhs", "weight"):
        operand = derived[role]
        metadata[operand["tensor"]] = {
            "format": "mxfp8",
            "block": 32,
            "codes": operand["codes"],
            "scale_codes": [value for row in operand["scale_codes"] for value in row],
        }
    cb = _cb()
    source = CODEGEN.emit_kernel_mlir(
        cb,
        selection_contract=SELECTION.load_selection_contract(CONTRACT),
        hardware_contract={"features": ["mx_mesh", "block_e8m0"]},
        model_quantization=metadata,
    )
    assert all(
        spec["source"] == "explicit_model_quantization_metadata"
        for spec in cb["params"]["mx_operand_abi"]["operands"].values()
    )
    ABI.bind_native_program(cb, source)


@pytest.mark.parametrize(
    "legacy",
    [
        {"mx_operands": {"A_bytes": [0]}},
        {"canonical_inputs": {"A0": {"values": [0.0]}}},
        {"reference_kernel_path": "answer.cpp"},
    ],
)
def test_native_route_rejects_legacy_golden_and_reference_kernel_fields(legacy):
    cb = _cb()
    cb.update(legacy)
    with pytest.raises(CODEGEN.MuonMlirCodegenError, match="rejects legacy"):
        _emit(cb, _actual())


def test_harness_refuses_a_program_whose_abi_binding_was_perturbed():
    cb = _cb()
    source = _emit(cb, _actual())
    cb["params"]["mx_operand_abi"]["operands"]["lhs"]["codes"][0] ^= 1
    with pytest.raises(ABI.NativeMxAbiError, match="digest mismatch"):
        HARNESS.program_from_cb(cb, source, model=None)


def test_selection_fails_closed_when_scale_relationships_are_missing():
    cb = _cb()
    del cb["tensors"]["A0_scale"]["scale_of"]
    with pytest.raises(SELECTION.KernelSelectionContractError, match="explicit scale tensor"):
        SELECTION.select_command_buffer_family(
            cb, {"features": ["mx_mesh", "block_e8m0"]}, SELECTION.load_selection_contract(CONTRACT)
        )


def test_selection_fails_closed_when_commit_does_not_consume_matmul():
    cb = _cb()
    commit = next(command for command in cb["commands"] if command["opcode"] == "COMMIT")
    commit["operands"]["src"] = "unrelated_accumulator"
    with pytest.raises(SELECTION.KernelSelectionContractError, match="commit must consume"):
        SELECTION.select_command_buffer_family(
            cb, {"features": ["mx_mesh", "block_e8m0"]}, SELECTION.load_selection_contract(CONTRACT)
        )
