import numpy as np
import pytest

from merlin.model_check import full_build as FB


def console(payload, *, build="fixture", rank="0"):
    h = 0xCBF29CE484222325
    for b in payload:
        h = ((h ^ b) * 0x100000001B3) & ((1 << 64) - 1)
    return (
        f"OUT_BIN_BEGIN v1 out0 1 {len(payload)} 1 u {len(payload)}\n".encode()
        + payload
        + f"OUT_BIN_END v1 {h:016x}\nMETRIC build_hash {build}\nMETRIC memref_rank_mismatch {rank}\nDONE\n".encode()
    )


def stage(tmp_path, n=1):
    p = tmp_path / "capture"
    p.mkdir()
    (p / "model.mlir").write_text(
        f"module {{ func.func @forward() -> tensor<{n}xi1> {{ "
        f"%x = arith.constant dense<true> : tensor<{n}xi1> return %x : tensor<{n}xi1> }} }}"
    )
    return p


@pytest.mark.parametrize("raw", [2, 255])
def test_wire_boolean_noncanonical_bytes_refuse_before_numerical_comparison(tmp_path, raw):
    with pytest.raises(FB.FullBuildError, match="noncanonical"):
        FB.read(console(bytes([raw])), stage(tmp_path), "fixture")


def test_canonical_boolean_exact_values_and_wire_bytes(tmp_path):
    values, diag = FB.read(console(bytes([0, 1, 1, 0])), stage(tmp_path, 4), "fixture")
    assert values[0].dtype == np.bool_ and values[0].tobytes() == bytes([0, 1, 1, 0])
    assert diag["metrics"]["memref_rank_mismatch"] == "0"


@pytest.mark.parametrize("change", [{"build": "wrong"}, {"rank": "1"}])
def test_clean_boolean_cannot_hide_build_or_runtime_refusal(tmp_path, change):
    with pytest.raises(ValueError):
        FB.read(console(bytes([1]), **change), stage(tmp_path), "fixture")
