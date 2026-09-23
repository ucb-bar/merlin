"""What this design's load does to an operand is read from its parameter header, or refused."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from merlin.common.paths import merlin_dir
from merlin.runtime.backends import base as _backends

# The backend package registers without loading its header readers; ask for the module by the name
# the registry gave the package, whichever way it was discovered.
operand_sum_contract = importlib.import_module(
    f"{_backends.get_backend('gemmini').__name__}.gemmini_readout_semantics"
).operand_sum_contract

_HEADER = (
    merlin_dir() / "experiments/capsule_bench/targets/gemmini/contracts/hwbringup_gemmini_v0/isa_include"
) / "gemmini_params.h"


def _mutated(tmp_path: Path, old: str, new: str) -> Path:
    text = _HEADER.read_text(encoding="utf-8")
    assert text.count(old) >= 1
    path = tmp_path / "gemmini_params.h"
    path.write_text(text.replace(old, new), encoding="utf-8")
    return path


def test_the_shipped_header_states_a_scaled_rounding_saturating_load() -> None:
    contract = operand_sum_contract(_HEADER)
    assert contract["schema"] == "operand_sum_contract_v1"
    assert (contract["operands"], contract["operand_dtype"], contract["scale_dtype"]) == (2, "i8", "f32")
    assert contract["operand_rounding"] == "half_even" and contract["operand_saturates"] is True
    assert len(contract["provenance"]["params_header_sha256"]) == 64


def test_a_design_generated_without_a_scaled_load_licenses_nothing(tmp_path: Path) -> None:
    assert operand_sum_contract(_mutated(tmp_path, "#define HAS_MVIN_SCALE\n", "\n")) is None


def test_a_load_whose_arithmetic_is_not_the_verified_one_is_refused_not_assumed(tmp_path: Path) -> None:
    # Truncation instead of round-half-even: the bound the facet computes would be wrong for it.
    with pytest.raises(ValueError, match="load-scale implementation"):
        operand_sum_contract(
            _mutated(
                tmp_path,
                "#define MVIN_SCALE(x, scale) \\\n    ({float y = ROUND_NEAR_EVEN(",
                "#define MVIN_SCALE(x, scale) \\\n    ({float y = (",
            )
        )
