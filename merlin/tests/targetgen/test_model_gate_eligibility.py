"""The whole-model gate must count what the hardware CAN do, not everything graded.

Measured: an int8 systolic target graded 12 bf16 capsules its contract declares no capability for
("input dtype 'bf16' not in contraction formats ['int8']"). Those can never pass, so the best reachable
op-pass fraction was 23/35 = 0.66 against a 0.8 gate -- the whole-model capsules were mathematically
unreachable and nothing reported it.
"""

from __future__ import annotations

import pathlib

import pytest
import yaml

from merlin.common.paths import repo_root
from merlin.targetgen.capsule_runner import _gate_counts


def _capsule(name: str) -> dict:
    base = pathlib.Path(repo_root()) / "merlin/contract/capsules"
    for d in base.glob(f"*/{name}"):
        if d.is_dir():
            return yaml.safe_load((d / "capsule.yaml").read_text())
    pytest.skip(f"capsule {name} not on disk")


@pytest.mark.parametrize("name", ["GF4_add_bf16_pt", "GC1_depthwise_bf16_pt"])
def test_a_dtype_the_target_cannot_do_is_out_of_the_gate(name):
    cap = _capsule(name)
    assert _gate_counts({"capsule": name}, [cap], "gemmini") is False


@pytest.mark.parametrize("name", ["A2_single_tile_matmul", "B0_quantized_linear_i8"])
def test_a_capsule_the_target_can_do_stays_in_the_gate(name):
    cap = _capsule(name)
    assert _gate_counts({"capsule": name}, [cap], "gemmini") is True


def test_an_unknown_capsule_counts():
    """Fails OPEN: a result we cannot match to a capsule must not silently shrink the denominator."""
    assert _gate_counts({"capsule": "nope"}, [], "gemmini") is True


def test_an_unresolvable_target_counts_everything():
    cap = _capsule("A2_single_tile_matmul")
    assert _gate_counts({"capsule": "A2_single_tile_matmul"}, [cap], "no_such_target_xyz") is True


def test_the_measured_fraction_flips_the_gate():
    """22 pass + 13 fail, 12 of the failures ineligible: 22/35 = 0.63 fails a 0.8 gate; excluding the
    ineligible gives 22/23 = 0.96, which passes. That is the difference between the whole-model capsules
    being reachable and being impossible."""
    assert 22 / 35 < 0.8
    assert 22 / 23 >= 0.8
