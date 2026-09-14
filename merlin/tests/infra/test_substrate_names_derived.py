"""Board/simulator labels in generic code are READ from what owns them, never restated.

``check_no_target_name.py`` bans substrate-name literals in generic modules. The generic modules below
used to spell the board label themselves; each now reads it from the board adapter (which records it)
or checks it against the contract that consumes it. These tests hold the two sides together, so a
relabelled adapter or a changed authority breaks here instead of silently yielding no wall time.
"""
from __future__ import annotations

import pytest

from merlin.compare import spec
from merlin.kernels.measurement import authority_for
from merlin.mining import beam_cli, k1, op_sweep, runner


def test_certify_legs_are_the_measurement_authority_the_record_names():
    """certify_rvv stamps its record ``target: rvv`` and the beam reads that target's authority to pick
    cycles and wall time, so the runner's default legs must BE that declared pair."""
    auth = authority_for("rvv")
    assert auth.declared, auth.gaps()
    assert runner.DEFAULT_TARGETS == (auth.cycles_from, auth.wall_from)


def test_board_leg_is_labelled_with_the_adapter_substrate():
    assert k1.SUBSTRATE in runner.DEFAULT_TARGETS
    assert beam_cli.DEFAULT_TARGETS == (k1.SUBSTRATE,)


def test_op_sweep_hands_the_beam_its_default_board_leg(tmp_path):
    seen = {}

    def beam(**kw):
        seen.update(kw)
        return {"best": {}, "parent_run_dir": None}

    cell = op_sweep.OpCell(op="matmul", dtype="f32", shape_regime="s", workload_dir=tmp_path,
                           expert_objdump=tmp_path / "x.objdump")
    op_sweep.run_cell(cell, beam_fn=beam)
    assert seen["targets"] == beam_cli.DEFAULT_TARGETS


def test_compare_implements_exactly_the_adapter_substrate():
    assert spec.implemented_targets() == (k1.SUBSTRATE,)
    assert spec.default_target() == k1.SUBSTRATE
    parsed = spec.Spec.parse({"configs": ["baseline"], "workloads": ["openvla"]})
    assert parsed.target == k1.SUBSTRATE
    assert spec.Spec(configs=(), workloads=()).target == k1.SUBSTRATE


def test_compare_refuses_a_default_when_the_implemented_set_is_ambiguous(monkeypatch):
    monkeypatch.setattr(spec, "implemented_targets", lambda: ("a", "b"))
    with pytest.raises(ValueError, match="name one"):
        spec.Spec.parse({"configs": ["baseline"], "workloads": ["openvla"]})
