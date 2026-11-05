"""The residual-state and multi-block axes: what the stores held before, and outputs larger than them."""

from __future__ import annotations

import pytest

from merlin.targetgen import capsule_golden as CG
from merlin.targetgen import corpus_spec as CS
from merlin.targetgen import corpus_synth as CSY
from merlin.targetgen.contract import interface_emit as IE


def _binding() -> CS.CorpusBinding:
    return CS.CorpusBinding(
        target="synthetic",
        tile_dim=16,
        operand_dtype="int8",
        accum_dtype="i32",
        integer=True,
        tiers=["L0"],
        compare="exact_int",
        classes_for=lambda **_: [],
    )


def _conv(**over) -> dict:
    entry = {
        "name": "C",
        "kind": "layer",
        "op": "conv2d",
        "ci": 4,
        "N": 16,
        "Himg": 6,
        "Wimg": 6,
        "kh": 3,
        "kw": 3,
        "padding": [1, 1, 1, 1],
        "stimulus_range": [-4, 3],
        "source_role": "derived_sweep",
        "source_reference": "residual-state test",
    }
    entry.update(over)
    return entry


@pytest.mark.parametrize(
    "entry",
    [
        _conv(),
        {
            "name": "M",
            "kind": "layer",
            "op": "matmul",
            "M": 16,
            "K": 31,
            "N": 15,
            "stimulus_range": [-4, 3],
            "source_role": "derived_sweep",
            "source_reference": "residual-state test",
        },
    ],
)
def test_a_prelude_runs_first_is_graded_and_leaves_the_operation_unchanged(entry):
    plain_capsule, _ = CS.build(entry, _binding())
    capsule, interface = CS.build({**entry, "prelude": {"M": 48, "K": 48, "N": 16}}, _binding())
    commands = IE.parse_interface_mlir(interface)["commands"]
    assert [c["opcode"] for c in commands][:3] == ["RES_PACK", "MATMUL_RESIDENT", "COMMIT"]
    outputs = CG.golden(capsule)
    assert outputs["Y0"] == CG.golden(plain_capsule)["Y0"]
    assert len(outputs["P_Y"]) == 48 and len(outputs["P_Y"][0]) == 16
    assert any(v != 0 for row in outputs["P_Y"] for v in row), "the prelude leaves non-zero state"
    # Two contractions in one module may not share an SSA accumulator.
    defined = [line.split("=")[0].strip() for line in interface.splitlines() if "= merlin_iface.matmul" in line]
    assert len(defined) == len(set(defined))


def test_a_prelude_is_declared_in_whole_integer_extents_with_distinct_names():
    with pytest.raises(ValueError, match="prelude"):
        CS.build(_conv(prelude={"M": 48, "K": 48}), _binding())
    with pytest.raises(ValueError, match="collide"):
        CS.build(_conv(ifm="P_A", prelude={"M": 48, "K": 48, "N": 16}), _binding())


def test_a_tile_relative_prelude_resolves_against_the_tile_edge():
    from merlin_experiments.phase0.sweeps import _resolve_flat_extents

    entry = _resolve_flat_extents(_conv(prelude={"M": "3*tile", "K": "2*tile", "N": "tile"}), _binding())
    assert entry["prelude"] == {"M": 48, "K": 32, "N": 16}


def _spec() -> dict:
    return {
        "target": "synthetic",
        "cells": [
            {"cell": f"contraction/i8/{a}", "family": "contraction", "dtype": "i8", "alignment": a}
            for a in ("aligned", "partial")
        ],
        "boundaries": {
            "tile_edge": 16,
            "extent_probes": [{"boundary": "tile_edge", "edge": 16, "points": [1, 15, 16, 17, 32]}],
        },
        "memory_mapping": {"regime_dtype": "i8", "required": {}},
        "accumulator_output_boundary": {
            "status": "resolved",
            "capacity_rows": 1024,
            "tile_edge": 16,
            "N_tiles": 65,
            "output_rows_if_resident": 1040,
        },
        "conv_geometry": {
            "required": [
                {"signature": "k3x3/s1x1/d1x1/pad1x1", "kernel": [3, 3], "stride": [1, 1], "dilation": [1, 1],
                 "pad_before": [1, 1], "pad_after": [1, 1], "pad_known": True},
                {"signature": "k4x4/s4x4/d1x1/pad0x0", "kernel": [4, 4], "stride": [4, 4], "dilation": [1, 1],
                 "pad_before": [0, 0], "pad_after": [0, 0], "pad_known": True},
            ]
        },
    }  # fmt: skip


def _axis(entries, axis):
    return [e for e in entries if (e.get("generalization") or {}).get("generalization_axis") == axis]


def test_only_padded_windows_and_a_partial_contraction_get_a_residual_state_sibling():
    entries = CSY.synthesize(_spec())["capsules"]
    stateful = _axis(entries, "residual_state")
    names = sorted(e["name"] for e in stateful)
    assert any("pad1x1" in n for n in names) and not any("pad0x0" in n for n in names)
    assert all(e["prelude"] and e["stimulus_range"][0] < 0 for e in stateful)


def test_multi_block_members_straddle_the_accumulator_capacity_with_tails():
    from merlin_experiments.phase0.sweeps import resolve_extent

    members = {e["name"]: e for e in _axis(CSY.synthesize(_spec())["capsules"], "multi_block")}
    assert set(members) == {"SY_multi_block_within", "SY_multi_block_beyond"}
    capacity_tiles = 1024 // 16

    def tiles(entry):
        m, n = (resolve_extent(entry[axis], 16) for axis in ("M", "N"))
        return -(-m // 16) * -(-n // 16), m, n, resolve_extent(entry["K"], 16)

    inside, *_ = tiles(members["SY_multi_block_within"])
    beyond, m, n, k = tiles(members["SY_multi_block_beyond"])
    assert inside <= capacity_tiles < beyond
    assert m % 16 and n % 16 and k % 16 and k > 16
