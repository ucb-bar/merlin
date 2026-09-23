"""A row from another engine may enter the bench only when a shared control licenses it.

The temptation this guards against is concrete: a measurement taken in-model on an FPGA and one taken
in isolation on a cycle-accurate model both print a number called "cycles", and putting them in one
column makes a comparison that was never measured look like one that was. The licence is a control
measured on BOTH sides; this module's job is to compute that licence rather than assert it, and to
refuse the row -- loudly, with the reason -- when it does not hold.
"""

from __future__ import annotations

import json
import sys

import pytest

from merlin.common.paths import repo_root

SCRIPTS = repo_root() / "merlin" / "experiments" / "gemmini_perf_bench" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

cross_engine_rows = pytest.importorskip("cross_engine_rows")


LAYER = {
    "op": "conv2d",
    "batch": 1,
    "in_dim": 56,
    "in_channels": 64,
    "out_channels": 64,
    "kernel": 3,
    "stride": 1,
    "padding": 1,
    "relu": True,
}

CAPSULES = {
    "entries": [
        {
            "groups": [3, 8, 12],
            "entry": {
                "op": "conv2d",
                "Himg": 56,
                "Wimg": 56,
                "ci": 64,
                "N": 64,
                "kh": 3,
                "kw": 3,
                "stride": [1, 1],
                "padding": [1, 1, 1, 1],
                "epilogue": ["bias_add", "acc_scale", "relu"],
            },
        },
        {  # the SAME groups also appear as the raw contraction; it is a different function
            "groups": [3, 8, 12],
            "entry": {
                "op": "conv2d",
                "Himg": 56,
                "Wimg": 56,
                "ci": 64,
                "N": 64,
                "kh": 3,
                "kw": 3,
                "stride": [1, 1],
                "padding": [1, 1, 1, 1],
                "epilogue": [],
            },
        },
        {
            "groups": [34],
            "entry": {"op": "matmul", "M": 196, "K": 256, "N": 1024, "epilogue": ["acc_scale"]},
        },
    ]
}


def _console(cycles: dict, checksums: dict | None = None, *, complete: bool = True) -> str:
    checksums = checksums or {g: str(1000 + g) for g in cycles}
    lines = ["MERLIN_PROFILE measured begin"]
    lines += [f"GM_GROUP {g} conv2d {c} sum={checksums[g]}" for g, c in sorted(cycles.items())]
    lines.append("GROUP_MODEL_TOTAL cycles: 25419657")
    if complete:
        lines.append("MERLIN_PROFILE measured end rc=0")
    return "\n".join(lines) + "\n"


def _run(tmp_path, name, cycles, checksums=None, *, complete=True, status="observed", oracle="71 of 71 equal"):
    """A measurement directory shaped the way `new_measurement` lays one out."""
    d = tmp_path / "firesim_some_board" / "resnet50" / name
    (d / "evidence").mkdir(parents=True)
    (d / "evidence" / "uartlog").write_text(_console(cycles, checksums, complete=complete), encoding="utf-8")
    (d / "checkpoint_row.json").write_text(
        json.dumps({"job_id": 727, "label": name, "status": status, "observed_cycles": 1, "oracle": oracle}),
        encoding="utf-8",
    )
    return d


OURS = {3: 473343, 8: 473320, 12: 473297}
CONTROL = {3: 523784, 8: 524048, 12: 523626}


def _pair(tmp_path, ours=None, control=None, **kw):
    a = cross_engine_rows.load_run(_run(tmp_path, "ours", ours or OURS, **kw))
    b = cross_engine_rows.load_run(_run(tmp_path, "control", control or CONTROL, oracle=None))
    return a, b


# --- reading the foreign run ---------------------------------------------------------------------


def test_the_console_is_parsed_structurally_and_completion_is_carried(tmp_path):
    run = cross_engine_rows.load_run(_run(tmp_path, "ours", OURS))
    assert run.groups == OURS
    assert run.completed is True
    assert run.substrate == "firesim_some_board"
    assert run.job_id == 727
    assert run.sealed is False  # status "observed"


def test_a_log_that_never_reached_the_end_is_marked_incomplete(tmp_path):
    run = cross_engine_rows.load_run(_run(tmp_path, "ours", OURS, complete=False))
    assert run.completed is False


def test_a_directory_without_both_evidence_streams_is_not_a_measurement(tmp_path):
    (tmp_path / "bare").mkdir()
    with pytest.raises(FileNotFoundError):
        cross_engine_rows.load_run(tmp_path / "bare")


def test_the_substrate_is_read_from_where_the_run_lives(tmp_path):
    """A run cannot be filed under a device it did not run on: the device comes from the layout."""
    run = cross_engine_rows.load_run(_run(tmp_path, "ours", OURS))
    assert run.substrate == "firesim_some_board"


# --- which groups are the layer -------------------------------------------------------------------


def test_only_the_groups_with_the_fused_epilogue_are_the_layer():
    """Each group appears twice in the inventory -- once fused, once raw. They are different
    functions, and matching the raw one would compare against work that skips the epilogue."""
    mapped = cross_engine_rows.groups_by_shape(CAPSULES)
    assert set(mapped) == {3, 8, 12, 34}
    assert mapped[3]["epilogue"] == ["bias_add", "acc_scale", "relu"]


def test_a_group_of_a_different_shape_does_not_match():
    mapped = cross_engine_rows.groups_by_shape(CAPSULES)
    assert cross_engine_rows.matches_layer(mapped[3], LAYER) is True
    assert cross_engine_rows.matches_layer(mapped[34], LAYER) is False
    assert cross_engine_rows.matches_layer(mapped[3], {**LAYER, "padding": 0}) is False
    assert cross_engine_rows.matches_layer(mapped[3], {**LAYER, "in_dim": 28}) is False
    assert cross_engine_rows.matches_layer(mapped[3], {**LAYER, "stride": 2}) is False


# --- the licence ------------------------------------------------------------------------------------


def test_a_licensed_row_carries_its_device_its_status_and_the_control_it_came_through(tmp_path):
    ours, control = _pair(tmp_path)
    row = cross_engine_rows.calibrate(
        ours, control, group_capsules=CAPSULES, layer=LAYER, own_control_cycles=522860, tolerance=0.02
    )
    assert row["arm"] == "compute_group"
    assert row["cycles"] == 473297 and row["cycles_max"] == 473343
    assert row["engine"] == "firesim_some_board" and row["job_id"] == 727
    assert row["sealed"] is False and row["status"] == "observed"
    cal = row["calibration"]
    assert cal["foreign_control_cycles"] == 523626
    assert cal["own_control_cycles"] == 522860
    assert cal["agreement_ratio"] == pytest.approx(523626 / 522860)
    assert cal["control_has_own_oracle"] is False  # the control run records no oracle of its own


def test_a_differing_requant_constant_is_reported_not_absorbed(tmp_path):
    """The two sides share the contract's shape but were captured separately, so the constant can
    differ. It does not move cycles -- same instruction either way -- but they are then not computing
    the identical function, and the row has to say so."""
    capsules = json.loads(json.dumps(CAPSULES))
    capsules["entries"][0]["entry"]["acc_scale"] = 0.005265
    ours, control = _pair(tmp_path)
    row = cross_engine_rows.calibrate(
        ours,
        control,
        group_capsules=capsules,
        layer={**LAYER, "scale": 0.003978},
        own_control_cycles=522860,
        tolerance=0.02,
    )
    assert row["acc_scale"] == {"foreign": 0.005265, "own": 0.003978, "identical": False}
    assert row["epilogue"] == ["bias_add", "acc_scale", "relu"]


def test_an_identical_requant_constant_reads_as_identical(tmp_path):
    capsules = json.loads(json.dumps(CAPSULES))
    capsules["entries"][0]["entry"]["acc_scale"] = 0.25
    ours, control = _pair(tmp_path)
    row = cross_engine_rows.calibrate(
        ours,
        control,
        group_capsules=capsules,
        layer={**LAYER, "scale": 0.25},
        own_control_cycles=522860,
        tolerance=0.02,
    )
    assert row["acc_scale"]["identical"] is True


def test_a_control_that_disagrees_beyond_tolerance_refuses_the_row(tmp_path):
    """This is the whole point. Without it the block would print a comparison between two engines
    nothing had shown to be on the same scale."""
    ours, control = _pair(tmp_path, control={g: c * 2 for g, c in CONTROL.items()})
    with pytest.raises(cross_engine_rows.NotLicensed) as caught:
        cross_engine_rows.calibrate(
            ours, control, group_capsules=CAPSULES, layer=LAYER, own_control_cycles=522860, tolerance=0.02
        )
    assert "not calibrated" in str(caught.value)
    assert "2.00%" in str(caught.value)


def test_the_tolerance_is_the_callers_and_it_actually_binds(tmp_path):
    ours, control = _pair(tmp_path, control={g: int(c * 1.05) for g, c in CONTROL.items()})
    kw = dict(group_capsules=CAPSULES, layer=LAYER, own_control_cycles=522860)
    with pytest.raises(cross_engine_rows.NotLicensed):
        cross_engine_rows.calibrate(ours, control, tolerance=0.02, **kw)
    assert cross_engine_rows.calibrate(ours, control, tolerance=0.10, **kw)["cycles"] == 473297


def test_a_prefix_log_is_refused_on_either_side(tmp_path):
    """A partial log reads better than the truth: the fast groups finish first."""
    kw = dict(group_capsules=CAPSULES, layer=LAYER, own_control_cycles=522860, tolerance=0.02)
    ours, control = _pair(tmp_path / "a", complete=False)
    with pytest.raises(cross_engine_rows.NotLicensed) as caught:
        cross_engine_rows.calibrate(ours, control, **kw)
    assert "prefix" in str(caught.value)

    ours2 = cross_engine_rows.load_run(_run(tmp_path / "b", "ours", OURS))
    control2 = cross_engine_rows.load_run(_run(tmp_path / "b", "control", CONTROL, complete=False, oracle=None))
    with pytest.raises(cross_engine_rows.NotLicensed) as caught2:
        cross_engine_rows.calibrate(ours2, control2, **kw)
    assert "prefix" in str(caught2.value)


def test_two_runs_that_computed_different_things_are_refused(tmp_path):
    """A cycle ratio between a kernel and a faster-but-different kernel is meaningless. The per-group
    output checksums in the two logs are what rules that out, and they are checked, not assumed."""
    ours = cross_engine_rows.load_run(_run(tmp_path, "ours", OURS, {3: "111", 8: "222", 12: "333"}))
    control = cross_engine_rows.load_run(
        _run(tmp_path, "control", CONTROL, {3: "111", 8: "999", 12: "333"}, oracle=None)
    )
    with pytest.raises(cross_engine_rows.NotLicensed) as caught:
        cross_engine_rows.calibrate(
            ours, control, group_capsules=CAPSULES, layer=LAYER, own_control_cycles=522860, tolerance=0.02
        )
    assert "different outputs" in str(caught.value)
    assert "[8]" in str(caught.value)


@pytest.mark.parametrize("missing_side", ["ours", "control", "both"])
def test_absent_checksums_never_count_as_agreement(tmp_path, missing_side):
    ours, control = _pair(tmp_path)
    if missing_side in {"ours", "both"}:
        ours.checksums.pop(8)
    if missing_side in {"control", "both"}:
        control.checksums.pop(8)
    with pytest.raises(cross_engine_rows.NotLicensed, match="printed no output checksum"):
        cross_engine_rows.calibrate(
            ours, control, group_capsules=CAPSULES, layer=LAYER, own_control_cycles=522860, tolerance=0.02
        )


@pytest.mark.parametrize("line", ["GM_GROUP", "GM_GROUP nope conv2d 42", "GM_GROUP 8 conv2d broken"])
def test_announced_but_unreadable_group_is_not_silently_dropped(line):
    with pytest.raises(cross_engine_rows.ConsoleUnreadable, match="missing evidence"):
        cross_engine_rows.parse_console(line + "\nMERLIN_PROFILE measured end\n")


def test_unrelated_console_noise_is_still_ignored():
    assert cross_engine_rows.parse_console("boot message\nMERLIN_PROFILE measured end\n") == ({}, {}, True)


def test_a_layer_no_group_computes_is_refused(tmp_path):
    ours, control = _pair(tmp_path)
    with pytest.raises(cross_engine_rows.NotLicensed) as caught:
        cross_engine_rows.calibrate(
            ours,
            control,
            group_capsules=CAPSULES,
            layer={**LAYER, "in_dim": 7},
            own_control_cycles=522860,
            tolerance=0.02,
        )
    assert "no compute group" in str(caught.value)


def test_a_group_missing_from_one_of_the_logs_is_refused(tmp_path):
    ours, control = _pair(tmp_path, ours={3: 473343, 8: 473320})
    with pytest.raises(cross_engine_rows.NotLicensed) as caught:
        cross_engine_rows.calibrate(
            ours, control, group_capsules=CAPSULES, layer=LAYER, own_control_cycles=522860, tolerance=0.02
        )
    assert "[12]" in str(caught.value)


def test_control_agreement_is_symmetric_and_at_least_one():
    assert cross_engine_rows.control_agreement(100, 100) == 1.0
    assert cross_engine_rows.control_agreement(110, 100) == cross_engine_rows.control_agreement(100, 110)
    with pytest.raises(ValueError):
        cross_engine_rows.control_agreement(0, 100)
