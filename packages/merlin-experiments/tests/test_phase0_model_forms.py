"""Model forms are derived from iteration workloads' stated groups, never from a held-out model."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase0 import model_forms as MF
from merlin_experiments.phase0.claim_boundary import assert_no_claim_capsules, held_out_models, is_held_out

from merlin.targetgen import group_capsule_entries as G

_TE = SimpleNamespace(target="synthetic", sim_via="sim")
_CONTRACT = {
    "name": "synthetic",
    "compute_units": [{"name": "array", "kind": "systolic", "dtypes": ["int8"]}],
    "capabilities": {"mesh": {"rows": 16, "cols": 16}},
    "encoding": {
        "semantic_class": {"a": "MVIN", "b": "COMPUTE", "c": "MVOUT"},
        "corpus_issue_order": ["MVIN", "COMPUTE", "MVOUT"],
    },
}
_FACTS = {"facts": {"datapaths": [{"name": "input", "dtype": "i8"}, {"name": "accumulator", "dtype": "i32"}]}}


def _binding(tiers=("L0", "L1", "L2", "L3")):
    datapath = {"required_oracle_tiers": list(tiers), "compare": "exact_int", "requant_output_dtype": "i8"}
    return G.group_binding(_TE, datapath, contract=_CONTRACT, facts=_FACTS, taxonomy={})


def _row(name, entry, count=1, groups=(0,)):
    return {"name": name, "count": count, "groups": list(groups), "entry": entry, "program": {"transposed": False}}


def _stated(*rows):
    return {"entries": list(rows), "accelerator_groups": len(rows), "stated": len(rows), "unstated": {}}


def test_scale_classes_and_extreme_representatives():
    gain = {
        "op": "residual_add",
        "M": 64,
        "N": 32,
        "lhs_scale": 1.7,
        "rhs_scale": 0.4,
        "bound_lsb": 1,
        "epilogue": ["relu"],
        "operand_dtype": "int8",
    }
    unit = {**gain, "lhs_scale": 0.9, "rhs_scale": 0.3}
    assert MF.scale_class(gain) == MF.GAIN_ABOVE_ONE and MF.scale_class(unit) == MF.GAIN_BELOW_ONE
    assert MF.scale_class({"op": "matmul", "epilogue": []}) == MF.UNSCALED
    assert MF.form_key(gain) != MF.form_key(unit)
    steeper = {**gain, "lhs_scale": 2.5, "rhs_scale": 0.6}
    assert MF.form_key(gain) == MF.form_key(steeper)
    picks = MF.representatives([_row("b", gain), _row("a", steeper)])
    assert [(extreme, row["name"]) for extreme, row in picks] == [("max_multiplier", "a"), ("min_multiplier", "b")]


def test_only_positions_are_reduced():
    entry = {"op": "matmul", "M": 3136, "K": 576, "N": 256, "epilogue": []}
    reduced = MF.reduce_entry(entry, 16)
    assert (reduced["M"], reduced["K"], reduced["N"]) == (32, 576, 256)
    assert MF.reduce_extent(7, 16) == 7 and MF.reduce_extent(100, 16) == 36
    conv = {
        "op": "conv2d",
        "ci": 64,
        "N": 64,
        "Himg": 56,
        "Wimg": 56,
        "kh": 3,
        "kw": 3,
        "stride": [1, 1],
        "padding": [1, 1, 1, 1],
        "epilogue": [],
    }
    small = MF.reduce_entry(conv, 16)
    assert (small["ci"], small["N"], small["kh"]) == (64, 64, 3) and small["Himg"] < 56


def test_stimulus_spans_the_operand_where_the_multiplier_is_applied_to_operands():
    gain = {"op": "residual_add", "M": 16, "N": 16, "bound_lsb": 1, "epilogue": []}
    assert MF.stimulus_range(gain, operand_range=(-128, 127), output_range=(-128, 127)) == [-128, 127]
    readout = {"op": "matmul", "M": 16, "K": 2048, "N": 16, "epilogue": ["acc_scale"], "acc_scale": 0.002}
    lo, hi = MF.stimulus_range(readout, operand_range=(-128, 127), output_range=(-128, 127))
    assert lo == -hi and 0 < hi <= 127


def test_derived_entries_name_their_iteration_source_and_split_for_certification():
    binding = _binding()
    big = {"op": "matmul", "M": 4096, "K": 64, "N": 512, "epilogue": [], "operand_dtype": "int8"}
    gain = {
        "op": "residual_add",
        "M": 256,
        "N": 16,
        "lhs_scale": 1.5,
        "rhs_scale": 0.5,
        "bound_lsb": 1,
        "epilogue": ["relu"],
        "operand_dtype": "int8",
    }
    got = MF.derive_application(
        "residual_cnn", _stated(_row("G_big", big), _row("G_add", gain)), binding, capture_sha256="0" * 64, ceiling=2048
    )
    names = [e["name"] for e in got["entries"]]
    add = next(e for e in got["entries"] if e["op"] == "residual_add")
    assert add["name"] == "MF_residual_cnn_residual_add_relu_gain_gt_1_max"
    assert add["model_form"]["stimulus_rule"] == "operand_full_range" and add["stimulus_range"] == [-128, 127]
    capped = next(e for e in got["entries"] if e["op"] == "matmul" and e.get("extends"))
    assert capped["max_oracle_tier"] == "L2" and capped["extends"] in names
    sibling = next(e for e in got["entries"] if e["name"] == capped["extends"])
    assert MF.written_elements(sibling) <= 2048 and sibling["model_form"]["certification"] == "certified"
    for entry in got["entries"]:
        assert entry["model"] == "residual_cnn" and entry["source_role"] == G.SOURCE_ROLE
        assert entry["model_form"]["workload_role"] == "iteration" and entry["cat"] == "layers"


def test_a_held_out_model_is_refused_by_name_before_any_capture_is_read(tmp_path: Path):
    missing = tmp_path / "never_read.mlir"
    with pytest.raises(MF.ModelFormRefusal, match="held-out model"):
        MF.derive_model_forms(
            "synthetic", {"resnet18": missing}, _binding(), iteration_roster=["resnet18"], held_out=["resnet18"]
        )
    with pytest.raises(MF.ModelFormRefusal, match="iteration roster only"):
        MF.derive_model_forms(
            "synthetic", {"residual_cnn": missing}, _binding(), iteration_roster=["coverage_mlp"], held_out=[]
        )


def test_a_renamed_held_out_form_is_caught_by_the_claim_boundary():
    assert is_held_out("resnet18_int8_consistent", ["resnet18", "resnet50"]) == "resnet18"
    assert is_held_out("residual_cnn", ["resnet18", "resnet50"]) is None
    leaked = {"name": "MF_resnet18_matmul_raw_unscaled", "model_form": {"model": "resnet18"}}
    with pytest.raises(ValueError, match="held-out claim model 'resnet18'"):
        assert_no_claim_capsules([leaked], ["resnet18"])
    with pytest.raises(ValueError, match="held-out"):
        assert_no_claim_capsules([{"name": "MF_resnet18_x"}], ["resnet18"])


def test_evaluation_only_models_join_the_held_out_set():
    te = SimpleNamespace(workload_spec={"models": ["resnet50"], "evaluation_only_models": ["wideresnet"]})
    held = held_out_models(te)
    assert {"resnet50", "wideresnet", "resnet18", "resnet34", "regnet_x_400mf"} <= set(held)


def test_the_declared_mac_width_bounds_every_partial_sum():
    semantics = {
        "internal_arithmetic": {
            "mac_result_bits": 20,
            "full_operation_overflow_policy": "bounded_exact_requires_each_partial_sum",
        }
    }
    limit = MF.mac_limit(semantics)
    assert limit == (1 << 19) - 1
    assert MF.mac_limit({"internal_arithmetic": {"mac_result_bits": 20}}) is None
    entry = {"op": "matmul", "M": 16, "K": 72, "N": 8, "epilogue": []}
    lo, hi = MF.stimulus_range(
        entry, operand_range=(-128, 127), output_range=(-(1 << 31), (1 << 31) - 1), partial_sum_limit=limit
    )
    assert lo == -hi and 72 * hi * hi + hi <= limit < 72 * (hi + 1) ** 2 + hi + 1
    add = {"op": "residual_add", "M": 16, "N": 16, "bound_lsb": 1, "epilogue": []}
    assert MF.stimulus_range(add, operand_range=(-128, 127), output_range=(-128, 127), partial_sum_limit=limit) == [
        -128,
        127,
    ]
    with pytest.raises(MF.ModelFormRefusal, match="MAC width"):
        MF.stimulus_range(
            {**entry, "K": 1 << 20}, operand_range=(-128, 127), output_range=(-128, 127), partial_sum_limit=limit
        )


def test_a_window_mean_form_is_minted_in_the_orientation_the_model_holds():
    from merlin.xdsl_dialects.lowering.group_command import STATIONARY_ACTIVATION, STATIONARY_KEY

    mean = {
        "op": "matmul",
        "M": 8,
        "K": 256,
        "N": 1,
        "epilogue": ["acc_scale"],
        "acc_scale": 1 / 256,
        "operand_dtype": "int8",
    }
    row = {"name": "G_mean", "count": 1, "groups": [16], "entry": mean, "program": {"stored_operand": None}}
    got = MF.derive_application("residual_cnn", _stated(row), _binding(), capture_sha256="0" * 64)
    (entry,) = got["entries"]
    assert (entry["M"], entry["K"], entry["N"]) == (1, 256, 8)
    assert entry[STATIONARY_KEY] == STATIONARY_ACTIVATION
    assert entry["model_form"]["form"]["stationary"] == STATIONARY_ACTIVATION
    stored = {**row, "entry": {**mean, "N": 4}, "program": {"stored_operand": 1}}
    (kept,) = MF.derive_application("residual_cnn", _stated(stored), _binding(), capture_sha256="0" * 64)["entries"]
    assert STATIONARY_KEY not in kept and kept["N"] == 4


def test_an_activation_stationary_form_is_named_apart_from_its_weight_stationary_twin():
    """``stationary`` is part of the form key, so it is part of the name; otherwise two forms that
    differ only in which operand stays resident mint one capsule name twice."""
    from merlin_experiments.phase0 import model_forms as MF

    from merlin.xdsl_dialects.lowering.group_command import STATIONARY_ACTIVATION, STATIONARY_KEY

    entry = {"op": "matmul", "M": 1, "K": 16, "N": 16, "epilogue": []}
    stored = MF.capsule_name("app", entry, "most_frequent")
    resident = MF.capsule_name("app", {**entry, STATIONARY_KEY: STATIONARY_ACTIVATION}, "most_frequent")
    assert stored != resident and "activation_stationary" in resident
    assert MF.form_key(entry) != MF.form_key({**entry, STATIONARY_KEY: STATIONARY_ACTIVATION})
