"""Column-gain requirements reach generated accelerator tests and coverage gates."""

import copy

import numpy as np
import pytest
import yaml

from merlin.common.paths import merlin_dir
from merlin.runtime.commandbuffer import materialize_inputs
from merlin.runtime.reference import reference_outputs
from merlin.runtime.simulator import simulate
from merlin.targetgen import column_readout_capsules as CR
from merlin.targetgen import conformance as CF
from merlin.targetgen import corpus_spec as CS
from merlin.targetgen import corpus_synth
from merlin.targetgen.contract.interface_emit import parse_interface_mlir
from merlin.targetgen.contract.schemas import validate_capsule


def _path(width):
    return {
        "exits": {"accumulator": True, "requant": True},
        "accumulator": {"dtype": "i32"},
        "accumulator_stages": [{"kind": "preload", "dtype": "i32", "range": [-(1 << 31), (1 << 31) - 1]}],
        "readout": [
            {"kind": "convert", "to": "f32", "rounding": "half_even"},
            {"kind": "scale", "dtype": "f32", "rounding": "half_even", "granularity": "per_command",
             "columns_per_scale": width},
            {"kind": "requant", "to": "i8", "rounding": "half_even", "clamp": [-128, 127]},
        ],
    }


def _binding(width):
    return CS.CorpusBinding(target="synthetic", tile_dim=width, operand_dtype="int8", accum_dtype="i32",
                            integer=True, tiers=["L0", "L1", "L2", "L3"], compare="exact_int",
                            requant_output_dtype="i8", classes_for=lambda **_: [])


@pytest.mark.parametrize("width", [3, 8])
def test_selected_width_derives_matrix_and_convolution_gain_blocks_and_tails(width):
    rows = CR.requirements(_path(width), ["int8"], ["matmul", "conv2d"])
    assert len(rows) == 8
    assert sum(row["column_readout"].get("logical_granularity") == "per_channel" for row in rows) == 4
    assert {(r["op"], r["column_readout"]["columns"]) for r in rows} == {
        (op, n) for op in ("matmul", "conv2d") for n in (2 * width, 2 * width + 1)
    }
    for row in rows:
        entry = CR.entry(row)
        cap, text = CS.build(entry, _binding(width))
        cap["lanes"] = entry["lanes"]
        validate_capsule(cap)
        assert CR.matches(cap, row)
        cb = parse_interface_mlir(text)
        inputs = {name: tensor.to_list() for name, tensor in materialize_inputs(cb).items()}
        expected = reference_outputs(cb, inputs=inputs)
        assert simulate(cb, inputs=inputs)["outputs"] == expected
        lhs = np.asarray(inputs["IFM" if row["op"] == "conv2d" else "A0"], dtype=np.int64).reshape(-1, width)
        raw = lhs @ np.asarray(inputs["W"], dtype=np.int64) + np.asarray(inputs["B"], dtype=np.int64)
        independent = np.clip(np.rint(raw.astype(np.float32)
                                     * np.asarray(entry["acc_scale_by_column"], dtype=np.float32)), -128, 127)
        assert expected["Y0"] == independent.astype(np.int64).tolist()
        wrong = copy.deepcopy(cb)
        command = next(c for c in wrong["commands"] if c["opcode"] in ("COMMIT", "CONV2D"))
        attrs = command["attributes"]
        attrs["acc_scale"] = attrs.pop("acc_scale_by_column")[0]
        assert reference_outputs(wrong, inputs=inputs) != expected  # scalar-only mutant is observable
        if row["column_readout"].get("logical_granularity") == "per_channel":
            blocked = copy.deepcopy(cb)
            command = next(c for c in blocked["commands"] if c["opcode"] in ("COMMIT", "CONV2D"))
            command["attributes"]["acc_scale_by_column"] = [
                entry["acc_scale_by_column"][(index // width) * width]
                for index in range(len(entry["acc_scale_by_column"]))
            ]
            assert reference_outputs(blocked, inputs=inputs) != expected
        assert entry["generalization"]["must_accelerate"] is True


@pytest.mark.parametrize("include_convolution", [False, True])
def test_group_derivation_and_synthesis_include_column_work_without_private_capture_shapes(
    monkeypatch, include_convolution,
):
    from test_derived_epilogue_groups import _oracle, _writable

    _writable(monkeypatch)
    oracle = _oracle()
    oracle.cap_map["contraction"].arbitrary_mnk = True
    from merlin.xdsl_dialects.lowering import compute_groups as CG

    monkeypatch.setattr(CG, "TargetOracle", lambda _target: oracle)
    monkeypatch.setattr(CF, "_declared_output_path", lambda _target: _path(5))
    groups = CF._group_axis({}, "synthetic")
    if include_convolution:
        groups["required"].extend(CR.requirements(_path(5), ["int8"], ["conv2d"]))
    column = [r for r in groups["required"] if r.get("column_readout")]
    assert len(column) == (8 if include_convolution else 4) and all(r["observed_in"] == [] for r in column)
    root = merlin_dir() / "contract/capsules/conformance"
    spec = yaml.safe_load(next(iter(sorted(root.glob("*.yaml")))).read_text())
    spec["groups"] = groups
    dtype = next(str(c["dtype"]) for c in spec["cells"] if c.get("family") == "contraction")
    for row in column:
        row["dtype"] = dtype
    members = [e for e in corpus_synth.synthesize(spec)["capsules"]
               if (e.get("generalization") or {}).get("generalization_axis") == "column_readout"]
    assert len(members) == len(column)
    assert all("acc_scale_by_column" in e and "acc_scale" not in e for e in members)
    for member in members:
        # These consumers run before the capsule builder can supply defaults.
        assert corpus_synth.pass_requirements_for(member, spec)
        assert corpus_synth.cap_to_affordable(member, {
            "boundaries": {"tile_edge": 5}, "cert_affordability": {"max_elements": 1000},
        }) is None


@pytest.mark.parametrize("mutation", ["scalar", "uniform", "inside_block", "host_only"])
def test_coverage_requires_actual_vector_arithmetic_and_accelerator_placement(tmp_path, mutation):
    rows = CR.requirements(_path(4), ["int8"], ["matmul", "conv2d"])
    for i, row in enumerate(rows):
        entry = CR.entry(row)
        cap, _ = CS.build(entry, _binding(4))
        cap["lanes"] = entry["lanes"]
        directory = tmp_path / str(i)
        directory.mkdir()
        (directory / "capsule.yaml").write_text(yaml.safe_dump(cap))
    result = CF._group_gap(rows, [tmp_path])
    assert result["n_required"] == result["n_covered"] == 8 and result["uncovered"] == []
    path = tmp_path / "0/capsule.yaml"
    cap = yaml.safe_load(path.read_text())
    attrs = cap["operation"]["attributes"]
    if mutation == "scalar":
        attrs["acc_scale"] = 0.5
        attrs.pop("acc_scale_by_column")
    elif mutation == "uniform":
        attrs["acc_scale_by_column"] = [0.5] * len(attrs["acc_scale_by_column"])
    elif mutation == "inside_block":
        attrs["acc_scale_by_column"][1] = 0.75
    else:
        cap["lanes"] = {"forbid": ["on_mesh"]}
    path.write_text(yaml.safe_dump(cap))
    result = CF._group_gap(rows, [tmp_path])
    assert result["n_covered"] == 7 and result["uncovered"] == [rows[0]["signature"]]


def test_unknown_width_format_and_undeclared_path_do_not_invent_hardware_support():
    assert CR.requirements(None, ["int8"]) == []
    for field, value in (("columns_per_scale", None), ("dtype", "bf16")):
        path = _path(4)
        path["readout"][1][field] = value
        assert CR.requirements(path, ["int8"]) == []
        assert "column_group_scale" in CR.unstated(path)


def test_block_cases_alone_do_not_cover_independent_column_gains(tmp_path):
    rows = CR.requirements(_path(4), ["int8"], ["matmul", "conv2d"])
    for index, row in enumerate(rows):
        if row["column_readout"].get("logical_granularity") == "per_channel":
            continue
        entry = CR.entry(row)
        cap, _ = CS.build(entry, _binding(4))
        cap["lanes"] = entry["lanes"]
        directory = tmp_path / str(index)
        directory.mkdir()
        (directory / "capsule.yaml").write_text(yaml.safe_dump(cap))
    gap = CF._group_gap(rows, [tmp_path])
    assert gap["n_required"] == 8 and gap["n_covered"] == 4
    assert len(gap["uncovered"]) == 4
    assert all("|independent_column_scale|" in name for name in gap["uncovered"])


def test_single_lane_physical_scale_blocks_already_exercise_independent_columns():
    rows = CR.requirements(_path(1), ["int8"], ["matmul", "conv2d"])
    assert len(rows) == 4
    assert not any(row["column_readout"].get("logical_granularity") for row in rows)


@pytest.mark.parametrize("evidence", ["derived", "UNKNOWN", "missing", "scaled_load"])
def test_prepared_contract_column_profile_uses_selected_facts_without_changing_inputs(
    monkeypatch, tmp_path, evidence,
):
    from test_derived_epilogue_groups import _oracle, _writable
    from test_output_epilogue import _SCALED, _accumulates, _contract, _facts, _preload_fir

    from merlin.targetgen import target_registry
    from merlin.targetgen.rtl import semantic_facts
    from merlin.targetgen.rtl.facts import observed_facts
    from merlin.xdsl_dialects.lowering import compute_groups as CG
    from merlin.xdsl_dialects.lowering import output_fusion as OF

    # Real output-path derivation over small FIRRTL; only the behavioral probe
    # result is supplied. A changed load must still fail the structural proof.
    contract = _contract()
    facts = _facts(_preload_fir(tmp_path, **({"lanes": _SCALED} if evidence == "scaled_load" else {})))
    original = copy.deepcopy((contract, facts))
    semantic = _accumulates() if evidence in ("derived", "scaled_load") else {
        "facts": {"accumulate_on_load": {"status": "UNKNOWN"}} if evidence == "UNKNOWN" else {},
    }
    monkeypatch.setattr(semantic_facts, "derive", lambda target, *, only: semantic)
    _writable(monkeypatch)
    oracle = _oracle(routed=False)
    oracle.cap_map["contraction"].arbitrary_mnk = True
    monkeypatch.setattr(CG, "TargetOracle", lambda target: oracle)
    # Foreign ambient files must not replace the derive caller's selection.
    monkeypatch.setenv("MERLIN_TARGET_CONTRACT", str(tmp_path / "absent-contract"))
    monkeypatch.setenv("MERLIN_RTL_FACTS", str(tmp_path / "absent-facts"))
    with target_registry.observed_contract("synthetic", contract), observed_facts("synthetic", facts):
        assert OF.declared_path("synthetic") is None
        requirement = {"groups": CF._group_axis({}, "synthetic")}
        if evidence == "derived":
            selected = CR.with_profile(requirement, CR.DISTINCT_SCALES_V2)
            row = selected["groups"]["required"][-1]
            assert row["column_readout"]["columns_per_scale"] == 2
            assert row["column_readout"]["columns"] == 6
            assert "bias_add" in row["epilogue"]
        else:
            with pytest.raises(ValueError, match="with bias preload"):
                CR.with_profile(requirement, CR.DISTINCT_SCALES_V2)
        assert "output_epilogue" not in target_registry.load_contract("synthetic")
    assert (contract, facts) == original


def test_prepared_contract_without_facts_does_not_regenerate_them(monkeypatch):
    from merlin.targetgen import target_registry
    from merlin.targetgen.rtl import facts

    def unavailable(target, *, regenerate):
        assert regenerate is False
        raise FileNotFoundError("no selected facts")

    monkeypatch.setattr(facts, "load_facts", unavailable)
    with target_registry.observed_contract("synthetic", {}):
        assert CF._declared_output_path("synthetic") is None
