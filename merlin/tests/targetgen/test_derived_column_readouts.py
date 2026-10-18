"""Column-gain requirements reach generated accelerator tests and coverage gates."""

import copy

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
        wrong = copy.deepcopy(cb)
        command = next(c for c in wrong["commands"] if c["opcode"] in ("COMMIT", "CONV2D"))
        attrs = command["attributes"]
        attrs["acc_scale"] = attrs.pop("acc_scale_by_column")[0]
        assert reference_outputs(wrong, inputs=inputs) != expected  # scalar-only mutant is observable
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
    assert len(column) == (4 if include_convolution else 2) and all(r["observed_in"] == [] for r in column)
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
    assert result["n_required"] == result["n_covered"] == 4 and result["uncovered"] == []
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
    assert result["n_covered"] == 3 and result["uncovered"] == [rows[0]["signature"]]


def test_unknown_width_format_and_undeclared_path_do_not_invent_hardware_support():
    assert CR.requirements(None, ["int8"]) == []
    for field, value in (("columns_per_scale", None), ("dtype", "bf16")):
        path = _path(4)
        path["readout"][1][field] = value
        assert CR.requirements(path, ["int8"]) == []
        assert "column_group_scale" in CR.unstated(path)
