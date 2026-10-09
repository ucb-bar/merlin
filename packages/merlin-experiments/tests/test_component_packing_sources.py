"""Native partitions prepare full bounded originals, never physical-tail grants."""

import copy
import dataclasses
import importlib.util
import os
from pathlib import Path

import pytest
import yaml
from merlin_experiments.phase0 import component_automatic as A
from merlin_experiments.phase0 import packing_intake as P
from merlin_experiments.phase0.component_source_binding import verify_prepared_sources
from merlin_experiments.phase0.rtl_intake import RtlIntakeRefusal

from merlin.targetgen import component_program, golden_store
from merlin.targetgen.capsule_inputs import materialize_capsule_leaves
from merlin.targetgen.rtl.source_selection import produce_selection


def _fixture(name):
    spec = importlib.util.spec_from_file_location("private_packing_" + name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


F = _fixture("test_component_automatic")
S = _fixture("test_component_source_binding")
automatic, independent = F.automatic, F.independent


@pytest.fixture
def tools():
    names = ("MERLIN_TEST_FIRTOOL", "MERLIN_TEST_CIRCT_OPT")
    if any(not os.environ.get(name) for name in names):
        pytest.skip("local packing controls need explicitly selected public native FIRRTL/CIRCT tools")
    return {name: Path(os.environ[name]).absolute() for name in names}


@pytest.fixture
def selected(tools, tmp_path, request):
    broken = getattr(request, "param", False)
    source = tmp_path / "source.fir"
    source.write_text(
        "FIRRTL version 3.2.0\ncircuit Unit :\n"
        "  module Unit : @[generators/test_unit/src/Partition.scala 1:1]\n"
        "    input arbitrary : UInt<32>\n"
        + "".join(f"    output p{index} : UInt<8>\n" for index in range(4))
        + "".join(
            f"    p{index} <= bits(arbitrary, {low + 7}, {low})\n"
            for index, low in enumerate([0, 8, 8 if broken else 16, 24])
        )
    )
    bundle = produce_selection(
        target="test_unit",
        firrtl=source,
        generator="test_unit",
        config="IndependentPacking",
        core_root="Unit",
        firtool=tools["MERLIN_TEST_FIRTOOL"],
        output=tmp_path / "original-production",
    )
    descriptor = F.write(tmp_path / "descriptor.yaml", {"target": "test_unit"})
    private = tmp_path / "protected"
    private.mkdir()
    return {
        "target": "test_unit",
        "descriptor": descriptor,
        "source_bundle": bundle,
        "forbidden_roots": (private,),
        "output": tmp_path / "unused-hardware",
    }


def _options(automatic, selected, tools, tmp_path):
    options = S._source_options(automatic, tmp_path, version=A.PACKING_POLICY_SCHEMA)
    packing = P.issue_independent_packing_intake(
        hardware=options["hardware_intake"],
        circt_opt=tools["MERLIN_TEST_CIRCT_OPT"],
        forbidden_roots=selected["forbidden_roots"],
        output=tmp_path / "packing-intake",
    )
    policy = yaml.safe_load(options["component_coverage"].read_bytes())
    policy["packing_intake_sha256"] = packing.sha256
    F.write(options["component_coverage"], policy)
    options["packing_intake"] = packing
    return options


def _original_copy(root):
    capsule = yaml.safe_load((root / "capsule.yaml").read_bytes())
    program = capsule["component_program"]
    assert [node["op"] for node in program["nodes"]] == ["copy"]
    leaves = materialize_capsule_leaves(capsule)
    m, n = leaves["A"].shape
    expected = [[leaves["A"].data[i * n + j] for j in range(n)] for i in range(m)]
    assert golden_store.load_golden(root)["outputs"] == {"Y": expected}
    return tuple(leaves["A"].shape)


def test_actual_packing_source_preparation_replays_all_originals_and_keeps_resource_unknowns(
    automatic, selected, tools, tmp_path
):
    options = _options(automatic, selected, tools, tmp_path)
    packing = options["packing_intake"]
    facts = packing.record()["facts"]
    assert [(row["packed_width"], row["slice_width"], row["slice_count"]) for row in facts["partitions"]] == [
        (32, 8, 4)
    ]
    with pytest.raises(RtlIntakeRefusal, match="live independently issued"):
        dataclasses.replace(packing).verify()
    changed = copy.deepcopy(packing.record())
    changed["facts"]["partitions"][0]["slice_count"] = 5
    with pytest.raises(RtlIntakeRefusal, match="original typed SSA"):
        P.verify_record(changed)
    report = F.run(options)
    A.verify(report["automatic_derivation"], report=report)
    verify_prepared_sources(
        options["output_root"], report, software=options["software_intake"], hardware=options["hardware_intake"]
    )
    assert report["status"] == "source_prepared_incomplete"
    rows = [row for row in report["obligations"] if row["id"].startswith("auto_storage_probe_")]
    assert len(rows) == 4 and {row["cohort"] for row in rows} == {"functional_guard", "withheld_transfer"}
    shapes, sources = set(), set()
    for row in rows:
        assert row["state"] == "source_generated" and row["resource_boundaries"] == {}
        for member in row["members"]:
            root = options["output_root"] / member["member"]
            shapes.add(_original_copy(root))
            sources.add((root / "capsule.interface.mlir").read_text())
    assert {(1, 3), (1, 4), (1, 5), (3, 1), (4, 1), (5, 1), (2, 4), (2, 5), (4, 2), (5, 2)} <= shapes
    assert len(sources) == 12
    unknowns = report["automatic_derivation"]["required_unknowns"]
    assert {row["kind"] for row in unknowns} >= {"packing_mapping", "packing_domain", "resource_role", "effect_domain"}
    assert all(row["state"] == "unavailable" for row in report["obligations"] if row["id"].startswith("auto_missing_"))
    substituted = copy.deepcopy(report)
    substituted["automatic_derivation"].pop("packing_intake")
    _resign(substituted)
    with pytest.raises(ValueError, match="lost its selected original packing"):
        A.verify(substituted["automatic_derivation"], report=substituted)
    wrong_origin = copy.deepcopy(report)
    wrong_origin["automatic_derivation"]["packing_intake"]["hardware_intake_sha256"] = "0" * 64
    _resign(wrong_origin)
    with pytest.raises(ValueError, match="protected original hardware"):
        A.verify(wrong_origin["automatic_derivation"], report=wrong_origin)
    policy = yaml.safe_load(options["component_coverage"].read_bytes())
    historical = {key: value for key, value in policy.items() if key != "packing_intake_sha256"}
    historical["schema"] = A.LOGICAL_POLICY_SCHEMA
    legacy_options = {
        **options,
        "component_coverage": F.write(tmp_path / "historical-policy.json", historical),
        "output_root": tmp_path / "historical-policy-new-intake",
    }
    with pytest.raises(ValueError, match="local packing requires the explicit versioned"):
        F.run(legacy_options)
    del options["packing_intake"]
    options["output_root"] = tmp_path / "missing-live-packing"
    with pytest.raises(ValueError, match="identical live independent hardware"):
        F.run(options)


def _resign(report):
    record = report["automatic_derivation"]
    record["sha256"] = A.digest({key: value for key, value in record.items() if key != "sha256"})
    report["generation_identity"]["automatic_derivation_sha256"] = A.digest(record)


@pytest.mark.parametrize("selected", [True], indirect=True)
def test_actual_missing_partition_cannot_prepare_or_mint_boundary_sources(automatic, selected, tools, tmp_path):
    options = _options(automatic, selected, tools, tmp_path)
    facts = options["packing_intake"].record()["facts"]
    assert facts["partitions"] == [] and facts["incomplete_candidates"] == 1
    report = F.run(options)
    assert not any(row["id"].startswith("auto_storage_probe_") for row in report["obligations"])
    assert any(row["kind"] == "packing_source" for row in report["automatic_derivation"]["required_unknowns"])
    assert report["status"] == "source_prepared_incomplete"
    A.verify(report["automatic_derivation"], report=report)


def test_partition_source_budget_retains_denied_frontiers_before_allocating(
    automatic, selected, tools, tmp_path, monkeypatch
):
    options = _options(automatic, selected, tools, tmp_path)
    policy = yaml.safe_load(options["component_coverage"].read_bytes())
    policy["execution_budget"]["max_tensor_payload_bytes"] = 2
    F.write(options["component_coverage"], policy)
    build = component_program.build

    def bounded(entry, binding):
        shape = entry["program"]["inputs"][0]["shape"]
        assert shape == [1, 1], "denied source boundary reached tensor/reference allocation"
        return build(entry, binding)

    monkeypatch.setattr(component_program, "build", bounded)
    report = F.run(options)
    rows = [row for row in report["obligations"] if row["id"].startswith("auto_storage_probe_")]
    assert len(rows) == 4 and all(row["state"] == "unavailable" and len(row["members"]) == 3 for row in rows)
    assert all("execution budget exceeded" in member["reason"] for row in rows for member in row["members"])
    assert not any(
        (options["output_root"] / member["requested_member"]).exists() for row in rows for member in row["members"]
    )
    A.verify(report["automatic_derivation"], report=report)
