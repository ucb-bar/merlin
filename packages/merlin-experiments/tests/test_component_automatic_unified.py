"""One ordinary source generation retains independently selected facets."""

import copy
import importlib.util
import json
import os
from pathlib import Path

import pytest
import yaml
from merlin_experiments.phase0 import arithmetic_intake as R
from merlin_experiments.phase0 import component_automatic as A
from merlin_experiments.phase0 import operator_schema_intake as O
from merlin_experiments.phase0 import packing_intake as P
from merlin_experiments.phase0.software_intake import issue_independent_software_intake

from merlin.common import invocation_record as I
from merlin.targetgen import golden_store
from merlin.targetgen.capsule_inputs import materialize_capsule_leaves
from merlin.targetgen.rtl.source_selection import produce_selection


def _fixture(name):
    spec = importlib.util.spec_from_file_location("private_unified_" + name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


T = _fixture("test_component_typed_add")
F = T.fixtures
automatic, independent, add_generation = T.automatic, T.independent, T.add_generation
_SCHEMA = "merlin.component_automatic_policy.v7"


@pytest.fixture(scope="module")
def native_add(tmp_path_factory):
    names = ("MERLIN_TEST_TORCH_PYTHON", "MERLIN_TEST_M2M_ROOT", "MERLIN_TEST_OPERATOR_DECLARATIONS")
    if any(not os.environ.get(name) for name in names):
        pytest.skip("unified controls need explicit public native framework/capture/declaration sources")
    python, capture, declarations = (Path(os.environ[name]).absolute() for name in names)
    owner = tmp_path_factory.mktemp("public-native-unified-source")
    script = owner / "capture.py"
    script.write_text("""import sys,json
sys.path.insert(0,sys.argv[1])
import torch
from m2m.capture.trace import snapshot_exported_program
class Original(torch.nn.Module):
 def forward(self,A,W,B):
  return A@W,A.clone(),torch.ops.aten.alias.default(A),torch.ops.aten.add.Tensor(A,B)
inputs=(torch.ones(5,7,dtype=torch.int8),torch.ones(7,11,dtype=torch.int8),torch.ones(5,7,dtype=torch.int8))
graph=snapshot_exported_program(torch.export.export(Original(),inputs),stage="original")
print(json.dumps({"valid":{"trace":{"schema":"m2m.frontend_trace.v1","graphs":{"original":graph}}}},sort_keys=True))
""")
    result = I.run(
        [str(python), "-I", str(script), str(capture)],
        directory=owner,
        stage="actual_original_unified_source",
        inputs=(script,),
        env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"},
        capture_output=True,
        timeout=60,
    )
    result.check_returncode()
    return json.loads(result.stdout), python, declarations


@pytest.fixture
def selected(tmp_path):
    names = ("MERLIN_TEST_FIRTOOL", "MERLIN_TEST_CIRCT_OPT")
    if any(not os.environ.get(name) for name in names):
        pytest.skip("unified controls need explicit public native FIRRTL/CIRCT tools")
    source = tmp_path / "original.fir"
    source.write_text(
        "FIRRTL version 3.2.0\ncircuit Unit :\n"
        "  module Unit : @[generators/test_unit/src/Independent.scala 1:1]\n"
        "    input x : SInt<8>\n    input y : SInt<8>\n    input z : SInt<32>\n"
        "    input packed : UInt<32>\n    output q : SInt<20>\n"
        + "".join(f"    output p{index} : UInt<8>\n" for index in range(4))
        + "    node product = mul(x, y)\n    q <= add(product, z)\n"
        + "".join(f"    p{index} <= bits(packed, {index * 8 + 7}, {index * 8})\n" for index in range(4))
    )
    bundle = produce_selection(
        target="test_unit",
        firrtl=source,
        generator="test_unit",
        config="IndependentFacets",
        core_root="Unit",
        firtool=Path(os.environ[names[0]]),
        output=tmp_path / "original-production",
    )
    forbidden = tmp_path / "absent-unified-private-inputs"
    return {"source_bundle": bundle, "forbidden_roots": (forbidden,)}


@pytest.fixture
def combined(add_generation, selected, tmp_path):
    options = add_generation
    previous = options["software_intake"]
    review = yaml.safe_load(
        Path(next(p.path for p in previous.source_pins if p.role == "protected-minimal-review")).read_bytes()
    )
    next(row for row in review["operation_basis"] if row["owner"] == "movement")["operations"].append(
        "aten.alias.default"
    )
    software = issue_independent_software_intake(
        hardware=previous.hardware,
        source=options["software_spec"],
        review=F.write(tmp_path / "combined-review.json", review),
        forbidden_roots=selected["forbidden_roots"],
        output_root=tmp_path / "combined-software",
    )
    previous_schema = options["operator_schema_intake"].record()
    selection = yaml.safe_load(Path(previous_schema["selection_path"]).read_bytes())
    selection["software_intake_sha256"] = software.sha256
    schema = O.issue_independent_operator_schema_intake(
        software=software,
        selection=F.write(tmp_path / "combined-schema-selection.yaml", selection),
        forbidden_roots=selected["forbidden_roots"],
        output=tmp_path / "combined-schemas",
    )
    options.update(software_intake=software, operator_schema_intake=schema)
    for name, issuer in (
        ("arithmetic", R.issue_independent_arithmetic_intake),
        ("packing", P.issue_independent_packing_intake),
    ):
        options[name + "_intake"] = issuer(
            hardware=options["hardware_intake"],
            circt_opt=Path(os.environ["MERLIN_TEST_CIRCT_OPT"]),
            forbidden_roots=selected["forbidden_roots"],
            output=tmp_path / ("combined-" + name),
        )
    policy = yaml.safe_load(options["component_coverage"].read_bytes())
    policy.update(
        schema=_SCHEMA,
        operator_schema_intake_sha256=schema.sha256,
        arithmetic_intake_sha256=options["arithmetic_intake"].sha256,
        packing_intake_sha256=options["packing_intake"].sha256,
    )
    policy["budget"]["max_members"] = 96
    options["component_coverage"] = F.write(tmp_path / "combined-policy.json", policy)
    return options


def _run(options, tmp_path, schema):
    policy = yaml.safe_load(options["component_coverage"].read_bytes())
    policy["schema"] = schema
    chosen = dict(options)
    if schema != A.PACKING_POLICY_SCHEMA and schema != _SCHEMA:
        policy.pop("packing_intake_sha256")
        chosen.pop("packing_intake")
    chosen.update(
        component_coverage=F.write(tmp_path / (schema + ".json"), policy),
        output_root=tmp_path / schema,
    )
    report = F.run(chosen)
    A.verify(report["automatic_derivation"], report=report)
    return chosen, report


def _all_original_outputs(root):
    capsule = yaml.safe_load((root / "capsule.yaml").read_bytes())
    values = {name: (tensor.shape, list(tensor.data)) for name, tensor in materialize_capsule_leaves(capsule).items()}
    program = capsule["component_program"]
    for node in program["nodes"]:
        args = [values[name] for name in node["actual_inputs"]]
        if node["op"] == "matmul":
            (lhs_shape, lhs), (rhs_shape, rhs) = args
            m, k, n = lhs_shape[0], lhs_shape[1], rhs_shape[1]
            values[node["name"]] = (
                (m, n),
                [sum(lhs[i * k + p] * rhs[p * n + j] for p in range(k)) for i in range(m) for j in range(n)],
            )
        elif node["op"] == "add":
            assert args[0][0] == args[1][0]
            values[node["name"]] = (args[0][0], [lhs + rhs for lhs, rhs in zip(args[0][1], args[1][1], strict=True)])
        else:
            assert node["op"] in {"copy", "alias"}
            values[node["name"]] = (args[0][0], list(args[0][1]))
    expected = {}
    for output in program["outputs"]:
        (m, n), data = values[output["actual_value"]]
        expected[output["name"]] = [data[i * n : (i + 1) * n] for i in range(m)]
    assert golden_store.load_golden(root)["outputs"] == expected
    return expected


def test_actual_unified_generation_preserves_every_historical_gap_and_complete_output(combined, tmp_path):
    historical = {}
    for schema in (A.LOGICAL_POLICY_SCHEMA, A.TYPED_POLICY_SCHEMA, A.PACKING_POLICY_SCHEMA):
        _, historical[schema] = _run(combined, tmp_path, schema)
    assert any(
        row["id"].startswith("auto_elementwise_add_") for row in historical[A.TYPED_POLICY_SCHEMA]["obligations"]
    )
    assert not any(
        row["id"].startswith("auto_elementwise_add_") for row in historical[A.PACKING_POLICY_SCHEMA]["obligations"]
    )
    options, report = _run(combined, tmp_path, _SCHEMA)
    record = report["automatic_derivation"]
    assert report["status"] == "source_prepared_incomplete"
    assert {"operator_schema_intake", "typed_add_sources", "arithmetic_intake", "packing_intake"} <= set(record)
    assert record["operator_effect_semantics"][0]["effect_classes"] == ["may_alias_result"]
    expected = {
        row["id"]: row for report in historical.values() for row in report["automatic_derivation"]["required_unknowns"]
    }
    assert {row["id"]: row for row in record["required_unknowns"]} == expected
    assert {row["kind"] for row in expected.values()} >= {
        "numeric_datapath",
        "numeric_domain",
        "physical_effect",
        "resource_role",
        "packing_mapping",
    }
    families = set()
    for row in report["obligations"]:
        if row["id"] in expected:
            assert row["mandatory"] and row["state"] == "unavailable" and row["members"] == []
        else:
            assert row["state"] == "source_generated" and row["resource_boundaries"] == {}
            for member in row["members"]:
                _all_original_outputs(options["output_root"] / member["member"])
            families.add(row["id"].split("_")[1])
    assert {"elementwise", "storage", "may", "contraction", "movement"} <= families


def test_unified_receipt_cannot_drop_selected_facets_or_any_previous_gap(combined, tmp_path):
    _, report = _run(combined, tmp_path, _SCHEMA)
    for field in (
        "operator_schema_intake",
        "operator_effect_semantics",
        "arithmetic_intake",
        "packing_intake",
        "typed_add_sources",
    ):
        changed = copy.deepcopy(report)
        changed["automatic_derivation"].pop(field)
        _resign(changed)
        with pytest.raises(ValueError, match="lost its"):
            A.verify(changed["automatic_derivation"], report=changed)
    downgraded = copy.deepcopy(report)
    downgraded["automatic_derivation"]["schema"] = A.PACKING_RECEIPT_SCHEMA
    _resign(downgraded)
    with pytest.raises(ValueError, match="versioned policy"):
        A.verify(downgraded["automatic_derivation"], report=downgraded)
    changed = copy.deepcopy(report)
    missing = changed["automatic_derivation"]["required_unknowns"]
    missing.remove(next(row for row in missing if row["kind"] == "semantic_family"))
    _resign(changed)
    with pytest.raises(ValueError, match="source factory"):
        A.verify(changed["automatic_derivation"], report=changed)


def _resign(report):
    record = report["automatic_derivation"]
    record["sha256"] = A.digest({key: value for key, value in record.items() if key != "sha256"})
    report["generation_identity"]["automatic_derivation_sha256"] = A.digest(record)
