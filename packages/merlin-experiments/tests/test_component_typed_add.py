"""Public native typed add premises reach ordinary bounded source generation."""

import copy
import hashlib
import importlib.util
import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml
from merlin_experiments.phase0 import component_automatic as A
from merlin_experiments.phase0 import component_coverage as C
from merlin_experiments.phase0 import operator_schema_intake as O
from merlin_experiments.phase0 import typed_add_sources as T
from merlin_experiments.phase0.component_semantic_basis import BasisSource, ComponentSemanticBasis
from merlin_experiments.phase0.evidence import select_evidence
from merlin_experiments.phase0.rtl_intake import RtlIntakeRefusal
from merlin_experiments.phase0.software_intake import issue_independent_software_intake

from merlin.common import invocation_record as I
from merlin.common.paths import module_source_path
from merlin.targetgen import golden_store
from merlin.targetgen.capsule_inputs import materialize_capsule_leaves
from merlin.targetgen.frontend_typed_add import add_forms

_fixture_path = Path(__file__).with_name("test_component_automatic.py")
_fixture_spec = importlib.util.spec_from_file_location("private_typed_add_fixtures", _fixture_path)
fixtures = importlib.util.module_from_spec(_fixture_spec)
_fixture_spec.loader.exec_module(fixtures)
automatic, independent, selected, write = fixtures.automatic, fixtures.independent, fixtures.selected, fixtures.write

NUMERIC = {"model": {"engine": "integer_reference"}, "operand_dtype": "int8", "overflow": "bounded_exact"}


@pytest.fixture(scope="module")
def native_add(tmp_path_factory):
    names = ("MERLIN_TEST_TORCH_PYTHON", "MERLIN_TEST_M2M_ROOT", "MERLIN_TEST_OPERATOR_DECLARATIONS")
    if any(not os.environ.get(name) for name in names):
        pytest.skip("typed add needs explicit selected public native framework/capture/declaration sources")
    python, capture, declarations = (Path(os.environ[name]).absolute() for name in names)
    owner = tmp_path_factory.mktemp("public-native-typed-add")
    script = owner / "capture.py"
    script.write_text("""import sys,json,importlib.util
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import torch
from m2m.capture.trace import snapshot_exported_program
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path)
 result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result
schemas=module("schemas",sys.argv[2]);defaults=module("defaults",sys.argv[3])
class Model(torch.nn.Module):
 def __init__(self,alpha): super().__init__();self.alpha=alpha
 def forward(self,A,W,B):
  return A@W,A.clone(),torch.ops.aten.add.Tensor(A,B,alpha=self.alpha)
out={}
for name,alpha,bshape,bdtype in [("valid",1,(5,7),torch.int8),("alpha",2,(5,7),torch.int8),
                                ("broadcast",1,(1,7),torch.int8),("promotion",1,(5,7),torch.int16)]:
 inputs=(torch.arange(35,dtype=torch.int8).reshape(5,7),torch.ones(7,11,dtype=torch.int8),
         torch.ones(bshape,dtype=bdtype))
 graph=snapshot_exported_program(torch.export.export(Model(alpha),inputs),stage="original")
 request={"namespace":"aten","captured_schemas":graph["operator_schemas"],
          "operations":sorted({n["target"] for n in graph["nodes"] if n["op"]=="call_function"})}
 observed=schemas.observe(request,declarations=Path(sys.argv[4]).read_bytes())
 default_request={"schema":"merlin.original_schema_defaults_request.v1","graph_sha256":graph["sha256"],
                  "rows":[{"target":r["target"],"schema":r["schema"]}
                          for r in observed["rows"] if r["status"]=="observed"]}
 out[name]={"trace":{"schema":"m2m.frontend_trace.v1","graphs":{"original":graph}},
            "observation":observed,"defaults":defaults.observe(default_request)}
out["numeric"]={"wrapped":(torch.tensor([127,-128],dtype=torch.int8)+torch.tensor([1,-1],dtype=torch.int8)).tolist(),
                "promoted_dtype":str((torch.ones(1,dtype=torch.int8)+torch.ones(1,dtype=torch.int16)).dtype)}
print(json.dumps(out,sort_keys=True))
""")
    schemas = module_source_path("merlin.targetgen.torch_schema_observer")
    defaults = module_source_path("merlin.targetgen.torch_schema_defaults_observer")
    result = I.run(
        [str(python), "-I", str(script), str(capture), str(schemas), str(defaults), str(declarations)],
        directory=owner,
        stage="actual_original_typed_add_controls",
        inputs=(script, schemas, defaults, declarations),
        env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"},
        capture_output=True,
        timeout=60,
    )
    result.check_returncode()
    return json.loads(result.stdout), python, declarations


def test_actual_default_input_result_types_and_wrap_are_preserved(native_add):
    sources, _, _ = native_add
    source = sources["valid"]
    (form,) = add_forms(source["trace"], source["observation"], source["defaults"], numerical_semantics=NUMERIC)
    assert form["status"] == "supported"
    assert form["operand_dtypes"] == ["int8", "int8"] and form["result_dtypes"] == ["int8"]
    call = next(node for node in source["trace"]["graphs"]["original"]["nodes"] if node["target"] == "aten.add.Tensor")
    assert "alpha" not in call["kwargs"]  # actual export elides the public unit default
    assert sources["numeric"] == {"wrapped": [-128, 127], "promoted_dtype": "torch.int16"}


@pytest.mark.parametrize("case", ["alpha", "broadcast", "promotion"])
def test_actual_wrong_alpha_broadcast_or_promotion_is_unknown(native_add, case):
    sources, _, _ = native_add
    source = sources[case]
    (form,) = add_forms(source["trace"], source["observation"], source["defaults"], numerical_semantics=NUMERIC)
    assert form["status"] == "unknown"


@pytest.mark.parametrize("defect", ["alpha_default", "bool_default", "result_dtype", "rank", "unknown_overflow"])
def test_source_and_default_substitutions_cannot_admit_add(native_add, defect):
    sources, _, _ = native_add
    source, numeric = copy.deepcopy(sources["valid"]), copy.deepcopy(NUMERIC)
    graph = source["trace"]["graphs"]["original"]
    call = next(node for node in graph["nodes"] if node["target"] == "aten.add.Tensor")
    if defect in {"alpha_default", "bool_default"}:
        row = next(row for row in source["defaults"]["rows"] if row["request"]["target"] == "aten.add.Tensor")
        row["defaults"][2]["default"]["value"] = 2 if defect == "alpha_default" else True
    elif defect == "unknown_overflow":
        numeric["overflow"] = "saturate"
    else:
        call["results"][0]["dtype" if defect == "result_dtype" else "shape"] = (
            "int32" if defect == "result_dtype" else [2, 3, 1]
        )
        # Rebuild every original output edge from the actual amended typed slot.
        for edge in graph["edges"]:
            if edge["producer_node_id"] == call["id"]:
                for key in ("dtype", "shape"):
                    edge[key] = call["results"][0][key]
        from merlin.targetgen.frontend_trace import _digest

        graph["sha256"] = _digest({key: value for key, value in graph.items() if key != "sha256"})
        source["defaults"]["graph_sha256"] = graph["sha256"]
    (form,) = add_forms(source["trace"], source["observation"], source["defaults"], numerical_semantics=numeric)
    assert form["status"] == "unknown"


@pytest.fixture
def add_generation(native_add, monkeypatch, request, tmp_path):
    sources, python, declarations = native_add
    choices = dict(getattr(request, "param", {}))
    source_case = choices.pop("source_case", "valid")
    selection_suffix = choices.pop("selection_suffix", ".json")
    if not os.environ.get("MERLIN_TEST_TORCH_SOURCE_ROOT"):
        pytest.skip("typed add source issuance requires explicit clean public declaration checkout")
    monkeypatch.setattr(fixtures, "example", lambda **kwargs: copy.deepcopy(sources[source_case]["trace"]))
    options = request.getfixturevalue("automatic")
    spec = yaml.safe_load(options["software_spec"].read_bytes())
    signature = {
        "ordered_operand_dtypes": ["int8", "int8"],
        "ordered_result_dtypes": ["int8"],
        "broadcasting": "none",
    }
    signature.update(choices)
    spec["operations"]["elementwise"] = {
        "ops": ["add"],
        "placement": "accelerator",
        "signature": signature,
    }
    write(options["software_spec"], spec)
    previous = options["software_intake"]
    review_path = next(Path(pin.path) for pin in previous.source_pins if pin.role == "protected-minimal-review")
    review = yaml.safe_load(review_path.read_bytes())
    review["source"]["sha256"] = hashlib.sha256(options["software_spec"].read_bytes()).hexdigest()
    review["operation_basis"].append(
        {"owner": "elementwise", "member": "source-example", "operations": ["aten.add.Tensor"]}
    )
    reviewed = write(tmp_path / "add-review.json", review)
    forbidden = tmp_path / "absent-add-private-prefix"
    software = issue_independent_software_intake(
        hardware=previous.hardware,
        source=options["software_spec"],
        review=reviewed,
        forbidden_roots=(forbidden,),
        output_root=tmp_path / "add-software",
    )
    options["software_intake"] = software
    checkout = Path(os.environ["MERLIN_TEST_TORCH_SOURCE_ROOT"])
    commit = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"]).decode().strip()
    selection = write(
        tmp_path / ("add-schema-selection" + selection_suffix),
        {
            "schema": O.SELECTION_SCHEMA,
            "status": "reviewed",
            "software_intake_sha256": software.sha256,
            "namespace": "aten",
            "python": str(python),
            "canonical_source": {"checkout": str(checkout), "commit": commit, "path": str(declarations)},
        },
    )
    schema = O.issue_independent_operator_schema_intake(
        software=software, selection=selection, forbidden_roots=(forbidden,), output=tmp_path / "add-schemas"
    )
    options["operator_schema_intake"] = schema
    options.pop("capability_contract")
    evidence = select_evidence(
        "fixture",
        descriptor=options["descriptor"],
        facts_path=options["rtl_facts"],
        software_spec=options["software_spec"],
        hardware_intake=options["hardware_intake"],
        software_intake=software,
        source_components=True,
    )
    recipe = yaml.safe_load(options["recipe"].read_bytes())
    recipe["component_performance"]["hardware"] = {
        key: evidence.derivation_identity[key] for key in ("contract_sha256", "raw_facts_sha256")
    }
    recipe["component_performance"]["objectives"] = []
    write(options["recipe"], recipe)
    template = yaml.safe_load(options["performance_template"].read_bytes())
    template["sweeps"], template["families"] = [], []
    write(options["performance_template"], template)
    policy = yaml.safe_load(options["component_coverage"].read_bytes())
    policy.update(
        schema=A.TYPED_POLICY_SCHEMA,
        operator_schema_intake_sha256=schema.sha256,
        software_spec_sha256=software.source.sha256,
        numerical_semantics_sha256=C.digest(evidence.software_spec["numerical_semantics"]),
        hardware=recipe["component_performance"]["hardware"],
    )
    write(options["component_coverage"], policy)
    return options


@pytest.mark.parametrize(
    "add_generation",
    [{"selection_suffix": ".json"}, {"selection_suffix": ".yaml"}],
    ids=["json", "yaml"],
    indirect=True,
)
def test_normal_generation_has_exact_i8_sources_budget_and_complete_independent_outputs(add_generation):
    report = fixtures.run(add_generation)
    A.verify(report["automatic_derivation"], report=report)
    rows = [row for row in report["obligations"] if row["id"].startswith("auto_elementwise_add_")]
    assert {row["cohort"] for row in rows} == {"functional_guard", "withheld_transfer"}
    shapes, count = set(), 0
    for row in rows:
        assert row["state"] == "source_generated"
        for member in row["members"]:
            count += 1
            root = add_generation["output_root"] / member["member"]
            capsule = yaml.safe_load((root / "capsule.yaml").read_bytes())
            assert capsule["component_program"]["nodes"][0]["dtype"] == "i8"
            assert capsule["integer_partial_sum_bound"]["status"] == "proven_safe"
            assert capsule["integer_partial_sum_bound"]["nodes"][0]["result_bits"] == 8
            source = (root / capsule["linalg_mlir"]).read_text()
            assert "arith.addi" in source and "i8" in source
            leaves = materialize_capsule_leaves(capsule)
            a, b = leaves["A"], leaves["B"]
            shapes.add(tuple(a.shape))
            expected = [
                [int(a.data[i * a.shape[1] + j]) + int(b.data[i * b.shape[1] + j]) for j in range(a.shape[1])]
                for i in range(a.shape[0])
            ]
            assert golden_store.load_golden(root)["outputs"] == {"Y": expected}
    assert count == 5 and shapes == {(1, 1), (1, 2), (2, 1), (2, 2), (3, 2)}
    assert ("numeric_domain", "aten.add.Tensor") in {
        (row["kind"], row["selector"]) for row in report["automatic_derivation"]["required_unknowns"]
    }
    with pytest.raises(ValueError, match="source-only preparation is not concrete"):
        C.verify_report(add_generation["output_root"])
    saved = copy.deepcopy(report)
    saved["automatic_derivation"].pop("typed_add_sources")
    saved["automatic_derivation"]["sha256"] = C.digest(
        {key: value for key, value in saved["automatic_derivation"].items() if key != "sha256"}
    )
    saved["generation_identity"]["automatic_derivation_sha256"] = C.digest(saved["automatic_derivation"])
    with pytest.raises(ValueError, match="lost its actual original"):
        A.verify(saved["automatic_derivation"], report=saved)


def test_v5_requires_live_schema_even_with_resigned_policy(add_generation):
    add_generation.pop("operator_schema_intake")
    with pytest.raises(ValueError, match="identical live independent schema"):
        fixtures.generation.generate_target("fixture", **add_generation)


@pytest.mark.parametrize(
    "add_generation",
    [
        {"ordered_result_dtypes": ["int32"]},
        {"ordered_operand_dtypes": ["int8", "int16"]},
        {"broadcasting": "numpy"},
        {"source_case": "alpha"},
        {"source_case": "promotion"},
        {"source_case": "broadcast"},
    ],
    indirect=True,
)
def test_mismatched_review_cannot_change_original_source_types_or_broadcast(add_generation):
    report = fixtures.run(add_generation)
    A.verify(report["automatic_derivation"], report=report)
    assert not any(row["id"].startswith("auto_elementwise_add_") for row in report["obligations"])
    assert ("source_operator_form", "aten.add.Tensor") in {
        (row["kind"], row["selector"]) for row in report["automatic_derivation"]["required_unknowns"]
    }


def test_resigned_forms_cannot_replace_actual_native_or_original_source(add_generation):
    report = fixtures.run(add_generation)
    for defect in ("alpha", "result_dtype"):
        saved = copy.deepcopy(report)
        form = saved["automatic_derivation"]["typed_add_sources"]["members"][0]["forms"][0]
        form["alpha" if defect == "alpha" else "result_dtypes"] = 2 if defect == "alpha" else ["int32"]
        record = saved["automatic_derivation"]
        record["sha256"] = C.digest({key: value for key, value in record.items() if key != "sha256"})
        saved["generation_identity"]["automatic_derivation_sha256"] = C.digest(record)
        with pytest.raises(ValueError, match="typed SSA/default replay"):
            A.verify(record, report=saved)


@pytest.mark.parametrize("add_generation", [{"selection_suffix": ".yaml"}], indirect=True)
def test_yaml_selection_unknown_version_refuses_both_native_observation_and_replay(
    add_generation, monkeypatch, tmp_path
):
    report = fixtures.run(add_generation)
    record = report["automatic_derivation"]
    schema_record = record["operator_schema_intake"]
    basis_pin = next(pin for pin in record["sources"] if pin["role"] == "semantic-basis-roster")
    basis = ComponentSemanticBasis.load(
        Path(basis_pin["path"]).read_bytes(),
        source=BasisSource(basis_pin["path"], basis_pin["sha256"], basis_pin["role"]),
        parent=Path(basis_pin["path"]).parent,
        routing={},
    )
    software = add_generation["software_intake"].public_facts()
    path = Path(schema_record["selection_path"])
    selected = yaml.safe_load(path.read_bytes())
    selected["schema"] = "merlin.independent_operator_schema_selection.unsupported"
    write(path, selected)
    monkeypatch.setattr(I, "run", lambda *args, **kwargs: pytest.fail("unsupported selection reached native process"))
    with pytest.raises(RtlIntakeRefusal, match="closed protected public source selection"):
        T.observe(
            schema_record=schema_record,
            basis=basis,
            numerical_semantics=software["numerical_semantics"],
            destination=tmp_path / "unsupported-observer",
        )
    with pytest.raises(RtlIntakeRefusal, match="closed protected public source selection"):
        T.verify(
            record["typed_add_sources"],
            schema_record=schema_record,
            basis=basis,
            numerical_semantics=software["numerical_semantics"],
        )
