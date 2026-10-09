"""Fresh original None metadata and actual public zero-result bridge controls."""

import copy
import importlib.util
import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml
from merlin_experiments.phase0 import component_automatic as A
from merlin_experiments.phase0 import operator_schema_intake as O
from merlin_experiments.phase0 import tensor_argument_intake as T
from merlin_experiments.phase0 import zero_return_intake as Z
from merlin_experiments.phase0.rtl_intake import RtlIntakeRefusal

from merlin.common import invocation_record as I
from merlin.common.paths import module_source_path
from merlin.targetgen import golden_store
from merlin.targetgen.capsule_inputs import materialize_capsule_leaves
from merlin.targetgen.frontend_operator_effects import original_operator_effects

_path = Path(__file__).with_name("test_component_operator_schemas.py")
_spec = importlib.util.spec_from_file_location("private_zero_return_fixtures", _path)
F = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(F)
automatic, independent, selected = F.automatic, F.independent, F.selected


@pytest.fixture(scope="module")
def native_zero_sources(tmp_path_factory):
    keys = (
        "MERLIN_TEST_TORCH_PYTHON",
        "MERLIN_TEST_M2M_ROOT",
        "MERLIN_TEST_OPERATOR_DECLARATIONS",
        "MERLIN_TEST_TORCH_SOURCE_ROOT",
        "MERLIN_TEST_HOST_CXX",
    )
    if any(not os.environ.get(key) for key in keys):
        pytest.skip("fresh zero-return controls require explicit public Torch/capture/compiler source selectors")
    python, capture, declarations, checkout, compiler = (Path(os.environ[key]).absolute() for key in keys)
    owner = tmp_path_factory.mktemp("zero-return-native")
    source = owner / "capture.py"
    source.write_text("""import sys,json,importlib.util
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import torch
from m2m.capture.trace import snapshot_exported_program
spec=importlib.util.spec_from_file_location("schema_observer",sys.argv[2])
observer=importlib.util.module_from_spec(spec);spec.loader.exec_module(observer)
class Model(torch.nn.Module):
 def forward(self,A,W0,W1):
  observed_none=torch.ops.aten._assert_tensor_metadata.default(A,dtype=torch.int8)
  return A@W0,A@W1,A.clone(),observed_none
inputs=(torch.arange(6,dtype=torch.int8).reshape(2,3),torch.ones(3,2,dtype=torch.int8),torch.ones(3,2,dtype=torch.int8))
ep=torch.export.export(Model(),inputs)
graph=snapshot_exported_program(ep,stage="original")
request={"namespace":"aten","captured_schemas":graph["operator_schemas"],
 "operations":sorted({node["target"] for node in graph["nodes"] if node["op"]=="call_function"})}
observation=observer.observe(request,declarations=Path(sys.argv[3]).read_bytes())
operation=torch.ops.aten._assert_tensor_metadata.default
checks={"correct_returns_none":operation(inputs[0],dtype=torch.int8) is None}
try: operation(inputs[0],dtype=torch.float32)
except RuntimeError: checks["incorrect_metadata_raises"]=True
else: checks["incorrect_metadata_raises"]=False
print(json.dumps({"trace":{"schema":"m2m.frontend_trace.v1","graphs":{"original":graph}},"observation":observation,"checks":checks},sort_keys=True))
""")
    observer = module_source_path("merlin.targetgen.torch_schema_observer")
    completed = I.run(
        [str(python), "-I", str(source), str(capture), str(observer), str(declarations)],
        directory=owner,
        stage="fresh_original_zero_return_controls",
        inputs=(source, declarations, observer, capture / "m2m/capture/trace.py"),
        env=T.ENVIRONMENT,
        capture_output=True,
        timeout=60,
    )
    completed.check_returncode()
    document = json.loads(completed.stdout)
    assertion = next(
        node
        for node in document["trace"]["graphs"]["original"]["nodes"]
        if node["target"] == "aten._assert_tensor_metadata.default"
    )
    if "result_metadata" not in assertion:
        pytest.skip("selected historical capture producer has no result-metadata observation provenance")
    commit = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"]).decode().strip()
    forbidden = owner / "private-author-exclusions"
    getter = T.prepare_getter(
        python=python,
        compiler=compiler,
        checkout=checkout,
        commit=commit,
        forbidden=(forbidden,),
        output=owner / "argument-getter",
    )
    zero_getter = Z.prepare_getter(tensor_getter=getter, forbidden=(forbidden,), output=owner / "zero-getter")
    Z.verify_getter(zero_getter)
    return document, getter, zero_getter, declarations, owner


def observe(document, getter, owner):
    owner.mkdir()
    member = Z.observe_returns(
        trace=document["trace"],
        schema_observation=document["observation"],
        getter=getter,
        output=owner,
    )
    output = Z.verify_returns(
        trace=document["trace"],
        schema_observation=document["observation"],
        getter=getter,
        member=member,
    )
    return member, output


def test_actual_zero_return_bridge_binds_fresh_none_without_purity(native_zero_sources, tmp_path):
    document, _, getter, _, _ = native_zero_sources
    assert document["checks"] == {"correct_returns_none": True, "incorrect_metadata_raises": True}
    original = copy.deepcopy(document["trace"])
    legacy = original_operator_effects(document["trace"], document["observation"])
    assert len(legacy.unknowns()) == 1
    member, observed = observe(document, getter, tmp_path / "positive")
    relation = original_operator_effects(document["trace"], document["observation"], zero_returns=observed)
    assert relation.unknowns() == [] and relation.effect_classes == ()
    assert len(relation.zero_returns()) == 1
    assert relation.zero_returns()[0]["native"] == {
        "schema": original["graphs"]["original"]["operator_schemas"]["aten._assert_tensor_metadata.default"],
        "return_count": 0,
        "empty_stack_is_none": True,
    }
    assert document["trace"] == original
    changed = copy.deepcopy(document)
    assertion = next(
        node
        for node in changed["trace"]["graphs"]["original"]["nodes"]
        if node["target"] == "aten._assert_tensor_metadata.default"
    )
    assertion["result_metadata"]["status"] = "unobserved"
    F.resign(changed["trace"])
    with pytest.raises(RtlIntakeRefusal, match="original schema or full metadata"):
        Z.verify_returns(
            trace=changed["trace"], schema_observation=changed["observation"], getter=getter, member=member
        )


@pytest.mark.parametrize(
    "defect", ["missing", "unobserved", "foreign_slot", "container", "kind", "shape", "extra_slot"]
)
def test_actual_native_refuses_unknown_or_mismatched_original_none(native_zero_sources, tmp_path, defect):
    document, _, getter, _, _ = native_zero_sources
    changed = copy.deepcopy(document)
    assertion = next(
        node
        for node in changed["trace"]["graphs"]["original"]["nodes"]
        if node["target"] == "aten._assert_tensor_metadata.default"
    )
    metadata = assertion["result_metadata"]
    if defect == "missing":
        assertion.pop("result_metadata")
    elif defect == "unobserved":
        assertion["result_metadata"] = {"schema": metadata["schema"], "status": "unobserved"}
    elif defect == "foreign_slot":
        metadata["values"][0]["result_id"] += "other"
    elif defect == "container":
        metadata["container"] = "tuple"
    elif defect == "kind":
        metadata["values"][0]["kind"] = "tensor"
    elif defect == "shape":
        assertion["results"][0]["shape"] = []
    else:
        extra = copy.deepcopy(assertion["results"][0])
        extra["id"] += "extra"
        assertion["results"].append(extra)
    F.resign(changed["trace"])
    _, observed = observe(changed, getter, tmp_path / defect)
    assert observed["rows"][0]["status"] == "unknown"
    relation = original_operator_effects(changed["trace"], changed["observation"], zero_returns=observed)
    assert len(relation.unknowns()) == 1 and relation.effect_classes == ()


@pytest.mark.parametrize("defect", ["removed", "foreign_graph", "boolean_count", "not_none"])
def test_saved_native_row_cannot_substitute_complete_result_relation(native_zero_sources, tmp_path, defect):
    document, _, getter, _, _ = native_zero_sources
    _, observed = observe(document, getter, tmp_path / defect)
    if defect == "removed":
        observed["rows"] = []
    elif defect == "foreign_graph":
        observed["graph_sha256"] = "0" * 64
    elif defect == "boolean_count":
        observed["rows"][0]["native"]["return_count"] = False
    else:
        observed["rows"][0]["native"]["empty_stack_is_none"] = False
    with pytest.raises(ValueError, match="zero-return"):
        original_operator_effects(document["trace"], document["observation"], zero_returns=observed)


@pytest.fixture
def fresh_zero_example(native_zero_sources, monkeypatch):
    document, _, _, _, _ = native_zero_sources
    monkeypatch.setattr(F.automatic_fixtures, "example", lambda **kwargs: copy.deepcopy(document["trace"]))


@pytest.fixture
def issued_zero_intake(native_zero_sources, fresh_zero_example, automatic, tmp_path):
    _, argument, _, declarations, _ = native_zero_sources
    selection = F.write(
        tmp_path / "zero-selection.json",
        {
            "schema": O.ZERO_SELECTION_SCHEMA,
            "status": "reviewed",
            "software_intake_sha256": automatic["software_intake"].sha256,
            "namespace": "aten",
            "python": argument["python"],
            "canonical_source": {
                "checkout": argument["checkout"],
                "commit": argument["commit"],
                "path": str(declarations),
            },
            "tensor_arguments": {"compiler": argument["compiler"]},
            "zero_returns": {"compiler": argument["compiler"]},
        },
    )
    return O.issue_independent_operator_schema_intake(
        software=automatic["software_intake"],
        selection=selection,
        forbidden_roots=(tmp_path / "private-author-exclusions",),
        output=tmp_path / "issued-zero-returns",
    ), automatic


def test_live_intake_and_ordinary_generation_keep_unreviewed_operation_required(issued_zero_intake):
    intake, options = issued_zero_intake
    record = intake.record()
    assert record["schema"] == O.ZERO_SCHEMA
    assert record["members"][0]["unknowns"] == []
    assert len(record["members"][0]["zero_return_bindings"]) == 1
    assert "non_schema_effects_and_whole_effect_domain" in record["unknowns"]
    policy = yaml.safe_load(options["component_coverage"].read_bytes())
    policy.update(schema=A.EFFECT_POLICY_SCHEMA, operator_schema_intake_sha256=intake.sha256)
    F.write(options["component_coverage"], policy)
    options["operator_schema_intake"] = intake
    report = F.automatic_fixtures.run(options)
    A.verify(report["automatic_derivation"], report=report)
    assert report["status"] == "incomplete"
    unknowns = {(row["kind"], row["selector"]) for row in report["automatic_derivation"]["required_unknowns"]}
    assert ("operation", "aten._assert_tensor_metadata.default") in unknowns
    assert ("operator_effect", "aten._assert_tensor_metadata.default") not in unknowns
    assert ("effect_domain", "original_operator_effects") in unknowns
    assert ("resource_role", "rtl_boundary_axis_mapping") in unknowns
    members = [member for row in report["obligations"] for member in row["members"]]
    assert members
    for member in members:
        root = options["output_root"] / member["member"]
        capsule = yaml.safe_load((root / "capsule.yaml").read_bytes())
        assert (root / capsule["linalg_mlir"]).is_file()
        assert "_assert" not in (root / capsule["linalg_mlir"]).read_text()
        leaves = materialize_capsule_leaves(capsule)
        values = {
            name: [[int(leaf.data[i * leaf.shape[1] + j]) for j in range(leaf.shape[1])] for i in range(leaf.shape[0])]
            for name, leaf in leaves.items()
        }
        program = capsule["component_program"]
        for node in program["nodes"]:
            inputs = [values[name] for name in node["actual_inputs"]]
            if node["op"] == "copy":
                values[node["name"]] = copy.deepcopy(inputs[0])
            else:
                assert node["op"] == "matmul"
                a, w = inputs
                values[node["name"]] = [
                    [sum(a[i][p] * w[p][j] for p in range(len(w))) for j in range(len(w[0]))] for i in range(len(a))
                ]
        expected = {row["name"]: values[row["actual_value"]] for row in program["outputs"]}
        assert golden_store.load_golden(root)["outputs"] == expected
    missing_facet = copy.deepcopy(record)
    missing_facet["members"][0].pop("zero_returns")
    with pytest.raises(RtlIntakeRefusal, match="complete closed original"):
        O.verify_record(missing_facet)
    wrong_version = copy.deepcopy(record)
    wrong_version["schema"] = O.TENSOR_SCHEMA
    with pytest.raises(RtlIntakeRefusal, match="record schema"):
        O.verify_record(wrong_version)
    substituted = copy.deepcopy(report["automatic_derivation"])
    substituted["operator_schema_intake"]["members"][0].pop("zero_returns")
    substituted["sha256"] = F._digest({key: value for key, value in substituted.items() if key != "sha256"})
    re_signed_report = copy.deepcopy(report)
    re_signed_report["generation_identity"]["automatic_derivation_sha256"] = F._digest(substituted)
    with pytest.raises(RtlIntakeRefusal, match="complete closed original"):
        A.verify(substituted, report=re_signed_report)
