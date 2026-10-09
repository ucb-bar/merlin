"""Real canonical schemas drive bounded original-source logical alias cases.

Native tests require explicitly selected public framework/capture sources;
absence is an explicit skip, never replacement observations or schema tables.
"""

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
from merlin_experiments.phase0.rtl_intake import RtlIntakeRefusal

from merlin.common import invocation_record as I
from merlin.common.paths import module_source_path
from merlin.targetgen.capsule_inputs import materialize_capsule_leaves
from merlin.targetgen.frontend_operator_effects import original_operator_effects, original_tensor_argument_requests
from merlin.targetgen.frontend_trace import _digest

_fixture_path = Path(__file__).with_name("test_component_automatic.py")
_fixture_spec = importlib.util.spec_from_file_location("private_automatic_source_fixtures", _fixture_path)
automatic_fixtures = importlib.util.module_from_spec(_fixture_spec)
_fixture_spec.loader.exec_module(automatic_fixtures)

automatic = automatic_fixtures.automatic
independent = automatic_fixtures.independent
selected = automatic_fixtures.selected
write = automatic_fixtures.write


@pytest.fixture(scope="module")
def native_sources(tmp_path_factory):
    names = ("MERLIN_TEST_TORCH_PYTHON", "MERLIN_TEST_M2M_ROOT", "MERLIN_TEST_OPERATOR_DECLARATIONS")
    if any(not os.environ.get(name) for name in names):
        pytest.skip("native schema controls need explicit protected public Python/capture/declaration sources")
    python, capture_root, declarations = (Path(os.environ[name]).absolute() for name in names)
    destination = tmp_path_factory.mktemp("actual-schema-source")
    script = destination / "capture.py"
    script.write_text("""import sys,json,importlib.util
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import torch
from m2m.capture.trace import snapshot_exported_program
spec=importlib.util.spec_from_file_location("schema_observer",sys.argv[2])
observer=importlib.util.module_from_spec(spec);spec.loader.exec_module(observer)
class Model(torch.nn.Module):
 def forward(self,A,W0,W1):
  return A@W0,A@W1,A.clone(),A.reshape(2,3)
class Mutation(torch.nn.Module):
 def forward(self,A,B):
  return torch.ops.aten.copy_.default(A,B)
class Scalars(torch.nn.Module):
 def forward(self,A,W0,W1):
  return (A@W0,A@W1,A.clone(),A.reshape(2,3),
          torch.ops.aten.div.Tensor(A,2.0),torch.ops.aten.mul.Tensor(A,2),
          torch.ops.aten.mul.Tensor(A,True))
out={}
cases=[("alias",Model(),(torch.arange(6,dtype=torch.int8).reshape(2,3),
        torch.ones(3,2,dtype=torch.int8),torch.ones(3,2,dtype=torch.int8))),
       ("mutation",Mutation(),(torch.ones(2,3,dtype=torch.int8),torch.zeros(2,3,dtype=torch.int8))),
       ("scalars",Scalars(),(torch.arange(6,dtype=torch.int8).reshape(2,3),
        torch.ones(3,2,dtype=torch.int8),torch.ones(3,2,dtype=torch.int8)))]
for name,model,inputs in cases:
 graph=snapshot_exported_program(torch.export.export(model,inputs),stage="original")
 operations=sorted({node["target"] for node in graph["nodes"] if node["op"]=="call_function"})
 request={"namespace":"aten","captured_schemas":graph["operator_schemas"],"operations":operations}
 observation=observer.observe(request,declarations=Path(sys.argv[3]).read_bytes())
 out[name]={"trace":{"schema":"m2m.frontend_trace.v1","graphs":{"original":graph}},"observation":observation}
print(json.dumps(out,sort_keys=True))
""")
    result = I.run(
        [
            str(python),
            "-I",
            str(script),
            str(capture_root),
            str(module_source_path("merlin.targetgen.torch_schema_observer")),
            str(declarations),
        ],
        directory=destination,
        stage="actual_original_schema_controls",
        inputs=(script, declarations, module_source_path("merlin.targetgen.torch_schema_observer")),
        env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"},
        capture_output=True,
        timeout=60,
    )
    result.check_returncode()
    return json.loads(result.stdout), python, declarations


def resign(trace):
    graph = trace["graphs"]["original"]
    graph["sha256"] = _digest({key: value for key, value in graph.items() if key != "sha256"})


def test_actual_capture_canonical_schema_and_registered_runtime_bind_alias(native_sources):
    documents, _, _ = native_sources
    source = documents["alias"]
    observed = original_operator_effects(source["trace"], source["observation"])
    assert observed.effect_classes == ("may_alias_result",)
    assert observed.unknowns() == []
    (witness,) = observed.witnesses()
    graph = source["trace"]["graphs"]["original"]
    call = next(node for node in graph["nodes"] if node["id"] == witness["node"])
    assert witness["input_value"] == call["args"][0]["value_id"]
    assert witness["result_value"] == call["results"][0]["id"]
    assert witness["argument_path"] == "args/0"
    assert set(observed.public_semantics()) == {"graph_sha256", "effect_classes"}
    assert "physical" not in observed.effect_classes


def test_actual_source_mutation_is_mandatory_information_without_an_arithmetic_permission(native_sources):
    documents, _, _ = native_sources
    source = documents["mutation"]
    observed = original_operator_effects(source["trace"], source["observation"])
    assert "may_write_argument" in observed.effect_classes
    assert any(row["kind"] == "may_write_argument" and row["argument_path"] == "args/0" for row in observed.witnesses())


def test_actual_native_observer_refuses_a_mismatched_captured_schema(native_sources, tmp_path):
    documents, python, declarations = native_sources
    graph = documents["alias"]["trace"]["graphs"]["original"]
    schemas = dict(graph["operator_schemas"])
    schemas["aten.reshape.default"] = schemas["aten.clone.default"]
    request = write(
        tmp_path / "mismatched-capture.json",
        {"namespace": "aten", "captured_schemas": schemas, "operations": sorted(schemas)},
    )
    observer = module_source_path("merlin.targetgen.torch_schema_observer")
    result = I.run(
        [str(python), "-I", str(observer), str(request), str(declarations)],
        directory=tmp_path,
        stage="mismatched_original_schema_control",
        inputs=(observer, request, declarations),
        env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"},
        capture_output=True,
        timeout=60,
    )
    result.check_returncode()
    row = next(row for row in json.loads(result.stdout)["rows"] if row["target"] == "aten.reshape.default")
    assert row["status"] == "unknown"
    assert "canonical public source declaration" in row["reason"]


@pytest.mark.parametrize(
    "defect",
    ["captured_schema", "missing_schema", "required_argument", "tuple_return", "wildcard_alias", "alias_transition"],
)
def test_original_or_schema_defects_cannot_mint_alias_relations(native_sources, defect):
    documents, _, _ = native_sources
    source = copy.deepcopy(documents["alias"])
    graph = source["trace"]["graphs"]["original"]
    call = next(node for node in graph["nodes"] if node["target"] == "aten.reshape.default")
    row = next(row for row in source["observation"]["rows"] if row["target"] == call["target"])
    if defect == "captured_schema":
        graph["operator_schemas"][call["target"]] = "different"
    elif defect == "missing_schema":
        graph["operator_schemas"].pop(call["target"])
    elif defect == "required_argument":
        call["args"].pop()
    elif defect == "tuple_return":
        row["returns"][0]["type"] = "List[Tensor]"
    elif defect == "wildcard_alias":
        row["arguments"][0]["alias"]["before"] = ["*"]
        row["arguments"][0]["alias"]["after"] = ["*"]
    else:
        row["arguments"][0]["alias"]["after"] = ["other"]
    resign(source["trace"])
    observed = original_operator_effects(source["trace"], source["observation"])
    assert observed.effect_classes == ()
    assert observed.unknowns()


@pytest.fixture
def effect_generation(native_sources, monkeypatch, request, tmp_path):
    documents, python, declarations = native_sources
    monkeypatch.setattr(automatic_fixtures, "example", lambda **kwargs: copy.deepcopy(documents["alias"]["trace"]))
    options = request.getfixturevalue("automatic")
    # Exact tracked fixture selection uses actual observed public bytes. It
    # does not attest the configured URL or historical installed-library build.
    checkout = tmp_path / "public-declarations"
    checkout.mkdir()
    source = checkout / "native_functions.yaml"
    source.write_bytes(declarations.read_bytes())
    for args in (
        ("init",),
        ("config", "user.name", "Independent fixture"),
        ("config", "user.email", "fixture@example.invalid"),
        ("remote", "add", "origin", "https://example.invalid/public-schema-fixture"),
        ("add", "native_functions.yaml"),
        ("commit", "-m", "Pin public declaration bytes"),
    ):
        subprocess.run(["git", "-C", str(checkout), *args], check=True, capture_output=True)
    commit = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"]).decode().strip()
    selection = write(
        tmp_path / "schema-selection.json",
        {
            "schema": O.SELECTION_SCHEMA,
            "status": "reviewed",
            "software_intake_sha256": options["software_intake"].sha256,
            "namespace": "aten",
            "python": str(python),
            "canonical_source": {"checkout": str(checkout), "commit": commit, "path": str(source)},
        },
    )
    forbidden = tmp_path / "issued-forbidden"
    forbidden.mkdir()
    intake = O.issue_independent_operator_schema_intake(
        software=options["software_intake"],
        selection=selection,
        forbidden_roots=(forbidden,),
        output=tmp_path / "issued-schemas",
    )
    policy = yaml.safe_load(options["component_coverage"].read_bytes())
    policy.update(schema=A.EFFECT_POLICY_SCHEMA, operator_schema_intake_sha256=intake.sha256)
    write(options["component_coverage"], policy)
    options["operator_schema_intake"] = intake
    return options


def test_actual_normal_bounded_generation_selects_complete_logical_alias_source(effect_generation):
    report = automatic_fixtures.run(effect_generation)
    A.verify(report["automatic_derivation"], report=report)
    assert report["status"] == "incomplete"
    rows = [row for row in report["obligations"] if row["id"].startswith("auto_may_alias_result_")]
    assert {row["cohort"] for row in rows} == {"functional_guard", "withheld_transfer"}
    for row in rows:
        assert row["state"] == "generated"
        for member in row["members"]:
            root = effect_generation["output_root"] / member["member"]
            capsule = yaml.safe_load((root / "capsule.yaml").read_bytes())
            source = (root / capsule["linalg_mlir"]).read_text()
            assert "linalg.copy" in source
            assert "return %A, %A," in source
            leaves = materialize_capsule_leaves(capsule)
            from merlin.targetgen import golden_store

            expected = golden_store.load_golden(root)["outputs"]
            assert set(expected) == {"Yinput", "Yview", "Ycopy"}
            a = leaves["A"]
            independently_read_input = [
                [int(a.data[i * a.shape[1] + j]) for j in range(a.shape[1])] for i in range(a.shape[0])
            ]
            for output in expected.values():
                assert output == independently_read_input
            assert max(leaves["A"].shape) <= 3
    assert ("physical_effect", "may_alias_result") in {
        (row["kind"], row["selector"]) for row in report["automatic_derivation"]["required_unknowns"]
    }
    assert any(
        row["selector"] == "original_operator_effects" for row in report["automatic_derivation"]["required_unknowns"]
    )


def test_saved_or_changed_schema_records_cannot_recreate_live_authority(effect_generation):
    intake = effect_generation["operator_schema_intake"]
    forged = O.IndependentOperatorSchemaIntake(intake.software, intake.source_pins, intake.receipt_json)
    with pytest.raises(RtlIntakeRefusal, match="live independent issuance"):
        forged.verify()
    native = next(
        pin
        for pin in intake.source_pins
        if pin.role == "operator-schema-native-evidence" and Path(pin.path).name.startswith("observation-")
    )
    Path(native.path).write_text("{}")
    with pytest.raises(RtlIntakeRefusal, match="source changed"):
        intake.verify()


def test_versioned_effect_policy_requires_actual_live_schema_input(effect_generation):
    effect_generation.pop("operator_schema_intake")
    with pytest.raises(ValueError, match="identical live independent schema"):
        automatic_fixtures.generation.generate_target("fixture", **effect_generation)


@pytest.fixture(scope="module")
def tensor_arguments(native_sources, tmp_path_factory):
    names = ("MERLIN_TEST_TORCH_SOURCE_ROOT", "MERLIN_TEST_HOST_CXX")
    if any(not os.environ.get(name) for name in names):
        pytest.skip("native Tensor conversion needs explicit public source checkout and compiler")
    documents, python, _ = native_sources
    checkout, compiler = (Path(os.environ[name]).absolute() for name in names)
    commit = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"]).decode().strip()
    owner = tmp_path_factory.mktemp("actual-native-tensor-arguments")
    forbidden = owner / "absent-private-prefix"
    getter = T.prepare_getter(
        python=python,
        compiler=compiler,
        checkout=checkout,
        commit=commit,
        forbidden=(forbidden,),
        output=owner / "getter",
    )
    T.verify_getter(getter)
    member = T.observe_arguments(
        trace=documents["scalars"]["trace"],
        schema_observation=documents["scalars"]["observation"],
        getter=getter,
        output=owner,
    )
    observation = T.verify_arguments(
        trace=documents["scalars"]["trace"],
        schema_observation=documents["scalars"]["observation"],
        getter=getter,
        member=member,
    )
    return documents["scalars"], observation, getter, member


def test_actual_original_tensor_scalar_bindings_preserve_literal_and_wrapped_relation(tensor_arguments):
    source, observation, _, _ = tensor_arguments
    legacy = original_operator_effects(source["trace"], source["observation"])
    assert {row["target"] for row in legacy.unknowns()} == {"aten.div.Tensor", "aten.mul.Tensor"}
    actual = original_operator_effects(source["trace"], source["observation"], tensor_arguments=observation)
    assert actual.unknowns() == []
    assert actual.effect_classes == ("may_alias_result",)
    bindings = actual.tensor_bindings()
    assert len(bindings) == 3
    assert {row["request"]["literal"]["type"] for row in bindings} == {"bool", "int", "float"}
    assert all(row["native"]["source_allows_number"] and row["native"]["wrapped_number"] for row in bindings)
    assert all(row["native"]["disjoint_from_prior_live_boxes"] for row in bindings)
    # No original literal is rewritten into a source SSA value or Tensor.
    request = original_tensor_argument_requests(source["trace"], source["observation"])
    for row in request["rows"]:
        call = next(node for node in source["trace"]["graphs"]["original"]["nodes"] if node["id"] == row["node"])
        assert not isinstance(call["args"][row["argument_index"]], dict)


@pytest.mark.parametrize("defect", ["literal", "slot", "schema", "guard", "wrapped", "shape", "roster"])
def test_scalar_binding_substitution_cannot_complete_original_argument_roster(tensor_arguments, defect):
    source, observation, _, _ = tensor_arguments
    altered = copy.deepcopy(observation)
    row = altered["rows"][0]
    if defect == "literal":
        row["request"]["literal"] = {"type": "float", "value_hex": float(3).hex()}
    elif defect == "slot":
        row["request"]["argument_index"] = 0
    elif defect == "schema":
        row["native"]["schema"] = "different"
    elif defect == "guard":
        row["native"]["source_allows_number"] = False
    elif defect == "wrapped":
        row["native"]["wrapped_number"] = False
    elif defect == "shape":
        row["native"]["shape"] = [1]
    else:
        altered["rows"].pop()
    with pytest.raises(ValueError, match="Tensor"):
        original_operator_effects(source["trace"], source["observation"], tensor_arguments=altered)


def test_public_guard_is_not_enabled_for_arbitrary_tensor_operator(tensor_arguments, tmp_path):
    source, _, getter, _ = tensor_arguments
    trace = copy.deepcopy(source["trace"])
    graph = trace["graphs"]["original"]
    call = next(node for node in graph["nodes"] if node["target"] == "aten.matmul.default")
    old = call["args"][1]
    graph["edges"] = [
        row
        for row in graph["edges"]
        if not (row["consumer_node_id"] == call["id"] and row["argument_path"] == "args/1")
    ]
    assert old["value_id"]
    call["args"][1] = 2
    resign(trace)
    member = T.observe_arguments(trace=trace, schema_observation=source["observation"], getter=getter, output=tmp_path)
    actual = T.verify_arguments(trace=trace, schema_observation=source["observation"], getter=getter, member=member)
    effects = original_operator_effects(trace, source["observation"], tensor_arguments=actual)
    assert any(row["target"] == "aten.matmul.default" and "unobserved" in row["reason"] for row in effects.unknowns())
    assert (
        next(row for row in actual["rows"] if row["request"]["target"] == "aten.matmul.default")["status"] == "unknown"
    )


def test_actual_wrapped_scalar_promotion_cannot_be_replaced_by_ordinary_zero_dim_tensor(tensor_arguments, tmp_path):
    _, _, getter, _ = tensor_arguments
    script = tmp_path / "promotion.py"
    script.write_text("""import sys,json,importlib.util
import torch
spec=importlib.util.spec_from_file_location("native_tensor_argument_getter",sys.argv[1])
getter=importlib.util.module_from_spec(spec);spec.loader.exec_module(getter)
rows=[];live=[]
for target in (torch.ops.aten.div.Tensor,torch.ops.aten.mul.Tensor):
 for dtype in (torch.int8,torch.float32):
  source=torch.arange(1,5,dtype=dtype).reshape(2,2)
  for value in (2,2.0,True):
   native=getter.observe(target._schema.name,target._schema.overload_name,1,value)
   boxed=native["tensor"];direct=getter.observe(target._schema.name,target._schema.overload_name,0,source)
   assert native["source_allows_number"] and native["wrapped_number"] and boxed.shape==torch.Size([])
   assert torch._C._is_alias_of(direct["tensor"],source) and not direct["wrapped_number"]
   assert not torch._C._is_alias_of(source,boxed)
   assert all(not torch._C._is_alias_of(boxed,prior) for prior in live)
   live.append(boxed)
   original=target(source,value);wrapped=target(source,boxed);ordinary=target(source,boxed.clone())
   assert original.dtype==wrapped.dtype and torch.equal(original,wrapped)
   assert torch.equal(source,torch.arange(1,5,dtype=dtype).reshape(2,2))
   rows.append({"target":str(target),"source_dtype":str(dtype),"literal_type":type(value).__name__,
     "literal_output_dtype":str(original.dtype),"wrapped_output_dtype":str(wrapped.dtype),
     "ordinary_tensor_output_dtype":str(ordinary.dtype)})
for value in (None,"2",[2]):
 try:getter.observe("aten::mul","Tensor",1,value)
 except (RuntimeError,ValueError,TypeError):pass
 else:raise AssertionError("unsupported argument unexpectedly accepted")
print(json.dumps(rows,sort_keys=True))
""")
    result = I.run(
        [getter["python"], "-I", str(script), getter["getter"]],
        directory=tmp_path,
        stage="actual_wrapped_scalar_promotion_controls",
        inputs=(script, Path(getter["getter"])),
        env=T.ENVIRONMENT,
        capture_output=True,
        timeout=60,
    )
    result.check_returncode()
    rows = json.loads(result.stdout)
    assert len(rows) == 12
    differences = [row for row in rows if row["literal_output_dtype"] != row["ordinary_tensor_output_dtype"]]
    assert len(differences) == 2
    assert all(row["source_dtype"] == "torch.int8" and row["literal_type"] == "float" for row in differences)


def test_compiler_dependency_parent_spellings_preserve_symlink_refusal(tmp_path):
    header = tmp_path / "include" / "value.h"
    header.parent.mkdir()
    header.write_text("// public fixture\n")
    nested = header.parent / "nested"
    nested.mkdir()
    assert T._dependency_path(nested / ".." / header.name) == header
    link = header.parent / "indirect"
    link.symlink_to(nested, target_is_directory=True)
    with pytest.raises(RtlIntakeRefusal, match="symlink"):
        T._dependency_path(link / ".." / header.name)


def test_v2_live_intake_reaches_ordinary_generation_without_numeric_owner_grants(
    native_sources,
    tensor_arguments,
    monkeypatch,
    request,
    tmp_path,
):
    source, _, getter, _ = tensor_arguments
    _, python, declarations = native_sources
    monkeypatch.setattr(automatic_fixtures, "example", lambda **kwargs: copy.deepcopy(source["trace"]))
    options = request.getfixturevalue("automatic")
    selection = write(
        tmp_path / "v2-selection.json",
        {
            "schema": O.TENSOR_SELECTION_SCHEMA,
            "status": "reviewed",
            "software_intake_sha256": options["software_intake"].sha256,
            "namespace": "aten",
            "python": str(python),
            "canonical_source": {"checkout": getter["checkout"], "commit": getter["commit"], "path": str(declarations)},
            "tensor_arguments": {"compiler": getter["compiler"]},
        },
    )
    intake = O.issue_independent_operator_schema_intake(
        software=options["software_intake"],
        selection=selection,
        forbidden_roots=(tmp_path / "absent-private-prefix",),
        output=tmp_path / "v2-issued",
    )
    record = intake.record()
    assert record["schema"] == O.TENSOR_SCHEMA
    assert len(record["members"][0]["tensor_bindings"]) == 3
    policy = yaml.safe_load(options["component_coverage"].read_bytes())
    policy.update(schema=A.LOGICAL_POLICY_SCHEMA, operator_schema_intake_sha256=intake.sha256)
    write(options["component_coverage"], policy)
    options["operator_schema_intake"] = intake
    report = automatic_fixtures.run(options)
    A.verify(report["automatic_derivation"], report=report)
    unknowns = {(row["kind"], row["selector"]) for row in report["automatic_derivation"]["required_unknowns"]}
    assert ("operation", "aten.div.Tensor") in unknowns and ("operation", "aten.mul.Tensor") in unknowns
    assert ("operator_effect", "aten.div.Tensor") not in unknowns
    assert ("operator_effect", "aten.mul.Tensor") not in unknowns
    assert ("effect_domain", "original_operator_effects") in unknowns
    assert ("resource_role", "rtl_boundary_axis_mapping") in unknowns
    for obligation in report["obligations"]:
        for member in obligation["members"]:
            capsule_root = options["output_root"] / member["member"]
            capsule = yaml.safe_load((capsule_root / "capsule.yaml").read_bytes())
            assert "div" not in (capsule_root / capsule["linalg_mlir"]).read_text()
    forged = O.IndependentOperatorSchemaIntake(intake.software, intake.source_pins, intake.receipt_json)
    with pytest.raises(RtlIntakeRefusal, match="live independent issuance"):
        forged.verify()
    altered = copy.deepcopy(record)
    altered["schema"] = O.SCHEMA
    altered.pop("tensor_argument_getter")
    with pytest.raises(RtlIntakeRefusal, match="version"):
        O.verify_record(altered)
