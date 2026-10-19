"""The authored default selects narrowly typed prospective host rules."""

import importlib.util
import json
from copy import deepcopy
from hashlib import sha256

import pytest
import yaml

from merlin.common import mlir_query as mq
from merlin.common.paths import repo_root
from merlin.common.tree_hash import hash_tree
from merlin.targetgen.host_capabilities import admit_host_operation, validate_host_capabilities
from merlin.targetgen.software_spec import admit_operation
from merlin.targetgen.target_experiment import HostLane, HostLaneMatrix


def _document():
    return yaml.safe_load((repo_root() / "examples/gemmini/platform/host-capabilities-attention-v2.yaml").read_bytes())


def _new_rows():
    return [
        row
        for row in _document()["operations"]
        if row["id"].startswith("host_review_attention_") or row["id"] == "host_review_i1_logical_not"
    ]


def test_existing_declaration_objects_are_preserved():
    historical = repo_root() / "examples/gemmini/platform/host-capabilities.yaml"
    assert sha256(historical.read_bytes()).hexdigest() == (
        "c504847fde594949d64d174c82ecaef1422fc74c8a409488a89e304b2de626a8"
    )
    old = [row for row in _document()["operations"] if row not in _new_rows()]
    assert len(old) == 91
    assert sha256(json.dumps(old, sort_keys=True, separators=(",", ":")).encode()).hexdigest() == (
        "a4fa6973c6be50ada44eccec4ac43d769096c9a3fd50ea11bcbbfbffa9ddb967"
    )


def test_default_descriptor_resolves_successor_without_changing_payload(tmp_path, monkeypatch):
    relative = "examples/gemmini/target/descriptor.yaml"
    descriptor = tmp_path / relative
    descriptor.parent.mkdir(parents=True)
    descriptor.write_bytes((repo_root() / relative).read_bytes())
    matrix = HostLaneMatrix.from_mapping(yaml.safe_load(descriptor.read_bytes())["host_lane"], descriptor=descriptor)
    lane = matrix.for_dtype(None)
    spec = tmp_path / lane.capability_spec
    spec.parent.mkdir(parents=True)
    spec.write_bytes((repo_root() / lane.capability_spec).read_bytes())
    # Package loading has its own tests. Exercise the real descriptor/spec
    # selection and byte pin, retaining the selected immutable payload identity.
    identity = {"package_sha256": _document()["compiler"]["package_sha256"], "dtype_strategy": "int8_w8a8"}
    monkeypatch.setattr(HostLane, "resolve", lambda self, **kwargs: (tmp_path / self.package, identity))
    path, document, observed = lane.resolve_capabilities(root=tmp_path, descriptor=descriptor)
    assert path == spec
    assert document["review"]["successor"] == "attention-host-placement-v2-20261019"
    assert observed["capability_spec_sha256"] == sha256(spec.read_bytes()).hexdigest()
    assert observed["package_sha256"] == identity["package_sha256"]
    assert len(_new_rows()) == 9


def test_real_host_view_export_pins_bytes_and_refuses_overwrite(tmp_path, monkeypatch):
    helper = repo_root() / "build_tools/scripts/derive_host_view.py"
    spec = importlib.util.spec_from_file_location("host_view_test_producer", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    descriptor = tmp_path / "descriptor.yaml"
    descriptor.write_bytes((repo_root() / "examples/gemmini/target/descriptor.yaml").read_bytes())
    selected = yaml.safe_load(descriptor.read_bytes())["host_lane"]
    package = tmp_path / selected["package"]
    package.mkdir(parents=True)
    (package / "manifest.yaml").write_text(
        "target: host\nrun_id: synthetic\nfamily: scalar_linalg\nstatus: unverified\n"
        "authoring: {mode: deterministic_generated_from_spec, generated_by_agent: false}\n"
        "inputs: {}\noutputs: {knobs: knobs.yaml}\n"
    )
    (package / "knobs.yaml").write_text(
        "backend: scalar\ndtype_strategy: int8_w8a8\ncflags: [-march=rv64gc, -mabi=lp64d]\n"
    )
    document = _document()
    document["compiler"]["package_sha256"] = hash_tree(package)["sha256"]
    capabilities = tmp_path / selected["capability_spec"]
    capabilities.parent.mkdir(parents=True)
    capabilities.write_text(yaml.safe_dump(document))
    output = tmp_path / "successor"
    receipt = module.derive(descriptor=descriptor, source_root=tmp_path, output=output)
    artifact = json.loads((output / "software/host-capabilities.json").read_bytes())
    view = artifact[receipt["default_profile"]]
    assert view["capability_spec"] == document
    assert view["package_sha256"] == hash_tree(package)["sha256"]
    assert view["capability_spec_sha256"] == sha256(capabilities.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="must be new"):
        module.derive(descriptor=descriptor, source_root=tmp_path, output=output)
    original = HostLane.resolve_capabilities

    def changed_after_read(self, **kwargs):
        result = original(self, **kwargs)
        capabilities.write_bytes(capabilities.read_bytes() + b"\n# concurrent modification\n")
        return result

    monkeypatch.setattr(HostLane, "resolve_capabilities", changed_after_read)
    with pytest.raises(ValueError, match="bytes changed"):
        module.derive(descriptor=descriptor, source_root=tmp_path, output=tmp_path / "mutated")
    assert not (tmp_path / "mutated").exists()
    monkeypatch.setattr(HostLane, "resolve_capabilities", original)
    calls = 0

    def changed_before_publication(self, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            knobs = package / "knobs.yaml"
            knobs.write_bytes(knobs.read_bytes() + b"\n# changed payload\n")
        return original(self, **kwargs)

    monkeypatch.setattr(HostLane, "resolve_capabilities", changed_before_publication)
    with pytest.raises(ValueError, match="different compiler package"):
        module.derive(descriptor=descriptor, source_root=tmp_path, output=tmp_path / "changed-package")
    assert not (tmp_path / "changed-package").exists()


@pytest.mark.parametrize("index", range(9))
def test_successor_signatures_are_exact_and_not_numerical_qualification(index):
    row = _new_rows()[index]
    document = {**_document(), "operations": [row]}
    validate_host_capabilities(document)
    signature = deepcopy(row["signature"])
    signature["rank"] = signature.pop("ranks")[0]
    assert admit_operation(document, row["ops"][0], signature, "host")["status"] == "admitted"
    assert row["numerical_contract"]["status"] == "unreviewed"
    signature["rank"] = 2
    assert admit_operation(document, row["ops"][0], signature, "host")["status"] == "unsupported"
    signature = deepcopy(row["signature"])
    signature["rank"] = signature.pop("ranks")[0]
    signature["ordered_operand_dtypes"][0] = "f64"
    assert admit_operation(document, row["ops"][0], signature, "host")["status"] == "unsupported"


def _unary(source_dtype, result_dtype, body):
    return next(
        mq.walk(
            mq.parse(f"""builtin.module {{
      func.func @forward(%x: tensor<2x3x4x5x{source_dtype}>, %init: tensor<2x3x4x5x{result_dtype}>)
          -> tensor<2x3x4x5x{result_dtype}> {{
        %out = "linalg.generic"(%x, %init) <{{
          indexing_maps = [affine_map<(a,b,c,d)->(a,b,c,d)>, affine_map<(a,b,c,d)->(a,b,c,d)>],
          iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>,
                            #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>],
          operandSegmentSizes = array<i32: 1, 1>}}> ({{
          ^bb0(%value: {source_dtype}, %acc: {result_dtype}):
            {body}
            "linalg.yield"(%result) : ({result_dtype}) -> ()
        }}) : (tensor<2x3x4x5x{source_dtype}>, tensor<2x3x4x5x{result_dtype}>) -> tensor<2x3x4x5x{result_dtype}>
        func.return %out : tensor<2x3x4x5x{result_dtype}>
      }}
    }}"""),
            "linalg.generic",
        )
    )


@pytest.mark.parametrize(
    ("rule", "source_dtype", "result_dtype", "body"),
    [
        ("host_review_attention_exp_f32", "f32", "f32", '%result = "math.exp"(%value) : (f32) -> f32'),
        ("host_review_attention_roundeven_f32", "f32", "f32", '%result = "math.roundeven"(%value) : (f32) -> f32'),
        ("host_review_attention_i32_to_f32", "i32", "f32", '%result = "arith.sitofp"(%value) : (i32) -> f32'),
        (
            "host_review_i1_logical_not",
            "i1",
            "i1",
            '%one = "arith.constant"() <{value = true}> : () -> i1\n'
            '%result = "arith.xori"(%value, %one) : (i1, i1) -> i1',
        ),
    ],
)
def test_selected_closed_sources_pass_and_accumulator_substitution_refuses(rule, source_dtype, result_dtype, body):
    declaration = next(row for row in _new_rows() if row["id"] == rule)
    document = {**_document(), "operations": [declaration]}
    selected = {
        "host": {
            "capability_spec": document,
            "package_sha256": document["compiler"]["package_sha256"],
            "capability_spec_sha256": "a" * 64,
            "dtype_strategy": "int8_w8a8",
        }
    }
    row = {"frontend_op": declaration["ops"][0], "mlir_operation": "linalg.generic", "count": 1}
    signature = {**declaration["signature"], "rank": 4}
    signature.pop("ranks")
    assert (
        admit_host_operation(selected, row, signature, source_operations=(_unary(source_dtype, result_dtype, body),))[
            "status"
        ]
        == "admitted"
    )
    if source_dtype == result_dtype:
        wrong = _unary(source_dtype, result_dtype, body.replace("(%value", "(%acc"))
        assert admit_host_operation(selected, row, signature, source_operations=(wrong,))["status"] == "unsupported"
