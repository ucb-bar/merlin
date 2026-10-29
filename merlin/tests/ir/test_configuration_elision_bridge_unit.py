"""SDK-free mapping controls; generated Lean obligations are not proof outcomes."""

import json

import pytest

from merlin.common.paths import repo_root
from merlin.verify.configuration_elision_bridge import Unsupported, project
from merlin.verify.configuration_elision_emit import emit


@pytest.fixture(scope="module")
def actual():
    folder = repo_root() / "out/artifacts/verification/merlin-passes/loop-scope-20261028/NEXT-CONFIGURATION/F"
    case = folder / "corpus-v1/positives/multifield"
    data = json.loads((case / "PRODUCER.json").read_text())
    return folder, case, data


def projected(actual, final=None):
    _, case, data = actual
    return project(
        (case / "stage0.mlir").read_text(),
        final or (case / "stage1.mlir").read_text(),
        isa=data["isa_record"],
        effects=data["effects"],
    )


def test_actual_full_terms_and_all_generated_declarations(actual):
    result = projected(actual)
    before, after = (result[key]["entries"][0]["code"] for key in ("before", "after"))
    assert len(before) == 16 and len(after) == 14
    assert result["fields"] == ["configuration:0", "configuration:1"]
    assert [atom["id"] for atom in before] == list(range(16))
    assert all(atom == before[atom["id"]] for atom in after)
    assert not result["proof_authority"]
    generated = emit(result)
    assert len(generated["roots"]) == len(set(generated["roots"])) == 17
    for name in ("accepted", "complete_entries", "complete_frame", "complete_observation"):
        assert "ActualConfigurationElision." + name in generated["roots"]
    assert "Entries.Admitted" in generated["query"]
    assert "Except (Error × State Value)" in generated["query"]


@pytest.mark.parametrize("name", ["wrong_same_typed_binding", "changed_constant", "drop_observer", "drop_first_write"])
def test_actual_mutant_terms_are_never_normalized_to_positive(actual, name):
    folder, _, data = actual
    case = folder / "corpus-v1/negatives" / name
    result = project(
        (case / "stage0.mlir").read_text(),
        (case / "stage1.mlir").read_text(),
        isa=data["isa_record"],
        effects=data["effects"],
    )
    assert result != projected(actual)
    assert emit(result)["query"] != emit(projected(actual))["query"]
    assert not result["proof_authority"]  # semantic rejection is for the real checker


def test_dynamic_fields_and_nested_regions_explicitly_unsupported(actual):
    _, _, data = actual
    dynamic = """builtin.module {func.func @f(%slot:i8,%a:i32,%b:i32) {
      "configuration_fixture_isa.set"(%slot,%a,%b):(i8,i32,i32)->()
      func.return
    }}"""
    with pytest.raises(Unsupported, match="constant"):
        project(dynamic, dynamic, isa=data["isa_record"], effects=data["effects"])
    nested = 'builtin.module {func.func @f() {"opaque.region"() ({}) : () -> () func.return}}'
    with pytest.raises(Unsupported, match="nested"):
        project(nested, nested, isa=data["isa_record"], effects=data["effects"])


def test_internal_call_scope_refused_and_all_independent_entries_retained(actual):
    _, _, data = actual
    text = "builtin.module {func.func @a() {func.return} func.func @b() {func.return}}"
    result = project(text, text, isa=data["isa_record"], effects=data["effects"])
    assert [e["name"] for e in result["before"]["entries"]] == ["a", "b"]
    internal = "builtin.module {func.func @a() {func.return} func.func @b() {func.call @a():()->() func.return}}"
    with pytest.raises(Unsupported, match="call-graph"):
        project(internal, internal, isa=data["isa_record"], effects=data["effects"])


def test_all_typed_properties_locations_and_operands_are_bound(actual):
    result = projected(actual)
    for entry in result["before"]["entries"]:
        for atom in entry["code"]:
            token = json.loads(atom["token"])
            assert set(token) == {"name", "attributes", "properties", "result_types", "location", "operands", "results"}
    _, case, _ = actual
    import io

    from xdsl.dialects import builtin
    from xdsl.printer import Printer

    from merlin.verify.configuration_elision_bridge import parse

    module = parse((case / "stage1.mlir").read_text())
    module.attributes["fixture.x"] = builtin.IntegerAttr(1, builtin.i32)
    stream = io.StringIO()
    Printer(stream=stream, print_generic_format=True).print_op(module)
    changed = stream.getvalue()
    assert projected(actual, changed)["after"]["frame"] != result["before"]["frame"]


@pytest.mark.parametrize("key", ["op_name__", '\"op_name__\"', '\"op_\\6eame__\"'])
@pytest.mark.parametrize("payload", ["", '= "changed"'])
def test_reserved_raw_attribute_keys_are_refused_before_loss(key, payload):
    from merlin.verify.configuration_elision_bridge import parse
    text = 'builtin.module {func.func @f() {"opaque.effect"() {' + key + payload + '}:()->() func.return}}'
    with pytest.raises(Unsupported, match="reserved raw"):
        parse(text)


def test_reserved_marker_as_ordinary_string_value_is_preserved():
    from merlin.verify.configuration_elision_bridge import parse
    parse('builtin.module {func.func @f() {"opaque.effect"() {ordinary="op_name__"}:()->() func.return}}')
