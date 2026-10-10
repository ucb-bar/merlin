"""Explicit conditional source-word relations and genuine source/selection refusals."""

import hashlib
import json
from dataclasses import replace

import pytest

from merlin.targetgen.rtl import plain_word_relation as W


def _span(path, first=1, last=None):
    if last is None:
        last = len(path.read_text().splitlines())
    return W.WordSourceSpan(path, hashlib.sha256(path.read_bytes()).hexdigest(), first, last)


def _inputs(tmp_path, body="val high = Cell(5.W)\nval mid = Cell(3.W)\nval low = Flag()"):
    declaration = tmp_path / "declaration.scala"
    declaration.write_text("class Parcel extends Packet {\n" + body + "\n}\n")
    cast = tmp_path / "cast.scala"
    cast.write_text("output.inst := input.word.retype(new Parcel())\n")
    primitive = tmp_path / "primitives.scala"
    primitive.write_text("width constructor\nfixed constructor\nexplicit width\nfixed width\n")
    packing = tmp_path / "packing.scala"
    roles = ("field_roster", "allocation_order", "field_order", "flatten", "cast", "slice")
    packing.write_text("\n".join(roles) + "\n")
    semantics = W.WordSourceSemantics(
        "Packet",
        "reverse_definition",
        "low_to_high",
        "retype",
        (
            W.WordPrimitive("Cell", "literal", "W", None, _span(primitive, 1, 1), _span(primitive, 3, 3)),
            W.WordPrimitive("Flag", "fixed", None, 1, _span(primitive, 2, 2), _span(primitive, 4, 4)),
        ),
        tuple(W.WordPackingPremise(role, _span(packing, index, index)) for index, role in enumerate(roles, 1)),
    )
    return {
        "declaration": W.WordDeclarationSelection(_span(declaration), "Parcel"),
        "cast": W.WordCastSelection(_span(cast), "input.word", "output.inst"),
        "semantics": semantics,
        "limits": W.WordRelationLimits(4096, 16384, 2048, 16, 128, 16),
    }


def test_complete_ordered_fields_cast_and_explicit_unknowns(tmp_path):
    inputs = _inputs(tmp_path)
    actual = W.observe_plain_word_relation(**inputs)
    assert actual["word_bits"] == 9
    assert actual["fields"] == [
        {"name": "high", "ordinal": 0, "constructor": "Cell", "width": 5, "low_bit": 4},
        {"name": "mid", "ordinal": 1, "constructor": "Cell", "width": 3, "low_bit": 1},
        {"name": "low", "ordinal": 2, "constructor": "Flag", "width": 1, "low_bit": 0},
    ]
    assert actual["cast"] == {"receiver": "input.word", "destination": "output.inst", "method": "retype"}
    assert len(actual["source_membership"]["files"]) == 4
    assert len(actual["source_membership"]["spans"]) == 12
    assert actual["status"] == "conditional_source_relation" and actual["capabilities_issued"] == 0
    assert "instruction_length_and_complete_executable_walk" in actual["required_unknowns"]
    assert "original_hw_word_and_field_occurrence_correspondence" in actual["required_unknowns"]
    assert "primitive_source_semantic_review" in actual["required_unknowns"]
    json.dumps(actual)


@pytest.mark.parametrize(
    ("field_order", "slice_order", "offsets"),
    [
        ("definition", "low_to_high", [0, 5, 8]),
        ("definition", "high_to_low", [4, 1, 0]),
        ("reverse_definition", "low_to_high", [4, 1, 0]),
        ("reverse_definition", "high_to_low", [0, 5, 8]),
    ],
)
def test_each_explicit_packing_rule_is_retained_as_conditional_data(tmp_path, field_order, slice_order, offsets):
    inputs = _inputs(tmp_path)
    inputs["semantics"] = replace(inputs["semantics"], field_order=field_order, slice_order=slice_order)
    actual = W.observe_plain_word_relation(**inputs)
    assert [row["low_bit"] for row in actual["fields"]] == offsets
    assert actual["selection"]["semantics"]["field_order"] == field_order
    assert actual["status"] == "conditional_source_relation"


def test_independent_names_order_widths_and_selected_fixed_width(tmp_path):
    inputs = _inputs(tmp_path, "val zeta = Cell(11.W); val alpha = Flag(); val beta = Cell(2.W)")
    primitives = inputs["semantics"].primitives
    inputs["semantics"] = replace(
        inputs["semantics"], primitives=(primitives[0], replace(primitives[1], fixed_width=4))
    )
    actual = W.observe_plain_word_relation(**inputs)
    assert actual["word_bits"] == 17
    assert [(row["name"], row["width"], row["low_bit"]) for row in actual["fields"]] == [
        ("zeta", 11, 6),
        ("alpha", 4, 2),
        ("beta", 2, 0),
    ]


@pytest.mark.parametrize("role", ["declaration", "cast", "construction", "width", "packing"])
def test_changed_full_original_source_bytes_refuse(tmp_path, role):
    inputs = _inputs(tmp_path)
    if role in {"declaration", "cast"}:
        path = inputs[role].source.path
    elif role in {"construction", "width"}:
        primitive = inputs["semantics"].primitives[0]
        path = (primitive.construction_source if role == "construction" else primitive.width_source).path
    else:
        path = inputs["semantics"].premises[0].source.path
    path.write_text(path.read_text() + "// changed full source outside selected span\n")
    with pytest.raises(W.WordRelationError, match="bytes changed"):
        W.observe_plain_word_relation(**inputs)


@pytest.mark.parametrize(
    "body",
    [
        "",
        "val x = Cell(0.W)",
        "val x = Cell(-2.W)",
        "val x = Cell(N.W)",
        "val x = Cell(4.W)\nval x = Flag()",
        "var x = Cell(4.W)",
        "val x = Missing(4.W)",
        "val x = Cell(4.W).cloneType",
        "val x = old",
        "val x = Option(Cell(4.W))",
        "val x = Cell(2.W)\ndef helper = Flag()",
        "val x = Cell(2.W)\nlaunch()",
        "val x = Cell(2.W)\nif (cond) { val y = Flag() }",
        "val x = Cell(1.W)\nval y = Cell(1.W)\n}",
    ],
)
def test_incomplete_or_computed_original_field_rosters_refuse(tmp_path, body):
    with pytest.raises(W.WordRelationError):
        W.observe_plain_word_relation(**_inputs(tmp_path, body))


@pytest.mark.parametrize(
    "source",
    [
        "class Parcel(n: Int) extends Packet { val x = Cell(2.W) }",
        "class Parcel extends Other { val x = Cell(2.W) }",
        "class Parcel extends Packet with Extra { val x = Cell(2.W) }",
        "class Parcel extends Packet { val x = Cell(2.W) }\nclass Parcel extends Packet { val y = Flag() }",
        'val text = """class Parcel extends Packet { val x = Cell(2.W) }"""',
        "/* class Parcel extends Packet { val x = Cell(2.W) } */",
    ],
)
def test_exact_plain_original_class_identity_and_lexical_membership_refuse(tmp_path, source):
    inputs = _inputs(tmp_path)
    path = inputs["declaration"].source.path
    path.write_text(source + "\n")
    inputs["declaration"] = replace(inputs["declaration"], source=_span(path))
    with pytest.raises(W.WordRelationError):
        W.observe_plain_word_relation(**inputs)


@pytest.mark.parametrize(
    "source",
    [
        "output.inst := other.word.retype(new Parcel())",
        "output.inst := input.word.retype(new Other())",
        "other.inst := input.word.retype(new Parcel())",
        "output.inst := input.word.retype(new Parcel(1))",
        "output.inst := input.word.retype(new Parcel()); launch()",
        "/* output.inst := input.word.retype(new Parcel()) */",
        'val text = "output.inst := input.word.retype(new Parcel())"',
    ],
)
def test_actual_cast_source_receiver_destination_and_return_class_refuse(tmp_path, source):
    inputs = _inputs(tmp_path)
    path = inputs["cast"].source.path
    path.write_text(source + "\n")
    inputs["cast"] = replace(inputs["cast"], source=_span(path))
    with pytest.raises(W.WordRelationError):
        W.observe_plain_word_relation(**inputs)


@pytest.mark.parametrize(
    "name,value",
    [
        ("source_bytes", 1),
        ("aggregate_source_bytes", 4),
        ("tokens", 2),
        ("fields", 2),
        ("word_bits", 8),
        ("nesting", 1),
    ],
)
def test_preallocation_and_complete_original_roster_budgets_refuse(tmp_path, name, value):
    inputs = _inputs(tmp_path)
    inputs["limits"] = replace(inputs["limits"], **{name: value})
    with pytest.raises(W.WordRelationError, match="budget"):
        W.observe_plain_word_relation(**inputs)


def test_giant_width_refuses_before_integer_conversion(tmp_path):
    inputs = _inputs(tmp_path, "val x = Cell(" + "9" * 3000 + ".W)")
    with pytest.raises(W.WordRelationError, match="width"):
        W.observe_plain_word_relation(**inputs)


@pytest.mark.parametrize(
    "change",
    [
        "missing_role",
        "duplicate_role",
        "duplicate_primitive",
        "bool_width",
        "bool_limit",
        "absent_source",
        "bad_span",
        "source_alias",
    ],
)
def test_absent_conflicting_and_scalar_alias_selections_refuse(tmp_path, change):
    inputs = _inputs(tmp_path)
    semantics = inputs["semantics"]
    if change == "missing_role":
        inputs["semantics"] = replace(semantics, premises=semantics.premises[:-1])
    elif change == "duplicate_role":
        inputs["semantics"] = replace(semantics, premises=(*semantics.premises[:-1], semantics.premises[0]))
    elif change == "duplicate_primitive":
        inputs["semantics"] = replace(semantics, primitives=(semantics.primitives[0], semantics.primitives[0]))
    elif change == "bool_width":
        inputs["semantics"] = replace(
            semantics, primitives=(semantics.primitives[0], replace(semantics.primitives[1], fixed_width=True))
        )
    elif change == "bool_limit":
        inputs["limits"] = replace(inputs["limits"], fields=True)
    elif change == "absent_source":
        inputs["declaration"].source.path.unlink()
    elif change == "bad_span":
        inputs["cast"] = replace(inputs["cast"], source=replace(inputs["cast"].source, last_line=2))
    else:
        original = inputs["declaration"].source
        alias = tmp_path / "alias.scala"
        alias.symlink_to(original.path)
        inputs["declaration"] = replace(inputs["declaration"], source=replace(original, path=alias))
    with pytest.raises(W.WordRelationError):
        W.observe_plain_word_relation(**inputs)


def test_mutation_during_actual_structural_parse_refuses(tmp_path, monkeypatch):
    inputs = _inputs(tmp_path)
    original = W._declaration
    path = inputs["cast"].source.path

    def changing(*args):
        actual = original(*args)
        path.write_text("output.inst := another.retype(new Parcel())\n")
        return actual

    monkeypatch.setattr(W, "_declaration", changing)
    with pytest.raises(W.WordRelationError, match="during observation"):
        W.observe_plain_word_relation(**inputs)


def test_comment_and_string_lookalikes_do_not_change_original_order(tmp_path):
    inputs = _inputs(
        tmp_path, "/* val phantom = Cell(99.W) */\nval high = Cell(5.W)\nval mid = Cell(3.W)\nval low = Flag()"
    )
    original = inputs["declaration"].source
    original.path.write_text(
        'val documentation = "class Parcel extends Packet { val phantom = Flag() }"\n' + original.path.read_text()
    )
    inputs["declaration"] = replace(inputs["declaration"], source=_span(original.path, 2))
    actual = W.observe_plain_word_relation(**inputs)
    assert [field["name"] for field in actual["fields"]] == ["high", "mid", "low"]
    assert actual["word_bits"] == 9


@pytest.mark.parametrize("change", ["order", "role", "numeric"])
def test_malformed_semantics_refuse_consistently(tmp_path, change):
    inputs = _inputs(tmp_path)
    semantics = inputs["semantics"]
    if change == "order":
        inputs["semantics"] = replace(semantics, field_order=[])
    elif change == "role":
        premises = (replace(semantics.premises[0], role=[]), *semantics.premises[1:])
        inputs["semantics"] = replace(semantics, premises=premises)
    else:
        primitive = replace(semantics.primitives[1], fixed_width=1.0)
        inputs["semantics"] = replace(semantics, primitives=(semantics.primitives[0], primitive))
    with pytest.raises(W.WordRelationError):
        W.observe_plain_word_relation(**inputs)
