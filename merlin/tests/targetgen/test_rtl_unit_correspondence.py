"""Accelerator-unit structural correspondence across two SoC configurations."""

import json

import pytest

from merlin.targetgen.rtl import unit_correspondence as uc
from merlin.targetgen.rtl.source_selection import SCHEMA, digest


def _generic(*, path="a.mlir", hint=True, gate="comb.xor", anchor=None):
    hint_attr = ' {sv.namehint = "flip"}' if hint else ""
    anchor_op = (
        f'  "hw.hierpath"() {{namepath = [#hw.innerNameRef<@{anchor}::@sym>, @Other], '
        'sym_name = "nla", sym_visibility = "private"} : () -> ()\n'
        if anchor
        else ""
    )
    return (
        f'#loc1 = loc("{path}":3:5)\n'
        f'#loc2 = loc("{path}":9:5)\n'
        '"builtin.module"() ({\n' + anchor_op + '  "hw.module"() ({\n'
        "  ^bb0(%arg0: i1, %arg1: i1):\n"
        f'    %0 = "{gate}"(%arg0, %arg1){hint_attr} : (i1, i1) -> i1\n'
        '    "hw.output"(%0) : (i1) -> ()\n'
        "  }) {module_type = !hw.modty<input a : i1, input b : i1, output y : i1>, parameters = [], "
        'result_locs = [#loc1], sym_name = "Leaf", sym_visibility = "private"} : () -> ()\n'
        '  "hw.module"() ({\n'
        "  ^bb0(%arg2: i1):\n"
        '    %1 = "hw.instance"(%arg2, %arg2) {argNames = ["a", "b"], instanceName = "leaf", '
        'moduleName = @Leaf, parameters = [], resultNames = ["y"]} : (i1, i1) -> i1\n'
        '    "hw.output"(%1) : (i1) -> ()\n'
        "  }) {module_type = !hw.modty<input x : i1, output z : i1>, parameters = [], "
        'result_locs = [#loc2], sym_name = "Unit"} : () -> ()\n'
        "}) : () -> ()\n"
    )


_FIRRTL = """FIRRTL version 3.3.0
circuit Top :%[[]]
  module Leaf : @[generators/accel/Leaf.scala 1:1]
    input a : UInt<1>
    {body}
  module Unit : @[generators/accel/Unit.scala 1:1]
    inst leaf of Leaf
  module Top : @[generators/soc/Top.scala 1:1]
    inst unit of Unit
"""


def test_locations_never_count_and_name_hints_are_the_only_ignored_attribute():
    base, _ = uc.hw_unit_view(_generic(path="one.mlir"))
    moved, _ = uc.hw_unit_view(_generic(path="elsewhere/two.mlir"))
    unhinted, _ = uc.hw_unit_view(_generic(hint=False))
    changed, _ = uc.hw_unit_view(_generic(gate="comb.and"))
    assert base["Leaf"]["exact"] == moved["Leaf"]["exact"]
    assert base["Unit"]["exact"] == moved["Unit"]["exact"]
    assert base["Leaf"]["exact"] != unhinted["Leaf"]["exact"]
    assert base["Leaf"]["structural"] == unhinted["Leaf"]["structural"]
    assert base["Leaf"]["structural"] != changed["Leaf"]["structural"]


def test_file_level_operations_report_their_module_symbols():
    _, rows = uc.hw_unit_view(_generic(anchor="Cache"))
    assert [(row["operation"], "Cache" in row["symbols"], "Other" in row["symbols"]) for row in rows] == [
        ("hw.hierpath", True, True)
    ]


def test_firrtl_bodies_are_exact_and_selected(tmp_path):
    fir = tmp_path / "x.fir"
    fir.write_text(_FIRRTL.format(body="output y : UInt<1>"))
    bodies = uc.firrtl_module_bodies(fir, {"Leaf", "Unit"})
    assert set(bodies) == {"Leaf", "Unit"}
    assert bodies["Unit"] == "  module Unit : @[generators/accel/Unit.scala 1:1]\n    inst leaf of Leaf\n"


def _facts(root, label, *, generic, body="output y : UInt<1>"):
    side = root / label
    side.mkdir()
    fir, hierarchy, core, soc, generic_path = (
        side / name for name in ("x.fir", "h.json", "c.mlir", "s.mlir", "g.mlir")
    )
    fir.write_text(_FIRRTL.format(body=body))
    hierarchy.write_text("{}")
    core.write_text(label)
    soc.write_text(label)
    generic_path.write_text(generic)
    selection = side / "source-selection.json"
    selection.write_text(
        json.dumps(
            {
                "schema": SCHEMA,
                "target": "accel",
                "config": f"{label.capitalize()}Config",
                "generator": "accel",
                "sources": {
                    role: {"path": str(path), "sha256": digest(path)}
                    for role, path in (("firrtl", fir), ("hierarchy", hierarchy), ("core_hw", core), ("soc_hw", soc))
                },
                "production": {"core_root": "Unit", "included_modules": ["Unit", "Leaf"], "tool": None},
            }
        )
    )
    return {
        "inputs": {"source_bundle_path": str(selection), "source_bundle_sha256": digest(selection)},
        "source_consistency": {
            "status": "verified",
            "genericization": {"output": {"path": str(generic_path), "sha256": digest(generic_path)}},
        },
    }


def test_unit_correspondence_separates_soc_anchors_from_unit_structure(tmp_path):
    target = _facts(tmp_path, "target", generic=_generic(path="t.mlir", hint=False, anchor="Cache"))
    simulation = _facts(tmp_path, "simulation", generic=_generic(path="s.mlir"))
    record = uc.compare_unit(target, simulation)
    assert record["status"] == "structurally_equivalent"
    assert record["configs"]["target"]["config"] == "TargetConfig"
    assert record["firrtl_module_bodies"] == {**record["firrtl_module_bodies"], "compared": 2, "differ": []}
    assert [row["module"] for row in record["hw_modules"]["differences"]] == ["Leaf"]
    assert record["hw_modules"]["differences"][0]["kind"] == "name_hints_only"
    assert [row["unit_modules_referenced"] for row in record["hw_file_level_operations"]["only_target"]] == [[]]

    same = uc.compare_unit(
        _facts(tmp_path, "a", generic=_generic(path="a.mlir")), _facts(tmp_path, "b", generic=_generic(path="b.mlir"))
    )
    assert same["status"] == "identical"


@pytest.mark.parametrize(
    "change",
    [{"generic": _generic(gate="comb.and")}, {"generic": _generic(anchor="Leaf")}, {"body": "output y : UInt<2>"}],
)
def test_unit_differences_are_never_equivalent(tmp_path, change):
    target = _facts(tmp_path, "target", **{"generic": _generic(), **change})
    simulation = _facts(tmp_path, "simulation", generic=_generic())
    assert uc.compare_unit(target, simulation)["status"] == "different"


def test_unverified_sources_refuse(tmp_path):
    target = _facts(tmp_path, "target", generic=_generic())
    target["source_consistency"]["status"] = "unverified"
    with pytest.raises(ValueError, match="verified source consistency"):
        uc.compare_unit(target, _facts(tmp_path, "simulation", generic=_generic()))
