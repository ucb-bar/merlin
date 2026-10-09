"""Coarse source production, hierarchy, precision and immutable-selection checks."""

import json
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

from merlin.targetgen.rtl import circt_introspect, source_selection
from merlin.targetgen.rtl.extract_module import extract
from merlin.targetgen.rtl.introspect import census_facts
from merlin.targetgen.rtl.source_selection import (
    SCHEMA,
    active_selection,
    circuit_root,
    derive_hierarchy,
    digest,
    firrtl_hierarchy_audit,
    load_selection,
    prepare_firtool_input,
    selected_sources,
)


class SourceSelectionTests(unittest.TestCase):
    def test_exact_hierarchy_and_memory_copy_banks(self):
        source = """FIRRTL version 3.3.0
circuit Top :%[[]]
  module Buffer : @[generators/demo/Buffer.scala 1:1]
    smem mem : UInt<16>[4] [8] @[generators/demo/Buffer.scala 3:1]
  module Buffer_1 : @[generators/demo/Buffer.scala 1:1]
    smem mem : UInt<16>[4] [8] @[generators/demo/Buffer.scala 3:1]
  module Unit : @[generators/demo/Unit.scala 1:1]
    inst buffer of Buffer
  module Unit_1 : @[generators/demo/Unit.scala 1:1]
    inst buffer of Buffer_1
  module RegisterFile : @[generators/demo/RegisterFile.scala 1:1]
    smem banks_0 : UInt<8>[4] [8] @[generators/demo/RegisterFile.scala 3:1]
    smem banks_1 : UInt<8>[4] [8] @[generators/demo/RegisterFile.scala 3:1]
  module Top : @[generators/demo/Top.scala 1:1]
    inst u0 of Unit
    inst u1 of Unit_1
    inst rf of RegisterFile
"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fir, hierarchy = root / "source.fir", root / "hierarchy.json"
            fir.write_text(source)
            hierarchy.write_text(json.dumps(derive_hierarchy(fir, "Top")))
            self.assertEqual(circuit_root(fir), "Top")
            self.assertEqual(derive_hierarchy(fir), derive_hierarchy(fir, "Top"))
            with self.assertRaises(ValueError):
                derive_hierarchy(fir, "UnrelatedConfig")
            self.assertEqual(firrtl_hierarchy_audit(fir, hierarchy)["status"], "verified")
            facts = census_facts(fir, hierarchy, generator="demo")
            memories = {row["name"]: row for row in facts["memories"]}
            self.assertEqual((memories["buffer.mem"]["banks"], memories["buffer.mem"]["copies"]), (1, 2))
            self.assertEqual(memories["registerfile.banks"]["banks"], 2)
            self.assertEqual(memories["registerfile.banks"]["bytes"], 64)

    def test_explicit_metadata_preparation_preserves_circuit(self):
        enum_class = "chisel3.experimental.EnumAnnotations$EnumComponentAnnotation"
        source = 'FIRRTL version 3.3.0\ncircuit T :%[[{"class":"' + enum_class + '"},{"class":"keep"}]]\n  module T :\n'
        prepared, count = prepare_firtool_input(source, [enum_class])
        self.assertEqual(count, 1)
        self.assertIn('"class":"keep"', prepared)
        self.assertTrue(prepared.endswith("\n  module T :\n"))
        with self.assertRaises(ValueError):
            prepare_firtool_input(source, ["semantic.annotation"])

    def test_verbatim_module_source_dependency_is_retained(self):
        source = """module {
  sv.verbatim.source private @external.v attributes {content = "module external; endmodule"}
  sv.verbatim.module private @external(out out : i32) attributes {source = @external.v}
  hw.module @Top(out result : i32) {
    %r = hw.instance "external" @external() -> (out: i32)
    hw.output %r : i32
  }
}
"""
        result, included, missing = extract(source, "Top")
        self.assertEqual(missing, [])
        self.assertIn("external.v", included)
        self.assertIn("sv.verbatim.source", result)

    def test_selection_is_exact_and_context_does_not_leak(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            member, bundle = root / "member", root / "selection.json"
            member.write_text("immutable")
            document = {
                "schema": SCHEMA,
                "target": "demo",
                "sources": {
                    role: {"path": str(member), "sha256": digest(member)}
                    for role in ("core_hw", "soc_hw", "firrtl", "hierarchy")
                },
            }
            bundle.write_text(json.dumps(document))
            selected = load_selection(bundle, target="demo")
            with selected_sources(selected):
                self.assertIs(active_selection("demo"), selected)
                with self.assertRaises(ValueError):
                    active_selection("foreign")
            self.assertIsNone(active_selection())
            member.write_text("changed")
            with self.assertRaises(ValueError):
                load_selection(bundle, target="demo")


if __name__ == "__main__":
    unittest.main()


def test_selected_generic_hw_is_generated_with_facts_not_into_source_bundle(tmp_path, monkeypatch):
    source_root = tmp_path / "selected"
    source_root.mkdir()
    member = source_root / "core.hw.mlir"
    member.write_text("module {}\n")
    bundle = source_root / "source-selection.json"
    bundle.write_text(
        json.dumps(
            {
                "schema": SCHEMA,
                "target": "demo",
                "sources": {
                    role: {"path": str(member), "sha256": digest(member)}
                    for role in ("core_hw", "soc_hw", "firrtl", "hierarchy")
                },
            }
        )
    )
    facts = tmp_path / "derived" / "facts.json"
    monkeypatch.setattr(source_selection, "production_consistency", lambda selected: {"sources": []})
    monkeypatch.setattr("merlin.integrations.modelir.discovery_imports", lambda root: nullcontext())
    monkeypatch.setattr("merlin.targetgen.rtl.mlc_bridge.mlc_dir", lambda: None)

    def inspect_selected(*args):
        selected = active_selection("demo")
        assert selected["_generic_hw_output"] == str(facts.parent / "core.hw.generic.mlir")
        assert selected["_generic_hw_output"] != str(member.with_suffix(".generic.mlir"))
        return {"inputs": {}, "generator": {}, "facts": {}}

    monkeypatch.setattr(circt_introspect, "_build_facts", inspect_selected)
    monkeypatch.setattr("merlin.targetgen.rtl.facts.write_facts_guarded", lambda out, record: None)
    circt_introspect.dump_facts(facts, target="demo", source_bundle=bundle)


def test_genericization_receipt_is_independent_of_launch_directory(tmp_path, monkeypatch):
    from merlin.targetgen.rtl import hw_graph

    monkeypatch.chdir(tmp_path)
    source, tool = Path("core.hw.mlir"), Path("circt-opt")
    source.write_text("module {}\n")
    tool.write_text("selected tool bytes\n")
    selected = {"target": "demo", "_generic_hw_output": "generated/core.generic.mlir"}
    discovery = ModuleType("mlc.discover.irgraph")
    discovery.HwGraph = lambda module: SimpleNamespace(modules={})
    discovery.to_generic = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "mlc.discover.irgraph", discovery)

    def execute(command, **kwargs):
        # Exercise the real producer, with the same relative output selection
        # that broke receipt verification in the installed Phase 0 launcher.
        assert all(Path(command[index]).is_absolute() for index in (0, 2, 4))
        Path(command[4]).write_text("module {}\n")

    monkeypatch.setattr(hw_graph.subprocess, "run", execute)
    with selected_sources(selected):
        hw_graph.load_hw_graph(source, circt_opt=tool)
    receipt = selected["_genericization"]
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    for index, member in ((0, "tool"), (2, "input"), (4, "output")):
        assert Path(receipt["command"][index]).resolve() == Path(receipt[member]["path"])
        assert digest(receipt[member]["path"]) == receipt[member]["sha256"]
