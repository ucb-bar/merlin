"""Public guidance describes exact selected code, never target support or authority."""

import json
import sys
from dataclasses import dataclass

import pytest

from merlin_experiments.phase2 import compiler_libraries as L
from merlin_experiments.phase2 import compiler_library_guidance as G

from merlin.targetgen.compiler_library import CompilerLibrarySelection, freeze_compiler_library, scoped_compiler_library


def selection(tmp_path):
    root = tmp_path / "reviewed"
    (root / "merlin").mkdir(parents=True)
    (root / "merlin/__init__.py").write_text("")
    (root / "merlin/portable.py").write_text('''from __future__ import annotations
from math import ceil as round_extent
raise RuntimeError("inventory must never import this implementation")
class TilePolicy:
    """Select legal tile alternatives from explicit resources."""
    def __init__(self, capacity: int): pass
    def choose(self, m: int, *, limit: int = 8):
        """Return a legal plan; independent emission checking is still required."""
        return min(m, limit)
    def _private(self): pass
def public_transform(plan): return plan
def _private_transform(plan): return plan
''')
    (root / "private-answer.txt").write_text("protected expected output must not enter guidance")
    contract = freeze_compiler_library(
        root,
        review_id="public-inventory-test",
        public_modules=("merlin.portable",),
        sources=(("merlin/__init__.py", "merlin"), ("merlin/portable.py", "merlin.portable")),
    )
    return CompilerLibrarySelection(contract, root)


def test_inventory_uses_exact_ast_source_without_importing_or_qualifying_it(tmp_path):
    owner = selection(tmp_path)
    assert G.selected_inventory() is None and G.prompt_text(None) == ""
    with scoped_compiler_library(owner):
        inventory = G.selected_inventory()
    assert "merlin.portable" not in sys.modules
    assert inventory["contract_sha256"] == owner.contract.sha256
    (module,) = inventory["public_modules"]
    rows = {row["symbol"]: row for row in module["declarations"]}
    assert set(rows) == {"round_extent", "TilePolicy", "TilePolicy.__init__", "TilePolicy.choose", "public_transform"}
    assert rows["round_extent"]["kind"] == "imported_name"
    assert rows["TilePolicy.choose"]["parameters_preview"] == "self, m: int, *, limit: int=8"
    assert rows["TilePolicy.choose"]["line"] > 1
    assert module["source"] == str(L.guest_root(owner) / "merlin/portable.py")
    assert module["source_sha256"] == owner.contract.members[1].sha256
    encoded = json.dumps(inventory)
    assert str(owner.root) not in encoded and "private-answer" not in encoded and "protected expected" not in encoded
    assert inventory["target_adapter_applicability"].startswith("UNKNOWN")
    assert "required" in inventory["independent_qualification"]
    text = G.prompt_text(inventory)
    assert "frozen_compiler_edit_authority" in text and "path/symbol" in text and "new-component" in text
    assert "reviewed authored code" in text and "selected software/ISA specification and RTL facts" in text
    assert "UNKNOWN or partial floors cannot establish convergence" in text


def test_inventory_limits_previews_and_records_omitted_declarations(tmp_path, monkeypatch):
    owner = selection(tmp_path)
    monkeypatch.setattr(G, "MAX_DECLARATIONS", 2)
    with scoped_compiler_library(owner):
        inventory = G.selected_inventory()
    (module,) = inventory["public_modules"]
    assert len(module["declarations"]) == 2 and module["declaration_count"] == 5
    assert module["omitted_declarations"] == 3


def test_inventory_resolves_guest_root_once_per_call_but_revalidates_each_call(tmp_path, monkeypatch):
    owner = selection(tmp_path)
    extra = owner.root / "merlin/second.py"
    extra.write_text("def second(value): return value\n")
    contract = freeze_compiler_library(
        owner.root, review_id="two-public-leaves",
        public_modules=(*owner.contract.public_modules, "merlin.second"),
        sources=tuple((m.path, m.module) for m in owner.contract.members)
                + (("merlin/second.py", "merlin.second"),),
    )
    owner = CompilerLibrarySelection(contract, owner.root)
    calls, original = [], L.guest_root

    def counted(selected):
        calls.append(selected.contract.sha256)
        return original(selected)

    monkeypatch.setattr(L, "guest_root", counted)
    with scoped_compiler_library(owner):
        first = G.selected_inventory()
        assert len(calls) == 1 and len(first["public_modules"]) == 2
        second = G.selected_inventory()
        assert second == first and len(calls) == 2
    assert all(row["source"].endswith(row["module"].replace(".", "/") + ".py")
               for row in first["public_modules"])
    assert "merlin.portable" not in sys.modules and "merlin.second" not in sys.modules


def test_inventory_still_refuses_source_mutation_during_derivation(tmp_path, monkeypatch):
    from merlin.targetgen.compiler_library import CompilerLibraryError

    owner = selection(tmp_path)
    path = owner.root / "merlin/portable.py"
    original_bytes, original = path.read_bytes(), G._declarations

    def mutate(body, prefix=""):
        path.write_bytes(original_bytes + b"\n# changed during inventory\n")
        return original(body, prefix)

    monkeypatch.setattr(G, "_declarations", mutate)
    with scoped_compiler_library(owner):
        try:
            with pytest.raises(CompilerLibraryError, match="bytes changed"):
                G.selected_inventory()
        finally:
            path.write_bytes(original_bytes)  # restore before the owner's independent exit check


def test_bounded_inventory_keeps_later_module_entrypoints_discoverable(tmp_path, monkeypatch):
    root = tmp_path / "reviewed"
    (root / "merlin").mkdir(parents=True)
    (root / "merlin/__init__.py").write_text("")
    (root / "merlin/early.py").write_text(
        "from math import ceil, floor, sqrt\n"
        + "class Policy:\n    def choose(self): pass\n"
        + "".join(f"def earlier_{index}(): pass\n" for index in range(12))
    )
    (root / "merlin/late.py").write_text(
        "raise RuntimeError('inventory must never execute source')\n"
        "def choose_traversal(shape, resources): pass\n"
    )
    contract = freeze_compiler_library(
        root,
        review_id="fair-inventory-test",
        public_modules=("merlin.early", "merlin.late"),
        sources=(("merlin/__init__.py", "merlin"), ("merlin/early.py", "merlin.early"),
                 ("merlin/late.py", "merlin.late")),
    )
    owner = CompilerLibrarySelection(contract, root)
    monkeypatch.setattr(G, "MAX_DECLARATIONS", 3)
    with scoped_compiler_library(owner):
        inventory = G.selected_inventory()
    early, late = inventory["public_modules"]
    assert inventory["schema"] == "merlin.phase2.public_compiler_library_inventory.v2"
    assert [row["symbol"] for row in late["declarations"]] == ["choose_traversal"]
    assert len(early["declarations"]) == 2
    assert all(row["kind"] != "imported_name" and "." not in row["symbol"]
               for row in early["declarations"])
    assert sum(len(row["declarations"]) for row in inventory["public_modules"]) == 3
    assert early["omitted_declarations"] + late["omitted_declarations"] == 15
    assert "merlin.late" not in sys.modules


def test_bound_prompt_input_and_stage_context_receive_only_public_inventory(tmp_path):
    owner = selection(tmp_path)

    @dataclass(frozen=True)
    class Inputs:
        allowed_paths: tuple = ("submission",)
        additional_compiler_library: dict | None = None

    original = Inputs()
    with scoped_compiler_library(owner):
        updated = L.prompt_inputs(original)
    assert updated.additional_compiler_library["review_id"] == owner.contract.review_id
    assert original.additional_compiler_library is None
    context = G.prompt_context(updated)
    context["additional_compiler_library"]["review_id"] = "forged projection"
    assert updated.additional_compiler_library["review_id"] == owner.contract.review_id
    assert G.prompt_context(original) == {}
