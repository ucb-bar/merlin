"""Next-library view normalization is explicit, source-closed and discoverable."""
import importlib.util

import pytest

from merlin.common.paths import repo_root
from merlin.compile.scheduling.export import export_shared_scheduling
from merlin.compile.scheduling.library_sources import library_modules
from merlin.targetgen.compiler_library import (
    CompilerLibrarySelection, freeze_compiler_library, scoped_compiler_library,
)
from merlin_experiments.phase2.compiler_library_guidance import selected_inventory

ROOT = repo_root()
SPEC = importlib.util.spec_from_file_location(
    "prepare_view_library", ROOT / "examples/gemmini/phase1/launch/prepare_author_library.py"
)
PREP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREP)
LEAF = "merlin.transforms.tensor_views"
OWNER = "merlin.transforms.certificates"


def test_view_selection_requires_explicit_review_and_existing_closure():
    assert PREP.reviewed_view_modules((), existing=(), source_modules=()) == ()
    with pytest.raises(ValueError, match="certificates"):
        PREP.reviewed_view_modules((), existing=(), source_modules=(), enabled=True)
    with pytest.raises(ValueError, match="Boolean"):
        PREP.reviewed_view_modules((), existing=(), source_modules=(OWNER,), enabled=1)
    assert PREP.reviewed_view_modules((), existing=(), source_modules=(OWNER,), enabled=True) == (LEAF,)
    assert PREP.reviewed_view_modules((LEAF,), existing=(), source_modules=(OWNER,), enabled=True) == (LEAF,)
    assert PREP.reviewed_view_modules((), existing=(LEAF,), source_modules=(OWNER,), enabled=True) == ()
    # This new preparation option must not retroactively change any SDK roster.
    assert all(LEAF not in library_modules(version) for version in (6, 7, 8))


def test_exact_view_source_is_admitted_and_has_callable_inventory(tmp_path):
    root = tmp_path / "reviewed"
    manifest = export_shared_scheduling(root, version=8, source_binding=True)
    sources = tuple((row["path"], row["module"]) for row in manifest["members"])
    requested = PREP.reviewed_view_modules(
        (), existing=manifest["public_modules"],
        source_modules={module for _, module in sources}, enabled=True,
    )
    additions = PREP.reviewed_modules(requested, existing=manifest["public_modules"])
    for path in additions:
        (root / path).write_bytes((ROOT / "src" / path).read_bytes())
    contract = freeze_compiler_library(
        root, review_id="editable-generic-compiler-v1:tensor-view-test",
        public_modules=(*manifest["public_modules"], *additions.values()),
        sources=(*sources, *additions.items()),
    )
    contract.verify(root)
    with scoped_compiler_library(CompilerLibrarySelection(contract, root)):
        inventory = selected_inventory()
    entry = next(row for row in inventory["public_modules"] if row["module"] == LEAF)
    declarations = {row["symbol"]: row for row in entry["declarations"]}
    callback = declarations["canonicalize_tensor_views"]
    assert callback["kind"] == "callable"
    assert "module: builtin.ModuleOp" in callback["parameters_preview"]
    assert "preserving every use" in callback["documentation_preview"]
    assert "TensorViewWitness" in declarations
    member = next(row for row in contract.members if row.module == LEAF)
    assert entry["source_sha256"] == member.sha256
    assert inventory["target_adapter_applicability"].startswith("UNKNOWN")
    assert set(row.module for row in contract.members) == {
        *(module for _, module in sources), LEAF,
    }
