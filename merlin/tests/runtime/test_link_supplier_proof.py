"""Actual native links prove caller-selected defining-symbol suppliers."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from merlin.llvmlower.compilation_recipe import CompilationRecipe, verify_completed_recipe
from merlin.llvmlower.link_supplier_trace import trace_symbol_flags
from merlin.runtime.backends import spike


def _native_link(
    tmp_path: Path,
    *,
    override: bool = False,
    reverse_archives: bool = False,
    requested_symbol: str = "selected_supplier",
    compiler=None,
    archiver=None,
):
    compiler = compiler or shutil.which("cc")
    archiver = archiver or shutil.which("ar")
    if compiler is None or archiver is None:
        pytest.skip("native C toolchain unavailable")
    main, supplier, alternate = (tmp_path / name for name in ("main.c", "supplier.c", "alternate.c"))
    main.write_text("extern int selected_supplier(void); int main(void) { return selected_supplier() != 17; }\n")
    supplier.write_text("int selected_supplier(void) { return 17; }\n")
    alternate.write_text("int selected_supplier(void) { return 23; }\n")
    recipe = CompilationRecipe(tmp_path, producer=Path(__file__))

    def run(argv):
        return subprocess.run(argv, capture_output=True, text=True)

    objects = []
    for source in (main, supplier, alternate):
        output = source.with_suffix(".o")
        recipe.run([compiler, "-c", source, "-o", output], runner=run, inputs=[source], output=output)
        objects.append(output)
    archive = tmp_path / "supplier.a"
    subprocess.run([archiver, "rcs", archive, objects[1]], check=True, capture_output=True)
    second_archive = tmp_path / "alternate.a"
    subprocess.run([archiver, "rcs", second_archive, objects[2]], check=True, capture_output=True)
    elf = tmp_path / "app"
    inputs = [
        objects[0],
        *([objects[2]] if override else []),
        *([second_archive] if reverse_archives else []),
        archive,
    ]
    extra = ["-nostdlib", "-nostartfiles", "-Wl,-e,main"] if Path(compiler).name.startswith("riscv64-") else []
    linked = recipe.run(
        [compiler, *extra, *inputs, *trace_symbol_flags([requested_symbol]), "-o", elf],
        runner=run,
        inputs=inputs,
        output=elf,
    )
    return recipe, linked, archive, elf


def _verify(recipe: CompilationRecipe, elf: Path, archive: Path):
    return verify_completed_recipe(
        recipe.path,
        executable=elf,
        expected_recipe_sha256=hashlib.sha256(recipe.path.read_bytes()).hexdigest(),
        expected_link_suppliers={"selected_supplier": archive},
    )


def test_native_archive_member_supplier_is_observed_and_bound(tmp_path):
    recipe, linked, archive, elf = _native_link(tmp_path)
    recipe.record_link_suppliers({"selected_supplier": archive}, linked)
    recipe.completed(elf)
    verified = _verify(recipe, elf, archive)
    assert "selected_supplier" in verified["link_suppliers"]["symbols"]
    assert "supplier.a(" in verified["link_suppliers"]["symbols"]["selected_supplier"]["definition"]
    emitter = verified["link_suppliers"]["emitter"]
    assert hashlib.sha256(Path(emitter["path"]).read_bytes()).hexdigest() == emitter["sha256"]
    subprocess.run([elf], check=True)

    different = tmp_path / "other.a"
    different.write_bytes(archive.read_bytes())
    with pytest.raises(ValueError, match="selected supplier"):
        _verify(recipe, elf, different)
    alias = tmp_path / "archive-alias.a"
    alias.symlink_to(archive)
    with pytest.raises(ValueError, match="selected supplier"):
        _verify(recipe, elf, alias)
    archive.write_bytes(archive.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="bytes changed"):
        _verify(recipe, elf, archive)


@pytest.mark.parametrize("mutation", ["trace_flag", "trace_line", "emitter_hash", "missing_proof"])
def test_rehashed_receipt_cannot_invent_link_supplier_observation(tmp_path, mutation):
    recipe, linked, archive, elf = _native_link(tmp_path)
    recipe.record_link_suppliers({"selected_supplier": archive}, linked)
    recipe.completed(elf)
    document = json.loads(recipe.path.read_text())
    if mutation == "trace_flag":
        document["commands"][-1]["argv"].remove("-Wl,--trace-symbol=selected_supplier")
    elif mutation == "trace_line":
        document["link_suppliers"]["stderr"] += linked.stderr.splitlines()[-1] + "\n"
    elif mutation == "emitter_hash":
        document["link_suppliers"]["emitter"]["sha256"] = "0" * 64
    elif mutation == "missing_proof":
        del document["link_suppliers"]
    recipe.path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="link supplier"):
        _verify(recipe, elf, archive)


def test_native_direct_object_override_refuses_archive_supplier_claim(tmp_path):
    recipe, linked, archive, elf = _native_link(tmp_path, override=True)
    with pytest.raises(ValueError, match="defining supplier"):
        recipe.record_link_suppliers({"selected_supplier": archive}, linked)
    with pytest.raises(ValueError, match="completed observation"):
        verify_completed_recipe(
            recipe.path,
            executable=elf,
            expected_recipe_sha256=hashlib.sha256(recipe.path.read_bytes()).hexdigest(),
            expected_link_suppliers={"selected_supplier": archive},
        )


def test_native_earlier_archive_overrides_same_named_later_supplier(tmp_path):
    recipe, linked, later_archive, elf = _native_link(tmp_path, reverse_archives=True)
    with pytest.raises(ValueError, match="defining supplier"):
        recipe.record_link_suppliers({"selected_supplier": later_archive}, linked)


def test_native_unreferenced_symbol_has_no_defining_supplier(tmp_path):
    recipe, linked, archive, elf = _native_link(tmp_path, requested_symbol="unused_symbol")
    with pytest.raises(ValueError, match="unique defining supplier"):
        recipe.record_link_suppliers({"unused_symbol": archive}, linked)


def test_selected_cross_linker_observes_archive_member_without_running_elf(tmp_path):
    compiler = spike.gcc_path()
    archiver = compiler.with_name("riscv64-unknown-elf-ar")
    if not compiler.is_file() or not archiver.is_file():
        pytest.skip("selected cross toolchain unavailable")
    recipe, linked, archive, elf = _native_link(tmp_path, compiler=compiler, archiver=archiver)
    recipe.record_link_suppliers({"selected_supplier": archive}, linked)
    recipe.completed(elf)
    assert _verify(recipe, elf, archive)["status"] == "completed"


@pytest.mark.parametrize("symbols", [[], ["x", "x"], ["x,-z"], ["0bad"]])
def test_trace_flags_refuse_empty_duplicate_or_injected_symbols(symbols):
    with pytest.raises(ValueError, match="symbol"):
        trace_symbol_flags(symbols)
