"""Original late-RNE reports bind actual emitted LLVM and explicit selection."""

import json
from pathlib import Path

import pytest

from merlin.common.digest import sha256_file
from merlin.llvmlower.compilation_recipe import CompilationRecipe
from merlin.llvmlower.host_bounded_rne import PREPARATION, bind_preparation, selected
from merlin.llvmlower.late_quant_rne import merlin_host_llvm_transform
from merlin.llvmlower.optional_passes import Selection, applied
from merlin.llvmlower.toolchain import clang
from merlin.tests.ir.test_late_quant_rne_intrinsic import source


def prepared(tmp_path, combine=True, host_isa="rv64gc"):
    original = tmp_path / "source.ll"
    original.write_text(source())
    work = tmp_path / "host_llvm"
    target = merlin_host_llvm_transform(
        Path(clang()).resolve().parent, host_isa=host_isa, combine_clamp=combine,
    )(original, work)
    observation = {"source_sha256": sha256_file(original), "selected_sha256": sha256_file(target)}
    return work, observation


def test_explicit_off_overrides_both_existing_api_and_environment_fallback(monkeypatch):
    monkeypatch.setenv("MERLIN_HOST_BOUNDED_RNE", "1")
    with applied(Selection.parse("-host-bounded-rne")):
        assert selected(True) is False
    with applied(Selection.parse("host-bounded-rne")):
        assert selected(False) is True


@pytest.mark.parametrize("host_isa,combine", [("rv64gc", False), ("rv64gc", True), ("portable", False)])
def test_existing_direct_callback_policies_keep_their_actual_recipe_marker(tmp_path, host_isa, combine):
    work, observation = prepared(tmp_path, combine, host_isa)
    recipe = CompilationRecipe(tmp_path, producer=Path(__file__))
    with applied(Selection.parse("-host-bounded-rne")):
        bind_preparation(recipe, work, observation)
    document = json.loads(recipe.path.read_bytes())
    assert document["preparation"][PREPARATION]["sha256"] == sha256_file(work / "receipt.json")


def test_selected_rv64gc_arm_refuses_a_legitimate_portable_callback(tmp_path):
    work, observation = prepared(tmp_path, False, "portable")
    recipe = CompilationRecipe(tmp_path, producer=Path(__file__))
    with applied(Selection.parse("host-bounded-rne")):
        with pytest.raises(ValueError, match="absent or changed"):
            bind_preparation(recipe, work, observation)
    assert "preparation" not in json.loads(recipe.path.read_bytes())


@pytest.mark.parametrize("field", ["source_sha256", "selected_sha256"])
def test_actual_lowering_and_selected_llvm_must_match_original_report(tmp_path, field):
    work, observation = prepared(tmp_path)
    observation[field] = "f" * 64
    recipe = CompilationRecipe(tmp_path, producer=Path(__file__))
    with applied(Selection.parse("host-bounded-rne")):
        with pytest.raises(ValueError, match="differs from actual emitted LLVM"):
            bind_preparation(recipe, work, observation)
    assert "preparation" not in json.loads(recipe.path.read_bytes())


@pytest.mark.parametrize("field,value", [
    ("host_isa", "portable"), ("legalizer_sha256", "f" * 64), ("routes", []), ("combine_clamp", False),
])
def test_selected_arm_requires_the_original_supported_scalar_policy(tmp_path, field, value):
    work, observation = prepared(tmp_path)
    report = work / "receipt.json"
    document = json.loads(report.read_bytes())
    document[field] = value
    report.write_text(json.dumps(document))
    recipe = CompilationRecipe(tmp_path, producer=Path(__file__))
    with applied(Selection.parse("host-bounded-rne")):
        with pytest.raises(ValueError, match="absent or changed"):
            bind_preparation(recipe, work, observation)


def test_default_off_needs_no_marker_but_selected_arm_refuses_missing_preparation(tmp_path):
    recipe = CompilationRecipe(tmp_path, producer=Path(__file__))
    before = recipe.path.read_bytes()
    with applied(Selection.parse("-host-bounded-rne")):
        bind_preparation(recipe, tmp_path / "host_llvm", None)
    assert recipe.path.read_bytes() == before
    with applied(Selection.parse("host-bounded-rne")):
        with pytest.raises(ValueError, match="no original preparation report"):
            bind_preparation(recipe, tmp_path / "host_llvm", None)


def test_existing_transform_chain_cannot_hide_its_actual_rne_terminal(tmp_path):
    from merlin.runtime.backends.spike_model import _transform_host_ir
    from merlin.tests.runtime.test_host_transform_chain import _chain, _stage

    original = tmp_path / "source.ll"
    original.write_text(source())
    work = tmp_path / "chain"
    stage = _stage(tmp_path, "unchanged")
    callback = merlin_host_llvm_transform(Path(clang()).resolve().parent, host_isa="rv64gc", combine_clamp=True)
    _selected, observation = _transform_host_ir(original, work, callback, chain=_chain(stage))
    recipe = CompilationRecipe(tmp_path, producer=Path(__file__))
    with applied(Selection.parse("host-bounded-rne")):
        bind_preparation(recipe, work, observation)
    document = json.loads(recipe.path.read_bytes())
    assert document["preparation"][PREPARATION]["sha256"] == sha256_file(work / "legacy_terminal/receipt.json")
