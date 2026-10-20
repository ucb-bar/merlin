"""Original source choices stay bound through real completed compiler recipes.

The tiny host objects below test transport/custody, not Gemmini qualification or
numerical execution of a transformed model.
"""

from __future__ import annotations

import hashlib
import json
import runpy
import shutil
import subprocess
from pathlib import Path

import pytest

from merlin.common.paths import merlin_dir
from merlin.llvmlower import automatic_destinations, automatic_gathers
from merlin.llvmlower.compilation_recipe import CompilationRecipe
from merlin.llvmlower.source_optimization_request import GATHER_PASS, PACKING_PASS, SourceOptimizationRequest
from merlin.llvmlower.source_optimization_selection import SourceCompilationOptimization, SourceOptimizationSelection
from merlin.runtime.source_optimization_build import SourceOptimizationBuild

SOURCE = "module {}\n"


def selected(source=SOURCE):
    request = SourceOptimizationRequest(hashlib.sha256(source.encode()).hexdigest(), False, False, 4096)
    return SourceOptimizationSelection(request, (PACKING_PASS, GATHER_PASS), 4096)


def completed(work, selection, *, producer=None):
    """Actual checked pass reports and a real compiler's linked file, no runner stubs."""
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("native compiler required for completed recipe transport")
    work.mkdir(parents=True)
    lower = work / "lower"
    lower.mkdir()
    source = work / "model.prepared.mlir"
    source.write_text(SOURCE)
    guard = SourceOptimizationBuild(selection)
    guard.bind(source)
    consumed = SourceCompilationOptimization(SOURCE, selection)
    after, gather = automatic_gathers.prepare_automatic_gathers(SOURCE, serial=True, **consumed.options(GATHER_PASS))
    _, packing = automatic_destinations.prepare_automatic_destinations(
        after, serial=True, permuted_outputs=True, **consumed.options(PACKING_PASS)
    )
    (lower / "source-optimization-selection.json").write_text(
        json.dumps(consumed.report(gather=gather, packing=packing))
    )
    recipe = CompilationRecipe(work, producer=producer or Path(__file__))
    guard.consumed(recipe, lower)
    c, obj, elf = (work / name for name in ("transport.c", "transport.o", "model.elf"))
    c.write_text("int main(void) { return 0; }\n")
    for argv, inputs, output in (
        ([compiler, "-c", c, "-o", obj], [c], obj),
        ([compiler, obj, "-o", elf], [obj], elf),
    ):
        recipe.run(
            argv, runner=lambda command: subprocess.run(command, capture_output=True), inputs=inputs, output=output
        )
    guard.finish(recipe, elf)
    return guard, recipe, elf


def test_default_selection_preserves_empty_transport_without_library_reads(monkeypatch):
    import merlin.runtime.source_optimization_build as transport

    monkeypatch.setattr(transport, "_pin", lambda path: pytest.fail("default transport read a selection artifact"))
    guard = SourceOptimizationBuild(None, backend="legacy")
    guard.bind("absent")
    guard.consumed(None, "absent")
    guard.accept(None, build_dir="absent", executable="absent")
    assert guard.options() == {} and guard.identity() == b"" and guard.record() is None
    called = []
    recipe = type("Recipe", (), {"completed": lambda self, elf: called.append(elf)})()
    guard.finish(recipe, "original.elf")
    assert called == ["original.elf"]


@pytest.mark.parametrize("backend", ["c", "unknown", "gsim"])
def test_unsupported_consumer_refuses_before_preparation(backend):
    with pytest.raises(ValueError, match="no source optimization consumer"):
        SourceOptimizationBuild(selected(), backend=backend)


@pytest.mark.parametrize("change", ["choice", "permission", "budget"])
def test_original_policy_cannot_change_during_prepare_callback(tmp_path, change):
    selection = selected()
    guard = SourceOptimizationBuild(selection)
    if change == "choice":
        object.__setattr__(selection.request, "padding_gather", True)
    elif change == "permission":
        object.__setattr__(selection, "permitted_passes", ())
    else:
        object.__setattr__(selection, "maximum_added_operations", 4097)
    with pytest.raises(ValueError, match="original.*changed"):
        guard.bind(tmp_path / "not-yet-created.mlir")


def test_current_prepared_source_must_match_original_request(tmp_path):
    path = tmp_path / "prepared.mlir"
    path.write_text(SOURCE + "// changed by routing\n")
    with pytest.raises(ValueError, match="current source"):
        SourceOptimizationBuild(selected()).bind(path)


def test_actual_prepared_source_remains_live_after_consumption(tmp_path):
    path = tmp_path / "prepared.mlir"
    path.write_text(SOURCE)
    guard = SourceOptimizationBuild(selected())
    guard.bind(path)
    path.write_text(SOURCE + "// later provider mutation\n")
    with pytest.raises(ValueError, match="actual prepared source changed"):
        guard.identity()


@pytest.mark.parametrize("wrapper", [automatic_gathers, automatic_destinations])
def test_late_algorithm_checker_mutation_refuses_before_completion(tmp_path, monkeypatch, wrapper):
    # Keep real implementations untouched. A genuine source file in the original
    # host-owned roster is changed after the original pre-callback snapshot.
    checker = tmp_path / "selected_checker.py"
    checker.write_text("# original reviewed checker\n")
    original = wrapper.implementation_paths
    monkeypatch.setattr(wrapper, "implementation_paths", lambda: (*original(), checker))
    guard = SourceOptimizationBuild(selected())
    checker.write_text("# changed by final link callback\n")
    with pytest.raises(ValueError, match="original.*changed"):
        guard.revalidate()


def test_real_completed_build_reopens_actual_recipe_and_detaches_returned_record(tmp_path):
    selection = selected()
    producer, recipe, elf = completed(tmp_path / "build", selection)
    record = producer.record()
    consumer = SourceOptimizationBuild(selection)
    consumer.accept(record, build_dir=recipe.path.parent, executable=elf)
    assert consumer.record() == record
    record["library"][0]["sha256"] = "0" * 64
    assert producer.record() == consumer.record()  # Returned proposal cannot edit retained snapshots.
    with pytest.raises(ValueError, match="receipt changed"):
        SourceOptimizationBuild(selection).accept(record, build_dir=recipe.path.parent, executable=elf)


@pytest.mark.parametrize("changed", ["source", "report", "executable", "compiler_input", "policy"])
def test_late_build_or_execution_callback_changes_are_rejected(tmp_path, changed):
    selection = selected()
    guard, recipe, elf = completed(tmp_path / "build", selection)
    record = guard.record()
    if changed == "policy":
        object.__setattr__(selection.request, "padding_gather", True)
    else:
        target = {
            "source": Path(record["input"]["path"]),
            "report": Path(record["report"]["path"]),
            "executable": elf,
            "compiler_input": recipe.path.parent / "transport.c",
        }[changed]
        target.write_bytes(target.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="changed"):
        guard.revalidate()


def test_valid_foreign_completed_build_cannot_replace_requested_product(tmp_path):
    selection = selected()
    first, recipe, elf = completed(tmp_path / "first", selection)
    other, _, other_elf = completed(tmp_path / "other", selection)
    with pytest.raises(ValueError, match="foreign consumed report"):
        SourceOptimizationBuild(selection).accept(other.record(), build_dir=recipe.path.parent, executable=elf)
    # The other file is independently valid, even when the compiler emitted the
    # same bytes. It is still not the requested invocation's exact product path.
    with pytest.raises(ValueError, match="final executable|receipt changed"):
        SourceOptimizationBuild(selection).accept(first.record(), build_dir=recipe.path.parent, executable=other_elf)


def test_prepared_recipe_is_not_a_completed_source_choice(tmp_path):
    selection = selected()
    producer, recipe, elf = completed(tmp_path / "build", selection)
    record = producer.record()
    document = json.loads(recipe.path.read_text())
    document["status"] = "prepared"
    recipe.path.write_text(json.dumps(document))
    record["recipe"]["sha256"] = hashlib.sha256(recipe.path.read_bytes()).hexdigest()
    record["recipe"]["bytes"] = recipe.path.stat().st_size
    with pytest.raises(ValueError, match="completed"):
        SourceOptimizationBuild(selection).accept(record, build_dir=recipe.path.parent, executable=elf)


@pytest.mark.parametrize("mutation", [None, "build", "execution"])
def test_highlevel_forwards_original_selection_and_verifies_real_completed_transport(tmp_path, monkeypatch, mutation):
    helpers = runpy.run_path(str(merlin_dir() / "tests/infra/test_baremetal_saved_model.py"))
    _fake_build, _fixture = helpers["_fake_build"], helpers["_fixture"]

    from merlin.compile import baremetal_model as BM

    inputs, out = _fixture(tmp_path, monkeypatch)
    selection = selected()
    seen = []

    def build(bundle, work, **kwargs):
        assert kwargs["source_optimizations"] is selection
        seen.append(selection)
        info = _fake_build(bundle, work, **kwargs)
        # The API seam is synthetic; the proof producer is an actual separate
        # tiny compiler invocation recorded under the requested ordinary work.
        elf = info["elf"]
        elf.unlink()
        work.rmdir()
        guard, _, actual = completed(work, selection)
        result = {**info, "elf": actual, "source_optimizations": guard.record()}
        if mutation == "build":
            object.__setattr__(selection.request, "padding_gather", True)
        return result

    def run(*args, **kwargs):
        object.__setattr__(selection, "maximum_added_operations", 4097)
        return {"console": helpers["_console"]()}

    monkeypatch.setattr(BM.spike_model, "build", build)
    monkeypatch.setattr(BM, "_spike_extension_backend", lambda target: None)
    monkeypatch.setattr(BM.spike_model, "run", run)
    options = {"run": "spike", "reference_file": "golden.npy"} if mutation == "execution" else {"run": "none"}
    if mutation is not None:
        with pytest.raises(ValueError, match="original.*changed"):
            BM.compile_saved_model(**inputs, output=out / "selected", source_optimizations=selection, **options)
        receipt = json.loads((out / "selected/baremetal_model.json").read_text())
        assert receipt["status"] == "failed"
        return
    result = BM.compile_saved_model(**inputs, output=out / "selected", source_optimizations=selection, **options)
    assert seen == [selection]
    assert result["inputs"]["source_optimizations"] == selection.record()
    assert result["output"]["source_optimizations"]["selection"] == selection.record()
    assert result["status"] == "compiled"
