"""Real helper-byte changes must invalidate every ordinary scheduling route."""

import importlib
from dataclasses import replace
from pathlib import Path

import pytest
from test_private_copy_elimination import ADAPTER, fixture
from test_program_limits import DECODER, LIMITS
from test_schedule_pipeline import encoding, limits_for
from test_selected_dense_contracts import inputs, schedule

from merlin.compile.scheduling.program_limits import finalize_program
from merlin.compile.scheduling.schedule_pipeline import compile_schedule


def test_selected_and_explicit_routes_use_one_verification_version():
    from merlin.compile.scheduling.compact_dense import _dense_library_sources
    from merlin.compile.scheduling.program_limits import _library_sources
    from merlin.compile.scheduling.schedule_pipeline import shared_library_sources
    from merlin.compile.scheduling.selected_contracts import _resolver_sources

    sources = shared_library_sources()
    assert sources == _dense_library_sources() == _library_sources() == dict(_resolver_sources())
    assert {
        "merlin.compile.scheduling.block_schedule",
        "merlin.transforms.pure_scalar",
        "merlin.transforms.scalar_witness",
        "merlin.semantic_compiler.model",
        "merlin.sched.ir",
        "merlin.sched.ir.parse",
    } <= sources.keys()


def private_source(tmp_path, monkeypatch, name):
    """Change an isolated copy of actual owner bytes, never the checkout."""
    module = importlib.import_module(name)
    path = tmp_path / (name + ".py")
    path.write_bytes(Path(module.__file__).read_bytes())
    monkeypatch.setattr(module, "__file__", str(path))
    return path


def change(path):
    path.write_bytes(path.read_bytes() + b"\n# changed implementation version\n")


def test_selected_declarations_reject_changed_block_semantics(tmp_path, monkeypatch):
    path = private_source(tmp_path, monkeypatch, "merlin.compile.scheduling.block_schedule")
    selected, *_ = inputs(tmp_path)
    selected.revalidate()
    change(path)
    with pytest.raises(ValueError, match="changed"):
        selected.revalidate()


def test_dense_rejects_changed_scalar_admission_during_encoding(tmp_path, monkeypatch):
    path = private_source(tmp_path, monkeypatch, "merlin.transforms.pure_scalar")
    model = inputs(tmp_path)
    original = model[3].encode

    def encode(operation):
        change(path)
        return original(operation)

    with pytest.raises(ValueError, match="changed"):
        schedule(model, adapter=replace(model[3], encode=encode))


def test_checked_dense_result_rejects_helper_change_before_emission(tmp_path, monkeypatch):
    path = private_source(tmp_path, monkeypatch, "merlin.transforms.certificates")
    result = schedule(inputs(tmp_path))
    change(path)
    with pytest.raises(ValueError, match="changed"):
        result.emit_c(None, symbol="stale")


@pytest.mark.parametrize("mutation", ["kernel", "index_width", "library"])
def test_dense_report_is_not_emission_authority(tmp_path, monkeypatch, mutation):
    from merlin.sched import codegen

    path = private_source(tmp_path, monkeypatch, "merlin.transforms.certificates")
    result = schedule(inputs(tmp_path))
    # The same custody is required without a selected-contract wrapper.
    result = replace(result, selected_contracts=None)
    result.realization.clear()
    if mutation == "kernel":
        result = replace(result, kernel=replace(result.kernel, name="unchecked"))
    elif mutation == "index_width":
        result = replace(result, index_bits=32)
    else:
        change(path)
    emitted = []
    monkeypatch.setattr(codegen, "emit_c_function", lambda *args, **kw: emitted.append(True) or "unchecked")
    with pytest.raises(ValueError, match="changed"):
        result.emit_c(None, symbol="stale")
    assert not emitted


def test_report_edits_preserve_checked_native_dense_execution(tmp_path, monkeypatch, record_property):
    import json

    from test_compact_dense import ADAPTER as DENSE_ADAPTER
    from test_compact_dense import fixture as dense_fixture
    from test_compact_dense import native_dense_pair
    from test_compact_dense import schedule as plan_dense

    from merlin.sched import codegen

    problem, resources, storage = dense_fixture(padded=True)
    result = plan_dense(problem, resources, storage)
    render = codegen.emit_c_function

    def edit_report(kernel, *args, **kwargs):
        if kernel is result.kernel:
            result.realization["pipeline"].clear()
        return render(kernel, *args, **kwargs)

    monkeypatch.setattr(codegen, "emit_c_function", edit_report)
    measured = native_dense_pair(
        tmp_path, problem, resources, storage, result, DENSE_ADAPTER,
        dtypes=("i8", "i8", "i32"), whole_graph=True,
    )
    assert result.realization["pipeline"] == {}
    record_property("native_execution", json.dumps(measured))


def test_schedule_rejects_changed_block_helper_during_rewrite(tmp_path, monkeypatch):
    path = private_source(tmp_path, monkeypatch, "merlin.compile.scheduling.block_schedule")
    kernel, objects, banks, model = fixture()
    codec, word = encoding()

    def rewrite(call, accesses):
        change(path)
        return ADAPTER.rewrite(call, accesses)

    with pytest.raises(ValueError, match="library changed"):
        compile_schedule(kernel, objects, banks, replace(ADAPTER, rewrite=rewrite), codec,
                         limits_for(banks, model, word), **model)


def test_finalizer_rejects_changed_pipeline_inside_decoder(tmp_path, monkeypatch):
    path = private_source(tmp_path, monkeypatch, "merlin.transforms.pipeline_state")

    def decode(image):
        change(path)
        return DECODER.decode(image)

    with pytest.raises(ValueError, match="changed"):
        finalize_program(bytes([1]), LIMITS, replace(DECODER, decode=decode))
