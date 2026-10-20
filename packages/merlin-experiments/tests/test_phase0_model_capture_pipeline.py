"""Bound source captures while ordinary model checks and publication stay ordered."""

from __future__ import annotations

import contextvars
import threading
from contextlib import closing

import pytest
import yaml
from merlin_experiments.phase0 import model_qualification as MQ
from merlin_experiments.phase0 import staging, writer
from merlin_experiments.phase0.parallel_capture import ordered_jobs
from merlin_experiments.phase0.sealed_generation import SealedCaptureSource
from test_phase0_model_qualification import _SEMANTICS, _binding, _row, _stated


def _entry(name):
    return {"name": name, "cat": "model", "kind": "model", "op": "model", "numerical_semantics": _SEMANTICS}


def _local_source(root, entry):
    member = root / "model" / entry["name"]
    member.mkdir(parents=True)
    (member / "capsule.interface.mlir").write_text("// local fixture source\n")
    (member / "capsule.yaml").write_text(
        yaml.safe_dump(
            {
                "name": entry["name"],
                "kind": "model",
                "interface_mlir": "capsule.interface.mlir",
                "semantic": {"generalization_axis": "fixture"},
            }
        )
    )
    (member / "golden.yaml").write_text("golden_source: host_torch_eager\noutputs: {Y: [[1]]}\n")
    return member


def _prepare(entry, out, source, finish):
    return entry, staging.prepare_staged(source, finish, out, member=f"model/{entry['name']}")


def _finish_jobs(jobs):
    results = []
    with closing(jobs):
        for entry, job in jobs:
            try:
                _, prepared = job.result()
                results.append((entry["name"], prepared.finish()))
            except ValueError as error:
                results.append((entry["name"], str(error)))
    return results


def _clean(out):
    assert not list(out.glob(staging.STAGING_PREFIX + "*"))
    assert not [t for t in threading.enumerate() if t.name.startswith("phase0-capture")]


def test_source_captures_overlap_real_qualification_reuses_new_sibling_and_context(tmp_path, monkeypatch):
    from merlin_experiments.phase0 import group_forms

    from merlin.common import mlir_query as mq
    from merlin.targetgen import group_capsule_entries as G

    selected = contextvars.ContextVar("selected_model", default="caller")
    barrier = threading.Barrier(2, timeout=5)
    captured, entered, qualified, group_writes = [], [], [], []
    main = threading.get_ident()
    source = SealedCaptureSource(
        {"venv": str(tmp_path), "m2m_root": str(tmp_path), "runs_root": str(tmp_path / "runs")}
    )
    out = tmp_path / "corpus"
    monkeypatch.setattr(MQ, "_weight_args", lambda *_: None)
    monkeypatch.setattr(mq, "parse", lambda *_: object())
    monkeypatch.setattr(G, "entries", lambda *_a, **_k: _stated(_row("G_shared", 4, 8, 4, [1])))
    original_group_write, original_qualify = group_forms.write_group_capsule, MQ.qualify

    def group_write(*args, **kwargs):
        group_writes.append(threading.get_ident())
        return original_group_write(*args, **kwargs)

    def qualify(entry, *args):
        assert threading.get_ident() == main and selected.get() == entry["name"]
        assert sorted(entered) == ["first", "second"]
        qualified.append(entry["name"])
        return original_qualify(entry, *args)

    monkeypatch.setattr(group_forms, "write_group_capsule", group_write)
    monkeypatch.setattr(MQ, "qualify", qualify)
    original_inner = writer._write_capsule_inner

    def local_capture(entry, _binding, root, *_args, **_kwargs):
        if entry.get("kind") != "model":
            return original_inner(entry, _binding, root, *_args, **_kwargs)
        assert threading.get_ident() != main
        assert not (root / "layers/G_shared").exists()
        start = len(source.attestations)
        source.attestations.append({"fixture_identity": entry["name"]})
        entered.append(entry["name"])
        barrier.wait()
        written = _local_source(root, entry)
        captured.append(entry["name"])
        return writer._with_attestations(written, source, start=start)

    monkeypatch.setattr(writer, "_write_capsule_inner", local_capture)

    def prepare(entry):
        token = selected.set(entry["name"])
        try:
            return _prepare(
                entry,
                out,
                lambda root: writer._write_capsule_source(entry, _binding(), root, capture=source),
                lambda written, root: writer._finish_capsule(entry, _binding(), root, written),
            )
        finally:
            selected.reset(token)

    def forbidden_full_write(_entry):
        pytest.fail("a selected fresh model must use source-only preparation")

    results = _finish_jobs(
        ordered_jobs([_entry("first"), _entry("second")], forbidden_full_write, workers=2, prepare_model=prepare)
    )
    assert qualified == ["first", "second"] and group_writes == [main]
    assert selected.get() == "caller" and source.attestations == []
    for name, directory in results:
        assert directory == out / "model" / name
        cap = yaml.safe_load((directory / "capsule.yaml").read_bytes())
        assert cap["capture_execution_attestations"] == [{"fixture_identity": name}]
        assert cap["model_qualification"]["covered_groups"] == 1
        assert (directory / "README.md").is_file()
    assert not (out / "layers/G_shared").is_symlink()
    _clean(out)


def test_window_bounds_retained_private_stages_and_finishes_in_declaration_order(tmp_path):
    out = tmp_path / "corpus"
    started, finished = [], []
    barrier = threading.Barrier(2, timeout=5)

    def prepare(entry):
        def source(root):
            started.append(entry["name"])
            if int(entry["name"]) < 2:
                barrier.wait()
            assert len(list(out.glob(staging.STAGING_PREFIX + "*"))) <= 2
            return _local_source(root, entry)

        def finish(written, _root):
            finished.append(entry["name"])
            assert not (out / "model" / entry["name"]).exists()
            return written

        return _prepare(entry, out, source, finish)

    entries = [_entry(str(i)) for i in range(5)]
    jobs = ordered_jobs(entries, lambda _e: pytest.fail("full write"), workers=2, prepare_model=prepare)
    first, job = next(jobs)
    _, prepared = job.result()
    assert sorted(started) == ["0", "1"] and not list(out.glob("model/*"))
    prepared.finish()
    assert first["name"] == "0"
    _finish_jobs(jobs)
    assert finished == [str(i) for i in range(5)]
    _clean(out)


@pytest.mark.parametrize("phase", ["source", "finish", "publication"])
def test_failure_keeps_original_error_and_no_partial_member_or_group(tmp_path, monkeypatch, phase):
    from merlin_experiments.phase0 import operator_exclusion

    out = tmp_path / "corpus"
    barrier = threading.Barrier(2, timeout=5)
    original_publication = operator_exclusion.check_publication

    def publication(selection, directory):
        if phase == "publication" and directory.name == "bad":
            raise ValueError("original publication refusal")
        return original_publication(selection, directory)

    monkeypatch.setattr(operator_exclusion, "check_publication", publication)

    def prepare(entry):
        def source(root):
            written = _local_source(root, entry)
            barrier.wait()
            if entry["name"] == "bad" and phase == "source":
                raise ValueError("original source refusal")
            return written

        def finish(written, root):
            if entry["name"] == "bad":
                group = root / "layers/G_partial"
                group.mkdir(parents=True)
                (group / "capsule.yaml").write_text("name: G_partial\n")
                if phase == "finish":
                    raise ValueError("original finish refusal")
            return written

        return _prepare(entry, out, source, finish)

    results = _finish_jobs(
        ordered_jobs([_entry("bad"), _entry("good")], lambda _e: None, workers=2, prepare_model=prepare)
    )
    assert results == [("bad", f"original {phase} refusal"), ("good", out / "model/good")]
    assert not (out / "model/bad").exists() and not (out / "layers/G_partial").exists()
    _clean(out)


def test_early_close_drains_owned_capture_threads_and_discards_all_pending_stages(tmp_path):
    out = tmp_path / "corpus"
    barrier = threading.Barrier(2, timeout=5)
    returned = []

    def prepare(entry):
        def source(root):
            barrier.wait()
            returned.append(entry["name"])
            return _local_source(root, entry)

        return _prepare(entry, out, source, lambda *_: pytest.fail("must not finalize"))

    jobs = ordered_jobs([_entry(str(i)) for i in range(6)], lambda _e: None, workers=2, prepare_model=prepare)
    _, job = next(jobs)
    _, prepared = job.result()
    jobs.close()
    assert sorted(returned) == ["0", "1"] and not (out / "model").exists()
    with pytest.raises(ValueError, match="closed"):
        prepared.finish()
    _clean(out)


def test_preparation_refuses_source_that_writes_a_sibling(tmp_path):
    def source(root):
        own = _local_source(root, _entry("own"))
        _local_source(root, _entry("foreign"))
        return own

    with pytest.raises(ValueError, match="sibling"):
        staging.prepare_staged(source, lambda written, _r: written, tmp_path, member="model/own")
    _clean(tmp_path)


def test_default_models_materialized_and_micro_routes_remain_serial(tmp_path):
    caller = threading.get_ident()
    entries = [
        _entry("first"),
        {**_entry("materialized"), "materialized_capture": {"selected": True}},
        {**_entry("micro"), "micro_model": True},
    ]
    observed = []

    def write(entry):
        assert threading.get_ident() == caller
        observed.append(entry["name"])
        return None

    list(ordered_jobs(entries, write, workers=2))
    assert observed == [e["name"] for e in entries]
    observed.clear()
    list(ordered_jobs(entries[1:], write, workers=2, prepare_model=lambda _e: pytest.fail("legacy model")))
    assert observed == ["materialized", "micro"]


@pytest.mark.parametrize("parallel_safe", [True, False, "unsupported truthy value"])
def test_generation_selects_split_only_from_actual_parallel_capability(tmp_path, monkeypatch, parallel_safe):
    from merlin_experiments.phase0 import generation, parallel_capture, sealed_generation
    from test_phase0_m2m_preflight import _generation_with_entries

    entries = [_entry("first"), _entry("second")]
    run, _, upstream = _generation_with_entries(monkeypatch, tmp_path, entries)
    source = SealedCaptureSource(
        {"venv": str(tmp_path), "m2m_root": str(upstream), "runs_root": str(tmp_path / "runs")}
    )
    source.parallel_safe = parallel_safe
    monkeypatch.setattr(sealed_generation, "bind_source", lambda *_a, **_k: source)
    monkeypatch.setattr(parallel_capture, "worker_limit", lambda: 2)
    main = threading.get_ident()
    barrier = threading.Barrier(2, timeout=5)
    phases = []

    def write_source(entry, _binding, root, *_a, **_k):
        assert (threading.get_ident() != main) is (parallel_safe is True)
        if parallel_safe is True:
            barrier.wait()
        phases.append(("source", entry["name"]))
        return _local_source(root, entry)

    def finish(entry, _binding, _root, written, **_kwargs):
        assert threading.get_ident() == main
        phases.append(("finish", entry["name"]))
        return written

    def full(entry, binding, root, *_a, **_k):
        assert parallel_safe is not True
        return finish(entry, binding, root, write_source(entry, binding, root))

    monkeypatch.setattr(generation, "_write_capsule_source", write_source)
    monkeypatch.setattr(generation, "_finish_capsule", finish)
    monkeypatch.setattr(generation, "_write_capsule", full)
    assert [p.name for p in run()] == ["first", "second"]
    assert [name for phase, name in phases if phase == "finish"] == ["first", "second"]
    _clean(tmp_path / "capsules")


def test_generation_fatal_publication_failure_drains_and_removes_pending_stages(tmp_path, monkeypatch):
    from merlin_experiments.phase0 import generation, operator_exclusion, parallel_capture, sealed_generation
    from test_phase0_m2m_preflight import _generation_with_entries

    run, _, upstream = _generation_with_entries(monkeypatch, tmp_path, [_entry("first"), _entry("second")])
    source = SealedCaptureSource(
        {"venv": str(tmp_path), "m2m_root": str(upstream), "runs_root": str(tmp_path / "runs")}
    )
    monkeypatch.setattr(sealed_generation, "bind_source", lambda *_a, **_k: source)
    monkeypatch.setattr(parallel_capture, "worker_limit", lambda: 2)
    barrier = threading.Barrier(2, timeout=5)
    completed = []

    def write_source(entry, _binding, root, *_a, **_k):
        barrier.wait()
        completed.append(entry["name"])
        return _local_source(root, entry)

    def refused(*_a, **_k):
        raise operator_exclusion.PublicationRefusal()

    monkeypatch.setattr(generation, "_write_capsule_source", write_source)
    monkeypatch.setattr(generation, "_finish_capsule", refused)
    with pytest.raises(operator_exclusion.PublicationRefusal):
        run()
    assert sorted(completed) == ["first", "second"]
    assert not (tmp_path / "capsules/model").exists()
    _clean(tmp_path / "capsules")
