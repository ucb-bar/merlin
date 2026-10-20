"""Parsed-source reuse never substitutes a source or linked-build decision."""

from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import dataclass

import pytest
from merlin_experiments.phase1.feedback import private_linked_source_preparation as preparation
from merlin_experiments.phase1.feedback import private_model_chain as chain
from xdsl.dialects.builtin import StringAttr

from merlin.common import mlir_query as mq

TEXT = "builtin.module { func.func @neutral(%x: tensor<2xf32>) -> tensor<2xf32> { func.return %x : tensor<2xf32> } }"


@dataclass
class Owner:
    pass


def _source(tmp_path):
    path = tmp_path / "model.mlir"
    path.write_text(TEXT)
    return path


def _count(monkeypatch):
    calls = []
    original = mq.parse

    def parse(*args, **kwargs):
        calls.append(None)
        return original(*args, **kwargs)

    monkeypatch.setattr(mq, "parse", parse)
    return calls


def test_explicit_scope_reuses_only_parse_and_returns_fresh_metadata(tmp_path, monkeypatch):
    path, calls = _source(tmp_path), _count(monkeypatch)
    original = preparation.source(path)
    assert preparation.source(path)[3] is not original[3]
    assert len(calls) == 2
    with preparation.validation_scope():
        first = preparation.source(path)
        first[2]["input_sha256"] = "forged"
        with preparation.validation_scope():
            again = preparation.source(path)
        assert first[3] is again[3]
        assert again[:2] == original[:2] and again[2] == original[2]
    assert len(calls) == 3
    preparation.source(path)
    assert len(calls) == 4


def test_owned_preparation_reopens_across_calls_but_not_owners(tmp_path, monkeypatch):
    path, calls, owner = _source(tmp_path), _count(monkeypatch), Owner()
    prepared = preparation.Preparation(owner)
    for _ in range(2):
        with prepared.activate(owner):
            assert preparation.source(path)[0] == path.read_bytes()
    assert len(calls) == 1
    with pytest.raises(preparation.PreparationChanged, match="foreign"):
        with prepared.activate(Owner()):
            pass


def test_unissued_and_expired_preparations_refuse(tmp_path):
    owner = Owner()
    with pytest.raises(preparation.PreparationChanged):
        with preparation.Preparation.__new__(preparation.Preparation).activate(owner):
            pass
    prepared = preparation.Preparation(owner)
    del owner
    with pytest.raises(preparation.PreparationChanged):
        with prepared.activate(Owner()):
            pass


@pytest.mark.parametrize("at_exit", [False, True])
def test_changed_raw_bytes_or_parsed_ir_refuse_even_after_last_consumer(tmp_path, at_exit):
    path, owner = _source(tmp_path), Owner()
    prepared = preparation.Preparation(owner)
    with pytest.raises(preparation.PreparationChanged):
        with prepared.activate(owner):
            module = preparation.source(path)[3]
            if at_exit:
                module.attributes["changed"] = StringAttr("different")
            else:
                path.write_text(TEXT + "\n")
                preparation.source(path)


def test_raw_snapshot_path_and_alias_are_independent(tmp_path, monkeypatch):
    path, calls = _source(tmp_path), _count(monkeypatch)
    other, alias = tmp_path / "other.mlir", tmp_path / "alias.mlir"
    other.write_bytes(path.read_bytes())
    alias.symlink_to(path)
    with preparation.validation_scope():
        first = preparation.source(path)
        assert preparation.source(other)[3] is not first[3]
        assert len(calls) == 2
        with pytest.raises(preparation.PreparationChanged, match="snapshot"):
            preparation.source(path, raw=TEXT.encode() + b"\n")
        with pytest.raises(preparation.PreparationChanged, match="canonical"):
            preparation.source(alias)


def test_copied_context_cannot_reuse_preparation_on_another_thread(tmp_path):
    path = _source(tmp_path)
    with preparation.validation_scope():
        preparation.source(path)
        context = copy_context()
        with ThreadPoolExecutor(max_workers=1) as pool:
            with pytest.raises(preparation.PreparationChanged, match="foreign"):
                pool.submit(context.run, preparation.source, path).result(timeout=5)


def test_saved_context_expires_even_while_original_owner_survives(tmp_path):
    path, owner = _source(tmp_path), Owner()
    prepared = preparation.Preparation(owner)
    with prepared.activate(owner):
        preparation.source(path)
        old = copy_context()
    with pytest.raises(preparation.PreparationChanged, match="expired"):
        old.run(preparation.source, path)
    with prepared.activate(owner):
        assert preparation.source(path)[0] == path.read_bytes()
        with pytest.raises(preparation.PreparationChanged, match="expired"):
            old.run(preparation.source, path)


def test_failed_validation_unwinds_scope_and_keeps_no_decision(tmp_path):
    path, owner = _source(tmp_path), Owner()
    prepared = preparation.Preparation(owner)
    with pytest.raises(RuntimeError, match="failed witness"):
        with prepared.activate(owner):
            preparation.source(path)
            raise RuntimeError("failed witness")
    with prepared.activate(owner):
        assert preparation.source(path)[0] == path.read_bytes()


def test_inherited_process_cannot_reuse_preparation(tmp_path):
    import multiprocessing
    import os

    path = _source(tmp_path)

    def inherited():
        try:
            preparation.source(path)
        except preparation.PreparationChanged:
            os._exit(0)
        os._exit(2)

    with preparation.validation_scope():
        preparation.source(path)
        process = multiprocessing.get_context("fork").Process(target=inherited)
        process.start()
        try:
            process.join(timeout=5)
            assert process.exitcode == 0
        finally:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)


def test_native_partition_rechecks_every_verdict_using_owner_local_parse(tmp_path, monkeypatch):
    path, calls = _source(tmp_path), _count(monkeypatch)
    record_path = tmp_path / "native-static.json"
    record_path.write_text("{}")
    selection = Owner()
    selection.verify = lambda: None
    partition = chain.NativePartition(record_path, {}, selection, tmp_path / "invocation.json")
    decisions = []
    monkeypatch.setattr(chain, "_producer", lambda *a: None)

    def verify(record):
        decisions.append(record)
        with preparation.validation_scope():
            preparation.source(path)

    monkeypatch.setattr(chain, "_verify_record", verify)
    partition.verify()
    partition.verify()
    assert len(decisions) == 2 and len(calls) == 1
    path.write_text(TEXT + "\n")
    with pytest.raises(preparation.PreparationChanged):
        partition.verify()
    assert len(decisions) == 2
