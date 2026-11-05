"""Live output accounting tolerates compiler cleanup, retaining product guards."""

import os
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2 import evaluation_pool as EP
from merlin_experiments.phase2.contracts import StageGateError


def census(root):
    return EP.ComponentEvaluationPool._disk_bytes(SimpleNamespace(grant=SimpleNamespace(files=20)), root)


class Scan:
    def __init__(self, original, entries):
        self.original, self.entries = original, entries

    def __enter__(self):
        self.original.__enter__()
        return self

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.entries)

    def __exit__(self, *args):
        return self.original.__exit__(*args)


def test_removed_assembly_does_not_refuse_a_live_evaluation(tmp_path, monkeypatch):
    stable, scratch = tmp_path / "kernel.o", tmp_path / "cc.s"
    stable.write_bytes(b"object")
    scratch.write_bytes(b"assembly")
    scandir = os.scandir

    class Entry:
        def __init__(self, entry):
            self.entry = entry

        def __getattr__(self, name):
            return getattr(self.entry, name)

        def stat(self, **kwargs):
            if self.entry.name == "cc.s":
                scratch.unlink()
            return self.entry.stat(**kwargs)

    def racing_scan(path):
        original = scandir(path)
        return Scan(original, (Entry(entry) for entry in original))

    with monkeypatch.context() as patch:
        patch.setattr(EP.os, "scandir", racing_scan)
        assert census(tmp_path) == len(b"object")
    assert not scratch.exists()


def test_removed_scratch_directory_does_not_refuse_a_live_evaluation(tmp_path, monkeypatch):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    (tmp_path / "kernel.o").write_bytes(b"object")
    scandir = os.scandir

    def racing_scan(path):
        if path == scratch:
            scratch.rmdir()
        return scandir(path)

    with monkeypatch.context() as patch:
        patch.setattr(EP.os, "scandir", racing_scan)
        assert census(tmp_path) == len(b"object")
    assert not scratch.exists()


@pytest.mark.parametrize("kind", ["symlink", "fifo"])
def test_live_census_still_refuses_linked_and_special_products(tmp_path, kind):
    product = tmp_path / "product"
    if kind == "symlink":
        product.symlink_to(tmp_path / "absent")
    else:
        os.mkfifo(product)
    with pytest.raises(StageGateError, match="linked product|special product"):
        census(tmp_path)
