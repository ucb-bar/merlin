"""Trusted startup selection controls; fake process output grants no RTL qualification."""

import copy
import json

import pytest
from test_bound_chipyard_rocc_backend import _bound, _console, _file, _gsim

from merlin.runtime import native_runtime as NR
from merlin.runtime.backends import chipyard_rocc_bound as B

pytest_plugins = ("test_bound_chipyard_rocc_backend",)


def test_selected_preload_uses_exact_verified_elf_and_records_fresh_mode(selection, tmp_path, monkeypatch):
    _gsim(selection, tmp_path)
    bound = _bound(selection)
    monkeypatch.delenv("MERLIN_GSIM_STARTUP_PRELOAD", raising=False)
    before = bound.execution_engine_identity("gsim")
    view_before = NR._backend_data(bound)
    monkeypatch.setenv("MERLIN_GSIM_STARTUP_PRELOAD", "1")
    after = bound.execution_engine_identity("gsim")
    assert after["startup_transport"] == "original_dram_elf_preload_v1"
    assert after != before and after["receipt"] == before["receipt"]
    view_after = NR._backend_data(bound)
    assert view_after["selection"] != view_before["selection"]
    assert view_after["argv"] == ("{elf}", "+loadmem={elf}")
    elf = _file(tmp_path, "requested.elf", "the original requested executable")
    assert bound.run_elf(elf, simulator="gsim", timeout=5) == _console()
    records = [json.loads(p.read_text()) for p in tmp_path.glob("invocations/*/invocation.json")]
    actual = next(row for row in records if row.get("stage") == "selected_rocc_process")
    assert actual["argv"][1:] == [str(elf), "+loadmem=" + str(elf)]


def test_unselected_startup_and_other_engine_keep_original_identity_and_argv(selection, tmp_path, monkeypatch):
    spike = _bound(copy.deepcopy(selection))
    spike_identity = spike.execution_engine_identity("spike")
    spike_argv = spike.execution_argv_template("spike")
    _gsim(selection, tmp_path)
    bound = _bound(selection)
    monkeypatch.delenv("MERLIN_GSIM_STARTUP_PRELOAD", raising=False)
    original = bound.execution_engine_identity("gsim")
    monkeypatch.setenv("MERLIN_GSIM_STARTUP_PRELOAD", "0")
    assert bound.execution_engine_identity("gsim") == original
    assert bound.execution_argv_template("gsim") == ("{elf}",)
    monkeypatch.setenv("MERLIN_GSIM_STARTUP_PRELOAD", "1")
    assert spike.execution_engine_identity("spike") == spike_identity
    assert spike.execution_argv_template("spike") == spike_argv


@pytest.mark.parametrize("selection_value", ["invalid", "-1", "2"])
def test_invalid_preload_selection_refuses_before_process(selection, tmp_path, monkeypatch, selection_value):
    _gsim(selection, tmp_path)
    bound = _bound(selection)
    monkeypatch.setenv("MERLIN_GSIM_STARTUP_PRELOAD", selection_value)
    with pytest.raises(B.RoCCExecutionError, match="startup preload must be 0 or 1"):
        bound.execution_engine_identity("gsim")


def test_conflicting_elf_preload_is_refused(selection, tmp_path, monkeypatch):
    engine, _, _ = _gsim(selection, tmp_path)
    engine["argv"].append("+loadmem=/another/program.elf")
    monkeypatch.setenv("MERLIN_GSIM_STARTUP_PRELOAD", "1")
    with pytest.raises(B.RoCCExecutionError, match="conflicts with another ELF"):
        _bound(selection).execution_engine_identity("gsim")


def test_preloaded_elf_mutation_still_refuses_before_result_publication(selection, tmp_path, monkeypatch):
    _gsim(selection, tmp_path)
    bound = _bound(selection)
    elf = _file(tmp_path, "requested.elf", "the original requested executable")
    monkeypatch.setenv("MERLIN_GSIM_STARTUP_PRELOAD", "1")
    original = B.subprocess.Popen

    def launch(*args, **kwargs):
        process = original(*args, **kwargs)
        elf.write_text("changed requested executable bytes")
        return process

    monkeypatch.setattr(B.subprocess, "Popen", launch)
    with pytest.raises(B.RoCCExecutionError, match="engine or ELF changed"):
        bound.run_elf(elf, simulator="gsim", timeout=5)
