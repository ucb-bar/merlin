"""A jailed full-call child reads a data-bound runtime from its declared inputs, never a support plugin.

With a contract-bound backend selected there is no support provider to mount. The jail instead exposes
the selected contract and facts files and, for an engine run, every member the backend's identity check
and process reopen. A plugin-selected backend keeps its support root.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from merlin.targetgen import full_call_build as FB


def _bound(paths, calls):
    def execution_input_paths(engine):
        calls.append(engine)
        return paths

    return SimpleNamespace(execution_input_paths=execution_input_paths)


def test_bound_runtime_exposes_its_selection_and_declared_engine_inputs(monkeypatch, tmp_path):
    contract, facts, simulator = tmp_path / "contract.json", tmp_path / "facts.json", tmp_path / "simulator"
    calls = []
    monkeypatch.setattr("merlin.runtime.backends.base._contract_backend", lambda target: _bound((simulator,), calls))
    monkeypatch.setattr(
        "merlin.targetgen.plugins.resolve_support", lambda target: pytest.fail("a bound runtime has no support plugin")
    )
    monkeypatch.setenv("MERLIN_TARGET_CONTRACT", str(contract))
    monkeypatch.setenv("MERLIN_RTL_FACTS", str(facts))
    assert FB._selected_roots("fixture", "rtl") == [str(contract), str(facts), simulator]
    assert FB._selected_roots("fixture") == [str(contract), str(facts)] and calls == ["rtl"]
    roots = FB._merlin_roots("fixture", "rtl")
    assert simulator in roots and str(contract) in roots


def test_bound_runtime_without_explicit_selection_files_refuses(monkeypatch):
    monkeypatch.setattr("merlin.runtime.backends.base._contract_backend", lambda target: _bound((), []))
    monkeypatch.delenv("MERLIN_RTL_FACTS", raising=False)
    monkeypatch.setenv("MERLIN_TARGET_CONTRACT", "/selected/contract.json")
    with pytest.raises(RuntimeError, match="explicitly selected contract and facts"):
        FB._selected_roots("fixture", "rtl")


def test_plugin_runtime_keeps_its_support_root(monkeypatch, tmp_path):
    monkeypatch.setattr("merlin.runtime.backends.base._contract_backend", lambda target: None)
    monkeypatch.setattr("merlin.targetgen.plugins.resolve_support", lambda target: SimpleNamespace(base=tmp_path))
    assert FB._selected_roots("fixture", "rtl") == [tmp_path]


@pytest.mark.skipif(
    not os.environ.get("MERLIN_TEST_BOUND_CONTRACTS"),
    reason="set MERLIN_TEST_BOUND_CONTRACTS (os.pathsep list) and MERLIN_RTL_FACTS to prepared bound contracts",
)
@pytest.mark.parametrize(
    "contract", [p for p in os.environ.get("MERLIN_TEST_BOUND_CONTRACTS", "").split(os.pathsep) if p]
)
def test_operator_bound_contract_resolves_full_call_rtl_roots(monkeypatch, contract):
    """With an operator's prepared contract, the L3 jail exposes the receipt-bound engine and its command inputs."""
    import json

    from merlin.runtime.backends import base

    monkeypatch.setenv("MERLIN_TARGET_CONTRACT", contract)
    monkeypatch.setenv("MERLIN_TARGET_PATH", "")
    selected = json.loads(Path(contract).read_text(encoding="utf-8"))
    target = selected["name"]
    backend = base.get_backend(target)
    for engine, block in selected["runner"]["chipyard_rocc"]["engines"].items():
        if engine == "gsim":
            continue
        roots = {Path(p) for p in FB._merlin_roots(target, engine) if p}
        assert {Path(block["binary"]["path"]), Path(block["cwd"]), Path(contract)} <= roots
        assert set(backend.execution_input_paths(engine)) <= roots
        if "receipt" in block:
            assert Path(block["receipt"]["path"]) in roots
