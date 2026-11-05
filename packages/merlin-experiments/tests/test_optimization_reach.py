"""The architecture-selected optimization advice reaches every author, and its hints follow measurements."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import yaml

from merlin_experiments.phase1 import compiler_library as CL
from merlin_experiments.phase1 import joint_control
from merlin_experiments.phase1 import optimization_suggestions as S
from merlin_experiments.phase1.feedback import mechanism_hints as M

REPO = Path(__file__).resolve().parents[3]


def _catalog():
    return json.loads((REPO / "docs/design/joint_optimization_catalog.json").read_text())


def _workspace(tmp_path):
    ws, run = tmp_path / "ws", tmp_path / "run"
    (ws / "public_declarations").mkdir(parents=True)
    run.mkdir()
    spec = {"numerical_semantics": {"operand_dtype": "int8", "accumulator_dtype": "int32"}}
    (ws / "public_declarations/software-spec.yaml").write_text(yaml.safe_dump(spec))
    (ws / "TASK.md").write_text("task\n")
    return ws, run


def test_full_source_runs_receive_the_architecture_selection(tmp_path, monkeypatch):
    ws, run = _workspace(tmp_path)
    full = SimpleNamespace(record=lambda: {"schema": "full"}, manifest_sha256="0" * 64,
                           members=(("a.py", "1" * 64, "merlin.compile.capacity"), ("README", "2" * 64, None)))
    monkeypatch.setattr("merlin_experiments.phase1.full_compiler_workspace.selected", lambda: full)
    monkeypatch.setattr(joint_control, "selected", lambda: object())
    CL.append_guidance(ws, run, None, None, repo=REPO, descriptor=None)
    report = json.loads((ws / "OPTIMIZATION_SUGGESTIONS.json").read_text())
    assert (run / "OPTIMIZATION_SUGGESTIONS.json").is_file() and (ws / "optimization_suggest.py").is_file()
    assert json.loads((ws / "OPTIMIZATION_API_SELECTION.json").read_text())["public_modules"] == ["merlin.compile.capacity"]
    assert "Capability-selected optimization suggestions" in (ws / "TASK.md").read_text()
    assert "Performance-driven compiler exploration" in (ws / "TASK.md").read_text()
    assert {row["id"] for row in report["all_mechanisms"]} >= {"capacity_tiles"}


def test_a_suggestion_failure_is_reported_not_fatal(tmp_path, monkeypatch, capsys):
    ws, run = _workspace(tmp_path)
    monkeypatch.setattr(joint_control, "selected", lambda: object())

    def broken(*_args, **_kwargs):
        raise ValueError("catalogue unreadable")

    monkeypatch.setattr(S, "prepare", broken)
    text = CL._performance_guidance(ws, run, {"public_modules": [], "contract_sha256": "0" * 64}, repo=REPO)
    assert "Performance-driven compiler exploration" in text
    assert "[optimization-suggestions] unavailable: ValueError" in capsys.readouterr().out
    assert CL._performance_guidance(ws, run, {}, repo=None) != "" and "no repository root" in capsys.readouterr().out


def test_declared_instruction_semantics_supply_host_and_transfer_traits():
    contract = {"endpoint_kind": "inline_asm_insn",
                "corpus_instruction_roles": {"by_role": {"memory": ["A"], "acc_readout": ["B"]}}}
    caps, evidence = S.contract_traits(contract)
    assert caps["host.scalar"] == caps["memory.explicit_dma"] == "supported"
    assert "endpoint_kind" in evidence["host.scalar"] and "by_role" in evidence["memory.explicit_dma"]
    caps, _ = S.contract_traits({"corpus_instruction_roles": {"by_role": {"memory": ["A"]}}})
    assert "memory.explicit_dma" not in caps and "host.scalar" not in caps


def test_ir_pending_mechanisms_are_kept_apart_from_capability_questions():
    catalog = _catalog()
    modules = [entry["compiler"]["module"] for entry in catalog["mechanisms"]]
    caps = {key: "supported" for key in ("compute.systolic", "memory.explicit_local_storage", "memory.scratchpad",
                                         "memory.local_memory", "memory.explicit_dma")}
    report = S.select(catalog, capabilities=caps, public_modules=modules, limit=64)
    pending = {row["id"] for row in report["pending_workload_check"]}
    discovery = {row["id"] for row in report["discovery"]}
    assert pending and not pending & discovery
    for row in report["all_mechanisms"]:
        if row["id"] in pending:
            assert all(v == "supported" for k, v in row["requirements"].items() if not k.startswith("workload."))


def _selection():
    catalog = _catalog()
    modules = [entry["compiler"]["module"] for entry in catalog["mechanisms"]]
    caps = {key: "supported" for key in ("compute.systolic", "memory.explicit_local_storage", "memory.scratchpad",
                                         "memory.local_memory", "memory.accumulator", "memory.explicit_dma",
                                         "host.scalar", "submission.cpu_issued")}
    return S.select(catalog, capabilities=caps, public_modules=modules, limit=64)


def test_hints_follow_the_measured_resource_and_only_name_available_mechanisms():
    selection = _selection()
    available = {row["id"] for row in M._available(selection)}
    storage = M.hints(selection, dominant={"name": "accelerator.storage.accumulator.all", "modeled_activity": 0.7})
    issue = M.hints(selection, dominant={"name": "cpu.issue", "modeled_activity": 0.9})
    assert storage and len({h["facet"] for h in storage}) > 1  # spans the bottleneck's facets
    # Mechanisms whose catalogue entry names the busy unit lead; the facet hints follow them.
    named = [h for h in storage if h["why"].startswith("names the busy accumulator unit")]
    assert named and storage[:len(named)] == named and len(named) < len(storage)
    assert {h["facet"] for h in storage[len(named):]} <= {"storage", "movement", "overlap"}
    assert all(h["id"] in available and ":" in h["api"] and h["evidence_to_request"] for h in storage + issue)
    assert all(h["facet"] in {"issue", "synchronization", "scheduling"} for h in issue)
    assert {h["facet"] for h in M.hints(selection, bound_by="read requests")} <= {"movement", "overlap", "storage"}
    assert M.hints(None, dominant={"name": "cpu.issue"}) == [] and M.hints(selection) == []


def test_unavailable_capabilities_never_become_hints():
    catalog = _catalog()
    report = S.select(catalog, capabilities={}, public_modules=[], limit=64)
    assert M.hints(report, dominant={"name": "accelerator.storage.scratchpad.0", "modeled_activity": 0.5}) == []


def test_an_accumulator_bound_case_is_offered_the_scheduling_library_accumulator_passes():
    """The readout-frontier and readout/compute-overlap passes name the accumulator in their catalogue
    entries, so a case whose busiest modeled resource is the accumulator is offered them first."""
    catalog = _catalog()
    capabilities = {f"{ns}.{term}": "supported" for ns, term in (
        ("memory", "accumulator"), ("memory", "scratchpad"), ("numeric", "int32_accumulation"),
        ("scheduling", "async_issue"), ("compute", "systolic"), ("memory", "resident_operands"))}
    modules = sorted({entry["compiler"]["module"] for entry in catalog["mechanisms"]})
    report = S.select(catalog, capabilities=capabilities, public_modules=modules, limit=64)
    hints = M.hints(report, dominant={"name": "accelerator.storage.accumulator.all", "modeled_activity": 0.75})
    named = {h["id"] for h in hints if h["why"].startswith("names the busy accumulator unit")}
    assert named & {"final_k_accumulator_readout_frontier", "accumulator_readout_compute_overlap"}
