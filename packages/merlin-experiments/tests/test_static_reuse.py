"""Retained static reuse admission uses actual compiler closure identities, without execution."""

import copy
import json
import subprocess

import pytest
from merlin_experiments.phase2 import contracts
from merlin_experiments.phase2 import static_identity as identity


@pytest.fixture(autouse=True)
def refuse_processes(monkeypatch):
    def refused(*args, **kwargs):
        pytest.fail("static reuse admission must not launch a process")

    monkeypatch.setattr(subprocess, "Popen", refused)


def seal_record(row, options):
    path = options["output"] / f"iteration_{row['iteration']:04d}.json"
    if path.exists():
        path.chmod(0o644)
    path.write_text(json.dumps(row))
    path.chmod(0o444)
    options["record_sha256"][row["iteration"]] = contracts.sha256_file(path)
    return path


@pytest.fixture
def retained(tmp_path):
    output = tmp_path / "journal"
    output.mkdir()
    shared = tmp_path / "shared"
    shared.mkdir()
    (shared / "__init__.py").write_text("")
    (shared / "helper.py").write_text("VALUE = 1\n")
    candidate = output / "candidate"
    candidate.mkdir()
    program = candidate / "compiler.py"
    program.write_text("from merlin.helper import VALUE\nraise RuntimeError('never execute candidate')\n")
    dependencies = identity.compiler_dependency_record(candidate, shared_source_root=shared)
    assert "helper.py" in dependencies["shared_sources"]
    program.chmod(0o444)
    candidate.chmod(0o555)
    binding = {"candidate_sha256": dependencies["candidate_sha256"], "compiler_dependencies": dependencies}
    capsules = ["a" * 64, "b" * 64]
    options = {
        "binding": binding,
        "output": output,
        "record_sha256": {},
        "compiler_shared_source_root": shared,
        "capsule_sha256s": capsules,
        "portfolio_sha256": "c" * 64,
    }
    analysis = {"candidate_sha256": binding["candidate_sha256"], "workload": {"capsule_sha256": capsules[0]}}
    row = {
        "schema": "global_perf_iteration_v1",
        "iteration": 0,
        "analysis_reuse_binding": copy.deepcopy(binding),
        **copy.deepcopy(binding),
        "readiness": {"status": "ready_for_probe_admission", "blockers": []},
        "elapsed_seconds": 1.0,
        "allocated_seconds": 2.0,
        "submitted_snapshot": str(candidate),
        "analysis": analysis,
        "portfolio": {
            "candidate_sha256": binding["candidate_sha256"],
            "portfolio_sha256": options["portfolio_sha256"],
            "members_total": 2,
            "members": [
                {
                    "identity": {"capsule_sha256": capsule},
                    "analysis": {
                        "candidate_sha256": binding["candidate_sha256"],
                        "workload": {"capsule_sha256": capsule},
                    },
                    "readiness": {"status": "ready_for_probe_admission"},
                }
                for capsule in capsules
            ],
        },
    }
    path = seal_record(row, options)
    return row, options, path, candidate


def test_reuse_reads_original_static_record_without_live_probe_additions(retained):
    row, options, path, _ = retained
    expected = copy.deepcopy(row)
    original_bytes = path.read_bytes()
    row["probe_receipts"] = [{"sha256": "d" * 64}]
    row["decision_feedback"] = {"status": "synthetic feedback"}
    admitted = identity.immutable_reusable_iteration(row, **options)
    assert admitted == expected
    assert admitted is not row
    assert "probe_receipts" not in admitted
    assert path.read_bytes() == original_bytes
    assert identity.find_reusable_iteration([row], **options) == expected


@pytest.mark.parametrize("mutation", ["bytes", "writable", "symlink", "missing"])
def test_record_storage_changes_are_cache_misses(retained, mutation):
    row, options, path, _ = retained
    if mutation == "bytes":
        path.chmod(0o644)
        path.write_bytes(path.read_bytes() + b" ")
        path.chmod(0o444)
    elif mutation == "writable":
        path.chmod(0o644)
    else:
        saved = path.with_suffix(".original")
        path.rename(saved)
        if mutation == "symlink":
            path.symlink_to(saved)
    assert identity.immutable_reusable_iteration(row, **options) is None


@pytest.mark.parametrize(
    "mutation", ["bytes", "writable_file", "writable_root", "symlink", "symlink_file", "outside", "ephemeral"]
)
def test_candidate_snapshot_changes_are_cache_misses(retained, tmp_path, mutation):
    row, options, _, candidate = retained
    program = candidate / "compiler.py"
    if mutation == "bytes":
        program.chmod(0o644)
        program.write_text("VALUE = 2\n")
        program.chmod(0o444)
    elif mutation == "writable_file":
        program.chmod(0o644)
    elif mutation == "writable_root":
        candidate.chmod(0o755)
    elif mutation == "symlink_file":
        candidate.chmod(0o755)
        saved = tmp_path / "original.py"
        program.rename(saved)
        program.symlink_to(saved)
        candidate.chmod(0o555)
    elif mutation == "symlink":
        saved = candidate.with_name("original")
        candidate.rename(saved)
        candidate.symlink_to(saved, target_is_directory=True)
    elif mutation == "outside":
        outside = tmp_path / "outside"
        candidate.chmod(0o755)
        candidate.rename(outside)
        outside.chmod(0o555)
        row["submitted_snapshot"] = str(outside)
        seal_record(row, options)
    elif mutation == "ephemeral":
        candidate.chmod(0o755)
        excluded = candidate / "__pycache__"
        excluded.mkdir()
        excluded.chmod(0o555)
        candidate.chmod(0o555)
    assert identity.immutable_reusable_iteration(row, **options) is None


@pytest.mark.parametrize("mutation", ["candidate", "dependencies", "shared_source", "binding", "live_analysis"])
def test_identity_changes_are_cache_misses(retained, mutation):
    row, options, _, _ = retained
    if mutation == "shared_source":
        (options["compiler_shared_source_root"] / "helper.py").write_text("VALUE = 2\n")
    elif mutation == "live_analysis":
        row["analysis"]["changed"] = True
    elif mutation == "binding":
        options["binding"] = {**options["binding"], "new_policy": True}
    else:
        if mutation == "candidate":
            row["candidate_sha256"] = "e" * 64
        else:
            row["compiler_dependencies"]["compiler_implementation_sha256"] = "e" * 64
        seal_record(row, options)
    assert identity.immutable_reusable_iteration(row, **options) is None


@pytest.mark.parametrize(
    "mutation",
    [
        "order",
        "member_readiness",
        "global_readiness",
        "count",
        "portfolio_identity",
        "member_candidate",
        "member_workload",
    ],
)
def test_portfolio_identity_order_and_readiness_are_required(retained, mutation):
    row, options, _, _ = retained
    portfolio = row["portfolio"]
    if mutation == "order":
        portfolio["members"].reverse()
    elif mutation == "member_readiness":
        portfolio["members"][1]["readiness"]["status"] = "blocked"
    elif mutation == "global_readiness":
        row["readiness"]["status"] = "blocked"
    elif mutation == "count":
        portfolio["members_total"] = 1
    elif mutation == "portfolio_identity":
        portfolio["portfolio_sha256"] = "e" * 64
    elif mutation == "member_candidate":
        portfolio["members"][1]["analysis"]["candidate_sha256"] = "e" * 64
    elif mutation == "member_workload":
        portfolio["members"][1]["analysis"]["workload"]["capsule_sha256"] = "e" * 64
    seal_record(row, options)
    assert identity.immutable_reusable_iteration(row, **options) is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("elapsed_seconds", True),
        ("allocated_seconds", False),
        ("elapsed_seconds", "1"),
        ("allocated_seconds", None),
        ("elapsed_seconds", float("nan")),
        ("allocated_seconds", float("inf")),
        ("elapsed_seconds", 3.0),
    ],
)
def test_invalid_or_exceeded_time_budget_is_a_cache_miss(retained, field, value):
    row, options, _, _ = retained
    row[field] = value
    seal_record(row, options)
    assert identity.immutable_reusable_iteration(row, **options) is None


def test_explicit_budget_blocker_is_a_cache_miss(retained):
    row, options, _, _ = retained
    row["readiness"]["blockers"] = ["iteration_wall_budget_exceeded"]
    seal_record(row, options)
    assert identity.immutable_reusable_iteration(row, **options) is None


def test_newest_invalid_entry_falls_back_to_older_sealed_evidence(retained):
    first, options, _, _ = retained
    second = copy.deepcopy(first)
    second["iteration"] = 1
    second_path = seal_record(second, options)
    assert identity.find_reusable_iteration([first, second], **options) == second
    second_path.chmod(0o644)
    assert identity.find_reusable_iteration([first, second], **options) == first
    assert identity.find_reusable_iteration([second], **options) is None


@pytest.mark.parametrize(
    "statement",
    [
        "from merlin.runtime import Tensor\n",
        "import merlin.runtime as runtime\nvalue = runtime.Tensor\n",
    ],
)
def test_dependency_identity_selects_lazy_leaf_without_executing_initializers(tmp_path, statement):
    shared = tmp_path / "shared"
    shared.mkdir()
    (shared / "__init__.py").write_text('raise RuntimeError("must not import")\n')
    runtime = shared / "runtime"
    runtime.mkdir()
    (runtime / "__init__.py").write_text(
        '_EXPORTS = {"Tensor": "tensor", "simulate": "simulator"}\n'
        "from importlib import import_module\n"
        "def __getattr__(name):\n"
        "    module = _EXPORTS.get(name)\n"
        "    if module is None:\n"
        "        raise AttributeError(name)\n"
        '    return getattr(import_module(f".{module}", __name__), name)\n'
        'raise RuntimeError("must not import lazy initializer")\n'
    )
    selected = runtime / "tensor.py"
    selected.write_text("class Tensor: pass\n")
    sibling = runtime / "simulator.py"
    sibling.write_text('raise RuntimeError("must not import sibling")\n')
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "compiler.py").write_text(statement)
    before = identity.compiler_dependency_record(candidate, shared_source_root=shared)
    assert before["selected_lazy_exports"] == {"merlin.runtime.Tensor": "merlin.runtime.tensor"}
    assert "runtime/tensor.py" in before["shared_sources"]
    assert "runtime/simulator.py" not in before["shared_sources"]
    sibling.write_text("# unrelated change\n")
    assert identity.compiler_dependency_record(candidate, shared_source_root=shared) == before
    selected.write_text("class Tensor: changed = True\n")
    after = identity.compiler_dependency_record(candidate, shared_source_root=shared)
    assert after["candidate_sha256"] == before["candidate_sha256"]
    assert after["compiler_implementation_sha256"] != before["compiler_implementation_sha256"]
