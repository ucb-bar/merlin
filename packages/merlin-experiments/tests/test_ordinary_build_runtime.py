"""Original normal policy routing; tiny fake tool files are never executed."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2 import build_view_configuration as CFG
from merlin_experiments.phase2 import ordinary_build_runtime as ROUTE
from merlin_experiments.phase2 import ordinary_fast_feedback as FAST
from merlin_experiments.phase2 import ordinary_search_policy as OS
from merlin_experiments.phase2.contracts import StageGateError
from merlin_experiments.phase2.search_configuration import read_search_configuration
from test_ordinary_fast_feedback import cheap_inputs as cheap_inputs
from test_ordinary_search_policy import ordinary_inputs as ordinary_inputs
from test_ordinary_search_policy import revisions
from test_search_configuration import selected as selected

from merlin.runtime import build_runtime as VIEW
from merlin.targetgen.contract import build_products as PRODUCTS


@pytest.fixture
def owners(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "storage"))
    system = tmp_path / "system"
    (system / "bin").mkdir(parents=True)
    (system / "usr/bin").mkdir(parents=True)
    tool, bwrap = tmp_path / "compiler", tmp_path / "bwrap"
    for path in (system / "bin/bash", system / "usr/bin/env", tool, bwrap):
        path.write_bytes(b"fake source-only executable; never launched")
        path.chmod(0o555)
    output = tmp_path / "output"
    output.mkdir()
    view = VIEW.prepare_build_runtime_view(
        tmp_path / "lookup",
        tools=(tool,),
        readonly_roots=((system / "bin", Path("/bin")), (system / "usr", Path("/usr"))),
        output_root=output,
        bwrap=bwrap,
        max_files=256,
        max_bytes=1 << 20,
        max_output_files=32,
        max_output_bytes=4096,
        max_console_bytes=4096,
    )
    yield SimpleNamespace(view=view, output=output)
    if not view._closed:
        view.close()


@pytest.mark.parametrize("reuse", [True, False])
def test_original_member_workers_receive_exact_build_owner_and_reuse_choice(
    ordinary_inputs, owners, tmp_path, monkeypatch, reuse
):
    original, seen = OS.PME.run_execution, []

    def execute(*args, **kwargs):
        assert VIEW.active_build_runtime() is owners.view
        active = PRODUCTS._ACTIVE.get()
        assert active is None if not reuse else active[0] is index
        seen.append(args[0].arm)
        return original(*args, **kwargs)

    # Select the harmless execution boundary before original admission.
    monkeypatch.setattr(OS.PME, "run_execution", execute)
    policy = OS.prepare_ordinary_search_policy(**ordinary_inputs.arguments)
    index = PRODUCTS.prepare_build_product_index(
        owners.output / "cache",
        context=policy.artifact_retest_context,
        build_owner=owners.view,
        max_entries=16,
        max_bytes=1 << 20,
        max_files=128,
    )
    original_context = policy.artifact_retest_context
    policy.install_build_runtime(owners.view, index)
    assert policy.artifact_retest_context == original_context
    assert policy.binding()["build_runtime_view"] == owners.view.verify()
    candidate, baseline = revisions(policy, tmp_path)
    try:
        result = policy.evaluate_retained(
            candidate,
            baseline,
            work_root=owners.output / "job",
            call_index=0,
            timeout_s=20,
            workers=1,
            reuse_artifacts=reuse,
        )
        assert result["summary"]["all_correct"]
        assert set(seen) == {"baseline", "candidate"}
        assert PRODUCTS._ACTIVE.get() is None and VIEW.active_build_runtime() is None
    finally:
        index.close()


def test_original_cheap_workers_adopt_view_before_creating_fresh_job(cheap_inputs, owners, tmp_path, monkeypatch):
    inputs, calls = cheap_inputs
    original = FAST.CR.run_capsule

    def run(*args, **kwargs):
        assert VIEW.active_build_runtime() is owners.view
        assert PRODUCTS._ACTIVE.get()[0] is index
        return original(*args, **kwargs)

    monkeypatch.setattr(FAST.CR, "run_capsule", run)
    policy = OS.prepare_ordinary_search_policy(**inputs.arguments)
    index = PRODUCTS.prepare_build_product_index(
        owners.output / "cache",
        context=policy.artifact_retest_context,
        build_owner=owners.view,
        max_entries=16,
        max_bytes=1 << 20,
        max_files=128,
    )
    policy.install_build_runtime(owners.view, index)
    candidate, baseline = revisions(policy, tmp_path)
    work = owners.output / "fresh_cheap"
    assert not work.exists()
    try:
        result = policy.evaluate_retained(
            candidate, baseline, kind=FAST.KIND, work_root=work, call_index=0, timeout_s=20, workers=1
        )
        assert len(calls) == 2 * len(policy.search_corpus.capsules)
        assert result["promotion"] == "SCREENING_ONLY"
        assert PRODUCTS._ACTIVE.get() is None and VIEW.active_build_runtime() is None
    finally:
        index.close()


def declaration():
    return {
        "tools": ["/original/compiler"],
        "readonly_roots": [{"source": "/original/libraries", "guest": "/lib"}],
        "bwrap": "/original/bwrap",
        "snapshot_files": 128,
        "snapshot_bytes": 16 << 20,
        "cache_entries": 16,
        "cache_bytes": 16 << 20,
    }


@pytest.mark.parametrize("change", [None, "callback", "budget", "relative", "foreign_root", "bool"])
def test_build_config_is_closed_and_stays_inside_original_reserve(selected, change):
    path, document = selected
    data = declaration()
    if change == "callback":
        data["callback"] = "unissued"
    elif change == "budget":
        data["snapshot_bytes"] = document["resources"]["disk_bytes"]
    elif change == "relative":
        data["tools"] = ["../compiler"]
    elif change == "foreign_root":
        data["readonly_roots"] = [{"source": "/original/libraries", "guest": "/lib", "role": "qualified"}]
    elif change == "bool":
        data["cache_entries"] = True
    document["build_runtime"] = data
    path.write_text(json.dumps(document))
    if change is None:
        configuration = read_search_configuration(path)
        CFG.validate(configuration)
        assert configuration.document["build_runtime"] == data
    else:
        with pytest.raises(StageGateError):
            read_search_configuration(path)


@pytest.mark.parametrize("terminal", [True, False])
def test_stage_cleanup_retains_owners_when_original_cleanup_is_unknown(terminal):
    closed = []
    policy = SimpleNamespace(
        build_runtime_view=SimpleNamespace(close=lambda: closed.append("view")),
        build_product_index=SimpleNamespace(close=lambda: closed.append("index")),
        evaluation_pool=SimpleNamespace(observations=lambda: [{"terminal": terminal}]),
    )
    if terminal:
        ROUTE.close_stage(policy)
        assert closed == ["index", "view"]
    else:
        with pytest.raises(StageGateError, match="cleanup UNKNOWN"):
            ROUTE.close_stage(policy)
        assert closed == []
