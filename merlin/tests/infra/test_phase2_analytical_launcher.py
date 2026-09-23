"""Native compatibility imports retain the installed fast-evaluation owner."""

import importlib

from merlin_experiments.phase2 import fast_evaluation_installation as F

from merlin.common.paths import repo_root


def test_native_fast_evaluation_helpers_are_installed_aliases(monkeypatch):
    scripts = repo_root() / "merlin/experiments/gemmini_perf_bench/scripts"
    monkeypatch.syspath_prepend(str(scripts))
    launcher = importlib.import_module("launch_global_agent_experiment")
    assert launcher._prepare_fast_evaluator_installation is F.prepare
    assert launcher._fast_evaluation_worker_arguments is F.worker_arguments
    assert launcher._validate_fast_evaluation_cli is F.validate_cli
