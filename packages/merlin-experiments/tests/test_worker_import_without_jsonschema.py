"""Broker workers run under interpreters (the model capture Python) that lack jsonschema."""

import os
import subprocess
import sys

from merlin.common.paths import repo_root


def test_model_check_worker_imports_without_jsonschema():
    root = repo_root()
    path = os.pathsep.join(str(root / p) for p in ("src", "packages/merlin-experiments/src", "packages/merlin-timing/src"))
    code = "import sys; sys.modules['jsonschema'] = None; import merlin_experiments.phase1.brokers.model_check_worker"
    result = subprocess.run([sys.executable, "-c", code], env={**os.environ, "PYTHONPATH": path},
                            capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr[-2000:]
