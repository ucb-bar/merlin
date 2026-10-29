"""Synthetic readiness metadata cannot inherit the real experiment's contract."""

import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from merlin_experiments.phase1 import author_profiles as A
from merlin_experiments.phase1 import fresh10_profiles as P
from merlin_experiments.phase1 import tooling_readiness as R
from test_fresh10_tooling_profiles import fresh10_baseline, selections  # noqa: F401 (fixtures)


@pytest.mark.parametrize('name', P.NAMES)
def test_five_synthetic_apis_select_and_restore_their_own_contract(
        name, selections, tmp_path, monkeypatch):
    selected = selections[name]
    monkeypatch.setattr(R, '_component_selection', lambda _bundle: (A.resolve(name), selected))
    generated = R._component_probe({}, schema_path=tmp_path / 'unused-schema')
    # Execute the exact tail emitted into the native sandbox. The preceding
    # import/schema/semantics checks have independent native probes.
    tail = generated[generated.index('import json, os\n_provider = '):]
    real = tmp_path / 'real-gemmini-contract.yaml'
    real.write_text('{"name":"gemmini","version":1}')
    script = (
        'import os, sys\nfrom pathlib import Path\n'
        f'sys.path.insert(0, {str(selected.root)!r})\n'
        "_before = {key: os.environ.get(key) for key in ('MERLIN_TARGET_PATH','MERLIN_TARGET_CONTRACT')}\n"
        + tail + '\n'
        "assert {key: os.environ.get(key) for key in _before} == _before\n")
    result = subprocess.run([sys.executable, '-I', '-B', '-c', script],
                            cwd=tmp_path, env={**os.environ, 'MERLIN_TARGET_PATH':'/original/provider',
                                              'MERLIN_TARGET_CONTRACT':str(real)},
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'SELECTED_FIVE_SYNTHESIS_APIS_OK' in result.stdout
