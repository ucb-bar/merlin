"""Measurement controllers are optional; runtime driver bytes remain core-owned."""

from __future__ import annotations

import subprocess
import sys

from merlin.common.paths import module_source_path, repo_root


def test_core_keeps_driver_resources_without_measurement_controllers(tmp_path):
    script = """
import importlib.util
import pathlib
import sys
sys.path.insert(0, sys.argv[1])
from merlin.kernels import ceiling_drivers
resources = pathlib.Path(ceiling_drivers.__file__).parent
for name in ('common.h', 'openblas_sgemm_driver.c', 'xnnpack_gemm_driver.c',
             'src/xnnpack/common.h'):
    assert (resources / name).is_file(), name
for name in ('run_expert_gemm', 'multishape_compare'):
    assert importlib.util.find_spec('merlin.kernels.ceiling_drivers.' + name) is None
"""
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", script, str(repo_root() / "src")],
        cwd=tmp_path,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr


def test_analysis_resolves_core_driver_resources_without_importing_engines(tmp_path):
    script = """
import pathlib
import subprocess
import sys
core, analysis = sys.argv[1:]
sys.path[:0] = [core, analysis]
def unexpected(*args, **kwargs):
    raise AssertionError('import launched an external process')
subprocess.run = subprocess.Popen = unexpected
from merlin.kernels.ceiling_drivers import multishape_compare, run_expert_gemm
for module in (multishape_compare, run_expert_gemm):
    assert pathlib.Path(module.__file__).is_relative_to(analysis)
    assert module.HERE == pathlib.Path(core) / 'merlin/kernels/ceiling_drivers'
    assert (module.HERE / 'ours_gemm_driver.c').is_file()
assert not any(name.startswith(('merlin.mining.workloads', 'torch', 'aet')) for name in sys.modules)
"""
    root = repo_root()
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", script, str(root / "src"), str(root / "packages/merlin-analysis/src")],
        cwd=tmp_path,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr


def test_expert_declarations_use_existing_shared_driver_bytes():
    from merlin.kernels.ceiling_drivers import multishape_compare, run_expert_gemm

    resources = module_source_path("merlin.kernels.ceiling_drivers").parent
    assert multishape_compare.HERE == run_expert_gemm.HERE == resources
    for record in run_expert_gemm._experts().values():
        assert record["driver"].is_file()
        assert record["driver"].parent == resources
        assert resources in record["incs"]
