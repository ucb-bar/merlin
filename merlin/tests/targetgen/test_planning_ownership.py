"""Planning remains usable without importing the command driver or lowering runner."""

from __future__ import annotations

import subprocess
import sys

from merlin.common.paths import repo_root


def test_host_candidate_query_does_not_import_the_compiler_driver():
    source = str(repo_root() / "src")
    code = """
import sys
sys.path.insert(0, sys.argv[1])
from merlin.compiler.backend.host import HOST_PRECISIONS
from merlin.compiler.planning.candidates import host_units
from merlin.compiler.planning.cost import measured_cost_for
from merlin.compiler.planning.placement import Placement
assert host_units(None)[0].dtypes == HOST_PRECISIONS
assert measured_cost_for(None) is None
assert Placement().to_dict() == {'placed': [], 'lanes': {}, 'n_emulated': 0, 'emulated': []}
assert 'merlin.compile_cli' not in sys.modules
assert 'merlin.llvmlower.pipeline' not in sys.modules
assert 'torch' not in sys.modules
"""
    result = subprocess.run([sys.executable, "-I", "-c", code, source], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_legacy_placement_imports_keep_class_and_function_identity():
    from merlin.compiler.planning import cost, placement
    from merlin.system import place as legacy

    assert legacy.Placed is placement.Placed
    assert legacy.Placement is placement.Placement
    assert legacy.place is placement.place
    assert legacy.device_selector is placement.device_selector
    assert legacy.measured_cost_for is cost.measured_cost_for
