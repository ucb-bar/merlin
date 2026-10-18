"""Expose checked block schedules only through an explicit new source grant."""

import subprocess
import sys

from merlin.common.paths import repo_root
from merlin.targetgen.tool_profile import (
    COMMON_SUBSTRATE_V4,
    COMMON_SUBSTRATE_V5,
    attach_common_substrate,
    substrate_data,
    substrate_inventory,
    tool_profile_prompt,
    verify_tool_profile,
)


def test_v5_adds_only_the_checked_scheduler_with_unchanged_authority():
    profiles = []
    bundles = []
    for version in (COMMON_SUBSTRATE_V4, COMMON_SUBSTRATE_V5):
        bundle = dict(arm="raw_baseline", tools=[], allowed=[], denied=[], host_inputs=[])
        attach_common_substrate(bundle, version=version, python_source_root=repo_root() / "src", host_index_bits=64)
        profiles.append(verify_tool_profile(bundle))
        bundles.append(bundle)
    old, new = profiles
    assert set(new["common_public_modules"]) - set(old["common_public_modules"]) == {
        "merlin.compile.scheduling.block_schedule"
    }
    assert new["proof_policy"] == old["proof_policy"]
    assert new["automatic_checker_sources"] == old["automatic_checker_sources"]
    assert substrate_data(COMMON_SUBSTRATE_V4) == substrate_data(COMMON_SUBSTRATE_V5)
    assert "check_residency" in tool_profile_prompt(bundles[1])
    assert "check_residency" not in tool_profile_prompt(bundles[0])


def test_scheduler_imports_and_checks_with_only_the_published_sources(tmp_path):
    root = repo_root() / "src"
    namespaces, modules, _ = substrate_inventory(COMMON_SUBSTRATE_V5)
    for module in (*namespaces, *modules):
        relative = module.replace(".", "/") + ".py"
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((root / relative).read_bytes())
    # Parent directories are implicit namespace packages. Their source init
    # files and the RTL derivation helpers are deliberately outside this grant.
    script = """import sys
from dataclasses import replace
sys.path.insert(0, sys.argv[1])
from merlin.compile.scheduling.block_schedule import (
    BlockScheduleError, Contraction, Geometry, Store,
    check_residency, schedule_contraction,
)
s = schedule_contraction(Contraction(5, 7, 6), Geometry(3, 128, 64, 64, True))
check_residency(s)
bad = replace(s, ops=tuple(op for op in s.ops if not isinstance(op, Store)))
try:
    check_residency(bad)
except BlockScheduleError:
    pass
else:
    raise AssertionError('missing stores admitted')
"""
    done = subprocess.run([sys.executable, "-I", "-S", "-B", "-c", script, str(tmp_path)],
                          capture_output=True, text=True, timeout=30)
    assert done.returncode == 0, done.stderr
