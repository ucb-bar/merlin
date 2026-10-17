"""New releases expose shared scheduling without changing historical grants."""

import hashlib
import subprocess
import sys

from merlin.common.paths import repo_root
from merlin.targetgen.tool_profile import (
    COMMON_SUBSTRATE_V3,
    COMMON_SUBSTRATE_V4,
    attach_common_substrate,
    substrate_data,
    substrate_inventory,
    tool_profile_prompt,
    verify_tool_profile,
)


def test_v4_publishes_only_the_shared_planner_and_preserves_v3_policy():
    root = repo_root() / "src"
    bundles = []
    for version in (COMMON_SUBSTRATE_V3, COMMON_SUBSTRATE_V4):
        bundle = {"arm": "raw_baseline", "tools": [], "allowed": [], "denied": [], "host_inputs": []}
        attach_common_substrate(bundle, version=version, python_source_root=root, host_index_bits=64)
        bundles.append(bundle)
    old, new = map(verify_tool_profile, bundles)
    assert old["proof_policy"] == new["proof_policy"]
    assert old["automatic_checker_sources"] == new["automatic_checker_sources"]
    assert substrate_data(COMMON_SUBSTRATE_V3) == substrate_data(COMMON_SUBSTRATE_V4)
    added = set(new["common_public_modules"]) - set(old["common_public_modules"])
    assert added == {"merlin.transforms.dense_planning", "merlin.transforms.dense_ir"}
    assert "dense_planning" in tool_profile_prompt(bundles[1])
    for row in new["common_sources"]:
        path = root / (row["module"].replace(".", "/") + ".py")
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]


def test_planner_runs_with_only_the_published_source_closure(tmp_path):
    root = repo_root() / "src"
    namespaces, modules, _ = substrate_inventory(COMMON_SUBSTRATE_V4)
    for name in (*namespaces, *modules):
        source = root / (name.replace(".", "/") + ".py")
        destination = tmp_path / source.relative_to(root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    # Import under an isolated interpreter; no local Merlin installation may fill gaps.
    script = """import sys
sys.path.insert(0, sys.argv[1])
from merlin.transforms.dense_planning import DenseProblem, DenseResources, choose_dense_block
from merlin.transforms.dense_ir import dense_schedule_op
p = DenseProblem(64, 64, 64, 16, 16, 16, 1, 1, 4)
r = DenseResources(4096, 8192)
c = choose_dense_block(p, r, static_budget=64, price=lambda b: 20)
assert c is not None and c.block.loads < 128
dense_schedule_op(p, c.block, r, c.static_commands, 64).verify()
"""
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", script, str(tmp_path)], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
