"""Optional workflow installation must not become a hidden compiler dependency."""

from __future__ import annotations

import json
import subprocess
import sys
import sysconfig
import tomllib
from pathlib import Path

from merlin.common.paths import repo_root


def test_core_renderer_and_package_loader_do_not_need_extension():
    root = repo_root()
    # -S suppresses editable .pth files: a sibling checkout must not supply a missing extension.
    import_paths = [str(root / "src"), sysconfig.get_path("purelib"), sysconfig.get_path("platlib")]
    code = (
        f"sys.path[:0] = {json.dumps(import_paths)}\n"
        + """
import importlib.util
from merlin.mining import registry, from_strategy
from merlin.compile import host_lane
assert importlib.util.find_spec('merlin.mining.beam') is None
assert importlib.util.find_spec('merlin.mining.tuning_agent') is None
assert 'transform.named_sequence' in from_strategy.render_schedule({})
assert registry.DTYPE_STRATEGIES
assert 'merlin.mining.fork' not in sys.modules
try:
    from_strategy.mint_fork('absent', {}, version=1, depth=1, timestamp='test',
                            source_evidence=[], lever='test')
except ModuleNotFoundError as exc:
    assert 'merlin-mining distribution' in str(exc)
else:
    raise AssertionError('core unexpectedly provided optional fork creation')
"""
    )
    done = subprocess.run(
        [sys.executable, "-S", "-c", "import sys\n" + code],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert done.returncode == 0, done.stderr


def test_extension_uses_the_core_objects_and_legacy_aliases():
    import merlin.mining.beam as beam
    import merlin.rvvgen.beam as legacy_beam
    from merlin.mining import fork, registry, tuning_agent

    root = repo_root()
    assert Path(beam.__file__).resolve().is_relative_to(root / "packages" / "merlin-mining" / "src")
    assert Path(registry.__file__).resolve().is_relative_to(root / "src")
    assert legacy_beam is beam
    assert fork.RvvPackage is registry.RvvPackage
    assert tuning_agent._DTYPE_STRATEGIES is registry.DTYPE_STRATEGIES


def test_mining_commands_have_exactly_one_distribution_owner():
    root = repo_root()
    core = tomllib.loads((root / "pyproject.toml").read_text())
    extension = tomllib.loads((root / "packages" / "merlin-mining" / "pyproject.toml").read_text())
    commands = extension["project"]["scripts"]
    assert len(commands) == 11
    assert not set(commands).intersection(core["project"]["scripts"])
    assert all(value.startswith("merlin.mining.") for value in commands.values())
    source = root / "packages" / "merlin-mining" / "src" / "merlin"
    assert not (source / "__init__.py").exists()
    assert not (source / "mining" / "__init__.py").exists()
