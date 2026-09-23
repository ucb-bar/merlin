"""A process must not silently reuse a loaded plugin after selecting another provider."""

import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("plugin", ["backend", "dialect", "sim_oracle", "sim_oracle_metadata"])
@pytest.mark.parametrize("change", ["switch", "remove", "missing_plugin", "malformed", "module"])
def test_same_target_provider_switch_refuses_without_replacing_loaded_objects(tmp_path, plugin, change):
    providers = []
    for marker in ("first", "second"):
        root = tmp_path / marker
        (root / "contracts").mkdir(parents=True)
        (root / "contracts/target_contract.yaml").write_text(
            "name: synthetic_switch\nplugin:\n"
            + (
                "  sim_oracle: plugin.py\n  sim_oracle_metadata: plugin.py\n"
                if plugin.startswith("sim_oracle")
                else f"  {plugin}: plugin.py\n"
            )
        )
        if plugin == "backend":
            source = (
                "from merlin.runtime.backends import base\n"
                "base.register(base.BackendInfo('synthetic_switch', base.TargetClass.NPU, "
                "base.BackendKind.KERNEL, __name__))\n"
                f"MARKER = {marker!r}\n"
            )
        elif plugin == "dialect":
            source = (
                "from types import SimpleNamespace\n"
                "from merlin.xdsl_dialects.lowering.target_lowering import register_dialect_spec\n"
                f"register_dialect_spec(SimpleNamespace(name='synthetic_switch'), {{'marker': {marker!r}}})\n"
            )
        else:
            source = (
                "from merlin.targetgen.oracle_policy import register_sim_oracle, OracleTierPlan\n"
                "register_sim_oracle('synthetic_engine', adapters=lambda t: {}, "
                "available=lambda t: (False, 'synthetic'), exclusive=True, "
                f"tier_plan=lambda t: OracleTierPlan(({marker!r},)))\n"
            )
        (root / "plugin.py").write_text(source)
        providers.append(str(root))
    code = f"""
import os, sys
from pathlib import Path
from merlin.runtime.backends import base
from merlin.xdsl_dialects.lowering import target_lowering as lowering
from merlin.targetgen import oracle_policy as oracle
base._discovered = True  # Only OOT discovery is in scope; no native backend imports.
os.environ['MERLIN_TARGET_PATH'] = {providers[0]!r}
plugin = {plugin!r}
def query():
    if plugin == 'backend':
        return base.get_backend('synthetic_switch')
    if plugin == 'dialect':
        lowering.plugin_opcodes()
        return lowering._PLUGIN_SPECS['synthetic_switch']
    if plugin == 'sim_oracle':
        result = oracle.sim_oracle_caps('synthetic_engine')
        oracle.sim_tier_plan('synthetic_engine', 'synthetic_switch')
        return result
    oracle.sim_tier_plan('synthetic_engine', 'synthetic_switch')
    return oracle.sim_oracle_caps('synthetic_engine')
original = query()
assert query() is original
alias = Path({str(tmp_path / "alias")!r})
alias.symlink_to({providers[0]!r}, target_is_directory=True)
os.environ['MERLIN_TARGET_PATH'] = str(alias)
assert query() is original  # Canonical aliases are the same provider, not a replacement.
os.environ['MERLIN_TARGET_PATH'] = {providers[0]!r}
change = {change!r}
contract = Path({providers[0]!r}) / 'contracts/target_contract.yaml'
if change == 'switch':
    os.environ['MERLIN_TARGET_PATH'] = {providers[1]!r}
elif change == 'remove':
    os.environ.pop('MERLIN_TARGET_PATH')
elif change == 'missing_plugin':
    contract.write_text('name: synthetic_switch\\n')
elif change == 'malformed':
    contract.write_text('broken: [')
else:
    other = contract.parent.parent / 'other.py'
    other.write_bytes(other.with_name('plugin.py').read_bytes())
    contract.write_text(contract.read_text().replace('plugin.py', 'other.py'))
for attempt in range(2):
    try:
        result = query()
    except base.PluginOwnershipError as exc:
        assert 'fresh process' in str(exc)
    else:
        raise AssertionError('provider switch silently reused old object: ' + str(result is original))
if plugin == 'backend':
    assert sys.modules['merlin._oot_backends.synthetic_switch'] is original
elif plugin == 'dialect':
    assert lowering._PLUGIN_SPECS['synthetic_switch'] is original
else:
    assert oracle._SIM_ORACLES['synthetic_engine'] is original
"""
    env = dict(os.environ)
    env.update(
        MERLIN_REPO_ROOT=str(tmp_path / "repo"),
        MERLIN_OUT_ROOT=str(tmp_path / "out"),
        MERLIN_TARGETS_DIR=str(tmp_path / "no-native-targets"),
    )
    env.pop("MERLIN_TARGET_CONTRACT", None)
    proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=20)
    assert proc.returncode == 0, proc.stdout + proc.stderr


@pytest.mark.parametrize("partial_registration", [False, True])
def test_failed_optional_plugin_keeps_owner_without_blocking_unchanged_selection(tmp_path, partial_registration):
    for marker in ("first", "second"):
        root = tmp_path / marker
        (root / "contracts").mkdir(parents=True)
        (root / "contracts/target_contract.yaml").write_text("name: failed_plugin\nplugin: {backend: backend.py}\n")
        registration = (
            "from merlin.runtime.backends import base\n"
            "base.register(base.BackendInfo('failed_plugin', base.TargetClass.NPU, "
            "base.BackendKind.KERNEL, __name__))\n"
        )
        (root / "backend.py").write_text(
            (registration if partial_registration or marker == "second" else "")
            + ("raise ImportError('optional dependency absent')\n" if marker == "first" else "")
        )
    code = f"""
import os
from merlin.runtime.backends import base
base._discovered = True
os.environ['MERLIN_TARGET_PATH'] = {str(tmp_path / "first")!r}
for _ in range(2):
    try:
        base.get_backend('failed_plugin')
    except (KeyError, ModuleNotFoundError):
        pass
    else:
        raise AssertionError('broken optional module unexpectedly available')
assert 'optional dependency absent' in base.load_failures()['failed_plugin']
os.environ['MERLIN_TARGET_PATH'] = {str(tmp_path / "second")!r}
for _ in range(2):
    try:
        base.get_backend('failed_plugin')
    except base.PluginOwnershipError:
        pass
    else:
        raise AssertionError('failed import lost its owner')
"""
    env = dict(os.environ)
    env.update(MERLIN_REPO_ROOT=str(tmp_path / "repo"), MERLIN_OUT_ROOT=str(tmp_path / "out"))
    env["MERLIN_TARGETS_DIR"] = str(tmp_path / "no-native")
    env.pop("MERLIN_TARGET_CONTRACT", None)
    proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=20)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_distinct_modules_cannot_claim_one_shared_oracle_namespace(tmp_path):
    root = tmp_path / "support"
    (root / "contracts").mkdir(parents=True)
    (root / "contracts/target_contract.yaml").write_text(
        "name: paired\nplugin: {sim_oracle_metadata: metadata.py, sim_oracle: runtime.py}\n"
    )
    (root / "metadata.py").write_text(
        "from merlin.targetgen.oracle_policy import register_sim_oracle, OracleTierPlan\n"
        "register_sim_oracle('paired', adapters=lambda t: {}, available=lambda t: (False, 'fixture'), "
        "exclusive=True, tier_plan=lambda t: OracleTierPlan(('synthetic',)))\n"
    )
    (root / "runtime.py").write_text("raise AssertionError('replacement must not execute')\n")
    code = """
from merlin.targetgen import oracle_policy as oracle
from merlin.runtime.backends.base import PluginOwnershipError
oracle.sim_tier_plan('paired', 'paired')
original = oracle._SIM_ORACLES['paired']
for _ in range(2):
    try:
        oracle.sim_oracle_caps('paired')
    except PluginOwnershipError:
        pass
    else:
        raise AssertionError('distinct runtime module silently ignored')
assert oracle._SIM_ORACLES['paired'] is original
"""
    env = dict(os.environ)
    env.update(
        MERLIN_REPO_ROOT=str(tmp_path / "repo"),
        MERLIN_OUT_ROOT=str(tmp_path / "out"),
        MERLIN_TARGETS_DIR=str(tmp_path / "no-native"),
        MERLIN_TARGET_PATH=str(root),
    )
    env.pop("MERLIN_TARGET_CONTRACT", None)
    proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=20)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_backend_import_and_metadata_discovery_have_one_lock_order(tmp_path):
    root = tmp_path / "support"
    (root / "contracts").mkdir(parents=True)
    (root / "contracts/target_contract.yaml").write_text(
        "name: interleaved\nplugin: {backend: backend.py, sim_oracle_metadata: metadata.py}\n"
    )
    (root / "backend.py").write_text(
        "from merlin.runtime.backends import base\n"
        "from merlin.targetgen import oracle_policy as oracle\n"
        "from fixture_sync import backend_started, oracle_waiting\n"
        "base.register(base.BackendInfo('interleaved', base.TargetClass.NPU, base.BackendKind.KERNEL, __name__))\n"
        "backend_started.set()\n"
        "assert oracle_waiting.wait(3)\n"
        "oracle.sim_tier_plan('interleaved', 'interleaved')\n"
    )
    (root / "metadata.py").write_text(
        "from merlin.targetgen.oracle_policy import register_sim_oracle, OracleTierPlan\n"
        "register_sim_oracle('interleaved', adapters=lambda t: {}, available=lambda t: (False, 'fixture'), "
        "exclusive=True, tier_plan=lambda t: OracleTierPlan(('synthetic',)))\n"
    )
    code = """
import sys, threading, types
from merlin.runtime.backends import base
from merlin.targetgen import oracle_policy as oracle
base._discovered = True
sync = types.ModuleType('fixture_sync')
sync.backend_started = threading.Event()
sync.oracle_waiting = threading.Event()
sys.modules['fixture_sync'] = sync
preflight = threading.Event()
real_lock = base._oot_lock
class TracedLock:
    def __enter__(self):
        if threading.current_thread().name == 'oracle' and sync.backend_started.is_set():
            sync.oracle_waiting.set()
        return real_lock.__enter__()
    def __exit__(self, *args):
        return real_lock.__exit__(*args)
base._oot_lock = TracedLock()
real_assert = base._assert_oot_plugin_ownership
def ownership():
    real_assert()
    if threading.current_thread().name == 'oracle' and not preflight.is_set():
        preflight.set()
        assert sync.backend_started.wait(3)
base._assert_oot_plugin_ownership = ownership
errors = []
def run(action):
    try:
        action()
    except Exception as exc:
        errors.append(repr(exc))
a = threading.Thread(name='oracle', daemon=True, target=run,
                     args=(lambda: oracle.sim_tier_plan('interleaved', 'interleaved'),))
b = threading.Thread(name='backend', daemon=True, target=run,
                     args=(lambda: base.get_backend('interleaved'),))
a.start()
assert preflight.wait(3)
b.start()
a.join(5)
b.join(1)
assert not a.is_alive() and not b.is_alive(), 'cross-consumer discovery deadlocked'
assert not errors, errors
assert oracle.sim_tier_plan('interleaved', 'interleaved').tiers == ('synthetic',)
"""
    env = dict(os.environ)
    env.update(
        MERLIN_REPO_ROOT=str(tmp_path / "repo"),
        MERLIN_OUT_ROOT=str(tmp_path / "out"),
        MERLIN_TARGETS_DIR=str(tmp_path / "no-native"),
        MERLIN_TARGET_PATH=str(root),
    )
    env.pop("MERLIN_TARGET_CONTRACT", None)
    proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=12)
    assert proc.returncode == 0, proc.stdout + proc.stderr
