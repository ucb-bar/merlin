"""Selected support owns the target-specific A/B predicate; no hardware runs."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from merlin.common.paths import repo_root


def _provider(root: Path, target: str, *, predicate: bool = True) -> Path:
    provider = root / target
    (provider / "contracts").mkdir(parents=True)
    (provider / "contracts/target_contract.yaml").write_text(f"name: {target}\nplugin:\n  backend: backend\n")
    backend = provider / "backend"
    (backend / "harness/include").mkdir(parents=True)
    (backend / "harness/include/gemmini.h").write_text(
        "#define gemmini_loop_conv_ws(downsample)\n#define gemmini_extended_config_ex(A_stride)\n"
    )
    (backend / "gemmini_sched.py").write_text(f"IDENTITY = {target!r}\n")
    code = f"""
from pathlib import Path
from types import SimpleNamespace
from merlin.runtime.backends.base import BackendInfo, BackendKind, TargetClass, register
register(BackendInfo({target!r}, TargetClass.NPU, BackendKind.KERNEL, __name__))
CALLS = []
def sched_instruction_set(): return SimpleNamespace(facts={{}})
def rocc_tests_dir(): return Path(__file__).parent / 'harness'
def gcc_path(): return Path('/synthetic/compiler')
"""
    if predicate:
        code += "def conv_downsample_flag(**kwargs):\n    CALLS.append(kwargs)\n    return 1\n"
    (backend / "__init__.py").write_text(code)
    return provider


def _run(tmp_path, source, *, provider_path=None):
    scripts = repo_root() / "merlin/experiments/gemmini_perf_bench/scripts"
    preamble = """
import json, pathlib, sys
from types import SimpleNamespace
from merlin.targetgen import target_registry
target_registry.list_targets = lambda: []
target_registry.generated_target_home = lambda: pathlib.Path(sys.argv[2]) / 'absent'
sys.path.insert(0, sys.argv[1])
import conv_downsample_ab as C
"""
    env = dict(os.environ, MERLIN_TARGET_PATH=str(provider_path or tmp_path / "providers"))
    result = subprocess.run(
        [sys.executable, "-c", preamble + source, str(scripts), str(tmp_path)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_two_selected_providers_route_and_emit_identical_programs(tmp_path):
    for target in ("fixture_alpha", "fixture_beta"):
        _provider(tmp_path / "providers", target)
    _run(
        tmp_path,
        """
calls = []
def compile_only(argv, **kwargs):
    assert argv[0] == '/synthetic/compiler'
    calls.append(argv)
    return SimpleNamespace(returncode=0, stdout='', stderr='')
C.subprocess.run = compile_only
C.provenance.verify = lambda pin: {}
C.provenance.record = lambda **kwargs: {'synthetic': True}
sys.modules['conv_shape_counter_probe'] = SimpleNamespace(
    PARTITION=(), counter_codes=lambda header, slots: {'RDMA_BYTES_REC': 0})
def case(sched, facts, iset, **kwargs):
    assert sched.IDENTITY in ('fixture_alpha', 'fixture_beta')
    return {'definition': 'void sample_ours(){ gemmini_loop_conv_ws( 0); gemmini_extended_config_ex( 2); }',
            'arrays': {}, 'ctype': 'elem_t', 'elements': 1, 'vendor': 'vendor();', 'attrs': {'stable': True}}
C.AB.conv_case = case
programs = []
for target in ('fixture_alpha', 'fixture_beta'):
    backend, sched, predicate = C._provider_components(target)
    assert sched.IDENTITY == target and predicate.__module__ == backend.__name__
    output = pathlib.Path(sys.argv[2]) / target
    assert C.main(['--target', target, '--name', 'sample', '--in-dim', '8', '--in-channels', '4',
                   '--out-channels', '4', '--kernel', '1', '--stride', '2', '--padding', '0',
                   '--out', str(output), '--build-only']) == 0
    assert len(backend.CALLS) == 1 and backend.CALLS[0]['in_rows'] == 8
    programs.append((output / 'probe.c').read_bytes())
    assert json.loads((output / 'case.json').read_text())['target'] == target
assert programs[0] == programs[1]
assert b'gemmini_loop_conv_ws( 1)' in programs[0]
assert b'gemmini_extended_config_ex( 1)' in programs[0]
assert len(calls) == 8
""",
    )


def test_selected_provider_missing_capability_refuses_without_native_fallback(tmp_path):
    _provider(tmp_path / "providers", "fixture_missing", predicate=False)
    _run(
        tmp_path,
        """
try:
    C._provider_components('fixture_missing')
except SystemExit as exc:
    assert 'lacks conv_downsample_flag' in str(exc)
else:
    raise AssertionError('missing selected capability was borrowed')
assert not any(name.endswith('gemmini_conv_downsample') for name in sys.modules)
""",
    )


def test_missing_selected_backend_cannot_borrow_registered_implementation(tmp_path):
    provider = _provider(tmp_path / "providers", "fixture_missing")
    (provider / "contracts/target_contract.yaml").write_text(
        "name: fixture_missing\nplugin:\n  backend: absent_backend\n"
    )
    _run(
        tmp_path,
        """
from merlin.runtime.backends.base import BackendInfo, BackendKind, TargetClass, register
register(BackendInfo('fixture_missing', TargetClass.NPU, BackendKind.KERNEL, 'sys'))
try:
    C._provider_components('fixture_missing')
except KeyError as exc:
    assert 'backend module failed to load' in str(exc)
else:
    raise AssertionError('missing selected backend was borrowed')
assert not any(name.endswith('gemmini_conv_downsample') for name in sys.modules)
""",
    )


def test_companion_export_preserves_original_predicate(tmp_path):
    from merlin.targetgen import target_registry

    provider = target_registry.explicit_targets().get("gemmini")
    if provider is None:
        pytest.skip("requires explicit Gemmini support on MERLIN_TARGET_PATH")
    assert target_registry.resolve("gemmini").base.resolve() == provider.resolve()
    _run(
        tmp_path,
        f"""
import hashlib, importlib, itertools
backend, sched, predicate = C._provider_components('gemmini')
helper = importlib.import_module(predicate.__module__)
assert hashlib.sha256(pathlib.Path(helper.__file__).read_bytes()).hexdigest() == (
    '3d87ad67d5e1caf90026366d3239adefcf2910af882ed6bc72b7e5c063222c51')
unknown = helper.DownsampleUnknown
assert pathlib.Path(backend.__file__).resolve().is_relative_to(pathlib.Path({str(provider)!r}).resolve())
assert pathlib.Path(helper.__file__).resolve().is_relative_to(pathlib.Path({str(provider)!r}).resolve())
header = 'const bool downsample = ' + ' && '.join(sorted(helper.IMPLEMENTED_CLAUSES)) + ';'
for kernel, stride, padding, dimension, pooled in itertools.product((1, 3), (1, 2), (0, 1), (7, 8), (False, True)):
    inputs = dict(kernel=kernel, stride=stride, padding=padding, in_rows=dimension,
                  in_cols=dimension, pooled=pooled, header_text=header)
    expected = int(kernel == 1 and stride == 2 and padding == 0 and dimension % 2 == 0 and not pooled)
    assert predicate(**inputs) == expected
try:
    predicate(kernel=1, stride=2, padding=0, in_rows=8, in_cols=8, pooled=False,
              header_text='const bool downsample = unknown_predicate;')
except unknown:
    pass
else:
    raise AssertionError('unrecognized header predicate was accepted')
""",
        provider_path=provider,
    )
