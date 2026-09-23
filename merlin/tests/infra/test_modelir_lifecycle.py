"""ModeLIR process-state policies without an upstream checkout or hardware."""

from __future__ import annotations

import subprocess
import sys

import pytest

from merlin.common.paths import module_source_path

_PROGRAM = r"""
import importlib, json, os, pathlib, sys, threading, types
sys.path.insert(0, sys.argv[1])
from merlin.integrations import modelir as M
from merlin.targetgen.rtl import mlc_bridge as B
root, mode = pathlib.Path(sys.argv[2]), sys.argv[3]
before_path, before_cwd = list(sys.path), os.getcwd()
assert importlib.util.find_spec('mlc') is None

if mode in ('retain', 'evict', 'retain_failure', 'evict_failure'):
    context = M.discovery_imports if mode.startswith('evict') else M.importable
    try:
        with context(root):
            import mlc.child
            borrowed = sys.modules['mlc.child']
            with context(root):
                assert sys.modules['mlc.child'] is borrowed
            assert sys.modules['mlc.child'] is borrowed
            if mode.endswith('failure'):
                raise RuntimeError('consumer failed')
    except RuntimeError as exc:
        assert str(exc) == 'consumer failed'
    assert ('mlc' in sys.modules) is not mode.startswith('evict')
    assert ('mlc.child' in sys.modules) is not mode.startswith('evict')
elif mode == 'installed':
    sys.path.insert(0, str(root))
    import mlc
    borrowed = mlc
    installed_path = list(sys.path)
    for context in (M.importable, M.discovery_imports):
        with context(root / 'other'):
            import mlc.child
            assert sys.path == installed_path
        assert sys.modules['mlc'] is borrowed
        assert 'mlc.child' in sys.modules  # no eviction when we did not insert
    sys.path.remove(str(root))
elif mode == 'no_root':
    for context in (M.importable, M.discovery_imports, M.artifact_context):
        with context(None):
            assert sys.path == before_path and os.getcwd() == before_cwd
    with M.artifact_context(None):
        os.chdir(root)
    assert os.getcwd() == before_cwd
elif mode == 'borrowed':
    borrowed = types.ModuleType('mlc.borrowed')
    sys.modules['mlc.borrowed'] = borrowed
    with M.discovery_imports(root):
        import mlc.child
    assert 'mlc' not in sys.modules and 'mlc.child' not in sys.modules
    assert sys.modules['mlc.borrowed'] is borrowed
elif mode == 'path_removed':
    for context in (M.importable, M.discovery_imports, M.artifact_context):
        with context(root):
            sys.path.remove(str(root))
elif mode == 'resolver_failure':
    def resolve():
        raise RuntimeError('root resolution failed')
    try:
        with M.artifact_context(resolve_root=resolve):
            raise AssertionError('entered with unavailable root')
    except RuntimeError as exc:
        assert str(exc) == 'root resolution failed'
    with M.artifact_context(root):  # failing resolver must release the lock
        assert os.getcwd() == str(root)
elif mode in ('artifact', 'artifact_failure'):
    def resolve():
        assert os.getcwd() == before_cwd
        return root
    try:
        with M.artifact_context(resolve_root=resolve):
            assert os.getcwd() == str(root) and sys.path[0] == str(root)
            import mlc.child
            with M.artifact_context(root):
                assert os.getcwd() == str(root)
            assert os.getcwd() == str(root)
            if mode.endswith('failure'):
                raise RuntimeError('consumer failed')
    except RuntimeError as exc:
        assert str(exc) == 'consumer failed'
    assert 'mlc.child' in sys.modules
elif mode == 'bridge':
    B.mlc_dir = lambda: root
    with B._mlc_cwd():
        assert os.getcwd() == str(root)
        import mlc.child
    assert 'mlc.child' in sys.modules
elif mode == 'discovery_consumers':
    B.mlc_dir = lambda: root
    assert B.opu_artifact_paths('synthetic') == {'artifact': root / 'result'}
    assert 'mlc' not in sys.modules
    assert B.compute_unit_dtypes('synthetic') == {'unit': ['i8']}
    assert 'mlc' not in sys.modules
elif mode == 'concurrent':
    other = root / 'other'
    other.mkdir()
    entered, attempted, resolved = [threading.Event() for _ in range(3)]
    errors = []
    def worker():
        attempted.set()
        try:
            def resolve():
                resolved.set()
                assert os.getcwd() == before_cwd
                return other
            with M.artifact_context(resolve_root=resolve):
                assert os.getcwd() == str(other)
                entered.set()
        except BaseException as exc:
            errors.append(exc)
    # The bridge and direct adapter share ONE lock, including deferred lookup.
    B.mlc_dir = lambda: root
    with B._mlc_cwd():
        thread = threading.Thread(target=worker)
        thread.start()
        assert attempted.wait(2)
        assert not resolved.wait(0.1)
        assert not entered.is_set()
        assert os.getcwd() == str(root)
    thread.join(3)
    assert not thread.is_alive() and not errors and entered.is_set()
else:
    raise AssertionError(mode)
assert sys.path == before_path
assert os.getcwd() == before_cwd
print(json.dumps({'mode': mode, 'restored': True}))
"""


@pytest.mark.parametrize(
    "mode",
    [
        "retain",
        "evict",
        "retain_failure",
        "evict_failure",
        "installed",
        "no_root",
        "artifact",
        "artifact_failure",
        "bridge",
        "discovery_consumers",
        "concurrent",
        "borrowed",
        "path_removed",
        "resolver_failure",
    ],
)
def test_lifecycle_in_cold_process(tmp_path, mode):
    root = tmp_path / "model"
    package = root / "mlc"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("IDENTITY = 'synthetic'\n")
    (package / "child.py").write_text("IDENTITY = 'child'\n")
    discovery = package / "discover"
    discovery.mkdir()
    (discovery / "__init__.py").write_text("")
    (discovery / "fingerprint.py").write_text(
        "def artifact_paths(target, *, base):\n"
        "    assert target == 'synthetic'\n"
        "    return {'artifact': base / 'result'}\n"
    )
    (discovery / "datapath_dtypes.py").write_text(
        "def compute_unit_dtypes_detail(target, *, base):\n"
        "    assert target == 'synthetic' and base.is_dir()\n"
        "    return {'supported': True, 'units': [{'unit': 'unit', 'dtypes': ['i8']}]}\n"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", _PROGRAM, str(module_source_path("merlin").parent.parent), str(root), mode],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"restored": true' in result.stdout
