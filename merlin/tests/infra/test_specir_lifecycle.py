"""SpecIR import ownership without numerical or upstream-checkout substitutes."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from merlin.common.paths import module_source_path, python_import_roots

_PROGRAM = r"""
import importlib, pathlib, sys, types
sys.path[:0] = sys.argv[1].split('|')
from merlin.integrations import specir as S
root, foreign, mode = pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3]), sys.argv[4]
before = list(sys.path)
if mode == 'installed':
    sys.path.insert(0, str(root))
    import specir.oracle.dtypes as D
    before = list(sys.path)
    with S.importable(None):
        import specir.oracle.dtypes as again
        assert again is D and sys.path == before
elif mode == 'missing':
    sys.path.insert(0, str(root))
    before = list(sys.path)
    try:
        with S.importable(root / 'missing'):
            raise AssertionError('explicit missing checkout fell through')
    except S.SpecIRImportError:
        pass
elif mode in ('foreign', 'foreign_child', 'foreign_path'):
    with S.importable(root):
        import specir.oracle.dtypes as D
    if mode == 'foreign_child':
        D.__spec__.origin = str(foreign / 'specir/oracle/dtypes.py')
    elif mode == 'foreign_path':
        sys.modules['specir'].__path__.append(str(foreign / 'specir'))
    try:
        with S.importable(foreign if mode == 'foreign' else root):
            raise AssertionError('mixed checkout was accepted')
    except S.SpecIRImportError:
        assert sys.modules['specir.oracle.dtypes'] is D
elif mode == 'retry':
    try:
        with S.importable(None):
            import specir
    except ModuleNotFoundError:
        pass
    else:
        raise AssertionError('unexpected installed specir')
    with S.importable(root):
        import specir
elif mode == 'postforeign':
    injected = types.ModuleType('specir.foreign')
    injected.__spec__ = importlib.util.spec_from_file_location('specir.foreign', foreign / 'specir/foreign.py')
    try:
        with S.importable(root):
            import specir
            sys.modules['specir.foreign'] = injected
    except S.SpecIRImportError:
        assert sys.modules['specir.foreign'] is injected
    else:
        raise AssertionError('successful body escaped postvalidation')
elif mode == 'threads':
    import threading
    started, entered = threading.Event(), threading.Event()
    failures = []
    def other():
        started.set()
        try:
            with S.importable(root):
                entered.set()
                import specir.oracle.dtypes as again
                assert again is D
        except BaseException as exc:
            failures.append(exc)
    with S.importable(root):
        import specir.oracle.dtypes as D
        active = list(sys.path)
        thread = threading.Thread(target=other)
        thread.start()
        assert started.wait(2) and not entered.wait(0.05)
        assert sys.path == active
    thread.join(2)
    assert not thread.is_alive() and entered.is_set() and not failures
elif mode in ('failure', 'nested', 'primary', 'removed'):
    try:
        with S.importable(root):
            import specir.oracle.dtypes as D
            if mode == 'nested':
                active = list(sys.path)
                with S.importable(root):
                    import specir.oracle.dtypes as again
                    assert again is D
                assert sys.path == active
            elif mode == 'removed':
                sys.path.remove(str(root))
            elif mode == 'primary':
                D.__spec__.origin = str(foreign / 'specir/oracle/dtypes.py')
                raise RuntimeError('primary body exception')
            else:
                raise RuntimeError('primary body exception')
    except RuntimeError as exc:
        assert str(exc) == 'primary body exception'
    assert sys.modules['specir.oracle.dtypes'] is D
elif mode in ('phase0', 'phase0_installed', 'phase0_dotenv'):
    import os
    from merlin_experiments.phase0 import numerics as N
    os.environ.pop('SPECIR_ROOT', None)
    N._dotenv = lambda: {}
    if mode == 'phase0_installed':
        sys.path.insert(0, str(root))
        before = list(sys.path)
    elif mode == 'phase0_dotenv':
        N._dotenv = lambda: {'SPECIR_ROOT': str(root)}
    else:
        os.environ['SPECIR_ROOT'] = str(root)
        N._dotenv = lambda: {'SPECIR_ROOT': str(foreign)}
    D, reduce = N._specir()
    assert D.VALUE == 'selected' and reduce() == 'selected'
    assert N._specir() == (D, reduce)
elif mode in ('capture', 'capture_unknown'):
    from merlin.targetgen.capsule_source import SpecRefSource, SpecProgramUnavailable
    source = SpecRefSource(str(root))
    if mode == 'capture_unknown':
        try:
            source.capture('unknown:op.matmul')
        except SpecProgramUnavailable as exc:
            assert 'not registered' in str(exc)
        else:
            raise AssertionError('unknown generator accepted')
    else:
        result = source.capture('synthetic:op.matmul', workload=(1,1,1))
        assert result.golden == {'out': [[6]]}
        assert result.operands == {'lhs': [[2]], 'weight': [[3]]}
        assert result.instructions == ['instruction'] and result.workload == (1,1,1)
assert sys.path == before and None not in sys.path
"""


@pytest.mark.parametrize(
    "mode",
    [
        "installed",
        "missing",
        "foreign",
        "foreign_child",
        "foreign_path",
        "retry",
        "postforeign",
        "threads",
        "failure",
        "nested",
        "primary",
        "removed",
        "phase0",
        "phase0_installed",
        "phase0_dotenv",
        "capture",
        "capture_unknown",
    ],
)
def test_scoped_imports_and_actual_consumers(tmp_path, mode):
    sources = {
        "__init__.py": "",
        "oracle/__init__.py": "",
        "oracle/dtypes.py": "VALUE = 'selected'\n",
        "oracle/refmodel.py": "def fp_reduce(): return 'selected'\n",
        "gate.py": "def load_targets(root): return [{'id':'synthetic','spec':'synthetic.mlir'}]\n",
        "registry.py": "from pathlib import Path\n_SPEC_ROOT = Path(__file__).parent\n",
        "loading.py": "def parse_spec_file(path): return None\n",
        "graph.py": (
            "from types import SimpleNamespace\n"
            "def all_nodes(module): return [SimpleNamespace(name='spec.op')]\n"
            "def name_of(node): return 'op.matmul'\n"
            "def attrs_of(node): return {}\n"
        ),
        "interface/__init__.py": "",
        "interface/emit_capsule.py": (
            "def emit_command_buffer(*args, **kwargs):\n"
            " return {'tensors':{'a':{'role':'lhs'},'b':{'role':'weight'}}}, {}\n"
        ),
        "interface/rocc_lower.py": (
            "from types import SimpleNamespace\n"
            "class RoccLoweringError(Exception): pass\n"
            "def lower_buffer(*args, **kwargs):\n"
            " return SimpleNamespace(operands={'a':[[2]],'b':[[3]]},"
            " golden={'y':[[6]]}, instructions=['instruction'])\n"
        ),
    }
    for name in ("selected", "foreign"):
        for relative, content in sources.items():
            path = tmp_path / name / "specir" / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            _PROGRAM,
            "|".join(map(str, (*python_import_roots(), *[Path(p) for p in sys.path if "site-packages" in p]))),
            str(tmp_path / "selected"),
            str(tmp_path / "foreign"),
            mode,
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_capture_discovery_keeps_configured_and_sibling_policy(monkeypatch, tmp_path):
    from merlin.targetgen import capsule_source as C

    monkeypatch.setattr(C, "_env", lambda key: None)
    monkeypatch.setattr(C, "repo_root", lambda: tmp_path / "merlin")
    assert C.SpecRefSource().root == str(tmp_path / "spec")
    monkeypatch.setattr(C, "_env", lambda key: "configured")
    assert C.SpecRefSource().root == "configured"
    assert C.SpecRefSource("explicit").root == "explicit"


def test_helper_source_changes_key_and_missing_source_really_bypasses_cache(monkeypatch, tmp_path):
    from merlin_experiments.phase0 import golden_cache as G

    from merlin.common import paths

    helper = tmp_path / "specir.py"
    helper.write_bytes(module_source_path("merlin.integrations.specir").read_bytes())
    original = paths.module_source_path
    monkeypatch.setattr(
        paths, "module_source_path", lambda name: helper if name == "merlin.integrations.specir" else original(name)
    )
    calls = []

    def engine(entry, binding):
        calls.append(entry)
        return {}, {}

    before = G._golden_cache_key(engine, {}, None)
    helper.write_bytes(helper.read_bytes() + b"\n# changed helper\n")
    assert G._golden_cache_key(engine, {}, None) != before
    helper.unlink()
    assert G.source_digest() == "unresolvable"
    monkeypatch.setattr(G, "_GOLDEN_CACHE_DISABLED", False)

    def forbid_cache(_namespace):
        raise AssertionError("unresolved source used cache")

    monkeypatch.setattr("merlin.common.artifacts.cache_dir", forbid_cache)
    assert G._golden_cached(engine, {}, None) == ({}, {})
    assert G._golden_cached(engine, {}, None) == ({}, {})
    assert calls == [{}, {}]


def test_phase0_and_shared_phase1_startup_pin_helper(tmp_path, monkeypatch):
    from merlin_experiments import adapters, runner
    from merlin_experiments.phase1 import source_inputs
    from merlin_experiments.spec import SpecError

    from merlin.common import paths

    helper = module_source_path("merlin.integrations.specir").resolve()
    assert adapters.phase0_startup_inputs()["specir_integration"] == helper
    assert runner._phase0_source_inputs()[1]["phase0:startup:specir_integration"] == str(helper)
    inputs = source_inputs.paths(repo=tmp_path, entrypoint=tmp_path / "driver.py")
    assert inputs["phase1:startup:specir_integration"] == str(helper)
    copied = tmp_path / "specir.py"
    copied.write_bytes(helper.read_bytes())
    original = paths.module_source_path
    monkeypatch.setattr(
        paths,
        "module_source_path",
        lambda name: copied if name == "merlin.integrations.specir" else original(name),
    )
    record = source_inputs.record(repo=tmp_path, entrypoint=tmp_path / "driver.py")
    copied.write_bytes(copied.read_bytes() + b"\n# startup drift\n")
    with pytest.raises(SpecError, match="source identity changed"):
        source_inputs.verify(record, repo=tmp_path, entrypoint=tmp_path / "driver.py")
