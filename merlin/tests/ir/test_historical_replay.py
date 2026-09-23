"""The replay's honesty properties, which are the only reason its number means anything.

A detection rate is trivially manufacturable: pick the commits, pick the denominator, rerun until the
number is good. Each test here pins one of the moves that would do that.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from merlin.common.paths import repo_root


@pytest.mark.parametrize("with_analysis", [False, True])
def test_replay_is_owned_only_by_analysis_without_loading_its_evaluator(tmp_path, with_analysis):
    root = repo_root()
    owner = root / "packages/merlin-analysis/src"
    script = """
import importlib.util
import pathlib
import sys
core, owner, enabled = sys.argv[1:]
sys.path.insert(0, core)
if enabled == 'yes':
    sys.path.insert(1, owner)
import merlin.verify
assert pathlib.Path(merlin.verify.__file__).is_relative_to(core)
for name in ('merlin.verify.replay', 'merlin.verify.replay_layers'):
    spec = importlib.util.find_spec(name)
    if enabled == 'yes':
        assert pathlib.Path(spec.origin).is_relative_to(owner), spec.origin
    else:
        assert spec is None, spec
if enabled == 'yes':
    from merlin.verify import replay, replay_layers
    assert replay.QUALIFICATION_POLICY == 'executed_checks_v2'
    assert replay_layers.ORACLE_CAPSULES == 12
    try:
        replay.main(['--help'])
    except SystemExit as exc:
        assert exc.code == 0
    else:
        raise AssertionError('help should exit without running a study')
assert not any(name.startswith(('merlin_experiments', 'merlin.targetgen.capsule_golden',
    'merlin.targetgen.capsule_runner', 'pytest', 'aet')) for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", script, str(root / "src"), str(owner), "yes" if with_analysis else "no"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    if with_analysis:
        assert "executed_checks_v2" in result.stdout
        assert "child commands lack qualified" in result.stdout


def test_analysis_replay_extra_declares_its_execution_dependencies():
    metadata = tomllib.loads((repo_root() / "packages/merlin-analysis/pyproject.toml").read_text())
    dependencies = metadata["project"]["optional-dependencies"]["replay"]
    assert {"merlin[verify]", "merlin-experiments==0.1.0", "pytest"} <= set(dependencies)
    assert "pytest" not in metadata["project"]["dependencies"], "ordinary analysis must not need the replay runner"


@pytest.mark.parametrize("missing", ["pytest", "a_pytest_dependency"])
def test_missing_pytest_has_install_guidance_without_mislabeling_other_import_errors(monkeypatch, missing):
    import builtins

    from merlin.verify import replay_layers

    original = builtins.__import__

    def unavailable(name, *args, **kwargs):
        if name == "pytest":
            raise ModuleNotFoundError(f"No module named {missing}", name=missing)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", unavailable)
    if missing == "pytest":
        with pytest.raises(RuntimeError, match=r"install merlin-analysis\[replay\]"):
            replay_layers._qualified_pytest([])
    else:
        with pytest.raises(ModuleNotFoundError) as error:
            replay_layers._qualified_pytest([])
        assert error.value.name == missing


def test_the_sample_is_reproducible_from_the_seed():
    """Two draws with one seed are the same draw. Without this the record is not checkable."""
    from merlin.verify.replay import draw

    pool = [f"c{i}" for i in range(200)]
    assert draw(pool, 20, 7) == draw(pool, 20, 7)
    assert draw(pool, 20, 7) != draw(pool, 20, 8), "different seeds must give different samples"


def test_a_larger_sample_extends_the_smaller_one():
    """Shuffle-then-take, not `random.sample`.

    This is what stops a bad result being quietly rerolled: with the same seed, n=40 is a SUPERSET of
    n=20, so a later, larger run cannot silently replace the earlier commits with friendlier ones. A
    reader can check the property directly from the two records.
    """
    from merlin.verify.replay import draw

    pool = [f"c{i}" for i in range(200)]
    assert draw(pool, 40, 11)[:20] == draw(pool, 20, 11)


def test_the_population_only_holds_commits_that_touched_an_observed_path():
    """The denominator's definition. A commit the layers cannot see is not a miss -- it is out of scope.

    Reported with the record rather than left implicit, because "detected 3 of 25" means nothing
    without knowing which 25.
    """
    from merlin.verify.replay import OBSERVED_ROOTS, population

    pool = population(repo_root())
    if not pool:
        pytest.skip("no history in this checkout")
    for sha, _subject, files in pool:
        assert files, f"{sha[:8]} is in the population with no observed file"
        for f in files:
            assert any(f.startswith(r) for r in OBSERVED_ROOTS), (
                f"{sha[:8]} contributes {f}, which no layer can observe"
            )


def _synthetic_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".venv").symlink_to(Path(sys.executable).parent.parent, target_is_directory=True)
    return repo


def test_relocated_source_owner_uses_its_adjacent_trusted_bootstrap(tmp_path):
    from merlin.common.paths import module_source_path

    repo = _synthetic_repo(tmp_path)
    test = repo / "test_synthetic.py"
    test.write_text("def test_comparison():\n    assert 1 == 1\n")
    owner = tmp_path / "installed-source"
    modules = owner / "merlin/verify"
    modules.mkdir(parents=True)
    for name in ("replay", "replay_layers"):
        shutil.copyfile(module_source_path(f"merlin.verify.{name}"), modules / f"{name}.py")
    shadow = tmp_path / "shadow"
    shadow.mkdir()
    script = """
import pathlib
import sys
core, owner, repo, shadow, test = map(pathlib.Path, sys.argv[1:])
sys.path[:0] = [str(core), str(owner)]
from merlin.verify import replay
assert pathlib.Path(replay.__file__).is_relative_to(owner)
replay.LAYERS = {'synthetic': replay._PYTEST + (str(test),)}
assert replay._run_layers(repo, str(shadow), 20) == {'synthetic': 'green'}
# A missing installed companion must not fall back to a live checkout's bootstrap.
helper = owner / 'merlin/verify/replay_layers.py'
helper.rename(helper.with_suffix('.unavailable'))
assert replay._run_layers(repo, str(shadow), 20) == {'synthetic': 'error'}
"""
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            script,
            str(repo_root() / "src"),
            str(owner),
            str(repo),
            str(shadow),
            str(test),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("def test_case():\n    assert True\n", "green"),
        ("def test_case():\n    assert False\n", "red"),
        ("import pytest\ndef test_case():\n    pytest.fail('numeric mismatch')\n", "red"),
        ("def test_case():\n    import absent_replay_fixture_dependency\n", "error"),
        ("import absent_replay_fixture_dependency\n", "error"),
        ("def test_case():\n    raise RuntimeError('not a measured rejection')\n", "error"),
        ("import pytest\ndef test_case():\n    pytest.skip('tool unavailable')\n", "error"),
        (
            "import pytest\n@pytest.fixture\ndef broken():\n    assert False\ndef test_case(broken):\n    pass\n",
            "error",
        ),
    ],
)
def test_pytest_qualification_requires_executed_assertions(tmp_path, monkeypatch, body, expected):
    from merlin.verify import replay

    repo = _synthetic_repo(tmp_path)
    test = repo / "test_fixture.py"
    test.write_text(body)
    monkeypatch.setattr(replay, "LAYERS", {"fixture": replay._PYTEST + (str(test),)})
    assert replay._run_layers(repo, str(tmp_path), 20) == {"fixture": expected}


def test_missing_oracle_import_is_unavailable_not_a_detection(tmp_path, monkeypatch):
    from merlin.verify import replay

    repo = _synthetic_repo(tmp_path)
    pkg = tmp_path / "shadow/merlin"
    (pkg / "runtime").mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "runtime/__init__.py").write_text("def simulate(cb): return {'outputs': {}}\n")
    (pkg / "runtime/reference.py").write_text("def reference_outputs(cb): return {}\n")
    monkeypatch.setattr(replay, "LAYERS", {"numeric-golden": ("-m", "merlin.verify.replay_layers", "oracle")})
    assert replay._run_layers(repo, str(pkg.parent), 20) == {"numeric-golden": "error"}


def test_replay_checks_parent_pinned_sources_before_execution(tmp_path, monkeypatch, capsys):
    from merlin.verify import replay

    repo = _synthetic_repo(tmp_path)
    pkg = tmp_path / "shadow/merlin"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    source = pkg / "value.py"
    source.write_text("VALUE=1\n")
    test = repo / "test_fixture.py"
    test.write_text("def test_case():\n    from merlin.value import VALUE\n    assert VALUE==1\n")
    monkeypatch.setattr(replay, "LAYERS", {"fixture": replay._PYTEST + (str(test),)})
    original_run = replay.subprocess.run

    def mutate_after_inventory(argv, **kwargs):
        assert argv[1:3] == ("-I", "-S")
        source.write_text("VALUE=2\n")
        result = original_run(argv, **kwargs)
        assert "frozen source hash mismatch" in result.stdout
        return result

    monkeypatch.setattr(replay.subprocess, "run", mutate_after_inventory)
    assert replay._run_layers(repo, str(pkg.parent), 20) == {"fixture": "error"}


def test_replay_dependencies_do_not_execute_pth_startup(tmp_path, monkeypatch):
    from merlin.verify import replay

    repo = _synthetic_repo(tmp_path)
    dependencies = tmp_path / "dependencies"
    dependencies.mkdir()
    marker = tmp_path / "startup-ran"
    (dependencies / "injection.pth").write_text(f"import pathlib; pathlib.Path({str(marker)!r}).touch()\n")
    (dependencies / "fixture_dependency.py").write_text("VALUE=42\n")
    monkeypatch.syspath_prepend(str(dependencies))
    test = repo / "test_fixture.py"
    test.write_text(
        "def test_case():\n    from fixture_dependency import VALUE\n    assert VALUE==42\n"
        f"    from pathlib import Path\n    assert not Path({str(marker)!r}).exists()\n"
    )
    monkeypatch.setattr(replay, "LAYERS", {"fixture": replay._PYTEST + (str(test),)})
    assert replay._run_layers(repo, str(tmp_path), 20) == {"fixture": "green"}
    assert not marker.exists()


@pytest.mark.parametrize("changed", ["helper", "context", "bootstrap", "python_version"])
def test_replay_invocation_pins_reject_instrument_drift(tmp_path, monkeypatch, changed):
    import hashlib
    import json

    from merlin.common import paths
    from merlin.verify import replay

    repo = _synthetic_repo(tmp_path)
    shadow = tmp_path / "shadow"
    shadow.mkdir()
    test = repo / "test_fixture.py"
    test.write_text("def test_case():\n    assert False\n")
    instrument = tmp_path / "instrument"
    instrument.mkdir()
    helper = instrument / "frozen_imports.py"
    layers = instrument / "replay_layers.py"
    resolver = paths.module_source_path
    shutil.copyfile(resolver("merlin.common.frozen_imports"), helper)
    shutil.copyfile(resolver("merlin.verify.replay_layers"), layers)
    monkeypatch.setattr(
        paths, "module_source_path", lambda name: helper if name == "merlin.common.frozen_imports" else resolver(name)
    )
    monkeypatch.setattr(replay, "__file__", str(instrument / "replay.py"))
    monkeypatch.setattr(replay, "LAYERS", {"fixture": replay._PYTEST + (str(test),)})
    original_run = replay.subprocess.run

    def mutate_after_inventory(argv, **kwargs):
        context = Path(kwargs["env"]["MERLIN_REPLAY_IMPORT_CONTEXT"])
        if changed in {"helper", "bootstrap"}:
            target = helper if changed == "helper" else layers
            target.write_text(target.read_text() + "\nraise RuntimeError('tampered instrument executed')\n")
        elif changed == "context":
            context.write_bytes(context.read_bytes() + b"\n")
        else:
            record = json.loads(context.read_bytes())
            record["python_version"] = [0, 0]
            payload = json.dumps(record).encode()
            context.write_bytes(payload)
            kwargs["env"]["MERLIN_REPLAY_IMPORT_CONTEXT_SHA256"] = hashlib.sha256(payload).hexdigest()
        result = original_run(argv, **kwargs)
        expected = "must match the parent's dependency environment" if changed == "python_version" else "hash mismatch"
        assert expected in result.stderr
        assert "tampered instrument executed" not in result.stderr
        return result

    monkeypatch.setattr(replay.subprocess, "run", mutate_after_inventory)
    assert replay._run_layers(repo, str(shadow), 20) == {"fixture": "error"}


def test_replay_lit_is_explicitly_unavailable_until_children_are_qualified(tmp_path, monkeypatch):
    from merlin.verify import replay

    repo = _synthetic_repo(tmp_path)
    monkeypatch.setattr(replay, "LAYERS", {"lit-pass-tests": ("-m", "merlin.verify.replay_layers", "lit")})
    assert replay._run_layers(repo, str(tmp_path), 20) == {"lit-pass-tests": "error"}
    assert "child commands" in replay.QUALIFICATION_LIMITS["lit-pass-tests"]


@pytest.mark.parametrize(
    ("want", "reference", "simulated", "expected"),
    [
        ({"Y": [123]}, {"Y": [123]}, {"Y": [123]}, 0),
        ({"Y": [123]}, {}, {}, 1),
        ({"Y": [123]}, {}, {"Y": [123]}, 1),
        ({"Y": [123]}, {"Y": [123]}, {}, 1),
        ({"Y": [123]}, {"Y": [122]}, {"Y": [123]}, 1),
        ({}, {}, {}, 3),
    ],
)
def test_numeric_layer_compares_every_expected_output(tmp_path, monkeypatch, want, reference, simulated, expected):
    from merlin import runtime
    from merlin.runtime import reference as reference_module
    from merlin.targetgen import capsule_golden
    from merlin.verify import replay_layers

    capsule = tmp_path / "merlin/contract/capsules/synthetic/capsule.yaml"
    capsule.parent.mkdir(parents=True)
    capsule.write_text("fixture: true\n")
    monkeypatch.setattr(capsule_golden, "golden", lambda *args: want)
    monkeypatch.setattr(replay_layers, "_lower", lambda *args: {})
    monkeypatch.setattr(reference_module, "reference_outputs", lambda *args: reference)
    monkeypatch.setattr(runtime, "simulate", lambda *args: {"outputs": simulated})
    assert replay_layers._oracle(tmp_path) == expected


def test_missing_frozen_module_cannot_fall_back_to_live_namespace_owner(tmp_path, monkeypatch):
    from merlin.verify import replay

    repo = _synthetic_repo(tmp_path)
    shadow, live = tmp_path / "shadow", tmp_path / "live"
    namespace = "from pkgutil import extend_path\n__path__ = extend_path(__path__, __name__)\n"
    for root in (shadow, live):
        pkg = root / "merlin/targetgen"
        pkg.mkdir(parents=True)
        (pkg.parent / "__init__.py").write_text(namespace)
        (pkg / "__init__.py").write_text(namespace)
        (pkg / "fixture_optional.py").write_text("VALUE = 'frozen'\n")
    test = repo / "test_fixture.py"
    test.write_text(
        f"import sys\nsys.path.append({str(live)!r})\n"
        "def test_case():\n"
        "    from merlin.targetgen.fixture_optional import VALUE\n"
        "    assert VALUE == 'frozen'\n"
    )
    monkeypatch.setattr(replay, "LAYERS", {"fixture": replay._PYTEST + (str(test),)})
    assert replay._run_layers(repo, str(shadow), 20) == {"fixture": "green"}
    (shadow / "merlin/targetgen/fixture_optional.py").unlink()
    assert replay._run_layers(repo, str(shadow), 20) == {"fixture": "error"}


@pytest.mark.parametrize(
    "receipt", [None, [], {"status": "red"}, {"qualification_policy": "executed_checks_v2", "status": "running"}]
)
def test_unqualified_process_exit_cannot_claim_rejection(tmp_path, monkeypatch, receipt):
    from merlin.verify import replay

    repo = _synthetic_repo(tmp_path)
    monkeypatch.setattr(replay, "LAYERS", {"fixture": ("-m", "merlin.verify.replay_layers", "oracle")})

    def crashed(argv, **kwargs):
        if receipt is not None:
            Path(argv[6]).write_text(json.dumps(receipt))
        return subprocess.CompletedProcess(argv, 1, "", "synthetic startup crash")

    monkeypatch.setattr(replay.subprocess, "run", crashed)
    assert replay._run_layers(repo, str(tmp_path), 20) == {"fixture": "error"}


def test_missing_replay_extra_diagnostic_reaches_the_operator(tmp_path, monkeypatch, capsys):
    from merlin.verify import replay

    repo = _synthetic_repo(tmp_path)
    monkeypatch.setattr(replay, "LAYERS", {"fixture": replay._PYTEST})
    detail = "RuntimeError: pytest is unavailable; install merlin-analysis[replay] in the replay interpreter"

    def unavailable(argv, **kwargs):
        Path(argv[6]).write_text(
            json.dumps({"qualification_policy": replay.QUALIFICATION_POLICY, "status": "error", "detail": detail})
        )
        return subprocess.CompletedProcess(argv, 3, "", detail)

    monkeypatch.setattr(replay.subprocess, "run", unavailable)
    assert replay._run_layers(repo, str(tmp_path), 20) == {"fixture": "error"}
    assert "install merlin-analysis[replay]" in capsys.readouterr().err


def test_new_qualification_policy_reports_unavailable_sample_without_changing_population(tmp_path, monkeypatch):
    from merlin.verify import replay

    repo = _synthetic_repo(tmp_path)
    historical = "merlin/python/merlin/runtime/reference.py"
    monkeypatch.setattr(replay, "_git", lambda *args, **kwargs: "synthetic")
    monkeypatch.setattr(replay, "population", lambda *args: [("synthetic", "fix(fixture): test only", [historical])])
    monkeypatch.setattr(replay, "_ancestors_of_layers", lambda *args: {"synthetic"})
    monkeypatch.setattr(replay, "_shadow", lambda *args, **kwargs: [])
    outcomes = iter(({"oracle": "green"}, {"oracle": "error"}))
    monkeypatch.setattr(replay, "_run_layers", lambda *args: next(outcomes))
    record = replay.replay(repo, n=1)
    assert record["qualification_policy"] == replay.QUALIFICATION_POLICY
    assert record["qualification_limits"] == replay.QUALIFICATION_LIMITS
    assert "lit child commands" in replay.render(record)
    assert record["population_definition"]["observed_roots"] == list(replay.OBSERVED_ROOTS)
    assert record["population_size"] == record["sample_size"] == 1
    assert record["counts"] == {"unreplayable": 1}
    assert record["detected_of_replayable_historical"] == "0/0"
    assert record["results"][0]["files"] == [historical]


def test_shadow_merges_namespace_owners_and_restores_historical_path(tmp_path, monkeypatch):
    from merlin.verify import replay

    repo = _synthetic_repo(tmp_path)
    core = repo / "src/merlin"
    (core / "targetgen").mkdir(parents=True)
    (core / "__init__.py").write_text("# core owns initializer\n")
    extension = repo / "packages/merlin-experiments/src/merlin/targetgen"
    extension.mkdir(parents=True)
    (extension / "capsule_golden.py").write_text("CURRENT = True\n")
    (extension / "golden_provenance.py").write_text("SIBLING = True\n")
    standalone = repo / "packages/merlin-experiments/src/merlin_experiments"
    standalone.mkdir()
    (standalone / "__init__.py").write_text("OWNER = True\n")
    monkeypatch.setattr(
        replay.subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, "PARENT = True\n", "")
    )
    shadow = tmp_path / "shadow"
    historical = "merlin/python/merlin/targetgen/capsule_golden.py"
    assert replay._shadow(repo, "synthetic", [historical], shadow) == []
    assert (shadow / "merlin/__init__.py").read_text() == "# core owns initializer\n"
    assert (shadow / "merlin/targetgen/capsule_golden.py").read_text() == "PARENT = True\n"
    assert (shadow / "merlin/targetgen/golden_provenance.py").is_file()
    assert (shadow / "merlin_experiments/__init__.py").is_file()
    assert (extension / "capsule_golden.py").read_text() == "CURRENT = True\n"


def test_real_conftest_preserves_explicit_replay_shadow(tmp_path):
    shadow = tmp_path / "shadow"
    pkg = shadow / "merlin/common"
    pkg.mkdir(parents=True)
    (pkg.parent / "__init__.py").write_text("")
    (pkg / "__init__.py").write_text("")
    (pkg / "paths.py").write_text(
        f"from pathlib import Path\ndef merlin_dir(): return Path({str(repo_root() / 'merlin')!r})\n"
    )
    script = """
import pathlib, runpy, sys
sys.path.insert(0, sys.argv[1])
runpy.run_path(sys.argv[2])
import merlin
assert pathlib.Path(merlin.__file__).is_relative_to(sys.argv[1]), merlin.__file__
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(shadow), str(repo_root() / "merlin/tests/conftest.py")],
        env=dict(os.environ, MERLIN_REPLAY_PYTHONPATH=str(shadow)),
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("code", [2, 3, -9])
def test_lit_process_errors_are_not_rejections(tmp_path, monkeypatch, code):
    from merlin.verify import replay_layers

    lit = tmp_path / "third_party/llvm-build/bin/llvm-lit"
    lit.parent.mkdir(parents=True)
    lit.write_text("never executed")
    (tmp_path / "merlin/tests/data/lit").mkdir(parents=True)
    monkeypatch.setattr(
        replay_layers.subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(args, code, "", "")
    )
    assert replay_layers._lit(tmp_path) == 3


@pytest.mark.parametrize(
    ("codes", "exit_code", "expected"),
    [(["PASS"], 0, 0), (["FAIL"], 1, 1), (["UNSUPPORTED"], 0, 3), (["UNRESOLVED"], 1, 3), ([], 0, 3)],
)
def test_lit_qualification_requires_structured_executed_results(tmp_path, monkeypatch, codes, exit_code, expected):
    from merlin.verify import replay_layers

    lit = tmp_path / "third_party/llvm-build/bin/llvm-lit"
    lit.parent.mkdir(parents=True)
    lit.write_text("never executed")
    (tmp_path / "merlin/tests/data/lit").mkdir(parents=True)

    def run(argv, **kwargs):
        Path(argv[argv.index("--output") + 1]).write_text(json.dumps({"tests": [{"code": code} for code in codes]}))
        return subprocess.CompletedProcess(argv, exit_code, "synthetic lit report", "")

    monkeypatch.setattr(replay_layers.subprocess, "run", run)
    assert replay_layers._lit(tmp_path) == expected


def test_the_shadow_replaces_code_but_never_the_data_paths():
    """The flaw that invalidated the first run, and the reason a negative control is not optional.

    `repo_root()` resolves from the package's own file location, so inside a shadow it points at the
    temp directory — where there is no capsule corpus, no lit suite and no llvm-build. Every layer then
    finds nothing to check and exits clean, and the replay records a MISS for a defect no layer ever
    looked at. The first 25-commit run reported 0 detections that way.

    Caught by running the shadow at the parent of the COMMIT readout fix (a defect whose effect is
    known and large) and finding every layer green. With MERLIN_REPO_ROOT pinned to the real checkout,
    the engines layer goes red, which is the answer the instrument is supposed to give.
    """
    import inspect

    from merlin.verify import replay

    src = inspect.getsource(replay._run_layers)
    assert "MERLIN_REPO_ROOT=str(repo)" in src, (
        "the shadow must pin the data root to the real checkout; without it every layer reports a "
        "clean pass because it cannot find its inputs"
    )


def test_the_instrument_contains_the_layers_that_exist():
    """A rate measured with a convenient subset of the layers understates the work it is describing.

    The first run wired three pytest files and left out the static layer (lit/FileCheck over the
    passes) and the numeric oracle over the real corpus — the two checks most likely to see a lowering
    defect. Both are in `LAYERS` now.
    """
    from merlin.verify.replay import LAYERS

    assert "lit-pass-tests" in LAYERS, "the static layer is missing from the instrument"
    assert "numeric-golden" in LAYERS, (
        "the numeric oracle is missing; without it a detection cannot be attributed to the new layer "
        "rather than to the dynamic check that already existed"
    )


def test_an_unreplayable_commit_is_reported_and_never_counted_as_a_miss():
    """The denominator again, from the other side.

    Two things make a commit unreplayable: its parent files are gone (deleted or renamed), or the
    shadowed package does not run. Both are reported. Folding either into `missed` would understate
    detection; dropping either from the record would overstate it by shrinking the denominator.
    """
    import inspect

    from merlin.verify import replay

    src = inspect.getsource(replay.replay)
    assert src.count('"unreplayable"') >= 2, (
        "both unreplayable paths (missing parent files, shadow that will not run) must be recorded"
    )
    rendered = replay.render(
        {
            "population_size": 101,
            "sample_size": 2,
            "seed": 1,
            "population_definition": {"ref": "0" * 40},
            "baseline": {"a-layer": "green"},
            "detected_of_replayable": "0/1",
            "detected_of_replayable_historical": "0/1",
            "layers_landed": "abc12345",
            "counts": {"missed": 1, "unreplayable": 1},
            "results": [
                {
                    "sha": "aaaaaaaa",
                    "subject": "fix(x): a",
                    "layers_red": [],
                    "outcome": "missed",
                    "predates_layers": True,
                },
                {
                    "sha": "bbbbbbbb",
                    "subject": "fix(y): b",
                    "layers_red": [],
                    "outcome": "unreplayable",
                    "predates_layers": True,
                },
            ],
        }
    )
    assert "1 unreplayable" in rendered, "the report must state the unreplayable count"
    assert "never folded into 'missed'" in rendered


def test_a_fix_that_postdates_the_layers_is_flagged_and_excluded_from_the_citable_rate():
    """A fix that shipped WITH its own regression test would be caught by that test, not by the layer.

    Such commits stay in the sample -- removing them after seeing the outcome is the exact move this
    module exists to prevent -- but the rate worth citing is computed without them.
    """
    from merlin.verify.replay import LAYERS_LANDED, _ancestors_of_layers

    hist = _ancestors_of_layers(repo_root())
    if not hist:
        pytest.skip("the commit that introduced the layers is not in this checkout")
    assert LAYERS_LANDED in hist, "a commit is its own ancestor; the boundary is off by one"
    import subprocess

    head = subprocess.run(
        ("git", "rev-parse", "HEAD"), cwd=repo_root(), capture_output=True, text=True, check=True
    ).stdout.strip()
    assert head not in hist, "HEAD postdates the layers; it must not count as historical"


def test_every_declared_layer_can_actually_run():
    """A layer wired to a name that does not exist reports `red` or `error` and is then disqualified.

    That happened twice: `merlin.verify.replay_lit` and `merlin/tests/ir/test_golden_engines_agree.py`
    were both invented in the LAYERS table and never created, and both runs quietly measured three
    layers while the surrounding text described five. This resolves each entry to a real file or
    importable module without running it, so the mistake fails here in milliseconds instead of an hour
    into a sweep.
    """
    import importlib.util

    from merlin.verify.replay import LAYERS

    for name, argv in LAYERS.items():
        argv = list(argv)
        if argv[:2] == ["-m", "pytest"]:
            target = repo_root() / argv[-1]
            assert target.is_file(), f"layer {name!r} runs {argv[-1]}, which does not exist"
        else:
            assert argv[0] == "-m", f"layer {name!r} has an unrecognised invocation: {argv}"
            assert importlib.util.find_spec(argv[1]), (
                f"layer {name!r} runs module {argv[1]!r}, which cannot be imported"
            )
