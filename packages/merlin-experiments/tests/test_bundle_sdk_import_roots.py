"""Frozen public SDK import selection changes search paths, never mounts."""

import shlex
import subprocess
from dataclasses import replace
from types import SimpleNamespace

import pytest

from merlin.common.paths import repo_root
from merlin.targetgen.sandbox import bwrap as BW
from merlin.targetgen.sandbox import preflight
from merlin.targetgen.sandbox import toolchain as TC
from merlin.targetgen.tool_profile import COMMON_SUBSTRATE_V8, attach_common_substrate
from merlin.targetgen.tool_profile_shared import prepare_shared_source_release


@pytest.fixture
def selected(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_BUNDLE_CAS", "")
    sdk = tmp_path / "public-sdk"
    prepare_shared_source_release(
        sdk,
        python_source_root=repo_root() / "src",
        license_directory=repo_root(),
        version=8,
    )
    bundle = dict(arm="raw_baseline", tools=[], allowed=[], denied=[], host_inputs=[])
    attach_common_substrate(bundle, version=COMMON_SUBSTRATE_V8, python_source_root=sdk)
    for row in bundle["allowed"]:
        if row["path"].startswith("merlin/python/"):
            row["path"] = str(sdk / row["path"].removeprefix("merlin/python/"))
    repo, ws = tmp_path / "operator", tmp_path / "run/workspace"
    repo.mkdir()
    ws.mkdir(parents=True)
    paths = replace(TC.ToolchainPaths.from_checkout(), repo=repo, python_import_roots=(str(repo / "src"),))
    yield SimpleNamespace(sdk=sdk, bundle=bundle, paths=paths, ws=ws, repo=repo)
    BW.remove_bundle_snapshot(ws)


def test_frozen_sdk_root_precedes_host_and_ambient_without_new_mounts(selected, monkeypatch):
    s = selected
    BW.materialize_bundle_inputs(s.ws, s.bundle, repo=s.repo)
    monkeypatch.setenv("PYTHONPATH", "/ambient/ungranted")
    paths = TC.paths_for_bundle(s.ws, s.bundle, paths=s.paths)
    assert paths.python_import_roots == (str(s.sdk), str(s.repo / "src"))
    target, sim = SimpleNamespace(target="neutral"), TC.SimToolchain()
    before = TC.toolchain_binds(target, paths=s.paths, sim=sim, harness="", memory_dir="")
    assert TC.toolchain_binds(target, paths=paths, sim=sim, harness="", memory_dir="") == before
    environment = TC.sandbox_env(target, s.ws, paths=s.paths, sim=sim, harness="", bundle=s.bundle)
    assert f"export PYTHONPATH={s.sdk}:{s.repo / 'src'}:{s.ws};" in environment
    assert "/ambient/ungranted" not in environment


@pytest.mark.parametrize("missing", ["merlin/__init__.py", "shared-library.json"])
def test_ungranted_sdk_namespace_or_export_cannot_select_import_root(selected, missing):
    s = selected
    s.bundle["allowed"] = [row for row in s.bundle["allowed"] if row["path"] != str(s.sdk / missing)]
    BW.materialize_bundle_inputs(s.ws, s.bundle, repo=s.repo)
    with pytest.raises(ValueError, match="exact frozen public grant bytes"):
        TC.paths_for_bundle(s.ws, s.bundle, paths=s.paths)


def test_changed_frozen_namespace_is_not_an_import_root_authority(selected):
    s = selected
    BW.materialize_bundle_inputs(s.ws, s.bundle, repo=s.repo)
    [source] = BW.snapshot_input_paths(s.ws, s.bundle, [s.sdk / "merlin/__init__.py"], repo=s.repo)
    source.chmod(source.stat().st_mode | 0o200)
    source.write_bytes(source.read_bytes() + b"# changed\n")
    with pytest.raises(ValueError, match="exact frozen public grant bytes"):
        TC.paths_for_bundle(s.ws, s.bundle, paths=s.paths)


def test_actual_bwrap_uses_frozen_sdk_origin_and_hides_ungranted_siblings(selected):
    if not preflight.probe_sandbox().usable:
        pytest.skip("this host cannot enter bwrap")
    s = selected
    BW.materialize_bundle_inputs(s.ws, s.bundle, repo=s.repo)
    # Mutating the live export cannot replace admitted snapshot bytes, and a
    # search path cannot make an ungranted sibling visible.
    (s.sdk / "merlin/__init__.py").write_text("raise AssertionError('live export used')\n")
    secret = s.sdk / "merlin/ungranted_private.py"
    secret.write_text("raise AssertionError('ungranted source used')\n")
    target, sim = SimpleNamespace(target="neutral"), TC.SimToolchain()
    argv = BW.base_argv(s.ws, s.bundle, repo=s.repo, include_claude_home=False)
    argv += TC.toolchain_binds(target, paths=s.paths, sim=sim, harness="", memory_dir="")
    argv = BW.reapply_bundle_snapshot(argv, s.ws, s.bundle, repo=s.repo)
    assert not BW.is_exposed(argv, secret)
    code = (
        "from pathlib import Path; import importlib.util; "
        "import merlin.transforms.certificates as module; "
        f"assert Path(module.__file__) == Path({str(s.sdk / 'merlin/transforms/certificates.py')!r}); "
        "assert importlib.util.find_spec('merlin.ungranted_private') is None; "
        f"assert not Path({str(secret)!r}).exists(); print('FROZEN_PUBLIC_IMPORT_OK')"
    )
    environment = TC.sandbox_env(target, s.ws, paths=s.paths, sim=sim, harness="", bundle=s.bundle)
    result = subprocess.run(
        [*argv, "bash", "-c", environment + " python3 -P -B -c " + shlex.quote(code)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert result.stdout.strip() == "FROZEN_PUBLIC_IMPORT_OK"


@pytest.mark.parametrize("level", [1, 2, 3])
def test_versioned_author_profile_keeps_legacy_sdk_host_only(selected, tmp_path, monkeypatch, level):
    from merlin_experiments.phase1 import author_profiles as A

    s = selected
    common = tmp_path / "raw-isa.h"
    common.write_text("/* common raw ISA input */\n")
    source = A.prepare_sources(repo_root(), tmp_path / "author-source",
                               profile=A.NAMES[level - 1], review_id="profile import-root regression")
    bundle = A.materialize_bundle(s.bundle, profile=A.NAMES[level - 1],
        common_allowed=[dict(path=str(common), mode="ro")], source_selection=source)
    assert bundle["compiler_tool_profile"]["shared_scheduling_resources"]
    assert all(row["path"] != str(s.sdk / "shared-library.json") for row in bundle["allowed"])
    # Independent judge metadata must never ask for a denied public SDK mount.
    monkeypatch.setattr(BW, "_snapshot_grants", lambda *args, **kwargs: pytest.fail("host-only SDK used as author import root"))
    assert TC.paths_for_bundle(s.ws, bundle, paths=s.paths) is s.paths
    environment = TC.sandbox_env(SimpleNamespace(target="neutral"), s.ws, paths=s.paths,
                                 sim=TC.SimToolchain(), harness="", bundle=bundle)
    assert str(s.sdk) not in environment
    assert str(s.repo / "src") in environment
    bundle["author_profile"]["live_cca"] = True
    with pytest.raises(ValueError, match="author profile differs"):
        TC.paths_for_bundle(s.ws, bundle, paths=s.paths)
