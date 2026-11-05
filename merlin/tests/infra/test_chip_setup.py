"""The chip examples' one-step setup (examples/lib/chip_setup.py): its decisions, without the network.

A pinned checkout is found, verified by content against a registry, and either recorded, reported as
drift with the commands that fix it, or reported missing with the commands that fetch it. These tests
build real git repositories (a superproject with a submodule, like a chipyard and its generators) and a
registry of their own, so the decisions are checked against ``merlin.common.provenance`` itself.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from merlin.common.paths import repo_root

GIT = ["git", "-c", "user.email=t@example.invalid", "-c", "user.name=t", "-c", "commit.gpgsign=false",
       "-c", "protocol.file.allow=always", "-c", "init.defaultBranch=main"]  # fmt: skip


def _load(name: str = "chip_setup"):
    path = repo_root() / "examples" / "lib" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"examples_lib_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # its dataclasses resolve their module through sys.modules
    spec.loader.exec_module(module)
    return module


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run([*GIT, *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def _repo(path: Path, files: dict[str, str]) -> str:
    path.mkdir(parents=True)
    _git(path, "init", "-q", "-b", "main")
    for rel, text in files.items():
        (path / rel).parent.mkdir(parents=True, exist_ok=True)
        (path / rel).write_text(text)
    _git(path, "add", "-A")
    _git(path, "commit", "-q", "-m", "init")
    return _git(path, "rev-parse", "HEAD")


def _registry(path: Path, pins: dict) -> Path:
    import yaml

    path.write_text(yaml.safe_dump({"version": 1, "pins": pins}, sort_keys=False))
    return path


@pytest.fixture()
def chipyard_like(tmp_path):
    """A superproject `top` recording submodule `gen/unit`, and a registry pinning both."""
    unit_sha = _repo(tmp_path / "unit_src", {"src/Unit.scala": "class Unit\n"})
    top = tmp_path / "top"
    top_sha0 = _repo(top, {"Configs.scala": "class Cfg\n"})
    assert top_sha0
    _git(top, "submodule", "add", "-q", str(tmp_path / "unit_src"), "gen/unit")
    _git(top, "commit", "-q", "-m", "add unit")
    top_sha = _git(top, "rev-parse", "HEAD")
    registry = _registry(
        tmp_path / "pins.yaml",
        {
            "demo_top": {
                "commit": top_sha,
                "repo_canonical": "https://example.invalid/top.git",
                "branch": "main",
                "root_env": "MERLIN_DEMO_TOP",
                "path": ".",
                "requires_paths": ["Configs.scala"],
            },
            "demo_unit": {
                "commit": unit_sha,
                # verify() compares a nested checkout's origin with this, and the submodule's origin is local
                "repo_canonical": str(tmp_path / "unit_src"),
                "root_env": "MERLIN_DEMO_TOP",
                "path": "gen/unit",
                "nested_in": "demo_top",
                "nested_path": "gen/unit",
                "requires_paths": ["src/Unit.scala"],
            },
        },
    )
    return SimpleNamespace(top=top, top_sha=top_sha, unit_sha=unit_sha, registry=registry, tmp=tmp_path)


def test_a_checkout_on_its_pin_is_verified_with_its_nested_pin_and_recorded(chipyard_like, monkeypatch):
    setup = _load()
    monkeypatch.delenv("MERLIN_DEMO_TOP", raising=False)
    item = setup.checkout_item("demo_top", chipyard_like.top, default=chipyard_like.tmp / "unused",
                               pins_path=chipyard_like.registry)  # fmt: skip
    assert item.state == setup.OK, item.detail
    assert item.settings == {"MERLIN_DEMO_TOP": str(chipyard_like.top)}
    assert "demo_unit" in item.detail


def test_the_default_location_is_looked_at_when_nothing_is_configured(chipyard_like, monkeypatch):
    setup = _load()
    monkeypatch.delenv("MERLIN_DEMO_TOP", raising=False)
    item = setup.checkout_item("demo_top", None, default=chipyard_like.top, pins_path=chipyard_like.registry)
    assert item.state == setup.OK, item.detail
    assert item.settings == {"MERLIN_DEMO_TOP": str(chipyard_like.top)}


def test_a_checkout_on_another_commit_is_drift_and_is_not_recorded(chipyard_like):
    setup = _load()
    (chipyard_like.top / "Configs.scala").write_text("class Cfg2\n")
    _git(chipyard_like.top, "commit", "-q", "-am", "moved")
    item = setup.checkout_item("demo_top", chipyard_like.top, default=chipyard_like.tmp / "unused",
                               pins_path=chipyard_like.registry)  # fmt: skip
    assert item.state == setup.DRIFT and not item.done
    assert item.settings == {}
    assert chipyard_like.top_sha[:12] in item.detail
    # The fix is printed, never run: the checkout is left exactly where it was.
    assert f"checkout -B main {chipyard_like.top_sha}" in "\n".join(item.fix)
    assert "submodule update --init gen/unit" in "\n".join(item.fix)
    assert _git(chipyard_like.top, "rev-parse", "HEAD") != chipyard_like.top_sha


def test_a_nested_checkout_off_its_gitlink_is_drift(chipyard_like):
    setup = _load()
    unit = chipyard_like.top / "gen" / "unit"
    (unit / "src" / "Unit.scala").write_text("class Other\n")
    _git(unit, "commit", "-q", "-am", "off the gitlink")
    item = setup.checkout_item("demo_top", chipyard_like.top, default=chipyard_like.tmp / "unused",
                               pins_path=chipyard_like.registry)  # fmt: skip
    assert item.state == setup.DRIFT
    assert "demo_unit:" in item.detail


def test_a_missing_required_path_is_drift_even_on_the_right_commit(chipyard_like):
    setup = _load()
    (chipyard_like.top / "Configs.scala").unlink()
    item = setup.checkout_item("demo_top", chipyard_like.top, default=chipyard_like.tmp / "unused",
                               pins_path=chipyard_like.registry)  # fmt: skip
    assert item.state == setup.DRIFT and "Configs.scala" in item.detail


def test_a_missing_checkout_prints_the_fetch_commands_derived_from_the_pin(chipyard_like, monkeypatch):
    setup = _load()
    monkeypatch.delenv("MERLIN_DEMO_TOP", raising=False)
    dest = chipyard_like.tmp / "out" / "ext" / "demo_top"
    item = setup.checkout_item("demo_top", None, default=dest, pins_path=chipyard_like.registry)
    assert item.state == setup.MISSING and item.settings == {}
    assert item.fix[:3] == [
        f"git clone https://example.invalid/top.git {dest}",
        f"git -C {dest} checkout -B main {chipyard_like.top_sha}",
        f"git -C {dest} submodule update --init gen/unit",
    ]
    assert not dest.exists()  # printed, not fetched


def test_a_configured_location_that_does_not_exist_is_reported_as_such(chipyard_like):
    setup = _load()
    gone = chipyard_like.tmp / "gone"
    item = setup.checkout_item("demo_top", gone, default=chipyard_like.tmp / "default",
                               pins_path=chipyard_like.registry)  # fmt: skip
    assert item.state == setup.MISSING
    assert str(gone) in item.detail and "$MERLIN_DEMO_TOP points there" in item.detail
    assert str(chipyard_like.tmp / "default") in item.fix[0]  # fetched to the default, which setup looks at


def test_a_pin_without_a_branch_is_checked_out_detached():
    setup = _load()
    pin = SimpleNamespace(name="p", repo_canonical="https://example.invalid/p.git", branch=None, commit="a" * 40)
    assert setup.fetch_commands(pin, [], Path("/d"))[1] == f"git -C /d checkout --detach {'a' * 40}"


def test_the_simulator_is_rebuilt_only_when_its_source_or_binary_changed(tmp_path):
    setup = _load()
    binary = tmp_path / "bin" / "spike"
    binary.parent.mkdir()
    binary.write_bytes(b"spike-1")
    stamp = {"source": "/src", "commit": "c" * 40, "binary_sha256": setup._sha256(binary)}
    assert setup.spike_build_needed(None, source=Path("/src"), commit="c" * 40, binary=binary)
    assert not setup.spike_build_needed(stamp, source=Path("/src"), commit="c" * 40, binary=binary)
    assert setup.spike_build_needed(stamp, source=Path("/src"), commit="d" * 40, binary=binary)
    assert setup.spike_build_needed(stamp, source=Path("/other"), commit="c" * 40, binary=binary)
    binary.write_bytes(b"spike-2")
    assert setup.spike_build_needed(stamp, source=Path("/src"), commit="c" * 40, binary=binary)


def test_the_simulator_is_never_built_from_an_unverified_source(tmp_path, monkeypatch):
    setup = _load()
    monkeypatch.setattr(setup, "_run", lambda *a, **k: pytest.fail("built from an unverified source"))
    for state in (setup.MISSING, setup.DRIFT):
        source = setup.Item(name="src", state=state)
        item = setup.build_spike("spike_zvt", "MERLIN_SPIKE_ZVT", source, prefix=tmp_path / "p")
        assert item.state == setup.BLOCKED and item.settings == {}


def test_a_recorded_build_is_reused_without_rebuilding(tmp_path, monkeypatch):
    setup = _load()
    from merlin.common import provenance as PROV

    pin = PROV.pin("spike_zvt")
    prefix = tmp_path / "spike_zvt"
    binary = prefix / "bin" / "spike"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"spike")
    (prefix / "merlin-build.json").write_text(
        json.dumps({"source": "/src", "commit": pin.commit, "binary_sha256": setup._sha256(binary)})
    )
    monkeypatch.setattr(setup, "_run", lambda *a, **k: pytest.fail("rebuilt an up-to-date simulator"))
    source = setup.Item(name="src", state=setup.OK, settings={pin.root_env: "/src"})
    item = setup.build_spike("spike_zvt", "MERLIN_SPIKE_ZVT", source, prefix=prefix)
    assert item.state == setup.OK and item.settings == {"MERLIN_SPIKE_ZVT": str(binary)}


def test_build_spike_needs_its_pin_among_the_checkouts(tmp_path):
    setup = _load()
    with pytest.raises(SystemExit, match="--checkout-pin"):
        setup.main(["--skip-capture-env", "--no-fetch", "--env-file", str(tmp_path / ".env"),
                    "--build-spike", "spike_zvt=MERLIN_SPIKE_ZVT"])  # fmt: skip


def test_the_report_lists_each_open_item_with_its_commands(capsys):
    setup = _load()
    items = [
        setup.Item(name="a", state=setup.OK, detail="/a"),
        setup.Item(name="b", state=setup.MISSING, detail="none", fix=["fetch b"], cost="1 GB"),
    ]
    assert setup.report(items) == 1
    out = capsys.readouterr().out
    assert "1. b  [1 GB]" in out and "    fetch b" in out
    assert setup.report(items[:1]) == 0


def test_missing_common_prerequisites_name_their_fix(tmp_path, monkeypatch):
    setup = _load()
    for key in ("MERLIN_CHIPYARD", "MERLIN_SPIKE", "MERLIN_RISCV_GCC"):
        monkeypatch.setenv(key, "")
    monkeypatch.setattr(setup, "REPO", tmp_path)  # no checkout-local LLVM install
    assert "llvm_toolchain.md" in "\n".join(setup.llvm_item({}).fix)
    assert "build-setup.sh riscv-tools" in "\n".join(setup.chipyard_item({"MERLIN_CHIPYARD": str(tmp_path)}).fix)
    sdk = setup.sdk_item({})
    assert sdk.state == setup.MISSING and "zephyr-sdk-0.17.0" in "\n".join(sdk.fix)


def test_the_zephyr_workspace_steps_are_printed_without_fetching(tmp_path, monkeypatch):
    setup = _load()
    monkeypatch.setattr(setup, "_run", lambda *a, **k: pytest.fail("fetched with --no-fetch"))
    item = setup.zephyr_item({"MERLIN_ZEPHYR_SW": str(tmp_path / "sw")}, fetch=False)
    text = "\n".join(item.fix)
    assert item.state == setup.MISSING
    assert "git clone -b dev https://github.com/ucb-bar/zephyr-chipyard-sw.git" in text
    assert "submodule update --init zephyr_ws/zephyr" in text
    assert "west init -l --mf west-riscv.yml" in text and "west update" in text


def test_a_configured_zephyr_tree_is_kept(tmp_path):
    setup = _load()
    (tmp_path / "zephyr").mkdir()
    (tmp_path / "zephyr" / "CMakeLists.txt").write_text("")
    item = setup.zephyr_item({"ZEPHYR_BASE": str(tmp_path / "zephyr")}, fetch=True)
    assert item.state == setup.OK and item.settings == {"ZEPHYR_BASE": str(tmp_path / "zephyr")}


def test_setup_writes_only_the_keys_it_was_given(tmp_path):
    setup = _load()
    env = tmp_path / ".env"
    env.write_text("# mine\nKEEP=1\nMERLIN_CHIPYARD=/old\n")
    setup._upsert_env(env, {"MERLIN_CHIPYARD": "/new", "ZEPHYR_BASE": "/z"})
    assert env.read_text() == "# mine\nKEEP=1\nMERLIN_CHIPYARD=/new\nZEPHYR_BASE=/z\n"


def test_exported_paths_are_adopted_and_flags_win(tmp_path, monkeypatch):
    setup = _load()
    monkeypatch.setenv("MERLIN_CHIPYARD", "/exported")
    monkeypatch.delenv("ZEPHYR_BASE", raising=False)
    monkeypatch.setenv("MERLIN_CLANG", "/llvm/bin/clang-23")
    args = SimpleNamespace(llvm=None, chipyard=tmp_path, zephyr_sw=None, zephyr_sdk=None)
    got = setup.adopt({"ZEPHYR_BASE": "/from/dotenv", "MERLIN_CHIPYARD": "/dotenv"}, args)
    assert got["MERLIN_CHIPYARD"] == str(tmp_path) and got["MERLIN_CLANG"] == "/llvm/bin/clang-23"
    assert got["ZEPHYR_BASE"] == "/from/dotenv"


def test_the_pinned_torch_mlir_is_documented_against_the_llvm_pin():
    setup = _load()
    guide = (repo_root() / "docs/guides/llvm_toolchain.md").read_text()
    assert "a47bddccec30" in guide  # the pin TORCH_MLIR is chosen against
    assert setup.TORCH_MLIR == "torch-mlir==20260531.828"
    assert Path(setup.__file__).read_text().count("a47bddcc") >= 1


def test_the_shell_loads_dotenv_without_overriding_the_environment(tmp_path):
    """examples/lib/common.sh exports .env for tools that read only os.environ; the environment wins."""
    (tmp_path / ".env").write_text('# c\nDEMO_A=from_dotenv\nDEMO_B="two words"\n# DEMO_C=commented\nnot a line\n')
    common = repo_root() / "examples" / "lib" / "common.sh"
    show = 'printf "%s|%s|%s" "$DEMO_A" "$DEMO_B" "${DEMO_C:-unset}"'
    script = f'. "{common}"; REPO_ROOT="{tmp_path}"; ex_load_dotenv; {show}'
    out = subprocess.run(["bash", "-c", script], env={"PATH": "/usr/bin:/bin", "DEMO_A": "from_env"},
                         capture_output=True, text=True, check=True).stdout  # fmt: skip
    assert out == "from_env|two words|unset"


@pytest.mark.parametrize("chip, flags", [("kodiak", ["--support", "--verify-pin saturn_opu_int8", "--spike-model"]),
                                         ("iris", ["--support", "--checkout-pin chipyard_iris",
                                                   "--checkout-pin spike_zvt",
                                                   "--build-spike spike_zvt=MERLIN_SPIKE_ZVT"])])  # fmt: skip
def test_each_chip_setup_asks_for_its_own_prerequisites(chip, flags):
    from merlin.common import provenance as PROV

    text = (repo_root() / "examples" / chip / "run.sh").read_text()
    for flag in flags:
        assert flag in text
    pins = PROV.load_pins()
    for flag in flags:
        name = flag.split()[1].partition("=")[0] if " " in flag else None
        if name:
            assert name in pins


def test_capture_libraries_are_installed_at_the_versions_model2mlir_locks(tmp_path):
    """model2MLIR calls xDSL's API directly; an unpinned xdsl resolved to a release without ``Printer.print``
    and every capture died in model2MLIR. The checkout's own uv.lock decides the version."""
    setup = _load()
    (tmp_path / "uv.lock").write_text(
        'version = 1\n[[package]]\nname = "xdsl"\nversion = "0.65.0"\n'
        '[[package]]\nname = "pyyaml"\nversion = "6.0.3"\n'
        '[[package]]\nname = "torchao"\nversion = "0.17.0"\n'
    )
    deps = setup.capture_deps(tmp_path)
    assert "xdsl==0.65.0" in deps and "PyYAML==6.0.3" in deps  # lock names are normalized (PEP 503)
    assert "structlog" in deps  # not in this lock: left to the resolver rather than guessed
    # built against torch, which here is the nightly matching torch-mlir, not the torch the lock resolved
    assert "torchao" in deps and not any(d.startswith("torchao==") for d in deps)


def test_capture_libraries_are_unpinned_without_a_lock(tmp_path):
    setup = _load()
    assert setup.capture_deps(tmp_path) == [*setup.CAPTURE_LOCKED_DEPS, *setup.CAPTURE_TORCH_DEPS]


def test_an_exported_value_that_shadows_a_recorded_setting_is_named():
    """run.sh lets the process environment win over .env, so a stale export silently replaces what setup
    verified (a ZEPHYR_BASE left pointing at a moved workspace made preflight fail right after setup said
    everything was configured)."""
    setup = _load()
    record = {"ZEPHYR_BASE": "/verified", "MERLIN_CHIPYARD": "/cy", "MERLIN_CLANG": "/c"}
    environ = {"ZEPHYR_BASE": "/stale", "MERLIN_CHIPYARD": "/cy"}
    assert setup.shadowed_settings(record, environ) == [("ZEPHYR_BASE", "/stale", "/verified")]


def test_a_relative_path_is_resolved_against_the_directory_the_user_ran_from(tmp_path):
    """Every run.sh cd's to the repo root; `--loader my/loader.py` typed in examples/kodiak must still
    name examples/kodiak/my/loader.py, and a built-in operator name must stay a name."""
    (tmp_path / "my").mkdir()
    (tmp_path / "my" / "loader.py").write_text("")
    common = repo_root() / "examples" / "lib" / "common.sh"
    script = f'. "{common}"; cd /; printf "%s|%s|%s" "$(ex_abs my/loader.py)" "$(ex_abs gemm)" "$(ex_abs /abs)"'
    out = subprocess.run(["bash", "-c", script], cwd=tmp_path, env={"PATH": "/usr/bin:/bin"},
                         capture_output=True, text=True, check=True).stdout  # fmt: skip
    assert out == f"{tmp_path}/my/loader.py|gemm|/abs"


def test_preflight_names_an_exported_path_that_does_not_exist(monkeypatch):
    flight = _load("preflight")
    monkeypatch.setenv("ZEPHYR_BASE", "/gone/zephyr")
    hint = flight._fix_hint("ZEPHYR_BASE", Path("/gone/zephyr"))
    assert "exported" in hint and "/gone/zephyr" in hint and "unset ZEPHYR_BASE" in hint
    monkeypatch.delenv("ZEPHYR_BASE")
    assert flight._fix_hint("ZEPHYR_BASE", Path("/x")).startswith("set ZEPHYR_BASE")
