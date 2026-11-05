"""Pinned support packages: one record, fetched by commit, one base for every package.

Chip packages are not copied into Merlin. ``build_tools/upstreams/target_support.json`` pins each one
(``pinned_support``: target, repo, commit, tree) and ONE base they all layer on, and
``merlin-target-fetch`` / the chip example's ``run.sh setup`` check a package out at exactly its pin.
The tracked record is checked here as data; the fetch, the verification and the refusals are checked
against local git repositories built in a temporary directory, so nothing touches the network and no
target is named.
"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

from merlin.common.paths import repo_root
from merlin.common.yaml import write_yaml
from merlin.targetgen import oot_fetch, provider_compose, support_pins, target_registry
from merlin.targetgen.oot_fetch import FetchError

GIT = ["git", "-c", "user.email=t@example.invalid", "-c", "user.name=t", "-c", "commit.gpgsign=false",
       "-c", "protocol.file.allow=always", "-c", "init.defaultBranch=main"]  # fmt: skip
BASE, CHIP = "fixture_vec", "fixture_chip"


# ── the tracked record ───────────────────────────────────────────────────────────────────────────


def _tracked_pins() -> support_pins.Pins:
    return support_pins.load_pins(repo_root() / support_pins.MANIFEST)


def test_the_tracked_pins_are_well_formed():
    document = json.loads((repo_root() / support_pins.MANIFEST).read_text(encoding="utf-8"))
    assert document["schema"] == support_pins.SCHEMA
    pins = _tracked_pins()
    assert pins.base is not None, "Merlin pins no base for its chip packages"
    assert pins.packages, "no chip package is pinned"
    for pin in pins.all():
        assert pin.repo.startswith("https://"), pin
        assert pin.tree is not None, f"{pin.target}: a pin records the tree a fetch is verified against"
        assert pin.branch, f"{pin.target}: a pin records where its commit was published"
    for pin in pins.packages.values():
        assert pin.extends == pins.base.target, f"{pin.target} does not layer on the single base"
        assert pin.example and (repo_root() / pin.example / "run.sh").is_file(), pin.example


def test_every_package_extends_the_single_base_pin_by_its_fetched_declaration():
    """Where a package is fetched, its own provider.yaml must name the base pin (the selection rule)."""
    pins = _tracked_pins()
    for pin in pins.packages.values():
        root = support_pins.available_root(pin)
        if root is None:
            continue
        from merlin.targetgen.providers import read_provider

        provider = read_provider(root)
        assert provider.extends.commit == pins.base.commit
        support_pins.check_single_base(provider, pins)


def test_the_in_repo_base_copy_is_the_base_pin():
    """The base keeps a tracked copy; the copy and the pin are one source of truth."""
    pins = _tracked_pins()
    base = pins.base
    if base.in_repo is None:
        pytest.skip("the base keeps no in-repo copy")
    root = repo_root() / base.in_repo
    record = yaml.safe_load((root.parent / "SOURCE.yaml").read_text(encoding="utf-8"))
    assert record["target"] == base.target
    assert record["source"]["commit"] == base.commit
    assert record["source"]["tree"] == base.tree
    assert record["source"]["repository"] == base.repo
    assert support_pins.available_root(base) == root.resolve()


def test_pinned_packages_are_not_vendored():
    """A fetched package has no tracked copy and no SOURCE.yaml left behind beside its example."""
    for pin in _tracked_pins().packages.values():
        example = repo_root() / pin.example
        assert not (example / target_registry.IN_REPO_SUPPORT_DIR).exists(), example
        assert not (example / "SOURCE.yaml").exists(), example
        assert pin.target not in target_registry.in_repo_support()


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda b: b["packages"][0].update(commit="abc"), "40-char sha"),
        (lambda b: b["packages"][0].update(extends="other"), "single base"),
        (lambda b: b["packages"].append(dict(b["packages"][0])), "pinned twice"),
        (lambda b: b["packages"][0].update(repo="git@host:x.git"), "https:// or file://"),
        (lambda b: b["packages"][0].update(in_repo="examples/x/support"), "never kept in-repo"),
        (lambda b: b["base"].update(surprise=1), "unknown fields"),
    ],
)
def test_a_malformed_pin_is_refused(mutate, match):
    document = json.loads((repo_root() / support_pins.MANIFEST).read_text(encoding="utf-8"))
    mutate(document[support_pins.BLOCK])
    with pytest.raises(support_pins.PinError, match=match):
        support_pins.parse(document)


# ── local fixture repositories ───────────────────────────────────────────────────────────────────


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run([*GIT, *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def _commit(repo: Path, message: str) -> str:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _base_repo(path: Path) -> str:
    path.mkdir(parents=True)
    _git(path, "init", "-q")
    write_yaml(path / "contracts/target_contract.yaml", {"name": BASE, "features": ["vector"]})
    write_yaml(
        path / "provider.yaml",
        {"schema": "merlin.provider.v1", "id": f"{BASE}-support", "target": BASE, "role": "support"},
    )
    return _commit(path, "base")


def _chip_repo(
    path: Path, base_repo: Path, base_commit: str, *, declared: str | None = None, submodule: bool = False
) -> str:
    """A layered package over the fixture base, its effective contract composed and committed.

    Its default branch keeps an unrelated commit, so the package commit is reachable only from a side
    branch: a fetch has to ask for the commit, not the default branch.
    """
    path.mkdir(parents=True)
    _git(path, "init", "-q")
    (path / "README.md").write_text("default branch\n")
    _commit(path, "default branch")
    _git(path, "checkout", "-q", "-b", "handwritten")
    write_yaml(
        path / "provider.yaml",
        {
            "schema": "merlin.provider.v2",
            "id": f"{CHIP}-support",
            "target": CHIP,
            "role": "support",
            "extends": {"target": BASE, "repo": base_repo.as_uri(), "commit": declared or base_commit},
        },
    )
    write_yaml(path / "contracts/overlay.yaml", {"features": ["matrix"]})
    (path / "boards.yaml").write_text("schema_version: 1\nboards: {}\n")
    if submodule:
        _git(path, "submodule", "add", "-q", base_repo.as_uri(), "base/fixture-base")
        _git(path / "base/fixture-base", "checkout", "-q", base_commit)
    if declared is None:
        provider_compose.materialize(path, search=[base_repo])
    else:  # declares a revision it was not composed from: a hand-written contract stands in
        write_yaml(path / "contracts/target_contract.yaml", {"name": CHIP})
    commit = _commit(path, "chip")
    _git(path, "checkout", "-q", "main")
    return commit


def _tree(repo: Path, commit: str) -> str:
    return _git(repo, "rev-parse", f"{commit}^{{tree}}")


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A base and a chip repository, a manifest pinning them, and a private generated home."""
    src = tmp_path / "src"
    base_commit = _base_repo(src / "fixture-base")
    chip_commit = _chip_repo(src / "fixture-chip", src / "fixture-base", base_commit)
    manifest = tmp_path / "pins.json"

    def write(*, chip_commit=chip_commit, chip_tree=None, base_commit=base_commit, chip_repo=None):
        block = {
            "base": {
                "target": BASE,
                "repo": (src / "fixture-base").as_uri(),
                "commit": base_commit,
                "tree": _tree(src / "fixture-base", base_commit),
            },
            "packages": [
                {
                    "target": CHIP,
                    "repo": (chip_repo or src / "fixture-chip").as_uri(),
                    "commit": chip_commit,
                    "tree": chip_tree or _tree(chip_repo or src / "fixture-chip", chip_commit),
                    "extends": BASE,
                    "example": "examples/fixture_chip",
                }
            ],
        }
        manifest.write_text(json.dumps({"schema": support_pins.SCHEMA, support_pins.BLOCK: block}))
        return support_pins.load_pins(manifest)

    monkeypatch.setenv(support_pins.ENV_MANIFEST, str(manifest))
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    monkeypatch.delenv("MERLIN_TARGET_PATH", raising=False)
    monkeypatch.setattr(target_registry, "in_repo_support", lambda: {})
    monkeypatch.setattr(target_registry, "reference_targets", lambda: {})
    return type(
        "World",
        (),
        {"src": src, "base": base_commit, "chip": chip_commit, "write": staticmethod(write), "tmp": tmp_path},
    )


# ── fetch by commit ──────────────────────────────────────────────────────────────────────────────


def test_fetch_checks_out_exactly_the_pinned_commit_off_the_default_branch(world):
    pins = world.write()
    roots = oot_fetch.fetch_pinned(CHIP, pins=pins)
    chip = support_pins.pinned_root(pins.packages[CHIP])
    assert roots[0] == chip and chip.name == f"{CHIP}@{world.chip[:12]}"
    assert _git(chip, "rev-parse", "HEAD") == world.chip
    assert _git(chip, "rev-parse", "HEAD^{tree}") == pins.packages[CHIP].tree
    # The base comes from the same single pin, into its own pinned home.
    assert roots[-1] == support_pins.pinned_root(pins.base)
    assert _git(roots[-1], "rev-parse", "HEAD") == world.base


def test_fetch_asks_for_the_commit_and_takes_one_commit_where_the_server_allows(world):
    _git(world.src / "fixture-chip", "config", "uploadpack.allowAnySHA1InWant", "true")
    pins = world.write()
    root = oot_fetch.fetch_pinned(CHIP, pins=pins, with_base=False)[0]
    assert _git(root, "rev-parse", "--is-shallow-repository") == "true"
    assert _git(root, "rev-list", "--count", "HEAD") == "1"


def test_a_tree_that_differs_from_the_record_is_refused(world):
    pins = world.write(chip_tree=_tree(world.src / "fixture-base", world.base))
    with pytest.raises(FetchError, match="carries tree"):
        oot_fetch.fetch_pinned(CHIP, pins=pins)


def test_a_commit_the_repository_lacks_is_refused(world):
    pins = world.write(chip_commit="0" * 40, chip_tree="0" * 40)
    with pytest.raises(FetchError):
        oot_fetch.fetch_pinned(CHIP, pins=pins)


def test_refetching_a_verified_checkout_is_a_no_op_without_network(world):
    pins = world.write()
    first = oot_fetch.fetch_pinned(CHIP, pins=pins)
    (world.src / "fixture-chip").rename(world.tmp / "gone")  # the origin disappears
    assert oot_fetch.fetch_pinned(CHIP, pins=pins) == first


def test_a_dirty_checkout_is_refused_not_reset(world):
    pins = world.write()
    root = oot_fetch.fetch_pinned(CHIP, pins=pins, with_base=False)[0]
    (root / "contracts/overlay.yaml").write_text("features: [edited]\n")
    with pytest.raises(FetchError, match="local changes"):
        oot_fetch.fetch_pinned(CHIP, pins=pins)
    assert (root / "contracts/overlay.yaml").read_text() == "features: [edited]\n"
    (root / "contracts/overlay.yaml").unlink()
    _git(root, "checkout", "-q", "--", "contracts/overlay.yaml")
    (root / "stray.txt").write_text("untracked\n")
    with pytest.raises(FetchError, match="local changes"):
        oot_fetch.fetch_pinned(CHIP, pins=pins)


def test_a_divergent_checkout_is_refused(world):
    pins = world.write()
    root = oot_fetch.fetch_pinned(CHIP, pins=pins, with_base=False)[0]
    (root / "local.txt").write_text("local work\n")
    local = _commit(root, "local work")
    with pytest.raises(FetchError, match="not fetched at"):
        oot_fetch.fetch_pinned(CHIP, pins=pins)
    assert _git(root, "rev-parse", "HEAD") == local
    # The default selection refuses it too, rather than selecting another revision's bytes.
    with pytest.raises(support_pins.PinError, match="not the pinned"):
        target_registry.resolve(CHIP)


def test_a_bump_fetches_the_new_commit_beside_the_old(world):
    pins = world.write()
    old = oot_fetch.fetch_pinned(CHIP, pins=pins, with_base=False)[0]
    chip = world.src / "fixture-chip"
    _git(chip, "checkout", "-q", "handwritten")
    (chip / "contracts/matrix_units.yaml").write_text("units: {}\n")
    bumped = _commit(chip, "bump")
    pins = world.write(chip_commit=bumped)
    new = oot_fetch.fetch_pinned(CHIP, pins=pins, with_base=False)[0]
    assert new != old and new.name == f"{CHIP}@{bumped[:12]}"
    assert target_registry.resolve(CHIP).base == new  # the pinned revision, not the stale one


# ── one base ─────────────────────────────────────────────────────────────────────────────────────


def test_a_package_composed_against_another_base_is_refused_with_the_fix(world):
    base = world.src / "fixture-base"
    (base / "NOTES").write_text("moved\n")
    moved = _commit(base, "moved base")
    other = world.tmp / "other-chip"
    stale = _chip_repo(other, base, moved)  # composed against the MOVED base
    pins = world.write(chip_commit=stale, chip_repo=other)
    want = (
        f"other-chip composed against {BASE}@{moved[:12]}; Merlin's base is {BASE}@{world.base[:12]}: "
        f"recompose other-chip against {world.base} and bump its pin"
    )
    with pytest.raises(FetchError) as refused:
        oot_fetch.fetch_pinned(CHIP, pins=pins)
    assert str(refused.value) == want
    # Selected explicitly instead, the same rule refuses it at resolution.
    root = oot_fetch.fetch_commit(other.as_uri(), stale, world.tmp / "explicit")
    os.environ["MERLIN_TARGET_PATH"] = str(root)
    try:
        with pytest.raises(support_pins.SingleBaseError, match="recompose other-chip"):
            target_registry.resolve(CHIP)
        with pytest.raises(provider_compose.ComposeError, match="recompose other-chip"):
            provider_compose.materialize(root, check=True, search=[base])
    finally:
        del os.environ["MERLIN_TARGET_PATH"]


def test_a_package_carrying_its_base_as_a_submodule_is_fetched_with_it_at_the_pin(world):
    chip = world.tmp / "sub-chip"
    commit = _chip_repo(chip, world.src / "fixture-base", world.base, submodule=True)
    pins = world.write(chip_commit=commit, chip_repo=chip)
    roots = oot_fetch.fetch_pinned(CHIP, pins=pins)
    root, carried = roots[0], roots[1]
    assert carried == root / "base/fixture-base"
    assert _git(carried, "rev-parse", "HEAD") == world.base
    # The carried base is the first base candidate, and --check holds against it.
    assert provider_compose.layer_bases(root) == [carried]
    fresh, comp = provider_compose.materialize(root, check=True)
    assert fresh and comp.base.root == carried.resolve() and comp.base_revision == world.base


def test_a_submodule_at_another_base_revision_is_refused(world):
    base = world.src / "fixture-base"
    first = world.base
    (base / "NOTES").write_text("moved\n")
    moved = _commit(base, "moved base")
    chip = world.tmp / "sub-chip"
    # provider.yaml names the pin, but the gitlink records another revision: two bases in one build.
    commit = _chip_repo(chip, base, moved, declared=first, submodule=True)
    pins = world.write(chip_commit=commit, chip_repo=chip, base_commit=first)
    with pytest.raises(FetchError, match=f"Merlin's base is {BASE}@{first[:12]}"):
        oot_fetch.fetch_pinned(CHIP, pins=pins)


# ── selection ────────────────────────────────────────────────────────────────────────────────────


def test_unset_selection_finds_the_fetched_package_and_an_explicit_one_wins(world, tmp_path):
    pins = world.write()
    with pytest.raises(support_pins.SupportNotFetched, match=r"examples/fixture_chip/run.sh setup"):
        target_registry.resolve(CHIP)
    from merlin.targetgen import plugins

    with pytest.raises(plugins.PluginError, match="merlin-target-fetch fixture_chip"):
        plugins.resolve_support(CHIP)
    root = oot_fetch.fetch_pinned(CHIP, pins=pins)[0]
    assert target_registry.default_support()[CHIP] == root
    assert target_registry.resolve(CHIP).base == root
    assert plugins.resolve_support(CHIP).base == root
    explicit = oot_fetch.fetch_commit((world.src / "fixture-chip").as_uri(), world.chip, tmp_path / "mine")
    os.environ["MERLIN_TARGET_PATH"] = str(explicit)
    try:
        assert target_registry.resolve(CHIP).base == explicit.resolve()
    finally:
        del os.environ["MERLIN_TARGET_PATH"]


def test_an_empty_selection_does_not_require_the_fetch(world, monkeypatch):
    world.write()
    monkeypatch.setenv("MERLIN_TARGET_PATH", "")
    assert target_registry.resolve(CHIP).kind == "generated"  # the explicit empty selection: nothing


def test_the_cli_prints_the_root_or_the_setup_command(world, capsys):
    pins = world.write()
    assert oot_fetch.main([CHIP, "--root"]) == 1
    assert "run.sh setup" in capsys.readouterr().err
    assert oot_fetch.main([CHIP]) == 0
    root = support_pins.pinned_root(pins.packages[CHIP])
    assert capsys.readouterr().out.splitlines()[0] == str(root)
    assert oot_fetch.main([CHIP, "--root"]) == 0
    assert capsys.readouterr().out.strip() == str(root.resolve())
    assert oot_fetch.main([CHIP, "--champion", "main"]) == 2


def test_an_in_repo_base_copy_must_record_the_pin(tmp_path, monkeypatch):
    copy = tmp_path / "examples/fixture_base/support"
    copy.mkdir(parents=True)
    write_yaml(copy.parent / "SOURCE.yaml", {"path": "support", "source": {"commit": "1" * 40}})
    pin = support_pins.Pin(target=BASE, repo="https://example.invalid/b.git", commit="2" * 40,
                           in_repo="examples/fixture_base/support")  # fmt: skip
    monkeypatch.setattr(support_pins, "checkout_root", lambda: tmp_path)
    with pytest.raises(support_pins.PinError, match="one source of truth"):
        support_pins.available_root(pin)


# ── the candidate boundary ───────────────────────────────────────────────────────────────────────


def test_fetched_packages_are_withheld_like_vendored_support(world, monkeypatch):
    """Sandboxes, bundle snapshots and clean rooms withhold every fetched checkout, stale ones too."""
    pytest.importorskip("merlin_experiments")
    from types import SimpleNamespace

    surfaces = importlib.import_module("merlin.targetgen.sandbox.answer_surfaces")
    bwrap = importlib.import_module("merlin.targetgen.sandbox.bwrap")
    cleanroom = importlib.import_module("merlin.targetgen.sandbox.cleanroom")
    pins = world.write()
    root = oot_fetch.fetch_pinned(CHIP, pins=pins)[0]
    base = support_pins.pinned_root(pins.base).resolve()
    stale = support_pins.pinned_shelf() / f"{CHIP}@{'f' * 12}"
    (stale / "backend").mkdir(parents=True)
    (stale / "backend/kernel.py").write_text("answer = 1\n")
    assert set(target_registry.fetched_support_dirs()) == {root, base, stale.resolve()}

    checkout = world.tmp  # the fetched shelf sits inside it, under out/build/generated/pinned
    monkeypatch.setattr(target_registry, "vendored_support_dirs", lambda: ())
    monkeypatch.setattr(target_registry, "targets_dir", lambda: checkout / "merlin/targets")
    monkeypatch.setenv("MERLIN_TARGETS_DIR", str(checkout / "merlin/targets"))
    monkeypatch.setattr(surfaces, "repo_root", lambda: checkout)
    monkeypatch.setattr(surfaces, "artifacts_dir", lambda: checkout / "out/artifacts")
    monkeypatch.setattr(surfaces, "MODULE_ACCESS", ())
    monkeypatch.setattr(surfaces, "contract_resource_roots", lambda *_: ())
    monkeypatch.setattr(surfaces, "experimenter_memory_dir", lambda: checkout / "absent-memory")
    te = SimpleNamespace(target="unrelated_device", capsule_corpus=None, corpus_siblings=lambda: (),
                         hidden_corpus=lambda: None, prior_backends=(), backend_package=None)  # fmt: skip
    derived = surfaces.answer_surfaces(te)
    exposed = ["--ro-bind", str(checkout), str(checkout)]
    masked = bwrap.apply_answer_masks(exposed, derived)
    denied = cleanroom.derive_deny_set(te)
    private = [root / "boards.yaml", root / "provider.yaml", base / "provider.yaml", stale / "backend/kernel.py"]
    for member in private:
        assert bwrap.is_exposed(exposed, member), member
        assert not bwrap.is_exposed(masked, member), member
        assert cleanroom.is_withheld(member, denied), member
    # Publication refuses a fetched support package as a candidate, as it refuses a vendored one.
    from merlin.targetgen import publish

    with pytest.raises(publish.PublishError, match="not support"):
        publish._export_role(SimpleNamespace(package_dir=root, target=CHIP))


# ── the fetched packages' own suites ─────────────────────────────────────────────────────────────
TRACKED_PACKAGES = sorted(_tracked_pins().packages)


@pytest.mark.parametrize("target", TRACKED_PACKAGES)
def test_a_fetched_package_suite_passes_against_this_checkout(target, tmp_path):
    """Each pinned package's own tests, from its fetched checkout, selected alone the way its README
    documents: a core change that breaks a package Merlin pins has to fail somewhere. Skipped, with the
    command, where the package is not fetched."""
    import sys

    from pinned_support import require_support

    root = require_support(target)
    if not (root / "tests").is_dir():
        pytest.skip(f"{target}: the package ships no tests/")
    env = dict(os.environ, MERLIN_TARGET_PATH=str(root), PYTHONDONTWRITEBYTECODE="1")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(root), os.environ.get("PYTHONPATH"))))
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", f"--rootdir={repo_root().resolve()}",
         f"--basetemp={tmp_path / 'basetemp'}", str(root / "tests")],
        cwd=tmp_path, env=env, capture_output=True, text=True,
    )  # fmt: skip
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-2000:]


@pytest.mark.parametrize("target", TRACKED_PACKAGES)
def test_the_recorded_core_references_are_the_fetched_packages(target):
    """check_wiring counts what a pinned package calls in the core as a production caller, reading it
    from the record; the record has to be what the pinned commit's code actually references."""
    import importlib.util

    from pinned_support import require_support

    root = require_support(target)
    path = repo_root() / "build_tools/scripts/check_wiring.py"
    spec = importlib.util.spec_from_file_location("check_wiring_for_pins", path)
    wiring = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wiring)
    document = json.loads((repo_root() / support_pins.MANIFEST).read_text(encoding="utf-8"))
    entry = next(p for p in document[support_pins.BLOCK]["packages"] if p["target"] == target)
    assert entry["core_references"] == wiring.package_core_references(root)
