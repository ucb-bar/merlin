"""The host lane is pinned infrastructure: readable by every arm, writable by none.

The experiment measures the TARGET lane — the target dialect, the target transforms, the partitioning
decision and the boundary/dispatch across it. It does not measure a second CPU backend, and a
submission that won by rewriting the host compiler would have answered a different question under this
one's name. So each target's descriptor declares a ``host_lane:`` block beside ``hardware_spec``, the
bundle generator grants it read-only to every arm, and denies the surface that would let it change.

Two failure modes are pinned here because this repo has shipped both:

  * a grant that resolves to nothing (an ``experiments/...`` path written without the ``merlin/``
    prefix bound no bytes while the manifest claimed otherwise). Checked by resolving every declared
    path the way the sandbox binder does and asserting it lands on real content.
  * a grant voided by its own denial. Deny wins in ``bwrap.base_argv``, so a path on both lists is
    tmpfs-masked and "read-only" silently becomes invisible. Checked by replaying the mount table.
"""
from __future__ import annotations

import pytest
import yaml

from merlin.common.paths import repo_root
from merlin.targetgen import generate_bundles as GB
from merlin.targetgen.sandbox import bwrap as BW
from merlin.targetgen.target_experiment import load_target_experiment

TARGETS_DIR = repo_root() / "merlin" / "experiments" / "capsule_bench" / "targets"


def _descriptors():
    return sorted(d / "target_experiment.yaml" for d in TARGETS_DIR.iterdir()
                  if d.is_dir() and (d / "target_experiment.yaml").is_file())


def _lane(descriptor):
    return (yaml.safe_load(descriptor.read_text(encoding="utf-8")) or {}).get("host_lane") or {}


DESCRIPTORS = _descriptors()


def test_there_are_descriptors_to_check():
    """Guard against a vacuous suite: an empty target list would pass every test below."""
    assert DESCRIPTORS


def _profiles(lane: dict) -> list[dict]:
    """Every declared lane, whether the descriptor uses the single or the keyed form."""
    if "profiles" not in lane:
        return [lane]
    shared = {k: v for k, v in lane.items() if k not in ("profiles", "default")}
    return [{**shared, **body} for body in lane["profiles"].values()]


@pytest.mark.parametrize("descriptor", DESCRIPTORS, ids=lambda p: p.parent.name)
def test_every_target_declares_a_frozen_host_lane(descriptor):
    lane = _lane(descriptor)
    assert lane, f"{descriptor}: no `host_lane:` block — the host compiler would be unpinned"
    for prof in _profiles(lane):
        for field in ("description", "repo_canonical", "package",
                      "requires_paths", "read_only", "deny_modification"):
            assert field in prof, f"{descriptor}: host_lane is missing `{field}`"
        assert prof["read_only"], f"{descriptor}: host_lane grants nothing read-only"
        assert prof["deny_modification"], f"{descriptor}: host_lane denies no modification surface"
        # WHICH PRECISION LANE, so the profile key means something and a package filed under the wrong
        # one is refused at load rather than discovered by a numeric failure.
        assert prof.get("dtype_strategy"), f"{descriptor}: host_lane declares no dtype_strategy"


@pytest.mark.parametrize("descriptor", DESCRIPTORS, ids=lambda p: p.parent.name)
def test_the_revision_pin_matches_how_the_package_came_to_exist(descriptor):
    """A pin's revision is never absent and never invented — but which pin is honest depends on the
    package's provenance, and conflating the two is what produced a fictional branch name.

    ``published``: the package was checked out of ``repo_canonical``, so ``branch`` names it and
    ``commit`` is a full sha or the word UNKNOWN. UNKNOWN is not "no pin needed" — it is "nobody could
    determine this", which is the true state here, because the branch lives on a remote this repo
    publishes to and does not vendor; the pin is then carried by content (``package`` +
    ``requires_paths`` + the tree digest).

    ``in_tree_minted``: the package was GENERATED here and never existed upstream, so there is no
    revision to name. Requiring one produced exactly one answer -- ``branch: UNKNOWN`` -- which reads
    as a failed lookup rather than as "this question does not apply". Such a lane must therefore NOT
    carry a branch, and its identity rests on the content digest, which is the stronger check anyway.
    """
    for prof in _profiles(_lane(descriptor)):
        provenance = prof.get("provenance", "published")
        assert provenance in ("published", "in_tree_minted"), \
            f"{descriptor}: unknown host_lane.provenance {provenance!r}"
        if provenance == "in_tree_minted":
            assert "branch" not in prof, (
                f"{descriptor}: an in-tree-minted lane must not name a branch; it never existed "
                f"upstream, so any value here is a fiction")
            continue
        assert prof.get("branch"), f"{descriptor}: a published lane must name its branch"
        commit = prof.get("commit")
        assert commit, f"{descriptor}: host_lane.commit is empty; write the sha or the word UNKNOWN"
        commit = str(commit)
        assert commit == "UNKNOWN" or (len(commit) == 40 and all(c in "0123456789abcdef" for c in commit)), \
            f"{descriptor}: host_lane.commit {commit!r} is neither a 40-char sha nor UNKNOWN"


@pytest.mark.parametrize("descriptor", DESCRIPTORS, ids=lambda p: p.parent.name)
def test_read_only_and_denied_never_name_the_same_path(descriptor):
    """Deny wins in the sandbox binder. A path on both lists grants nothing at all, so "read-only"
    would quietly become "not there" — the arm would lose the frozen lane and nothing would say so."""
    for prof in _profiles(_lane(descriptor)):
        clash = set(prof["read_only"]) & set(prof["deny_modification"])
        assert not clash, f"{descriptor}: {sorted(clash)} is both granted and denied"


@pytest.mark.parametrize("descriptor", DESCRIPTORS, ids=lambda p: p.parent.name)
def test_every_declared_path_resolves_to_real_content(descriptor):
    """The grant-path trap: an ``experiments/...`` path written without the ``merlin/`` prefix has
    silently granted nothing in this repo before. Resolve exactly as the sandbox binder does."""
    lane = _lane(descriptor)
    for rel in list(lane["read_only"]) + list(lane["deny_modification"]):
        assert BW.path_kind(BW.resolve_grant(rel, repo_root())) != "missing", \
            f"{descriptor}: host_lane path {rel!r} resolves to nothing — the grant would bind no bytes"


@pytest.mark.parametrize("descriptor", DESCRIPTORS, ids=lambda p: p.parent.name)
def test_the_pin_verifies_by_content(descriptor):
    """``requires_paths`` names the files whose PRESENCE is what the lane actually is.

    Verified by content and not by branch name, exactly as merlin/contract/hardware_pins.yaml requires:
    branches move, and the same branch name on a fork and on upstream are different histories.
    """
    lane = _lane(descriptor)
    pkg = repo_root() / str(lane["package"])
    if not pkg.is_dir():
        # The package is generated output (purgeable), so a clean clone may not carry it. Reported as
        # a SKIP, never as a pass: a check that could not run has established nothing.
        pytest.skip(f"host-lane package {lane['package']} is not materialized here — pin UNVERIFIED")
    missing = [r for r in lane["requires_paths"] if not (pkg / r).exists()]
    assert not missing, (f"{descriptor}: host-lane package {lane['package']} is missing {missing} — it "
                         f"is the right path and not the right content")


@pytest.mark.parametrize("descriptor", DESCRIPTORS, ids=lambda p: p.parent.name)
def test_every_arm_gets_the_lane_read_only_and_denies_its_implementation(descriptor):
    lane = _lane(descriptor)
    te = load_target_experiment(descriptor)
    bundles = GB.generate_bundles(te)
    assert bundles, f"{descriptor}: no arms generated"
    for bid, manifest in bundles.items():
        allowed = {e["path"]: e for e in manifest["allowed"]}
        denied = {e["path"] for e in manifest["denied"]}
        for rel in lane["read_only"]:
            assert rel in allowed, f"{bid}: frozen host lane {rel!r} not granted"
            assert allowed[rel].get("mode") == "ro", f"{bid}: host lane {rel!r} is not read-only"
        for rel in lane["deny_modification"]:
            assert rel in denied, f"{bid}: host-lane implementation {rel!r} is not denied"


@pytest.mark.parametrize("descriptor", DESCRIPTORS, ids=lambda p: p.parent.name)
def test_the_sandbox_actually_exposes_the_lane_and_hides_its_implementation(descriptor, tmp_path):
    """The end-to-end proof, replayed off the real mount table rather than off the manifest.

    ``is_exposed`` walks the ordered bwrap ops the way bwrap applies them, so this catches the case the
    manifest cannot: a grant that is present, correct, and then masked by a broader denial written
    after it.
    """
    lane = _lane(descriptor)
    te = load_target_experiment(descriptor)
    bundles = GB.generate_bundles(te)
    ws = tmp_path / "ws"
    ws.mkdir()
    for bid, manifest in bundles.items():
        argv = BW.base_argv(ws, manifest, repo=repo_root(),
                            _policy_test_live_inputs=True)
        for rel in lane["read_only"]:
            p = BW.resolve_grant(rel, repo_root())
            assert BW.is_exposed(argv, p), f"{bid}: the frozen host lane {rel!r} is NOT readable"
        if manifest["arm"] == "raw_baseline":
            # This arm denies all of merlin/, so its host-lane denial is subsumed rather than distinct;
            # the assertion below would be true for a reason that has nothing to do with this pin.
            continue
        for rel in lane["deny_modification"]:
            p = BW.resolve_grant(rel, repo_root())
            assert not BW.is_exposed(argv, p), \
                f"{bid}: the host compiler {rel!r} is reachable — it could be rewritten"


def test_a_lane_granted_and_denied_at_once_is_refused(tmp_path):
    """The generator must refuse the self-voiding declaration rather than emit it.

    Written as a construction rather than an assertion about the shipped descriptors, so it keeps
    testing after every descriptor is correct.
    """
    src = DESCRIPTORS[0]
    doc = yaml.safe_load(src.read_text(encoding="utf-8"))
    doc["host_lane"]["deny_modification"] = list(doc["host_lane"]["read_only"])
    bad = tmp_path / "target_experiment.yaml"
    bad.write_text(yaml.safe_dump(doc, sort_keys=False))

    te = load_target_experiment(bad)
    with pytest.raises(ValueError, match="Deny wins"):
        GB._host_lane_grants(te)


def test_a_lane_that_pins_nothing_is_refused(tmp_path):
    src = DESCRIPTORS[0]
    doc = yaml.safe_load(src.read_text(encoding="utf-8"))
    doc["host_lane"]["read_only"] = []
    doc["host_lane"]["deny_modification"] = []
    bad = tmp_path / "target_experiment.yaml"
    bad.write_text(yaml.safe_dump(doc, sort_keys=False))

    te = load_target_experiment(bad)
    with pytest.raises(ValueError, match="not pinned"):
        GB._host_lane_grants(te)
