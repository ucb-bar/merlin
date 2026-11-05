"""The positive native gate compares promised aliases with frozen public data."""
import hashlib
import json
import subprocess

import pytest
from merlin_experiments.phase1 import public_contract_preflight as Q
from merlin_experiments.phase1 import public_contract_staging as P

from merlin.targetgen.sandbox import bwrap as BW


@pytest.fixture
def selected(tmp_path, monkeypatch):
    repo, snapshot, workspace = (tmp_path / name for name in ("repo", "snapshot", "workspace"))
    workspace.mkdir()
    entries = [dict(path=P.PREFIX + name, mode="ro", **{"as": name}) for name in sorted(P.PUBLIC_FILES)]
    entries.append(dict(path=P.SCHEMAS, mode="ro", **{"as": "schemas"}))
    bundle = dict(author_profile=dict(name="xs-el-v1/el1"), allowed=entries)
    sources = []
    for entry in entries:
        frozen = snapshot / entry["path"]
        frozen.parent.mkdir(parents=True, exist_ok=True)
        if entry["path"] == P.SCHEMAS:
            frozen.mkdir()
            for number in range(16):
                (frozen / f"public{number}.schema.json").write_text(json.dumps(dict(public_schema=number)))
        else:
            frozen.write_text("frozen public declaration: " + entry["path"])
        sources.append(frozen)
    calls = []

    def frozen_paths(ws, selected_bundle, live_paths, *, repo):
        calls.append((ws, selected_bundle, live_paths, repo))
        return sources

    monkeypatch.setattr(BW, "snapshot_input_paths", frozen_paths)
    return repo, workspace, bundle, sources, calls


@pytest.mark.parametrize("profile", ("xs-el-v1/el1", "xs-el-v1/el2", "xs-el-v1/el3",
                                     "fresh10-tooling-v1/infra", "fresh10-tooling-v1/scheduling",
                                     "fresh10-tooling-v1/no-reference-kernels"))
def test_all_public_files_are_bound_to_the_verified_snapshot(selected, profile):
    repo, workspace, bundle, sources, calls = selected
    bundle["author_profile"]["name"] = profile
    # A changed live checkout is not the native probe's expected-data owner.
    for entry in bundle["allowed"][:-1]:
        live = repo / entry["path"]
        live.parent.mkdir(parents=True, exist_ok=True)
        live.write_text("changed ambient checkout")
    expected = Q.expected_reads(workspace, bundle, repo=repo)
    assert expected["files_count"] == 23
    assert expected["expected_files"]["VERSION"] == hashlib.sha256(sources[0].read_bytes()).hexdigest()
    assert calls == [(workspace, bundle, [repo / row["path"] for row in bundle["allowed"]], repo)]


@pytest.mark.parametrize("profile", (None, "xs-el-v1/el4", "xs-f6-v1/inherited"))
def test_historical_and_other_profiles_keep_their_existing_preflight(profile, monkeypatch):
    monkeypatch.setattr(BW, "snapshot_input_paths", lambda *_a, **_kw: pytest.fail("unselected gate"))
    assert Q.expected_reads(None, dict(author_profile=dict(name=profile)), repo=None) is None


@pytest.mark.parametrize("change", ("missing", "duplicate", "mode", "alias", "extra", "empty-schemas", "linked-schema"))
def test_invalid_or_incomplete_expected_inventory_fails_closed(selected, change):
    repo, workspace, bundle, sources, _ = selected
    if change == "missing":
        bundle["allowed"].pop()
    elif change == "duplicate":
        bundle["allowed"].append(bundle["allowed"][0].copy())
    elif change == "mode":
        bundle["allowed"][0]["mode"] = "rw"
    elif change == "alias":
        bundle["allowed"][0]["as"] = "different"
    elif change == "extra":
        bundle["allowed"].append(dict(path=P.PREFIX + "private.py", mode="ro"))
    elif change == "empty-schemas":
        for member in sources[-1].iterdir():
            member.unlink()
    else:
        member = next(sources[-1].iterdir())
        member.unlink()
        member.symlink_to(sources[0])
    with pytest.raises(ValueError):
        Q.expected_reads(workspace, bundle, repo=repo)


def test_restricted_profile_cannot_guess_its_repository(selected):
    _, workspace, bundle, _, _ = selected
    with pytest.raises(ValueError, match="composition owner's repository"):
        Q.expected_reads(workspace, bundle, repo=None)


def test_native_payload_reads_every_byte_and_refuses_a_changed_alias(selected, tmp_path):
    repo, workspace, bundle, sources, _ = selected
    for entry, source in zip(bundle["allowed"], sources, strict=True):
        P.copy_grant(entry, source, workspace / entry["as"])
    expected = Q.expected_reads(workspace, bundle, repo=repo)
    # A candidate module must not substitute the trusted JSON/hash implementation.
    (workspace / "hashlib.py").write_text("raise RuntimeError('candidate module executed')")
    import shlex

    result = subprocess.run(shlex.split(Q.command(expected)), cwd=workspace, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    rounds = tmp_path / "rounds"
    rounds.mkdir()
    Q.record_success(expected, result.stdout, rounds, 0)
    receipt = rounds / "round_00.public_contract_preflight.json"
    assert json.loads(receipt.read_bytes())["native_readability_verified"] is True
    assert receipt.stat().st_mode & 0o777 == 0o600
    (workspace / "VERSION").chmod(0o644)
    (workspace / "VERSION").write_text("changed author-visible bytes")
    failed = subprocess.run(shlex.split(Q.command(expected)), cwd=workspace, capture_output=True, text=True)
    assert failed.returncode != 0 and "differs from frozen snapshot: VERSION" in failed.stderr


@pytest.mark.parametrize("stdout", ("", Q.MARKER + "{}", Q.MARKER + "{}\n" + Q.MARKER + "{}"))
def test_zero_exit_without_exact_native_positive_receipt_is_not_success(selected, tmp_path, stdout):
    repo, workspace, bundle, _, _ = selected
    expected = Q.expected_reads(workspace, bundle, repo=repo)
    with pytest.raises(RuntimeError, match="did not verify"):
        Q.record_success(expected, stdout, tmp_path, 0)
    assert not (tmp_path / "round_00.public_contract_preflight.json").exists()
