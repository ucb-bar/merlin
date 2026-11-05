"""Private judge facts preserve the evaluator without becoming author inputs.

These tests create real verified input snapshots and inspect mount/environment
plans. They execute no sandbox, model, compiler or simulator.
"""
from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path

import pytest

from merlin.targetgen.sandbox import bwrap as B


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def selected(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_BUNDLE_CAS", "")
    repo = tmp_path / "repo"
    repo.mkdir()
    workspace = repo / "run/workspace"
    workspace.mkdir(parents=True)
    facts = repo / "private/facts.json"
    facts.parent.mkdir()
    facts.write_text('{"facts":{"marker":"private-judge-only"}}\n')
    bundle = {
        "allowed": [],
        "host_inputs": [{"path": str(facts)}],
        "author_profile": {
            "name": "xs-el-raw-rtl-v2/el1",
            "level": 1,
            "disclosure": {"rtl_facts": False},
        },
        "judge_rtl_facts_file": {"path": str(facts), "sha256": _sha(facts)},
    }
    return repo, workspace, facts, bundle


@pytest.mark.parametrize("level", [1, 2])
def test_private_snapshot_feeds_judge_but_author_env_is_unset(selected, monkeypatch, level):
    repo, workspace, facts, bundle = selected
    bundle["author_profile"].update(name=f"xs-el-raw-rtl-v2/el{level}", level=level)
    B.materialize_bundle_inputs(workspace, bundle, repo=repo)
    frozen = B.frozen_selected_rtl_facts(workspace, bundle, repo=repo)
    assert frozen != facts
    assert frozen.is_relative_to(B.bundle_snapshot_root(workspace))
    assert frozen.read_bytes() == facts.read_bytes()
    # An operator's host-side facts selection is deliberately not inherited.
    monkeypatch.setenv("MERLIN_RTL_FACTS", str(frozen))
    argv = B.base_argv(workspace, bundle, repo=repo, include_claude_home=False)
    assert ["--unsetenv", "MERLIN_RTL_FACTS"] == argv[argv.index("MERLIN_RTL_FACTS") - 1:argv.index("MERLIN_RTL_FACTS") + 1]
    assert B._selected_rtl_facts_file(bundle, repo=repo) is None
    assert not B.is_exposed(argv, facts)
    assert not B.is_exposed(argv, frozen)
    assert not any(op == "expose" and Path(destination) == facts for op, _source, destination in B._mounts(argv))


def test_private_host_directory_can_cover_the_exact_pinned_leaf(selected):
    repo, workspace, facts, bundle = selected
    bundle["host_inputs"] = [{"path": str(facts.parent)}]
    B.materialize_bundle_inputs(workspace, bundle, repo=repo)
    assert B.frozen_selected_rtl_facts(workspace, bundle, repo=repo).read_bytes() == facts.read_bytes()


def test_judge_reads_frozen_pin_after_live_original_changes(selected):
    repo, workspace, facts, bundle = selected
    original = facts.read_bytes()
    B.materialize_bundle_inputs(workspace, bundle, repo=repo)
    facts.write_text('{"facts":{"marker":"changed-live-original"}}\n')
    assert B.frozen_selected_rtl_facts(workspace, bundle, repo=repo).read_bytes() == original
    # Reusing the verified snapshot also remains independent of live original bytes.
    B.materialize_bundle_inputs(workspace, bundle, repo=repo)


@pytest.mark.parametrize("digest", [None, "", "0" * 63, "G" * 64, "A" * 64, "0" * 65, 17])
def test_malformed_pin_rejected_before_freeze(selected, digest):
    repo, workspace, _facts, bundle = selected
    bundle["judge_rtl_facts_file"]["sha256"] = digest
    with pytest.raises(RuntimeError, match="exact SHA256"):
        B.materialize_bundle_inputs(workspace, bundle, repo=repo)
    assert not B.bundle_snapshot_root(workspace).exists()


def test_wrong_digest_does_not_publish_snapshot(selected):
    repo, workspace, _facts, bundle = selected
    bundle["judge_rtl_facts_file"]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="digest differs"):
        B.materialize_bundle_inputs(workspace, bundle, repo=repo)
    assert not B.bundle_snapshot_root(workspace).exists()
    assert not B.bundle_snapshot_root(workspace).with_name("bundle_inputs.pending").exists()


def test_changed_pin_cannot_relabel_an_existing_snapshot(selected):
    repo, workspace, _facts, bundle = selected
    B.materialize_bundle_inputs(workspace, bundle, repo=repo)
    changed = copy.deepcopy(bundle)
    changed["judge_rtl_facts_file"]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="digest differs"):
        B.frozen_selected_rtl_facts(workspace, changed, repo=repo)
    with pytest.raises(RuntimeError, match="digest differs"):
        B.materialize_bundle_inputs(workspace, changed, repo=repo)


@pytest.mark.parametrize("public", [None, "/irrelevant/facts.json"])
def test_dual_selector_is_rejected_even_when_public_value_is_null(selected, public):
    repo, workspace, _facts, bundle = selected
    bundle["selected_rtl_facts_file"] = public
    with pytest.raises(RuntimeError, match="also select public"):
        B.materialize_bundle_inputs(workspace, bundle, repo=repo)


@pytest.mark.parametrize("change", ["missing-policy", "disclosed", "wrong-level", "boolean-level", "missing-selector"])
def test_restricted_profile_policy_fails_closed(selected, change):
    repo, workspace, _facts, bundle = selected
    if change == "missing-policy":
        bundle["author_profile"].pop("disclosure")
    elif change == "disclosed":
        bundle["author_profile"]["disclosure"]["rtl_facts"] = True
    elif change == "wrong-level":
        bundle["author_profile"]["level"] = 2
    elif change == "boolean-level":
        bundle["author_profile"]["level"] = True
    else:
        bundle.pop("judge_rtl_facts_file")
    with pytest.raises(RuntimeError, match="disclosure policy|require pinned"):
        B.materialize_bundle_inputs(workspace, bundle, repo=repo)


@pytest.mark.parametrize("name", [None, "xs-el-v1/el1", "xs-el-no-expert-v1/el1", "xs-el-raw-rtl-v2/el3", "arbitrary-profile"])
def test_historical_unrestricted_and_el3_profiles_cannot_select_private_facts(selected, name):
    repo, workspace, _facts, bundle = selected
    if name is None:
        bundle.pop("author_profile")
    else:
        bundle["author_profile"]["name"] = name
    with pytest.raises(RuntimeError, match="restricted raw-RTL author profile"):
        B.materialize_bundle_inputs(workspace, bundle, repo=repo)


def test_private_selector_must_have_declared_host_coverage(selected):
    repo, workspace, _facts, bundle = selected
    bundle["host_inputs"] = []
    with pytest.raises(RuntimeError, match="declared under host_inputs"):
        B.materialize_bundle_inputs(workspace, bundle, repo=repo)


@pytest.mark.parametrize("kind", ["exact", "ancestor", "resolved-directory-alias", "hardlink", "absolute-alias"])
def test_public_grants_and_aliases_cannot_cover_private_facts(selected, kind):
    repo, workspace, facts, bundle = selected
    if kind == "exact":
        row = {"path": str(facts)}
    elif kind == "ancestor":
        row = {"path": str(facts.parent)}
    elif kind == "resolved-directory-alias":
        alias = repo / "public-alias"
        alias.symlink_to(facts.parent, target_is_directory=True)
        row = {"path": str(alias)}
    elif kind == "hardlink":
        alias = repo / "unrelated-public.json"
        os.link(facts, alias)
        row = {"path": str(alias)}
    else:
        public = repo / "public.txt"
        public.write_text("ordinary public data")
        row = {"path": str(public), "as": str(facts)}
    bundle["allowed"] = [row]
    with pytest.raises(RuntimeError, match="public grant"):
        B.materialize_bundle_inputs(workspace, bundle, repo=repo)


@pytest.mark.parametrize("kind", ["symlink", "hardlink"])
def test_nested_public_tree_aliases_still_fail_original_host_privacy_guard(selected, kind):
    repo, workspace, facts, bundle = selected
    public = repo / "public"
    public.mkdir()
    alias = public / "innocent.json"
    alias.symlink_to(facts) if kind == "symlink" else os.link(facts, alias)
    bundle["allowed"] = [{"path": str(public)}]
    with pytest.raises(RuntimeError, match="alias|host-only"):
        B.materialize_bundle_inputs(workspace, bundle, repo=repo)


@pytest.mark.parametrize("kind", ["relative", "traversal", "wrong-name", "symlink", "trailing-slash"])
def test_private_selector_path_cannot_be_indirect_or_escaping(selected, kind):
    repo, workspace, facts, bundle = selected
    if kind == "relative":
        path = "private/facts.json"
    elif kind == "traversal":
        path = str(facts.parent / "../private/facts.json")
    elif kind == "wrong-name":
        path = str(facts.with_name("other.json"))
    elif kind == "symlink":
        alias = repo / "alias"
        alias.symlink_to(facts.parent, target_is_directory=True)
        path = str(alias / "facts.json")
    else:
        path = str(facts) + "/"
    bundle["judge_rtl_facts_file"]["path"] = path
    with pytest.raises(RuntimeError, match="canonical.*facts.json"):
        B.materialize_bundle_inputs(workspace, bundle, repo=repo)


def test_missing_or_extra_private_pin_fields_rejected(selected):
    repo, workspace, _facts, bundle = selected
    for value in [None, "facts.json", {"path": "/private/facts.json"}, {**bundle["judge_rtl_facts_file"], "alias": "facts.json"}]:
        bundle["judge_rtl_facts_file"] = value
        with pytest.raises(RuntimeError, match="exactly path and sha256"):
            B.materialize_bundle_inputs(workspace, bundle, repo=repo)


def test_historical_public_selection_and_unselected_behavior_unchanged(selected):
    repo, workspace, facts, _private_bundle = selected
    historical = {
        "allowed": [{"path": str(facts.parent) + "/"}],
        "selected_rtl_facts_file": str(facts),
    }
    B.materialize_bundle_inputs(workspace, historical, repo=repo)
    assert B.frozen_selected_rtl_facts(workspace, historical, repo=repo).read_bytes() == facts.read_bytes()
    argv = B.base_argv(workspace, historical, repo=repo, include_claude_home=False)
    at = argv.index("MERLIN_RTL_FACTS")
    assert argv[at-1:at+2] == ["--setenv", "MERLIN_RTL_FACTS", str(facts)]
    assert B.frozen_selected_rtl_facts(workspace, {"allowed": []}, repo=repo) is None
