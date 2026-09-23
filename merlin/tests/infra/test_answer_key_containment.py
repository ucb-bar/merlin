"""Answer-key containment WITHOUT kernel isolation: the clean room, the read audit, the sandbox probe.

Each of the three mechanisms is tested twice: once that it accepts what it should, and once with the
mechanism MUTATED so the property it exists to hold no longer holds. A containment check that cannot
fail is the thing this whole area is guarding against, so every mutation below is asserted to be
caught, not merely exercised.

Nothing here names a hardware target: the answer surfaces are injected as synthetic paths, exactly as
the real ones are derived from a descriptor.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

from merlin.targetgen.sandbox import cleanroom as CR
from merlin.targetgen.sandbox import preflight as PF
from merlin.targetgen.sandbox import read_audit as RA
from merlin.targetgen.sandbox.answer_surfaces import AnswerSurface


# --------------------------------------------------------------------------- fixtures
@dataclass(frozen=True)
class _FakeTE:
    """Stand-in for a loaded descriptor. Only ``target`` is read by the code under test."""

    target: str = "t0"


@pytest.fixture
def fake_repo(tmp_path: Path) -> Path:
    """A miniature checkout: a granted tree, an answer-bearing tree inside it, and a target package
    that is denied as a blanket while DECLARING one grantable sub-tree."""
    repo = tmp_path / "repo"
    (repo / "public" / "nested").mkdir(parents=True)
    (repo / "public" / "iface.yaml").write_text("op: matmul\n", encoding="utf-8")
    (repo / "public" / "nested" / "notes.md").write_text("derive it\n", encoding="utf-8")
    # The answer key: a directory INSIDE the granted tree, plus a single answer file beside it.
    (repo / "public" / "answers" / "deep").mkdir(parents=True)
    (repo / "public" / "answers" / "golden.yaml").write_text("result: 42\n", encoding="utf-8")
    (repo / "public" / "answers" / "deep" / "more.yaml").write_text("result: 43\n", encoding="utf-8")
    (repo / "public" / "oracle.py").write_text("def answer(): return 42\n", encoding="utf-8")
    # The target package: hand-authored derivations the arm must not see, beside the contract it must.
    (repo / "public" / "package" / "derivations").mkdir(parents=True)
    (repo / "public" / "package" / "derivations" / "handwritten.py").write_text("# the answer\n", encoding="utf-8")
    (repo / "public" / "package" / "contracts").mkdir()
    (repo / "public" / "package" / "contracts" / "facts.yaml").write_text("mesh: 16\n", encoding="utf-8")
    return repo


@pytest.fixture
def wired(monkeypatch: pytest.MonkeyPatch, fake_repo: Path, tmp_path: Path):
    """Point the clean room at the miniature checkout and at a synthetic answer-surface set.

    Only the package surface declares a grantable sub-path, exactly as the real derivation does: the
    exemption is a property of the surface, not something a bundle can assert for itself.
    """
    surfaces = [
        AnswerSurface("answers", fake_repo / "public" / "answers", "dir", "golden"),
        AnswerSurface("oracle", fake_repo / "public" / "oracle.py", "file", "oracle"),
        AnswerSurface("package", fake_repo / "public" / "package", "dir", "backend", grantable=("contracts",)),
    ]
    monkeypatch.setattr(CR, "repo_root", lambda: fake_repo)
    monkeypatch.setattr(CR, "answer_surfaces", lambda _te: list(surfaces))
    monkeypatch.setenv(content_store_env(), str(tmp_path / "cas"))
    return _FakeTE()


def content_store_env() -> str:
    from merlin.common import content_store

    return content_store.LOCATION_ENV


# --------------------------------------------------------------------------- 1. clean room, accepted
def test_clean_room_materializes_the_allow_set_and_excises_the_answer_key(wired, tmp_path, fake_repo):
    """A broad grant that CONTAINS the answer key yields a room in which the key is absent."""
    bundle = {"allowed": [{"path": "public"}]}
    room = CR.build_clean_room(wired, tmp_path / "home", bundle)

    assert room.verdict.ok and room.verdict.complete
    inputs = room.inputs / "repo" / "public"
    assert (inputs / "iface.yaml").is_file()
    assert (inputs / "nested" / "notes.md").is_file()
    # Not denied-but-present: ABSENT.
    assert not (inputs / "answers").exists()
    assert not (inputs / "oracle.py").exists()
    assert not (inputs / "package" / "derivations").exists()
    excised = {Path(row["source"]).name for row in room.manifest["excised"]}
    # The package declares a grantable sub-tree, so it is walked into and excised entry by entry —
    # and with only the broad grant in hand, the contract sub-tree is excised too.
    assert {"answers", "oracle.py", "derivations", "contracts"} <= excised
    assert not (inputs / "package" / "contracts").exists()
    # The host-only manifest is a SIBLING of the room, never inside it.
    assert (tmp_path / "home" / CR.MANIFEST_NAME).is_file()
    assert not (room.root / CR.MANIFEST_NAME).exists()


def test_a_deeper_grant_does_not_reopen_a_surface(wired, tmp_path, fake_repo):
    """THE PRECEDENCE DECISION, asserted.

    A bundle that grants a subdirectory of an answer surface does NOT get it. Under a
    longest-prefix-wins rule it would, and that is the route by which an answer key is re-admitted: a
    grant added for a perfectly good reason silently re-opens what the deny existed to hide. Here the
    grant is simply ineffective, because ``public/answers`` declares no way in.
    """
    bundle = {"allowed": [{"path": "public"}, {"path": "public/answers/deep"}]}
    room = CR.build_clean_room(wired, tmp_path / "home", bundle)
    assert room.verdict.ok, room.verdict.describe()
    assert not (room.inputs / "repo" / "public" / "answers").exists()


def test_a_surface_may_declare_a_grantable_subtree(wired, tmp_path, fake_repo):
    """...and where the SURFACE declares a way in, a grant reaches it.

    This is not a softening of the rule above, it is the same rule: the exemption is derived with the
    surface, so the arm that is required to derive from its target's contract can still read it, while
    the hand-authored derivations beside it stay absent. Widening this takes an edit to the surface
    derivation, not to a manifest.
    """
    bundle = {"allowed": [{"path": "public"}, {"path": "public/package/contracts"}]}
    room = CR.build_clean_room(wired, tmp_path / "home", bundle)
    assert room.verdict.ok, room.verdict.describe()
    assert (room.inputs / "repo" / "public" / "package" / "contracts" / "facts.yaml").is_file()
    assert not (room.inputs / "repo" / "public" / "package" / "derivations").exists()


def test_an_undeclared_subtree_of_an_exempting_surface_stays_withheld(wired, tmp_path, fake_repo):
    """The exemption is scoped to what the surface named, not to the surface as a whole."""
    bundle = {"allowed": [{"path": "public/package/derivations"}]}
    room = CR.build_clean_room(wired, tmp_path / "home", bundle)
    assert room.verdict.ok, room.verdict.describe()
    assert not (room.inputs / "repo" / "public" / "package" / "derivations").exists()


def test_inputs_are_not_writable_through_a_shared_inode(wired, tmp_path, fake_repo):
    bundle = {"allowed": [{"path": "public"}]}
    room = CR.build_clean_room(wired, tmp_path / "home", bundle)
    placed = room.inputs / "repo" / "public" / "iface.yaml"
    assert not (placed.stat().st_mode & stat.S_IWUSR)
    # And the original is untouched by anything the room does.
    assert (fake_repo / "public" / "iface.yaml").read_text(encoding="utf-8") == "op: matmul\n"


# --------------------------------------------------------------------------- 1. clean room, MUTATED
def test_mutation_disabling_the_deny_rule_is_caught_by_verification(wired, tmp_path, monkeypatch):
    """THE MUTATION TEST FOR PART 1.

    Neutralise the containment rule so construction happily copies the answer key in, and assert the
    room is REFUSED anyway. This is what makes verification a second line of defence rather than a
    restatement of the builder: it re-derives the surfaces and inspects the bytes actually present, so
    a builder bug cannot certify itself.
    """
    monkeypatch.setattr(CR._Precedence, "decide", lambda self, path: CR._DECISION_ALLOW)
    with pytest.raises(CR.CleanRoomRefused) as excinfo:
        CR.build_clean_room(wired, tmp_path / "home", {"allowed": [{"path": "public"}]})
    assert "answer_surface_content" in str(excinfo.value)
    # A refused room is REMOVED, so no later run can find a half-room and treat it as usable.
    assert not (tmp_path / "home").exists()


def test_a_room_containing_an_answer_key_file_is_refused(wired, tmp_path, fake_repo):
    """The same property from the other direction: contamination that arrives AFTER the build — a stray
    copy, an operator convenience, an agent that fetched it — is refused by verification alone."""
    bundle = {"allowed": [{"path": "public"}]}
    room = CR.build_clean_room(wired, tmp_path / "home", bundle)
    assert room.verdict.ok

    smuggled = room.work / "notes" / "copied.yaml"
    smuggled.parent.mkdir(parents=True)
    smuggled.write_bytes((fake_repo / "public" / "answers" / "golden.yaml").read_bytes())

    verdict = CR.verify_clean_room(room.root, wired, bundle=bundle, repo=fake_repo)
    assert not verdict.ok
    kinds = {v.kind for v in verdict.violations}
    assert "answer_surface_content" in kinds
    assert any(str(smuggled) == v.path for v in verdict.violations)


def test_a_symlink_that_walks_out_of_the_room_is_refused(wired, tmp_path, fake_repo):
    bundle = {"allowed": [{"path": "public"}]}
    room = CR.build_clean_room(wired, tmp_path / "home", bundle)
    (room.work / "door").symlink_to(fake_repo)

    verdict = CR.verify_clean_room(room.root, wired, bundle=bundle, repo=fake_repo)
    assert not verdict.ok
    assert "symlink_escapes_room" in {v.kind for v in verdict.violations}


def test_a_symlink_to_an_allowed_file_inside_the_room_is_accepted(wired, tmp_path, fake_repo):
    """A link is not the problem; a link that lets the tree be walked UPWARD is."""
    bundle = {"allowed": [{"path": "public"}]}
    room = CR.build_clean_room(wired, tmp_path / "home", bundle)
    (room.work / "iface.yaml").symlink_to(room.inputs / "repo" / "public" / "iface.yaml")

    verdict = CR.verify_clean_room(room.root, wired, bundle=bundle, repo=fake_repo)
    assert verdict.ok, verdict.describe()


def test_a_room_inside_the_checkout_is_refused(wired, fake_repo):
    """An agent CLI detects its project root by walking up for a .git; a room under the checkout hands
    it the checkout no matter how tidy the room is."""
    with pytest.raises(CR.CleanRoomRefused) as excinfo:
        CR.build_clean_room(wired, fake_repo / "inside" / "home", {"allowed": [{"path": "public"}]})
    assert "room_inside_checkout" in str(excinfo.value)


def test_a_gitdir_pointer_file_is_refused(wired, tmp_path, fake_repo):
    bundle = {"allowed": [{"path": "public"}]}
    room = CR.build_clean_room(wired, tmp_path / "home", bundle)
    (room.work / ".git").write_text(f"gitdir: {fake_repo}/.git\n", encoding="utf-8")
    verdict = CR.verify_clean_room(room.root, wired, bundle=bundle, repo=fake_repo)
    assert not verdict.ok
    assert "git_pointer_out_of_room" in {v.kind for v in verdict.violations}


def test_an_unresolvable_grant_means_no_room(wired, tmp_path):
    with pytest.raises(CR.CleanRoomRefused) as excinfo:
        CR.build_clean_room(wired, tmp_path / "home", {"allowed": [{"path": "public/absent"}]})
    assert "unresolvable" in str(excinfo.value)
    assert not (tmp_path / "home").exists()


def test_an_underivable_deny_set_means_no_room(wired, tmp_path, monkeypatch):
    """FAIL CLOSED: if the answer surfaces cannot be derived there is no room, never an empty deny set."""

    def _boom(_te):
        raise RuntimeError("descriptor unreadable")

    monkeypatch.setattr(CR, "answer_surfaces", _boom)
    with pytest.raises(CR.CleanRoomRefused) as excinfo:
        CR.build_clean_room(wired, tmp_path / "home", {"allowed": [{"path": "public"}]})
    assert "could not be derived" in str(excinfo.value)


def test_a_verification_that_cannot_complete_is_not_a_pass(wired, tmp_path, monkeypatch):
    """An incomplete check is not a clean check, for the same reason an unauditable run is not a clean
    run. Simulated by making the content comparison report itself incomplete."""
    bundle = {"allowed": [{"path": "public"}]}
    room = CR.build_clean_room(wired, tmp_path / "home", bundle)
    assert room.verdict.ok

    monkeypatch.setattr(CR, "_content_matches", lambda *_a, **_k: ([], False, "surface unreadable"))
    verdict = CR.verify_clean_room(room.root, wired, bundle=bundle, repo=CR.repo_root())
    assert not verdict.complete
    assert not verdict.ok
    assert "surface unreadable" in verdict.describe()


# --------------------------------------------------------------------------- 3. the sandbox probe
def _fake_bwrap(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)
    return path


@pytest.fixture(autouse=True)
def _clear_probe_cache():
    PF.reset_probe_cache()
    yield
    PF.reset_probe_cache()


def test_probe_reports_ok_for_a_sandbox_that_runs(tmp_path):
    binary = _fake_bwrap(tmp_path, "bwrap-ok", "#!/bin/sh\nexit 0\n")
    probe = PF.probe_sandbox(binary)
    assert probe.status == PF.SANDBOX_OK
    assert probe.usable


def test_probe_reports_inoperable_and_names_the_cause(tmp_path):
    """The observed condition on a host whose policy forbids unprivileged user namespaces."""
    binary = _fake_bwrap(
        tmp_path,
        "bwrap-denied",
        "#!/bin/sh\necho 'bwrap: setting up uid map: Permission denied' >&2\nexit 1\n",
    )
    probe = PF.probe_sandbox(binary)
    assert probe.status == PF.SANDBOX_INOPERABLE
    assert probe.reason == "userns_uid_map_denied"
    assert not probe.usable
    assert "uid map" in probe.describe()


def test_mutation_presence_on_path_is_not_operability(tmp_path, monkeypatch):
    """THE MUTATION TEST FOR PART 3.

    This is the exact condition today's refusals miss: the binary is installed and ``which`` finds it,
    so a presence check passes and no refusal fires — while the sandbox cannot actually be built. The
    test asserts the two answers DISAGREE, which is the whole reason the probe exists. A probe that
    regressed to a presence check would make both sides agree and fail here.
    """
    import shutil as _shutil

    binary = _fake_bwrap(
        tmp_path, PF.SANDBOX_BINARY_NAME, "#!/bin/sh\necho 'bwrap: setting up uid map: Permission denied' >&2\nexit 1\n"
    )
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ.get('PATH', '')}")

    assert _shutil.which(PF.SANDBOX_BINARY_NAME) == str(binary)  # the old question says "yes"
    probe = PF.probe_sandbox()  # the new question says "no"
    assert probe.status == PF.SANDBOX_INOPERABLE
    assert not probe.usable


def test_absent_and_inoperable_are_distinguishable(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    assert PF.probe_sandbox().status == PF.SANDBOX_ABSENT
    inoperable = PF.probe_sandbox(_fake_bwrap(tmp_path, "bwrap-bad", "#!/bin/sh\nexit 3\n"))
    assert inoperable.status == PF.SANDBOX_INOPERABLE
    assert inoperable.reason == "nonzero_exit"


def test_unknown_is_never_usable(tmp_path, monkeypatch):
    """A probe that could not be carried out is UNKNOWN, and UNKNOWN does not read as a working
    sandbox anywhere."""

    def _timeout(*_a, **_k):
        raise subprocess.TimeoutExpired(cmd="bwrap", timeout=1)

    monkeypatch.setattr(PF.subprocess, "run", _timeout)
    probe = PF.probe_sandbox(_fake_bwrap(tmp_path, "bwrap-slow", "#!/bin/sh\nsleep 99\n"))
    assert probe.status == PF.SANDBOX_UNKNOWN
    assert not probe.usable
    with pytest.raises(PF.SandboxUnavailable):
        PF.require_working_sandbox(_fake_bwrap(tmp_path, "bwrap-slow2", "#!/bin/sh\nsleep 99\n"))


def test_require_working_sandbox_carries_the_condition(tmp_path):
    binary = _fake_bwrap(
        tmp_path, "bwrap-denied2", "#!/bin/sh\necho 'bwrap: setting up uid map: Permission denied' >&2\nexit 1\n"
    )
    with pytest.raises(PF.SandboxUnavailable) as excinfo:
        PF.require_working_sandbox(binary, context="agent round")
    assert excinfo.value.probe.status == PF.SANDBOX_INOPERABLE
    assert "agent round" in str(excinfo.value)
    record = excinfo.value.probe.as_record()
    assert record["usable"] is False and record["reason"] == "userns_uid_map_denied"


# --------------------------------------------------------------------------- 2. the read audit
@pytest.fixture
def audited(monkeypatch: pytest.MonkeyPatch, fake_repo: Path):
    """Point the auditor at the same synthetic withheld set the clean-room tests use."""
    surfaces = [
        AnswerSurface("answers", fake_repo / "public" / "answers", "dir", "golden"),
        AnswerSurface("oracle", fake_repo / "public" / "oracle.py", "file", "oracle"),
    ]
    monkeypatch.setattr(RA, "repo_root", lambda: fake_repo)
    monkeypatch.setattr(RA, "answer_surfaces", lambda _te: list(surfaces))
    monkeypatch.setattr(
        RA,
        "audit_tokens",
        lambda _te: {"answer": ("golden.yaml",), "grader": ("decode",), "oracle_subpath": ()},
    )
    return _FakeTE()


def _command(seq: int, command: str, *, exit_code: int | None = 0, output: str = "data") -> list[dict]:
    """The started/completed pair the driver emits for one shell tool call."""
    item = {"id": f"item_{seq}", "type": "command_execution", "command": command}
    started = {
        "seq": seq,
        "arrived_at": "2026-09-20T00:00:00+00:00",
        "event": {
            "type": "item.started",
            "item": {**item, "aggregated_output": "", "exit_code": None, "status": "in_progress"},
        },
    }
    if exit_code is None:
        return [started]
    completed = {
        "seq": seq + 1,
        "arrived_at": "2026-09-20T00:00:01+00:00",
        "event": {
            "type": "item.completed",
            "item": {
                **item,
                "aggregated_output": output,
                "exit_code": exit_code,
                "status": "completed" if exit_code == 0 else "failed",
            },
        },
    }
    return [started, completed]


def _write_log(tmp_path: Path, rows: list, name: str = "round_00.codex_events.timestamped.jsonl") -> Path:
    log = tmp_path / "rounds" / name
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("".join(row if isinstance(row, str) else json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return log


def test_a_read_of_a_withheld_path_is_contaminated(audited, tmp_path, fake_repo):
    """The positive control: a plain, successful read of an answer file is found and named."""
    target = fake_repo / "public" / "answers" / "golden.yaml"
    log = _write_log(tmp_path, [*_command(1, f'/bin/bash -lc "cat {target}"')])
    audit = RA.audit_event_logs([log], audited, repo=fake_repo)

    assert audit.verdict == RA.CONTAMINATED
    assert not audit.is_clean and not audit
    assert [hit.kind for hit in audit.violations] == ["path_read"]
    assert audit.violations[0].path == str(target)
    with pytest.raises(RA.AnswerKeyExposure):
        audit.require_clean()


def test_mutation_an_unparseable_event_log_is_unknown_not_clean(audited, tmp_path):
    """THE MUTATION TEST FOR PART 2, and the single most important property of this component.

    The log is corrupt, so the auditor learns nothing. The one answer it must NOT give is the one that
    reads as a clearance. A run that cannot be audited has not been shown to be clean, so UNKNOWN is
    asserted here through every route a caller might take it: the verdict string, the only affirmative
    accessor, truthiness, and the raising form.
    """
    log = _write_log(tmp_path, ["{this is not json\n", "neither is this\n"])
    audit = RA.audit_event_logs([log], audited)

    assert audit.verdict == RA.UNKNOWN
    assert audit.verdict != RA.CLEAN
    assert audit.is_clean is False
    assert bool(audit) is False
    assert audit.events_unparsed == 2
    assert "NOT been shown to be clean" in audit.describe()
    assert audit.as_record()["verdict"] == RA.UNKNOWN
    with pytest.raises(RA.AnswerKeyExposure):
        audit.require_clean()


def test_a_partly_unparseable_log_does_not_become_clean(audited, tmp_path):
    """A log that is MOSTLY fine is still not auditable. The clean half must not carry the verdict."""
    log = _write_log(tmp_path, [*_command(1, '/bin/bash -lc "cat README.md"'), "<<<truncated\n"])
    audit = RA.audit_event_logs([log], audited)
    assert audit.verdict == RA.UNKNOWN
    assert audit.events_parsed == 2 and audit.events_unparsed == 1


def test_a_missing_event_log_is_unknown(audited, tmp_path):
    (tmp_path / "run").mkdir()
    audit = RA.audit_run(tmp_path / "run", audited)
    assert audit.verdict == RA.UNKNOWN
    assert audit.reason == "no_event_log"
    assert not audit.is_clean


def test_an_unrecognised_event_kind_forces_unknown(audited, tmp_path):
    """FAIL CLOSED on the schema itself: an item kind this module has never seen may be carrying a read
    it cannot see, so it is UNKNOWN rather than skipped."""
    row = {
        "seq": 1,
        "arrived_at": "2026-09-20T00:00:00+00:00",
        "event": {"type": "item.completed", "item": {"id": "i1", "type": "quantum_tool_call"}},
    }
    audit = RA.audit_event_logs([_write_log(tmp_path, [row])], audited)
    assert audit.verdict == RA.UNKNOWN
    assert any("quantum_tool_call" in kind for kind in audit.unknown_kinds)


def test_an_unrecognised_collab_tool_forces_unknown_while_a_known_one_does_not(audited, tmp_path):
    """The fail-closed boundary sits at the TOOL, not at the item kind whose vocabulary it opens."""

    def _collab(tool: str) -> dict:
        return {
            "seq": 1,
            "arrived_at": "2026-09-20T00:00:00+00:00",
            "event": {"type": "item.completed", "item": {"id": "i1", "type": "collab_tool_call", "tool": tool}},
        }

    assert RA.audit_event_logs([_write_log(tmp_path, [_collab("wait")])], audited).verdict == RA.CLEAN
    later = _write_log(tmp_path, [_collab("fetch_file")], name="round_01.codex_events.timestamped.jsonl")
    assert RA.audit_event_logs([later], audited).verdict == RA.UNKNOWN


def test_an_exclusion_pattern_is_not_a_read(audited, tmp_path, fake_repo):
    """``find ... ! -name golden.yaml`` EXCLUDES the answer key. Reading that as a read of it accuses an
    agent of the exact opposite of what it did — a cry-wolf regression guard."""
    log = _write_log(tmp_path, [*_command(1, "/bin/bash -lc \"find . -type f ! -name 'golden.yaml' | sort\"")])
    audit = RA.audit_event_logs([log], audited, repo=fake_repo)
    assert audit.verdict == RA.CLEAN, audit.describe()
    assert {hit.kind for hit in audit.hits} <= {"pattern_mention"}


def test_a_filter_program_mentioning_a_grader_stem_is_not_a_read(audited, tmp_path, fake_repo):
    """A bare module stem substring-matched against shell text produced 30 of 34 accusations on the
    archived transcripts, every one of them against an agent doing what it was asked to. A stem has to
    BE a path component to count."""
    log = _write_log(tmp_path, [*_command(1, "/bin/bash -lc \"jq -r '.instructions[] | {class,decoded}' trace.json\"")])
    audit = RA.audit_event_logs([log], audited, repo=fake_repo)
    assert audit.verdict == RA.CLEAN, audit.describe()
    assert not audit.violations


def test_a_blocked_probe_is_advisory(audited, tmp_path, fake_repo):
    """The agent reached for the answer key and got nothing. The protection worked; record it, do not
    fail the run on it."""
    target = fake_repo / "public" / "answers" / "golden.yaml"
    log = _write_log(tmp_path, [*_command(1, f'/bin/bash -lc "cat {target}"', exit_code=1, output="")])
    audit = RA.audit_event_logs([log], audited, repo=fake_repo)
    assert audit.verdict == RA.CLEAN
    assert [hit.kind for hit in audit.hits] == ["blocked_probe"]


def test_a_compound_command_outcome_is_indeterminate_rather_than_guessed(audited, tmp_path, fake_repo):
    """Captured output belongs to the whole pipeline, so it cannot be attributed to the one simple
    command that named a withheld path. That is UNKNOWN, not a verdict either way."""
    target = fake_repo / "public" / "answers" / "golden.yaml"
    log = _write_log(tmp_path, [*_command(1, f'/bin/bash -lc "echo hi && cat {target}"')])
    audit = RA.audit_event_logs([log], audited, repo=fake_repo)
    assert audit.verdict == RA.UNKNOWN
    assert [hit.kind for hit in audit.indeterminate] == ["indeterminate_outcome"]
    assert not audit.violations


def test_a_write_to_a_withheld_path_is_a_violation(audited, tmp_path, fake_repo):
    """A kind that has not been declared advisory counts as a violation, which is what makes a NEW hit
    kind disqualifying until somebody deliberately says otherwise."""
    row = {
        "seq": 1,
        "arrived_at": "2026-09-20T00:00:00+00:00",
        "event": {
            "type": "item.completed",
            "item": {
                "id": "i1",
                "type": "file_change",
                "status": "completed",
                "changes": [{"path": str(fake_repo / "public" / "answers" / "golden.yaml"), "kind": "update"}],
            },
        },
    }
    audit = RA.audit_event_logs([_write_log(tmp_path, [row])], audited, repo=fake_repo)
    assert audit.verdict == RA.CONTAMINATED
    assert [hit.kind for hit in audit.violations] == ["answer_surface_write"]


def test_a_run_that_touched_nothing_withheld_is_clean(audited, tmp_path, fake_repo):
    log = _write_log(
        tmp_path,
        [
            {
                "seq": 1,
                "arrived_at": "2026-09-20T00:00:00+00:00",
                "event": {"type": "thread.started", "thread_id": "t"},
            },
            *_command(2, "/bin/bash -lc \"sed -n '1,20p' public/iface.yaml\""),
            {
                "seq": 9,
                "arrived_at": "2026-09-20T00:00:02+00:00",
                "event": {"type": "item.completed", "item": {"id": "m", "type": "agent_message", "text": "done"}},
            },
        ],
    )
    audit = RA.audit_event_logs([log], audited, repo=fake_repo)
    assert audit.verdict == RA.CLEAN and audit.is_clean and audit
    audit.require_clean()


def test_an_underivable_withheld_set_is_unknown(audited, tmp_path, monkeypatch):
    """FAIL CLOSED: if the answer surfaces cannot be derived, nothing can be cleared against them."""

    def _boom(_te):
        raise RuntimeError("descriptor unreadable")

    monkeypatch.setattr(RA, "answer_surfaces", _boom)
    audit = RA.audit_event_logs([_write_log(tmp_path, [*_command(1, '/bin/bash -lc "ls"')])], audited)
    assert audit.verdict == RA.UNKNOWN
    assert audit.reason == "withheld_set_underivable"


def test_the_raw_envelope_is_accepted_too(audited, tmp_path, fake_repo):
    """The raw sidecar holds the bare event object rather than the seq/arrived_at wrapper."""
    target = fake_repo / "public" / "answers" / "golden.yaml"
    rows = [
        {
            "type": "item.completed",
            "item": {
                "id": "i1",
                "type": "command_execution",
                "command": f'/bin/bash -lc "cat {target}"',
                "aggregated_output": "data",
                "exit_code": 0,
                "status": "completed",
            },
        }
    ]
    audit = RA.audit_event_logs([_write_log(tmp_path, rows)], audited, repo=fake_repo)
    assert audit.verdict == RA.CONTAMINATED
