"""Pin verification is tested on the mistake it exists to prevent.

A session certified a microkernel against the only saturn revision containing the outer-product unit while
believing it was the revision named for the tapeout — which does not contain that unit at all. The
registry's job is to make that detectable before the work runs, so the central test builds exactly that
situation (right repo, plausible branch, wrong content) and requires it to be caught.

Everything here uses throwaway git repositories, so it runs with no hardware and no external checkout.
"""
from __future__ import annotations

import subprocess

import pytest

from merlin.common import provenance as P


def _repo(tmp_path, name="src"):
    """A tiny git repo with one commit; returns (path, sha)."""
    root = tmp_path / name
    root.mkdir(parents=True, exist_ok=True)
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@t"}
    def git(*args):
        return subprocess.run(("git", "-C", str(root)) + args, capture_output=True, text=True,
                              env={**env, "PATH": "/usr/bin:/bin"}, check=False)
    git("init", "-q", "-b", "main")
    (root / "kept.txt").write_text("hello", encoding="utf-8")
    git("add", "-A")
    git("commit", "-qm", "one")
    git("remote", "add", "origin", "https://example.invalid/canonical.git")
    sha = subprocess.run(("git", "-C", str(root), "rev-parse", "HEAD"), capture_output=True,
                         text=True).stdout.strip()
    return root, sha


def _pins(tmp_path, body: str):
    p = tmp_path / "pins.yaml"
    p.write_text("version: 1\npins:\n" + body, encoding="utf-8")
    return p


class TestTheRegistry:
    def test_an_unquoted_numeric_commit_is_refused(self, tmp_path):
        # YAML reads 40 digits as a NUMBER and drops leading zeros, so the pin would verify against a
        # different revision than the one written down. Caught by the tests below needing quotes.
        pins = _pins(tmp_path, "  x:\n    commit: " + "0" * 40 + "\n")
        with pytest.raises(P.PinsError, match="must be quoted"):
            P.load_pins(pins)

    def test_an_abbreviated_commit_is_refused(self, tmp_path):
        pins = _pins(tmp_path, '  x:\n    commit: "ea37380"\n')
        with pytest.raises(P.PinsError, match="40-character"):
            P.load_pins(pins)

    def test_a_pin_without_a_commit_is_refused(self, tmp_path):
        # A pin with no revision pins nothing, so it must not load as if it did.
        pins = _pins(tmp_path, "  x:\n    repo_canonical: https://e.invalid/r.git\n")
        with pytest.raises(P.PinsError, match="no commit"):
            P.load_pins(pins)

    def test_a_missing_registry_raises(self, tmp_path):
        with pytest.raises(P.PinsError, match="no pin registry"):
            P.load_pins(tmp_path / "absent.yaml")

    def test_an_unknown_pin_name_raises_and_lists_what_exists(self, tmp_path):
        pins = _pins(tmp_path, '  a:\n    commit: "' + "a" * 40 + '"\n')
        with pytest.raises(P.PinsError, match="declared"):
            P.pin("nope", pins)

    def test_the_shipped_registry_loads_and_declares_full_shas(self):
        got = P.load_pins()
        assert got, "the repo's own registry must load"
        for name, p in got.items():
            assert len(p.commit) == 40, f"{name} should pin a full sha, got {p.commit!r}"
            assert p.description, f"{name} needs to say what it is for"


class TestObservation:
    def test_it_reads_the_actual_revision(self, tmp_path):
        root, sha = _repo(tmp_path)
        got = P.observe(root)
        assert got.present and got.commit == sha and got.branch == "main"
        assert got.dirty_files == 0 and got.dirty is False

    def test_an_absent_path_is_not_present_rather_than_an_error(self, tmp_path):
        got = P.observe(tmp_path / "nothing")
        assert not got.present and got.commit == P.UNKNOWN

    def test_a_non_git_directory_reports_unknown_not_a_guess(self, tmp_path):
        (tmp_path / "plain").mkdir()
        got = P.observe(tmp_path / "plain")
        assert got.present and got.commit == P.UNKNOWN

    def test_uncommitted_changes_are_counted(self, tmp_path):
        root, _ = _repo(tmp_path)
        (root / "kept.txt").write_text("changed", encoding="utf-8")
        assert P.observe(root).dirty is True


class TestVerificationCatchesTheRealMistake:
    """Right repository, plausible branch, wrong content."""

    def test_a_forbidden_path_present_is_caught(self, tmp_path):
        # THE test. The tapeout pin declares the outer-product unit ABSENT; a checkout that has it is not
        # the revision it claims to be, however the revision was spelled.
        root, sha = _repo(tmp_path)
        (root / "OuterProductUnit.scala").write_text("// the unit", encoding="utf-8")
        pins = _pins(tmp_path, f"""  tapeout:
    commit: {sha}
    forbids_paths: [OuterProductUnit.scala]
""")
        got = P.verify("tapeout", checkout=root, path=pins)
        assert not got.ok
        assert got.forbidden_present == ("OuterProductUnit.scala",)

    def test_a_missing_required_path_is_caught(self, tmp_path):
        # The mirror: the OPU pin needs the unit, and a checkout lacking it cannot support the work.
        root, sha = _repo(tmp_path)
        pins = _pins(tmp_path, f"""  opu:
    commit: {sha}
    requires_paths: [src/OuterProductUnit.scala]
""")
        got = P.verify("opu", checkout=root, path=pins)
        assert not got.ok and got.missing_paths == ("src/OuterProductUnit.scala",)

    def test_a_wrong_commit_is_caught(self, tmp_path):
        root, _ = _repo(tmp_path)
        pins = _pins(tmp_path, '  x:\n    commit: "' + "0" * 40 + '"\n')
        got = P.verify("x", checkout=root, path=pins)
        assert not got.ok and any("commit is" in d for d in got.drift)

    def test_a_wrong_branch_is_caught(self, tmp_path):
        root, sha = _repo(tmp_path)
        pins = _pins(tmp_path, f"  x:\n    commit: {sha}\n    branch: opu-int8\n")
        got = P.verify("x", checkout=root, path=pins)
        assert any("branch is" in d for d in got.drift)

    def test_a_dirty_tree_with_nothing_declared_as_read_is_drift(self, tmp_path):
        # Fail closed: with no read set there is no basis for calling the dirt harmless, so "does the
        # declared revision describe what would be read" is UNKNOWN — and an UNKNOWN is drift, never
        # agreement.
        root, sha = _repo(tmp_path)
        (root / "kept.txt").write_text("edited", encoding="utf-8")
        pins = _pins(tmp_path, f"  x:\n    commit: {sha}\n")
        got = P.verify("x", checkout=root, path=pins)
        assert not got.ok and any("uncommitted" in d and "UNKNOWN" in d for d in got.drift)

    def test_an_edit_to_a_source_that_is_read_is_drift(self, tmp_path):
        # THE case a pin exists for: the commit still looks right, but the bytes a derivation consumes are
        # not the ones that revision contains, so anything emitted from them is unattributable.
        root, sha = _repo(tmp_path)
        (root / "kept.txt").write_text("edited", encoding="utf-8")
        pins = _pins(tmp_path, f"  x:\n    commit: {sha}\n    requires_paths: [kept.txt]\n")
        got = P.verify("x", checkout=root, path=pins)
        assert not got.ok
        assert any("kept.txt" in d for d in got.drift)

    def test_an_edit_to_something_never_read_is_a_note_not_drift(self, tmp_path):
        # A checkout on a shared host is almost never pristine. Reporting a stray build log as drift makes
        # the check fire on every build and teaches people to ignore it, which is worse than not checking —
        # so it is recorded and does not clear `ok`.
        root, sha = _repo(tmp_path)
        (root / "stray.log").write_text("build noise", encoding="utf-8")
        pins = _pins(tmp_path, f"  x:\n    commit: {sha}\n    requires_paths: [kept.txt]\n")
        got = P.verify("x", checkout=root, path=pins)
        assert got.ok, got.drift
        assert any("none of them a source this reads" in n for n in got.notes)
        assert got.observed.dirty is True, "the fact itself must not be lost"

    def test_the_read_set_can_be_narrowed_by_the_caller(self, tmp_path):
        # A build reads a specific set of files, which is usually narrower than everything the pin requires
        # to be present. Answering about THIS build is what makes the verdict actionable.
        root, sha = _repo(tmp_path)
        (root / "a.txt").write_text("x", encoding="utf-8")     # the only dirty file
        pins = _pins(tmp_path, f"  x:\n    commit: {sha}\n")
        assert P.verify("x", checkout=root, path=pins, reads=["b.txt"]).ok
        assert not P.verify("x", checkout=root, path=pins, reads=["a.txt"]).ok

    def test_an_untracked_directory_covers_the_files_inside_it(self, tmp_path):
        # git reports a newly-added directory as ONE entry with a trailing slash, so comparing only for
        # equality would call an edited source inside it clean.
        root, sha = _repo(tmp_path)
        (root / "src").mkdir()
        (root / "src" / "Consts.scala").write_text("// new", encoding="utf-8")
        pins = _pins(tmp_path, f"  x:\n    commit: {sha}\n")
        got = P.verify("x", checkout=root, path=pins, reads=["src/Consts.scala"])
        assert not got.ok and any("src/Consts.scala" in d for d in got.drift)

    def test_the_dirty_paths_are_recorded_not_just_counted(self, tmp_path):
        root, _sha = _repo(tmp_path)
        (root / "one.txt").write_text("1", encoding="utf-8")
        got = P.observe(root)
        assert "one.txt" in got.dirty_paths and got.dirty_files == len(got.dirty_paths)

    def test_a_different_origin_is_caught_unless_the_pin_explains_it(self, tmp_path):
        root, sha = _repo(tmp_path)
        strict = _pins(tmp_path, f"  x:\n    commit: {sha}\n"
                                 "    repo_canonical: https://example.invalid/other.git\n")
        assert any("origin is" in d for d in P.verify("x", checkout=root, path=strict).drift)
        # A pin that documents the fork does not re-report it as a surprise.
        explained = tmp_path / "explained.yaml"
        explained.write_text("version: 1\npins:\n"
                             f"  x:\n    commit: {sha}\n"
                             "    repo_canonical: https://example.invalid/other.git\n"
                             "    repo_observed_note: lives on a fork\n", encoding="utf-8")
        assert not any("origin is" in d for d in P.verify("x", checkout=root, path=explained).drift)

    def test_a_clean_matching_checkout_verifies(self, tmp_path):
        root, sha = _repo(tmp_path)
        pins = _pins(tmp_path, f"  x:\n    commit: {sha}\n    branch: main\n"
                               "    requires_paths: [kept.txt]\n")
        got = P.verify("x", checkout=root, path=pins)
        assert got.ok and not got.drift

    def test_require_raises_and_names_every_disagreement(self, tmp_path):
        root, _ = _repo(tmp_path)
        (root / "bad.scala").write_text("x", encoding="utf-8")
        pins = _pins(tmp_path, '  x:\n    commit: "' + "0" * 40 + '"\n'
                     "    requires_paths: [absent.txt]\n    forbids_paths: [bad.scala]\n")
        with pytest.raises(P.PinsError) as exc:
            P.require("x", checkout=root, path=pins)
        msg = str(exc.value)
        assert "commit is" in msg and "missing required path" in msg and "declares them absent" in msg

    def test_an_unset_root_env_is_drift_not_a_crash(self, tmp_path):
        pins = _pins(tmp_path, '  x:\n    commit: "' + "0" * 40 + '"\n'
                     "    root_env: MERLIN_DEFINITELY_UNSET_VAR_XYZ\n")
        got = P.verify("x", path=pins)
        assert not got.ok and any("unset" in d for d in got.drift)


class TestWhatGetsRecorded:
    def test_the_source_digest_changes_when_a_read_file_changes(self, tmp_path):
        f = tmp_path / "a.scala"
        f.write_text("one", encoding="utf-8")
        before = P.source_digest([f])
        f.write_text("two", encoding="utf-8")
        assert P.source_digest([f]) != before

    def test_the_source_digest_is_order_independent(self, tmp_path):
        a, b = tmp_path / "a", tmp_path / "b"
        a.write_text("1", encoding="utf-8")
        b.write_text("2", encoding="utf-8")
        assert P.source_digest([a, b]) == P.source_digest([b, a])

    def test_an_unreadable_source_does_not_silently_hash_as_empty(self, tmp_path):
        missing = tmp_path / "gone"
        present = tmp_path / "here"
        present.write_text("", encoding="utf-8")
        # An empty file and an absent one must not produce the same digest, or a vanished source would
        # look like an unchanged one.
        assert P.source_digest([missing]) != P.source_digest([present])

    def test_a_binary_digest_identifies_which_build_ran(self, tmp_path):
        exe = tmp_path / "sim"
        exe.write_bytes(b"build-A")
        first = P.file_digest(exe)
        exe.write_bytes(b"build-B")
        assert P.file_digest(exe) != first
        assert P.file_digest(tmp_path / "absent") == P.UNKNOWN

    def test_the_record_carries_the_verdict_and_the_drift(self, tmp_path):
        root, sha = _repo(tmp_path)
        pins = _pins(tmp_path, '  x:\n    commit: "' + "0" * 40 + '"\n')
        v = P.verify("x", checkout=root, path=pins)
        got = P.record(pins={"x": v}, sources=[root / "kept.txt"], artifacts={"sim": root / "kept.txt"})
        assert got["all_pins_ok"] is False
        assert got["hardware_pins"]["x"]["drift"], "drift must survive into the record"
        assert got["source_digest"] and got["artifact_digests"]["sim"] != P.UNKNOWN
        assert "commit" in got["merlin"]

    def test_the_record_is_json_serialisable(self):
        import json
        json.dumps(P.record(pins={}, sources=[]))


class TestArtifactsAreIdentifiedByContent:
    """A pin answers "which checkout was this read from". A built thing has no answer to that.

    An FPGA bitstream, a compiled simulator and a packaged image have no commit of their own. What
    identifies them is their bytes plus the revisions they were elaborated from — so those are separate
    fields, and the check has to distinguish "absent", "present but different" and "present with nothing
    declared to compare against". The last one is the interesting case: treating it as a pass is how an
    artifact ends up certifying itself.
    """

    def _reg(self, tmp_path, body: str):
        p = tmp_path / "pins.yaml"
        p.write_text("version: 1\npins: {}\nartifacts:\n" + body, encoding="utf-8")
        return p

    def test_a_matching_digest_verifies(self, tmp_path):
        blob = tmp_path / "firesim.tar.gz"
        blob.write_bytes(b"a bitstream")
        digest = P.file_digest(blob)
        reg = self._reg(tmp_path, f'  bit:\n    path: "{blob}"\n    digest: "{digest}"\n')
        got = P.verify_artifact("bit", path=reg)
        assert got.ok and got.matches is True and got.gaps == ()

    def test_a_changed_artifact_is_caught(self, tmp_path):
        blob = tmp_path / "firesim.tar.gz"
        blob.write_bytes(b"a bitstream")
        digest = P.file_digest(blob)
        blob.write_bytes(b"a DIFFERENT bitstream")
        reg = self._reg(tmp_path, f'  bit:\n    path: "{blob}"\n    digest: "{digest}"\n')
        got = P.verify_artifact("bit", path=reg)
        assert not got.ok and got.matches is False
        assert any("digest is" in g for g in got.gaps)

    def test_an_absent_artifact_is_not_a_pass(self, tmp_path):
        reg = self._reg(tmp_path, f'  bit:\n    path: "{tmp_path / "nope.tar.gz"}"\n    digest: "{"a"*64}"\n')
        got = P.verify_artifact("bit", path=reg)
        assert not got.ok and not got.present
        assert any("no file at" in g for g in got.gaps)

    def test_an_undeclared_digest_is_a_gap_not_agreement(self, tmp_path):
        # THE case. An artifact registered before its build finishes is honest; one that verifies against
        # nothing is self-certifying, which is what this registry exists to prevent.
        blob = tmp_path / "firesim.tar.gz"
        blob.write_bytes(b"whatever")
        reg = self._reg(tmp_path, f'  bit:\n    path: "{blob}"\n')
        got = P.verify_artifact("bit", path=reg)
        assert got.present and got.matches is None and not got.ok
        assert any("no digest declared" in g for g in got.gaps)
        assert got.digest != P.UNKNOWN, "the digest found must still be reported"

    def test_it_records_what_it_was_built_from(self, tmp_path):
        # A bitstream's identity is its bytes AND the revisions it was elaborated from; naming the pins
        # rather than repeating their shas keeps one source of truth for each.
        blob = tmp_path / "b.tar.gz"
        blob.write_bytes(b"x")
        reg = self._reg(tmp_path, f'  bit:\n    path: "{blob}"\n    built_from: [saturn_opu_int8]\n'
                                  f'    config: FireSimOPUV256D128ShuttleConfig\n')
        got = P.load_artifacts(reg)["bit"]
        assert got.built_from == ("saturn_opu_int8",)
        assert got.config == "FireSimOPUV256D128ShuttleConfig"

    def test_an_artifact_without_a_path_is_refused(self, tmp_path):
        reg = self._reg(tmp_path, "  bit:\n    digest: \"" + "a" * 64 + "\"\n")
        with pytest.raises(P.PinsError, match="no path"):
            P.load_artifacts(reg)

    def test_an_unquoted_digest_is_refused(self, tmp_path):
        # The same trap the commit shas have: an all-digit digest is read by YAML as a number.
        reg = self._reg(tmp_path, "  bit:\n    path: /x\n    digest: 12345678\n")
        with pytest.raises(P.PinsError, match="quoted string"):
            P.load_artifacts(reg)

    def test_an_unknown_artifact_lists_what_exists(self, tmp_path):
        blob = tmp_path / "b"; blob.write_bytes(b"x")
        reg = self._reg(tmp_path, f'  bit:\n    path: "{blob}"\n')
        with pytest.raises(P.PinsError, match="declared"):
            P.verify_artifact("other", path=reg)

    def test_the_shipped_registry_still_loads_with_no_artifacts_declared(self):
        # Purely additive: the existing pin path must not care that the section is absent.
        assert P.load_artifacts() == {} or all(a.path for a in P.load_artifacts().values())
        assert P.load_pins(), "pins must still load"
