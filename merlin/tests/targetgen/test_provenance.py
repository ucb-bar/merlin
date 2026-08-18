"""Provenance capture must be honest about absence and stable about content.

Two failure modes these tests exist to prevent, both of which make an evolution result
unattributable rather than merely incomplete:

* a missing source silently omitted from the record, so a reader cannot distinguish "not applicable"
  from "we could not tell";
* a content hash that changes with dict construction order, which would make every run look like a
  different one.
"""
from __future__ import annotations

import subprocess

import pytest

from merlin.common.paths import repo_root
from merlin.targetgen import provenance as prov


def _git(root, *args):
    subprocess.run(("git", "-C", str(root), *args), check=True, capture_output=True)


@pytest.fixture
def tiny_repo(tmp_path):
    """A real one-commit git checkout — so the git-reading code is exercised, not simulated."""
    root = tmp_path / "hw"
    root.mkdir()
    _git(root, "init", "-q", "-b", "trunk")
    _git(root, "config", "user.email", "t@example.com")
    _git(root, "config", "user.name", "T")
    (root / "Consts.scala").write_text("object OPMFunct6 { val opmacc = 0 }\n", encoding="utf-8")
    _git(root, "add", "Consts.scala")
    _git(root, "commit", "-q", "-m", "add consts")
    return root


class TestContentSha:
    def test_is_insensitive_to_key_order(self):
        assert prov.content_sha({"a": 1, "b": 2}) == prov.content_sha({"b": 2, "a": 1})

    def test_distinguishes_different_content(self):
        assert prov.content_sha({"a": 1}) != prov.content_sha({"a": 2})

    def test_survives_non_json_values(self):
        # Paths and similar leak into these records constantly; hashing must not raise on them.
        from pathlib import Path
        assert prov.content_sha({"p": Path("/x")})


class TestGitProvenance:
    def test_reads_head_and_branch(self, tiny_repo):
        rec = prov.git_provenance(tiny_repo)
        assert rec["available"] is True
        assert len(rec["head"]) == 40
        assert rec["branch"] == "trunk"
        assert rec["dirty"] is False

    def test_reports_dirty_when_tree_has_changes(self, tiny_repo):
        (tiny_repo / "Consts.scala").write_text("edited\n", encoding="utf-8")
        assert prov.git_provenance(tiny_repo)["dirty"] is True

    def test_untracked_file_counts_as_dirty(self, tiny_repo):
        # A generated-but-uncommitted RTL fact file is exactly this case, and it does affect results.
        (tiny_repo / "scratch.json").write_text("{}", encoding="utf-8")
        assert prov.git_provenance(tiny_repo)["dirty"] is True

    def test_non_repo_directory_is_unavailable_with_a_reason(self, tmp_path):
        rec = prov.git_provenance(tmp_path)
        assert rec["available"] is False and rec["reason"]

    def test_missing_path_is_unavailable_with_a_reason(self, tmp_path):
        rec = prov.git_provenance(tmp_path / "nope")
        assert rec["available"] is False and "not a directory" in rec["reason"]


class TestSimulatorProvenance:
    def test_missing_binary_names_itself_not_built(self, tmp_path):
        rec = prov.simulator_provenance({"rtl": tmp_path / "simulator-absent"})
        assert rec["rtl"]["available"] is False and "not built" in rec["rtl"]["reason"]

    def test_present_binary_records_size_and_a_scoped_sha(self, tmp_path):
        sim = tmp_path / "simulator-x"
        sim.write_bytes(b"\x7fELF" + b"\0" * 64)
        rec = prov.simulator_provenance({"rtl": sim})["rtl"]
        assert rec["available"] is True
        assert rec["bytes"] == 68
        # The sha must not claim more than it covers -- it is name+size+mtime, not the ELF contents.
        assert "not file contents" in rec["sha_covers"]

    def test_sha_changes_when_the_binary_is_rebuilt_bigger(self, tmp_path):
        sim = tmp_path / "simulator-x"
        sim.write_bytes(b"a")
        before = prov.simulator_provenance({"s": sim})["s"]["sha"]
        sim.write_bytes(b"aa")
        assert prov.simulator_provenance({"s": sim})["s"]["sha"] != before


class TestRtlProvenance:
    def test_maps_every_requested_label(self, tiny_repo, tmp_path):
        rec = prov.rtl_provenance({"hw": tiny_repo, "other": tmp_path / "gone"})
        assert set(rec) == {"hw", "other"}
        assert rec["hw"]["available"] is True
        assert rec["other"]["available"] is False


class TestRepoProvenance:
    def test_records_merlins_own_head_and_submodule_pins(self):
        rec = prov.repo_provenance()
        assert rec["available"] is True
        assert len(rec["head"]) == 40
        # This repo vendors baselines as submodules; the pins are what makes an RTL/IREE claim
        # reproducible, so an empty map here means the parser regressed.
        assert rec["submodules"], "expected submodule pins for this checkout"
        for path, sha in rec["submodules"].items():
            assert path and sha and sha[0].isalnum(), (path, sha)

    def test_submodule_pins_are_repo_relative_paths(self):
        pins = prov.repo_provenance()["submodules"]
        root = repo_root()
        assert any((root / p).exists() for p in pins), pins


class TestRecord:
    def test_never_raises_and_always_carries_its_own_hash(self):
        rec = prov.record("no_such_target_xyz")
        assert rec["provenance_sha"]
        assert rec["parent"]["available"] is False, "an unknown target has no champion"
        assert rec["parent"]["reason"]

    def test_unset_env_source_is_recorded_as_unset_rather_than_dropped(self):
        rec = prov.record("no_such_target_xyz",
                          env_rtl_sources=[("hw", "MERLIN_NO_SUCH_ENV_VAR_XYZ", "generators/x")])
        assert "hw" in rec["rtl"], "an unresolvable source must still appear in the record"
        assert rec["rtl"]["hw"]["available"] is False
        assert "MERLIN_NO_SUCH_ENV_VAR_XYZ" in rec["rtl"]["hw"]["reason"]

    def test_hash_covers_the_subrecords(self, tiny_repo, tmp_path):
        a = prov.record("t", rtl_sources={"hw": tiny_repo})
        b = prov.record("t", rtl_sources={"hw": tmp_path / "gone"})
        assert a["provenance_sha"] != b["provenance_sha"]

    def test_toolchain_is_asked_for_its_own_version(self):
        rec = prov.record("no_such_target_xyz")["toolchain"]
        assert "python" in rec and "clang" in rec
        if rec["clang"].get("available"):
            # Whatever it is, it must be the compiler's own words, not ours.
            assert "clang" in rec["clang"]["version"].lower()
