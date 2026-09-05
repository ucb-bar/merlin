"""The wheel bundle ships six curated trees twice, and the gate comparing them must be able to FAIL.

``merlin/{schemas,prompts,benchmarks,contract,targets,runtime}`` are copied into a gitignored
``merlin/python/merlin/_data/`` at build time, and ``merlin.common.paths.data_path`` resolves the same
logical file to the top-level copy in a checkout and to the bundled copy in an installed wheel. A
divergence therefore never shows up where the work happens.

These tests build both sides in a temp root and INJECT each kind of divergence, rather than trusting
the gate's own green line. The direction that matters most is "only in package": the bundle on disk
carried 1192 answer-surface files (``golden.yaml``, ``expected_instruction_coverage.yaml``,
``*.hidden.yaml``) that the build had since been taught to exclude, and the narrower contract-copies
gate reported that state as clean because it exempts ``capsules/`` wholesale.
"""
from __future__ import annotations

import importlib.util
import sys

from merlin.common.paths import repo_root

_SETUP = '''
_BUNDLE = {kind: None for kind in ("schemas", "contract")}
_EXCLUDE = {"contract": ("golden",)}
_EXCLUDE_SUFFIXES = {"contract": (".hidden.yaml",)}
_CODE_SUFFIXES = (".py", ".sh")
'''


def _gate():
    path = repo_root() / "build_tools" / "scripts" / "check_bundled_data.py"
    spec = importlib.util.spec_from_file_location("_check_bundled_data", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _fake_repo(tmp_path, *, source: dict, packaged: dict, setup_src: str = _SETUP,
               bundle: bool = True):
    """A miniature repo: top-level trees, a bundle, and a ``setup.py`` carrying the copy rules.

    ``bundle=False`` leaves out ``_data/`` entirely -- the fresh-clone state, which the gate must
    report as its own thing rather than as a comparison that found nothing.
    """
    (tmp_path / "setup.py").write_text(setup_src)
    if bundle:
        tmp_path.joinpath("merlin", "python", "merlin", "_data").mkdir(parents=True, exist_ok=True)
    for rel, text in source.items():
        p = tmp_path.joinpath("merlin", rel)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    for rel, text in packaged.items():
        p = tmp_path.joinpath("merlin", "python", "merlin", "_data", rel)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    return tmp_path


class TestTheGateCanFail:
    def test_identical_trees_pass(self, tmp_path):
        root = _fake_repo(tmp_path, source={"contract/abi.yaml": "a: 1\n"},
                          packaged={"contract/abi.yaml": "a: 1\n"})
        rep = _gate().audit(root)
        assert rep["status"] == "compared" and rep["n_compared"] == 1
        assert _gate().findings(rep) == []

    def test_a_divergence_is_reported(self, tmp_path):
        # THE FALSIFIER. A gate that cannot produce this line establishes nothing.
        root = _fake_repo(tmp_path, source={"contract/abi.yaml": "a: 1\n"},
                          packaged={"contract/abi.yaml": "a: 2\n"})
        assert _gate().audit(root)["differing"] == ["contract/abi.yaml"]

    def test_it_sees_every_bundled_tree_not_just_one(self, tmp_path):
        root = _fake_repo(tmp_path,
                          source={"contract/abi.yaml": "a: 1\n", "schemas/s.yaml": "b: 1\n"},
                          packaged={"contract/abi.yaml": "a: 1\n", "schemas/s.yaml": "b: 2\n"})
        assert _gate().audit(root)["differing"] == ["schemas/s.yaml"]

    def test_a_file_only_the_reviewer_sees_is_reported(self, tmp_path):
        root = _fake_repo(tmp_path,
                          source={"contract/abi.yaml": "a: 1\n", "contract/new.yaml": "b: 1\n"},
                          packaged={"contract/abi.yaml": "a: 1\n"})
        assert _gate().audit(root)["only_in_source"] == ["contract/new.yaml"]

    def test_a_stale_answer_key_in_the_bundle_is_reported(self, tmp_path):
        # The direction the narrow contract-copies gate is structurally unable to see. The build
        # excludes ``golden*``, so a golden sitting in the bundle is a leftover from an older build --
        # and it is the answer to a graded capsule.
        root = _fake_repo(tmp_path, source={"contract/capsules/x/capsule.yaml": "a: 1\n"},
                          packaged={"contract/capsules/x/capsule.yaml": "a: 1\n",
                                    "contract/capsules/x/golden.yaml": "the answer\n"})
        assert _gate().audit(root)["only_in_packaged"] == ["contract/capsules/x/golden.yaml"]


class TestTheBuildIsTheAuthorityOnWhatMayDiffer:
    def test_an_excluded_prefix_is_not_a_divergence(self, tmp_path):
        # ``golden*`` is excluded by the build, so its absence from the bundle is intended.
        root = _fake_repo(tmp_path,
                          source={"contract/abi.yaml": "a: 1\n", "contract/x/golden.yaml": "s\n"},
                          packaged={"contract/abi.yaml": "a: 1\n"})
        assert _gate().findings(_gate().audit(root)) == []

    def test_an_excluded_suffix_is_not_a_divergence(self, tmp_path):
        root = _fake_repo(tmp_path,
                          source={"contract/abi.yaml": "a: 1\n", "contract/t.hidden.yaml": "s\n"},
                          packaged={"contract/abi.yaml": "a: 1\n"})
        assert _gate().findings(_gate().audit(root)) == []

    def test_code_is_never_bundled_and_that_is_not_a_divergence(self, tmp_path):
        root = _fake_repo(tmp_path,
                          source={"contract/abi.yaml": "a: 1\n", "contract/helper.py": "x = 1\n"},
                          packaged={"contract/abi.yaml": "a: 1\n"})
        assert _gate().findings(_gate().audit(root)) == []

    def test_the_exclusion_is_per_tree_not_global(self, tmp_path):
        # ``golden`` is excluded under contract only; the same name under schemas is real drift.
        root = _fake_repo(tmp_path, source={"schemas/golden.yaml": "a: 1\n"}, packaged={})
        assert _gate().audit(root)["only_in_source"] == ["schemas/golden.yaml"]

    def test_the_rules_come_from_setup_py_not_from_a_second_copy(self, tmp_path):
        # Widen the build's exclusion and the gate must follow it, which is only possible if it reads
        # setup.py instead of keeping its own list.
        setup = _SETUP.replace('{"contract": ("golden",)}', '{"contract": ("golden", "draft")}')
        root = _fake_repo(tmp_path, source={"contract/draft_abi.yaml": "a: 1\n"}, packaged={},
                          setup_src=setup)
        assert _gate().findings(_gate().audit(root)) == []


class TestUnknownNeverReadsAsNo:
    def test_unparseable_build_rules_are_unknown_not_clean(self, tmp_path):
        root = _fake_repo(tmp_path, source={"contract/abi.yaml": "a: 1\n"},
                          packaged={"contract/abi.yaml": "a: 2\n"}, setup_src="def f(:\n")
        rep = _gate().audit(root)
        assert rep["status"] == "unknown" and rep["reason"]

    def test_missing_build_rules_are_unknown_not_clean(self, tmp_path):
        # The build changed shape; the gate no longer knows what it is allowed to ignore.
        root = _fake_repo(tmp_path, source={"contract/abi.yaml": "a: 1\n"},
                          packaged={"contract/abi.yaml": "a: 1\n"},
                          setup_src='_BUNDLE = {k: None for k in ("contract",)}\n')
        rep = _gate().audit(root)
        assert rep["status"] == "unknown" and "UNKNOWN" in rep["reason"]

    def test_an_unbuilt_bundle_is_its_own_state_not_a_pass(self, tmp_path):
        root = _fake_repo(tmp_path, source={"contract/abi.yaml": "a: 1\n"}, packaged={},
                          bundle=False)
        rep = _gate().audit(root)
        assert rep["status"] == "not-built" and rep["reason"]
        assert rep["differing"] == [] and rep["only_in_source"] == []   # nothing compared, no claim

    def test_a_partial_bundle_is_a_divergence_not_an_absence(self, tmp_path):
        # ``_data/`` exists but a whole tree under it is gone: the wheel would ship a partial corpus.
        root = _fake_repo(tmp_path,
                          source={"contract/abi.yaml": "a: 1\n", "schemas/s.yaml": "b: 1\n"},
                          packaged={"contract/abi.yaml": "a: 1\n"})
        assert _gate().audit(root)["only_in_source"] == ["schemas/s.yaml"]


class TestTheAllowlistIsDeclaredNotBlanket:
    def test_every_declared_divergence_carries_a_reason(self):
        declared = _gate()._INTENDED_DIVERGENCE
        assert all(str(v).strip() for v in declared.values()), declared

    def test_a_reasonless_entry_makes_the_gate_refuse(self, tmp_path, monkeypatch):
        # "Allowlisted" must never be reachable by silently adding a key.
        mod = _gate()
        monkeypatch.setitem(mod._INTENDED_DIVERGENCE, "contract/abi.yaml", "")
        root = _fake_repo(tmp_path, source={"contract/abi.yaml": "a: 1\n"},
                          packaged={"contract/abi.yaml": "a: 2\n"})
        assert mod.audit(root)["status"] == "unknown"

    def test_a_declared_entry_is_reported_as_allowed_not_hidden(self, tmp_path, monkeypatch):
        mod = _gate()
        monkeypatch.setitem(mod._INTENDED_DIVERGENCE, "contract/abi.yaml", "deliberately trimmed")
        root = _fake_repo(tmp_path, source={"contract/abi.yaml": "a: 1\n"},
                          packaged={"contract/abi.yaml": "a: 2\n"})
        rep = mod.audit(root)
        assert mod.findings(rep) == []
        assert rep["allowed"]["contract/abi.yaml"] == "deliberately trimmed"


class TestTheRealTreeAgrees:
    def test_the_bundle_matches_the_trees_it_is_built_from(self):
        mod = _gate()
        rep = mod.audit()
        if rep["status"] == "not-built":
            # Fresh clone / CI: the build has not run here. Not a pass, and not a failure either --
            # it is recorded so it cannot be mistaken for a clean comparison.
            assert rep["reason"]
            return
        assert rep["status"] == "compared", rep.get("reason")
        assert rep["n_compared"] > 0, "comparing zero files is not a passing gate"
        assert mod.findings(rep) == [], mod.findings(rep)[:20]


def _git_repo_with_staged(tmp_path, paths):
    """A real repo with ``paths`` staged — the pre-commit scope decision reads git, so the test does
    too rather than faking the answer it wants."""
    import subprocess
    run = lambda *a: subprocess.run(a, cwd=tmp_path, capture_output=True, check=True)
    run("git", "init", "-q")
    run("git", "config", "user.email", "t@example.invalid")
    run("git", "config", "user.name", "t")
    for rel in paths:
        f = tmp_path / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("x\n")
        run("git", "add", rel)
    return tmp_path


class TestPreCommitScopeNarrowsWhetherNotWhat:
    """``--staged`` decides WHETHER the gate runs, never WHAT it compares."""

    def test_a_commit_touching_a_bundled_corpus_is_checked(self, tmp_path):
        root = _git_repo_with_staged(tmp_path, ["merlin/contract/abi.yaml", "README.md"])
        run_it, why = _gate().staged_touches_a_bundled_tree(("contract", "schemas"), root)
        assert run_it and "merlin/contract/abi.yaml" in why

    def test_a_commit_touching_nothing_bundled_is_skipped_with_a_reason(self, tmp_path):
        root = _git_repo_with_staged(tmp_path, ["README.md", "merlin/python/merlin/x.py"])
        run_it, why = _gate().staged_touches_a_bundled_tree(("contract", "schemas"), root)
        assert run_it is False and why

    def test_an_unreadable_staged_list_checks_everything(self, tmp_path):
        # THE FALSIFIER for the scope narrowing: "git said nothing" must never read as "nothing to
        # do". Not a repo at all -> the gate widens, it does not go quiet.
        run_it, why = _gate().staged_touches_a_bundled_tree(("contract",), tmp_path / "not-a-repo")
        assert run_it, why
