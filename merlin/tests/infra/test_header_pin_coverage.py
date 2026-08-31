"""A pin must cover the files the claims attributed to it are actually derived from.

THE MEASURED FAILURE THESE TESTS ENCODE. ``verify()`` on the systolic generator's pin returned
``ok=True`` while every int8 dtype claim about that target came out of ISA headers the pin did not
mention, in a git repository NESTED inside the pinned one, sitting on a revision the container does not
record. Three independent reasons it stayed invisible, one test each below:

1. The headers were in no ``requires_paths`` anywhere, so nothing looked at them.
2. The container reported the whole nested tree as ONE dirty entry ``software/gemmini-rocc-tests`` — a
   gitlink, which git spells WITHOUT a trailing slash — and the read-set intersection matched only
   equality or a trailing-slash directory prefix. It matched nothing, and ``verify`` concluded
   "8 uncommitted change(s), none of them a source this reads".
3. ``git status`` was the wrong question, because HEAD was not the pinned revision. Measured on the host
   this was written against::

     include/gemmini.h         CLEAN vs HEAD   bytes 007826db == e6df8b9f  -> OFF-PIN, silently
     include/gemmini_params.h  MODIFIED vs HEAD bytes 3758ae96 == 7c540b3a -> PINNED by content

   The dirty check gets both cases exactly backwards.

Most of these tests are hermetic: they build throwaway git repositories in ``tmp_path`` and a throwaway
registry, so they assert the MECHANISM and keep running on a host that has never cloned the hardware.
The few that read the live registry assert only its shape (a pin exists that covers the declared ISA
headers), never a particular checkout's state — that state is someone else's working tree and is expected
to drift.
"""
from __future__ import annotations

import hashlib
import subprocess
import textwrap
from pathlib import Path

import pytest
import yaml

from merlin.common import paths as P
from merlin.common import provenance as PROV

# ---------------------------------------------------------------------------------------------------
# Hermetic fixtures: real git repositories, small enough to reason about.
# ---------------------------------------------------------------------------------------------------

_ENV = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t"}


def _git(repo: Path, *args: str) -> str:
    got = subprocess.run(("git", "-C", str(repo)) + args, capture_output=True, text=True,
                         env={**_ENV, "PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": str(repo)})
    assert got.returncode == 0, f"git {args}: {got.stderr}"
    return got.stdout.strip()


def _init(repo: Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "commit.gpgsign", "false")


def _commit(repo: Path, msg: str) -> str:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", msg)
    return _git(repo, "rev-parse", "HEAD")


def _write(p: Path, body: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(body), encoding="utf-8")


def _digest(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


@pytest.fixture()
def nested_tree(tmp_path, monkeypatch):
    """A container repo with a nested repo inside it, mirroring the measured shape.

    Returns ``(root, container, nested, rev_pinned, rev_moved)`` where ``rev_pinned`` is the revision the
    container records as its gitlink and ``rev_moved`` is where the nested working tree actually sits.
    """
    root = tmp_path / "root"
    nested = root / "generator" / "software" / "tests"
    container = root / "generator"

    _init(nested)
    _write(nested / "include" / "params.h", "typedef int8_t elem_t;\ntypedef int32_t acc_t;\n")
    _write(nested / "include" / "isa.h", "#define CONFIG_EX 0\n")
    rev_pinned = _commit(nested, "int8")
    # A later revision that changes BOTH headers -- the fp6-style rebuild.
    _write(nested / "include" / "params.h", "typedef uint8_t elem_t;\ntypedef uint64_t acc_t;\n")
    _write(nested / "include" / "isa.h", "#define CONFIG_EX 0\n#define CONFIG_SCALE_MEM 26\n")
    rev_moved = _commit(nested, "lowprec float")

    _init(container)
    _write(container / "src" / "Configs.scala", "class C\n")
    # Record the gitlink at rev_pinned, then leave the nested working tree on rev_moved. That is the
    # state the real checkout is in, and the state nothing detected.
    _git(container, "-c", "protocol.file.allow=always", "submodule", "add", "-q",
         str(nested), "software/tests")
    _git(container / "software" / "tests", "checkout", "-q", rev_pinned)
    _commit(container, "generator + gitlink at the int8 revision")
    _git(container / "software" / "tests", "checkout", "-q", rev_moved)

    monkeypatch.setenv("TESTROOT", str(root))
    return root, container, nested, rev_pinned, rev_moved


def _registry(tmp_path, container_rev: str, nested_rev: str, *, local_edits=None,
              covers=True, content_check=None) -> Path:
    """A throwaway pin registry. Pin NAMES are generic here on purpose: library code takes the pin as a
    parameter, so a test of the mechanism must not need a target's name to exercise it."""
    body = {
        "version": 1,
        "pins": {
            "container": {
                "commit": container_rev, "root_env": "TESTROOT", "path": "generator",
                "requires_paths": ["src/Configs.scala"],
                **({"covers": ["headers"]} if covers else {}),
            },
            "headers": {
                "commit": nested_rev, "root_env": "TESTROOT",
                "path": "generator/software/tests",
                "nested_in": "container", "nested_path": "software/tests",
                "requires_paths": ["include/isa.h", "include/params.h"],
                **({"content_check": content_check} if content_check is not None else {}),
                **({"local_edits": local_edits} if local_edits else {}),
            },
        },
    }
    reg = tmp_path / "pins.yaml"
    reg.write_text(yaml.safe_dump(body), encoding="utf-8")
    return reg


# ---------------------------------------------------------------------------------------------------
# 1. A dirty GITLINK covers the files beneath it.
# ---------------------------------------------------------------------------------------------------

def test_dirty_gitlink_covers_paths_beneath_it():
    """The exact silent miss: git spells a dirty submodule WITHOUT a trailing slash.

    Before the fix, ``_touches`` matched only equality or a trailing-slash prefix, so the one dirty entry
    ``software/tests`` intersected none of the header paths under it and the tree was declared to touch
    nothing this reads.
    """
    reads = ["software/tests/include/params.h", "src/Configs.scala"]
    assert PROV._touches(["software/tests"], reads) == ("software/tests/include/params.h",)
    # And a trailing slash (git's spelling for an untracked directory) still works.
    assert PROV._touches(["software/tests/"], reads) == ("software/tests/include/params.h",)


def test_a_sibling_with_a_shared_prefix_is_not_covered():
    """The fix must compare path COMPONENTS: ``foo`` never covers ``foobar/x``. A prefix check on the raw
    string would make every dirty file a false positive for its alphabetical neighbours."""
    assert PROV._touches(["software/test"], ["software/tests/include/params.h"]) == ()


# ---------------------------------------------------------------------------------------------------
# 2. Content, not `git status`, decides whether a file is the pinned revision's.
# ---------------------------------------------------------------------------------------------------

def test_clean_file_on_an_off_pin_head_is_reported_off_pin(nested_tree, tmp_path):
    """THE FALSIFIER THAT MUST FIRE. ``include/isa.h`` is CLEAN — and its bytes are a different
    revision's, because HEAD moved off the gitlink. Nothing keyed on dirtiness can see this."""
    root, container, nested, rev_pinned, rev_moved = nested_tree
    reg = _registry(tmp_path, _git(container, "rev-parse", "HEAD"), rev_pinned)

    assert _git(nested.parent / "tests", "status", "--porcelain", "--", "include/isa.h") == "", \
        "precondition: git must consider this file clean"

    st = PROV.source_status("headers", "include/isa.h", path=reg)
    assert st.status == PROV.OFF_PIN, st
    assert st.digest != st.pinned_digest
    assert "CONFIG_SCALE_MEM" in (nested.parent / "tests" / "include" / "isa.h").read_text()


def test_modified_file_whose_bytes_are_the_pinned_revisions_is_pinned(nested_tree, tmp_path):
    """The mirror case, equally wrong under the dirty check: the file is reported MODIFIED (HEAD is the
    later revision) and its bytes ARE the pinned revision's. Content says pinned; git status says dirty.
    """
    root, container, nested, rev_pinned, rev_moved = nested_tree
    live = container / "software" / "tests"
    pinned_bytes = subprocess.run(("git", "-C", str(live), "cat-file", "blob",
                                   f"{rev_pinned}:include/params.h"), capture_output=True).stdout
    (live / "include" / "params.h").write_bytes(pinned_bytes)
    assert _git(live, "status", "--porcelain", "--", "include/params.h") != "", \
        "precondition: git must consider this file modified"

    reg = _registry(tmp_path, _git(container, "rev-parse", "HEAD"), rev_pinned)
    st = PROV.source_status("headers", "include/params.h", path=reg)
    assert st.status == PROV.PINNED, st
    assert st.digest == st.pinned_digest == hashlib.sha256(pinned_bytes).hexdigest()


def test_an_undeclared_edit_is_refused(nested_tree, tmp_path):
    """THE REQUIRED FALSIFIER. A declared local edit pins CONTENT; edit the file again and it must be
    refused, with a message that names the reviewed digest rather than blending into "tree is dirty"."""
    root, container, nested, rev_pinned, rev_moved = nested_tree
    live = container / "software" / "tests"
    target = live / "include" / "params.h"

    # Declare the CURRENT bytes as a reviewed local edit -> off-pin but accounted for.
    declared = _digest(target)
    reg = _registry(tmp_path, _git(container, "rev-parse", "HEAD"), rev_pinned,
                    local_edits={"include/params.h": declared})
    st = PROV.source_status("headers", "include/params.h", path=reg)
    assert st.status == PROV.OFF_PIN
    assert st.declared_digest == declared
    assert "declares" in st.reason and "REVIEWED" in st.reason

    # Now a DIFFERENT local edit. The declaration no longer describes the bytes, and it must say so.
    target.write_text(target.read_text() + "\n#define SOMETHING_ELSE 1\n", encoding="utf-8")
    st2 = PROV.source_status("headers", "include/params.h", path=reg)
    assert st2.status == PROV.OFF_PIN
    assert st2.digest != declared
    assert "differ" in st2.reason and "local edit" in st2.reason

    # And it must FAIL, not merely be recorded.
    with pytest.raises(PROV.PinsError) as e:
        PROV.require("headers", path=reg)
    assert "include/params.h" in str(e.value)


def test_unreadable_pinned_blob_is_undeterminable_not_offpin(nested_tree, tmp_path):
    """Three states, never two. When the pinned revision's bytes cannot be read at all, the answer is
    UNDETERMINABLE — the state that says "do not publish this claim", distinct from "wrong revision"."""
    root, container, nested, rev_pinned, rev_moved = nested_tree
    absent = "0" * 40                                   # a well-formed sha that is not in this repo
    reg = _registry(tmp_path, _git(container, "rev-parse", "HEAD"), absent)
    st = PROV.source_status("headers", "include/isa.h", path=reg)
    assert st.status == PROV.UNDETERMINABLE, st
    assert st.status != PROV.OFF_PIN
    assert "UNKNOWN" in st.reason


def test_a_missing_file_is_undeterminable(nested_tree, tmp_path):
    root, container, nested, rev_pinned, rev_moved = nested_tree
    reg = _registry(tmp_path, _git(container, "rev-parse", "HEAD"), rev_pinned)
    st = PROV.source_status("headers", "include/nope.h", path=reg)
    assert st.status == PROV.UNDETERMINABLE


# ---------------------------------------------------------------------------------------------------
# 3. verify() / require() must FAIL, and the container must not hide it.
# ---------------------------------------------------------------------------------------------------

def test_verify_reports_the_gitlink_disagreement(nested_tree, tmp_path):
    root, container, nested, rev_pinned, rev_moved = nested_tree
    reg = _registry(tmp_path, _git(container, "rev-parse", "HEAD"), rev_pinned)
    v = PROV.verify("headers", path=reg)
    assert not v.ok
    assert v.nested_recorded == rev_pinned
    assert v.observed.commit == rev_moved
    assert any("OFF THE RECORDED GITLINK" in d for d in v.drift), v.drift


def test_verifying_only_the_container_cannot_miss_the_headers(nested_tree, tmp_path):
    """The whole point of ``covers``. A caller that verified the coarse pin and then reported a
    header-derived dtype as pinned is the failure; this makes that impossible rather than discouraged."""
    root, container, nested, rev_pinned, rev_moved = nested_tree
    reg = _registry(tmp_path, _git(container, "rev-parse", "HEAD"), rev_pinned)

    (tmp_path / "no_cover").mkdir(exist_ok=True)
    without = _registry(tmp_path / "no_cover", _git(container, "rev-parse", "HEAD"), rev_pinned,
                        covers=False)
    assert PROV.verify("container", path=without).ok, \
        "precondition: without `covers` the container verifies clean — that IS the measured bug"

    v = PROV.verify("container", path=reg)
    assert not v.ok
    assert [c.pin for c in v.covered] == ["headers"]
    assert any(d.startswith("[headers]") for d in v.drift), v.drift
    with pytest.raises(PROV.PinsError, match="headers"):
        PROV.require("container", path=reg)


def test_a_pinned_read_set_verifies_clean(nested_tree, tmp_path):
    """The check must be able to PASS, or it says nothing. Put the nested tree back on the gitlink and
    both headers become pinned claims."""
    root, container, nested, rev_pinned, rev_moved = nested_tree
    _git(container / "software" / "tests", "checkout", "-q", rev_pinned)
    reg = _registry(tmp_path, _git(container, "rev-parse", "HEAD"), rev_pinned)
    v = PROV.verify("headers", path=reg)
    assert v.ok, v.drift
    assert {s.rel: s.status for s in v.sources} == {
        "include/isa.h": PROV.PINNED, "include/params.h": PROV.PINNED}
    assert PROV.verify("container", path=reg).ok


def test_content_check_with_an_empty_read_set_fails_closed(nested_tree, tmp_path):
    """A check that cannot run must not report success. ``content_check`` on with nothing declared to
    check is UNKNOWN, and UNKNOWN is not clean."""
    root, container, nested, rev_pinned, rev_moved = nested_tree
    reg = _registry(tmp_path, _git(container, "rev-parse", "HEAD"), rev_pinned)
    doc = yaml.safe_load(reg.read_text(encoding="utf-8"))
    doc["pins"]["headers"]["requires_paths"] = []
    reg.write_text(yaml.safe_dump(doc), encoding="utf-8")
    v = PROV.verify("headers", path=reg)
    assert not v.ok
    assert any("nothing" in d and "UNKNOWN" in d for d in v.drift), v.drift


# ---------------------------------------------------------------------------------------------------
# 4. The registry itself: malformed containment must not load.
# ---------------------------------------------------------------------------------------------------

def test_nested_in_naming_an_undeclared_pin_is_refused(tmp_path):
    reg = tmp_path / "p.yaml"
    reg.write_text(yaml.safe_dump({"version": 1, "pins": {"a": {
        "commit": "0" * 40, "nested_in": "ghost", "nested_path": "x"}}}), encoding="utf-8")
    with pytest.raises(PROV.PinsError, match="not a declared pin"):
        PROV.load_pins(reg)


def test_nested_in_without_a_path_is_refused(tmp_path):
    """Without the gitlink path the comparison cannot run, and a comparison that cannot run reads as
    "nothing to report" — which is exactly how the nested headers stayed invisible."""
    reg = tmp_path / "p.yaml"
    reg.write_text(yaml.safe_dump({"version": 1, "pins": {
        "a": {"commit": "0" * 40},
        "b": {"commit": "1" * 40, "nested_in": "a"}}}), encoding="utf-8")
    with pytest.raises(PROV.PinsError, match="nested_path"):
        PROV.load_pins(reg)


def test_covers_naming_an_undeclared_pin_is_refused(tmp_path):
    reg = tmp_path / "p.yaml"
    reg.write_text(yaml.safe_dump({"version": 1, "pins": {
        "a": {"commit": "0" * 40, "covers": ["ghost"]}}}), encoding="utf-8")
    with pytest.raises(PROV.PinsError, match="covers"):
        PROV.load_pins(reg)


def test_a_coverage_cycle_terminates(tmp_path):
    reg = tmp_path / "p.yaml"
    reg.write_text(yaml.safe_dump({"version": 1, "pins": {
        "a": {"commit": "0" * 40, "covers": ["b"], "root_env": "NOPE_UNSET_ENV"},
        "b": {"commit": "1" * 40, "covers": ["a"], "root_env": "NOPE_UNSET_ENV"}}}), encoding="utf-8")
    v = PROV.verify("a", path=reg)          # must return, not recurse forever
    assert v.pin == "a"


def test_content_check_must_be_a_boolean_or_absent(tmp_path):
    """Absent means "decided by the pin's shape", which is a THIRD value; spelling it as a string would
    silently pick one of the two."""
    reg = tmp_path / "p.yaml"
    reg.write_text(yaml.safe_dump({"version": 1, "pins": {
        "a": {"commit": "0" * 40, "content_check": "yes"}}}), encoding="utf-8")
    with pytest.raises(PROV.PinsError, match="content_check"):
        PROV.load_pins(reg)


# ---------------------------------------------------------------------------------------------------
# 5. The LIVE registry: every declared ISA header must be covered by some pin's read set.
#    Shape only — never a particular checkout's state, which is someone else's working tree.
# ---------------------------------------------------------------------------------------------------

def _external_isa_sources() -> list[tuple[str, str]]:
    """``[(target, resolved absolute path)]`` for every declared ISA source that resolves OUTSIDE this
    repository.

    The outside-ness is the whole criterion. An in-repo ISA contract (atlas ships its green card and ISA
    definition as tracked files) is versioned by merlin's own commit, which ``record()`` already captures,
    so it needs no hardware pin. A source that resolves into an external checkout does not: merlin's
    commit says nothing about which revision those bytes are, and that is precisely the gap the systolic
    headers fell through — they are reached through a tracked SYMLINK, so they LOOK in-repo at the
    declared path and resolve into a nested submodule.

    Targets are discovered from the descriptors on disk; nothing here names one.
    """
    from merlin.targetgen import capability_discovery as CD
    root = Path(P.repo_root()).resolve()
    found: list[tuple[str, str]] = []
    exp = Path(P.merlin_dir()) / "experiments"
    for desc in sorted(exp.glob("*/targets/*/target_experiment.yaml")):
        doc = yaml.safe_load(desc.read_text(encoding="utf-8")) or {}
        target = str(doc.get("target") or desc.parent.name)
        if not ((doc.get("hardware_spec") or {}).get("isa_headers") or []):
            continue
        for src in CD.isa_sources(target):
            if not src.path:
                continue
            rp = Path(src.path).resolve()
            if root not in rp.parents:
                found.append((target, str(rp)))
    return found


def test_the_live_registry_loads_and_its_containment_resolves():
    pins = PROV.load_pins()
    assert pins, "the registry must declare pins"
    for name, p in pins.items():
        if p.nested_in:
            assert p.nested_in in pins and p.nested_path, name
            assert p.checks_content, \
                f"{name} is nested; content checking is what catches an off-gitlink header"
        for child in p.covers:
            assert child in pins, (name, child)


def test_every_external_isa_source_is_in_some_pins_read_set():
    """An ISA source that resolves outside this repo must appear in a pin's ``requires_paths``.

    This is the first of the three reasons the failure was invisible: the headers were in no pin at all,
    so nothing looked at them. Matched by resolved ABSOLUTE path against each pin's checkout plus its
    declared read paths — exact, so a basename that merely coincides with an unrelated pin's file cannot
    be mistaken for coverage.

    Skipped, not passed, when no external checkout is present: this host simply has no hardware sources
    to attribute, and a check that could not run must not report success.
    """
    pins = PROV.load_pins()
    covered: set[str] = set()
    for p in pins.values():
        co = p.checkout()
        if co is None:
            continue
        for rel in p.requires_paths:
            cand = Path(co) / str(rel).lstrip("./")
            try:
                covered.add(str(cand.resolve()))
            except OSError:                              # noqa: PERF203 — one bad path loses nothing
                continue

    external = _external_isa_sources()
    if not external:
        pytest.skip("no declared ISA source resolves outside this repo on this host; nothing to attribute")

    missing = sorted(f"{t}: {path}" for t, path in external if path not in covered)
    # Reported as one list rather than one assert per target: which sources lack coverage is the
    # actionable fact, and failing on the first hides the rest.
    assert not missing, ("ISA source(s) a dtype claim is derived from that resolve OUTSIDE this repo and "
                         "that NO pin's read set mentions, so nothing verifies which hardware revision "
                         "they belong to:\n  " + "\n  ".join(missing))


def test_an_absent_checkout_yields_no_per_file_verdicts(tmp_path, monkeypatch):
    """A host that simply does not have the hardware sources must get NO per-file verdicts at all.

    This is what keeps the gate's UNDETERMINABLE-is-a-failure rule from firing for everyone who has not
    cloned every external repo: with nothing to read, ``sources`` is empty, so there is no file whose
    revision could be called undeterminable. The absence itself is reported as drift, which is the honest
    answer, and the gate treats it as a note.
    """
    monkeypatch.delenv("TESTROOT", raising=False)
    reg = _registry(tmp_path, "0" * 40, "1" * 40)
    v = PROV.verify("headers", path=reg)
    assert v.sources == ()
    assert not v.observed.present
    assert not v.ok
