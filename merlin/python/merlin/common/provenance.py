"""Which external hardware revision a result belongs to — declared, verified, and recorded.

A result attributed to the wrong hardware is worse than no result. This session produced one: a
microkernel was certified against the only saturn revision containing the outer-product unit, while the
revision named for the tapeout does not contain that unit at all. Both are "saturn-vectors", both had been
checked out in the same tree, and nothing recorded which one the numbers belonged to.

Three operations, deliberately separate:

* :func:`pin` — read the DECLARED revision from ``merlin/contract/hardware_pins.yaml``. Tracked and
  reviewed, so changing what a result is measured against is a diff someone sees.
* :func:`observe` — read what is ACTUALLY in a checkout: commit, branch, remote, dirtiness. No
  interpretation, no comparison; just the facts as found.
* :func:`verify` — compare the two and say precisely how they differ.

**Verification is by content, not by name.** A pin lists ``requires_paths`` whose presence is what the
work actually needs, and may list ``forbids_paths`` whose ABSENCE is the point. That is what catches the
failure above: a checkout can be the right repository, on a plausible branch, and still be missing the
unit. Comparing a SHA would also have caught it, but only if someone had written the SHA down — whereas a
missing file is detectable without knowing what the revision was supposed to be.

**A pin covers only its own checkout, so containment is declared.** Second measured failure: the ISA
headers every int8 dtype claim about the systolic target is derived from live in a git repository NESTED
inside a pinned one. ``verify`` reported the containing pin CLEAN — the three Scala files it listed were
clean — while the nested checkout sat on a different revision than the gitlink its container records, with
one of the headers locally edited on top. Presence, HEAD and dirtiness all said nothing was wrong. So a
pin may declare ``nested_in`` (read the gitlink the container records), ``covers`` (fold a child pin's
verification into this one's, so verifying the coarse pin cannot miss the fine one), and ``content_check``
(compare the read paths' BYTES against the same paths at the pin's commit — see :func:`source_status`,
which is the only comparison that is right when HEAD itself is off the pin).

**Three states, never two.** A file is :data:`PINNED`, :data:`OFF_PIN`, or :data:`UNDETERMINABLE`. The
last is not a softer version of the second: "this is the wrong revision" tells you to reconcile a
checkout, "nobody could tell which revision this is" tells you not to publish the claim at all, and a
check that renders the second as either the first or as OK is how an off-pin header read as pinned.

**Nothing here mutates a checkout.** It verifies and records. On a shared host other people are working in
those trees, and a tool that quietly moves someone's HEAD to satisfy a pin would be a worse failure than
the one it prevents.

**Drift is reported, never averaged away.** :func:`verify` returns every disagreement it finds;
:func:`require` raises on any of them. A caller that wants to proceed anyway has to say so explicitly and
the recorded provenance keeps the drift, so a result produced against an unexpected revision is
identifiable afterwards rather than indistinguishable.
"""
from __future__ import annotations

import hashlib
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["Artifact", "ArtifactCheck", "Observation", "Pin", "PinsError", "SourceStatus",
           "Verification", "load_artifacts", "load_pins", "observe", "pin", "pins_path", "record",
           "require", "source_digest", "source_status", "verify", "verify_artifact"]

#: Recorded where a fact could not be read. Never compares equal to a real value, and callers must not
#: treat it as "unchanged" — see :mod:`merlin.targetgen.artifact_dag` for the same convention.
UNKNOWN = "UNKNOWN"


class PinsError(RuntimeError):
    """A pin is missing, malformed, or the checkout disagrees with it."""


def pins_path() -> Path:
    from .paths import merlin_dir
    return Path(merlin_dir()) / "contract" / "hardware_pins.yaml"


@dataclass(frozen=True)
class Pin:
    """One declared external revision."""

    name: str
    commit: str
    repo_canonical: str = ""
    branch: str | None = None
    root_env: str = ""
    path: str = ""
    requires_paths: tuple[str, ...] = ()
    forbids_paths: tuple[str, ...] = ()
    description: str = ""
    notes: str = ""
    used_by: tuple[str, ...] = ()
    repo_observed_note: str = ""
    #: Files this checkout is KNOWN to carry uncommitted, as ``(repo-relative path, sha256)`` pairs.
    #:
    #: The registry's rule is that a commit must describe the bytes that were read. Sometimes it cannot:
    #: work lives in a fork we do not control, or has not landed upstream yet, and the derivation reads it
    #: anyway. Until now the only way to say so was prose in ``notes`` (see the caveat under
    #: ``firesim_opu_v256d128``), which no check can read -- so the choice was between an unexplained drift
    #: and a comment nobody enforces.
    #:
    #: Declaring the DIGEST makes "<commit> plus exactly these bytes" a nameable revision. It does NOT say
    #: the file matches upstream, and it does not clear the fact that the tree is dirty: the edit is still
    #: reported, now in ``notes`` with its digest, so any report carries it. What it buys is that the
    #: content is pinned -- edit the file again and the digest stops matching, which is drift with a much
    #: more specific message than "uncommitted changes".
    #:
    #: This is a declaration to be REVIEWED, not a way to silence a check. An undeclared edit to a source
    #: a derivation reads is still drift, exactly as before.
    local_edits: tuple[tuple[str, str], ...] = ()
    #: The pin whose checkout CONTAINS this one at a gitlink, and the gitlink path inside it.
    #:
    #: WHY: a pin verifies its OWN checkout and says nothing about a git repository nested inside it. The
    #: measured failure -- the one this pair exists for -- is that the ISA headers every int8 dtype claim
    #: about the systolic target is derived from live in a nested submodule, `verify` reported the
    #: containing pin CLEAN (its three Scala files were clean), and the containing repo reported the whole
    #: nested tree as one dirty entry `software/gemmini-rocc-tests` that intersected no declared read path.
    #: So the headers were off the recorded gitlink AND locally edited, and the verdict was "ok".
    #:
    #: Declaring the containment lets `verify` read the gitlink sha the PARENT records and compare it
    #: against both this pin's declared commit and the nested checkout's actual HEAD -- three revisions
    #: that can all disagree, and did.
    nested_in: str = ""
    nested_path: str = ""
    #: Child pins whose verification is PART of this one's. A caller that verifies only the coarse pin
    #: must not be able to miss a nested surface its claims depend on: the ISA headers were exactly that,
    #: and the only pin anything declared was the generator's.
    covers: tuple[str, ...] = ()
    #: Compare the BYTES of each read path against the same path AT THIS PIN'S COMMIT, read out of the
    #: git object store (never a checkout mutation). None = on iff `nested_in` is set.
    #:
    #: WHY THIS AND NOT THE DIRTY CHECK: `git status` answers "does this file differ from HEAD", which is
    #: the wrong question when HEAD is not the pinned revision. Measured on this host: the instruction
    #: header `include/gemmini.h` is CLEAN -- and carries the off-pin fp6 revision's ISA surface (a
    #: CONFIG_SCALE_MEM funct value and MX-format fields the pinned RTL does not implement), while
    #: `include/gemmini_params.h` is reported MODIFIED and its bytes are byte-identical to the pinned
    #: revision's. The dirty check gets both cases exactly backwards; content gets both right.
    #:
    #: Default-off for pins that are not nested, deliberately. Turning it on everywhere in one change
    #: would re-litigate every existing pin's declared dirty-tree debt at once, and the ratchet
    #: convention says that debt shrinks on purpose, not in a burst.
    content_check: bool | None = None

    @property
    def checks_content(self) -> bool:
        return bool(self.nested_in) if self.content_check is None else self.content_check

    def declared_edit(self, rel: str) -> str | None:
        """The declared digest for ``rel``, or None if this pin declares no edit to it."""
        norm = str(rel).lstrip("./")
        for path, digest in self.local_edits:
            if str(path).lstrip("./") == norm:
                return digest
        return None

    def checkout(self) -> Path | None:
        """Where this pin's sources should be, or None when the root env var is unset."""
        from .paths import env as _env
        root = _env(self.root_env) if self.root_env else None
        if not root:
            return None
        return Path(root) / self.path if self.path else Path(root)


@dataclass(frozen=True)
class Observation:
    """What a checkout actually contains. Facts only."""

    path: str
    commit: str = UNKNOWN
    branch: str = UNKNOWN
    remote: str = UNKNOWN
    dirty_files: int = -1                 # -1 = could not be determined
    present: bool = False
    #: Repo-relative paths git reports as modified or untracked. Recorded, not just counted, because
    #: whether a dirty tree matters depends entirely on WHICH files are dirty: an edited RTL source
    #: changes what a derivation emits, while a stray build log in a benchmarks directory changes nothing.
    dirty_paths: tuple[str, ...] = ()

    @property
    def dirty(self) -> bool | None:
        return None if self.dirty_files < 0 else self.dirty_files > 0

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "commit": self.commit, "branch": self.branch,
                "remote": self.remote, "dirty_files": self.dirty_files, "present": self.present,
                "dirty_paths": list(self.dirty_paths)}


#: The only status that licenses calling a claim derived from a file a PINNED claim.
PINNED = "pinned"
#: The file is readable and its provenance is known — and it is not the pinned revision's.
OFF_PIN = "off_pin"
#: Neither could be established. NEVER merged into ``OFF_PIN``: "this is the wrong revision" and "nobody
#: could tell which revision this is" call for different actions, and a check that reports the second as
#: the first is the same collapse that let a header-derived dtype read as pinned in the first place.
UNDETERMINABLE = "undeterminable"


@dataclass(frozen=True)
class SourceStatus:
    """Whether ONE file a claim depends on belongs to the revision its pin declares.

    Answered by CONTENT: the file's bytes are compared against the same path at the pin's commit, read
    from the git object store. That is the only comparison that is right in both directions of the
    measured failure — a file reported modified whose bytes ARE the pinned revision's, and a file
    reported clean whose bytes are a DIFFERENT revision's because HEAD moved off the pin.

    Three states, never two: :data:`PINNED`, :data:`OFF_PIN`, :data:`UNDETERMINABLE`.
    """

    pin: str
    rel: str
    status: str = UNDETERMINABLE
    digest: str = UNKNOWN                  # sha256 of the bytes on disk
    pinned_digest: str = UNKNOWN           # sha256 of the same path at the pin's commit
    declared_digest: str = ""              # the pin's local_edits entry, when it declares one
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.status == PINNED

    def to_dict(self) -> dict[str, Any]:
        return {"pin": self.pin, "rel": self.rel, "status": self.status, "ok": self.ok,
                "digest": self.digest, "pinned_digest": self.pinned_digest,
                "declared_digest": self.declared_digest, "reason": self.reason}


@dataclass(frozen=True)
class Verification:
    """How a checkout differs from its pin. ``ok`` only when nothing MATERIAL differs.

    ``notes`` holds differences that are real but do not affect what was read — chiefly a dirty tree whose
    dirty files are none of the ones this work consumes. They are recorded rather than dropped, and they do
    not clear ``ok``.

    That distinction is deliberate. A checkout on a shared host is almost never pristine, so reporting
    every stray file as drift makes the check fire on every build and teaches people to ignore it — which
    is worse than not checking, because a REAL drift then also gets ignored. The narrow question a pin
    exists to answer is whether the declared revision describes the bytes that were read, and
    :func:`source_digest` records those bytes regardless.
    """

    pin: str
    observed: Observation
    drift: tuple[str, ...] = ()
    missing_paths: tuple[str, ...] = ()
    forbidden_present: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    #: The gitlink sha the CONTAINING pin's repository records for this nested checkout, when this pin
    #: declares ``nested_in``. Recorded because it is a third revision that can disagree with both the
    #: declared commit and the nested checkout's HEAD — and on this host all three do.
    nested_recorded: str = ""
    #: Per-file content verdicts for the read set, when the pin does content checking. This is what a
    #: caller must consult before calling a file-derived claim a pinned claim.
    sources: tuple[SourceStatus, ...] = ()
    #: Verifications of the pins this one ``covers``. Their material findings are also folded into
    #: ``drift`` (name-prefixed) so ``ok`` and :func:`require`'s message cannot silently omit them.
    covered: tuple["Verification", ...] = ()

    @property
    def ok(self) -> bool:
        return not (self.drift or self.missing_paths or self.forbidden_present)

    def source(self, rel: str) -> SourceStatus | None:
        """The content verdict for one read path, or None when this pin did not check it."""
        norm = str(rel).lstrip("./")
        for s in self.sources:
            if str(s.rel).lstrip("./") == norm:
                return s
        return None

    def to_dict(self) -> dict[str, Any]:
        return {"pin": self.pin, "ok": self.ok, "observed": self.observed.to_dict(),
                "drift": list(self.drift), "missing_paths": list(self.missing_paths),
                "forbidden_present": list(self.forbidden_present), "notes": list(self.notes),
                "nested_recorded": self.nested_recorded,
                "sources": [s.to_dict() for s in self.sources],
                "covered": [v.to_dict() for v in self.covered]}


@dataclass(frozen=True)
class Artifact:
    """A BUILT thing whose identity is its contents, not a revision.

    A pin answers "which checkout was this read from". That question has no answer for an FPGA bitstream, a
    compiled simulator or a packaged image: they have no commit of their own. What identifies them is the
    bytes, plus the revisions they were ELABORATED FROM — so those are separate fields, and `built_from`
    names other pins rather than repeating their shas.

    ``digest`` may be empty while a build is still running. That is honest and it is not the same as
    verifying: :func:`verify_artifact` reports an undeclared digest as a gap, so an artifact can be
    registered before it exists without silently becoming self-certifying.
    """

    name: str
    path: str                              # absolute, or relative to root_env
    description: str = ""
    digest: str = ""                       # sha256 of the file; empty = not yet recorded
    root_env: str = ""
    built_from: tuple[str, ...] = ()       # pin names this was elaborated from
    config: str = ""                       # the elaborated configuration, when there is one
    notes: str = ""

    def resolve(self) -> Path | None:
        from .paths import env as _env
        if not self.root_env:
            return Path(self.path)
        root = _env(self.root_env)
        return None if not root else Path(root) / self.path


@dataclass(frozen=True)
class ArtifactCheck:
    """Whether a built artifact is present and is the one that was declared."""

    artifact: str
    path: str
    present: bool = False
    digest: str = UNKNOWN
    matches: bool | None = None            # None when nothing was declared to match against
    gaps: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.present and self.matches is True

    def to_dict(self) -> dict[str, Any]:
        return {"artifact": self.artifact, "path": self.path, "present": self.present,
                "digest": self.digest, "matches": self.matches, "ok": self.ok,
                "gaps": list(self.gaps)}


def load_artifacts(path: "str | Path | None" = None) -> dict[str, Artifact]:
    """Every declared built artifact. An empty mapping when the registry declares none."""
    import yaml

    p = Path(path) if path is not None else pins_path()
    if not p.is_file():
        raise PinsError(f"no pin registry at {p}")
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    entries = raw.get("artifacts") or {}
    if not isinstance(entries, dict):
        raise PinsError(f"{p}: 'artifacts' must be a mapping of name -> declaration")
    out: dict[str, Artifact] = {}
    for name, body in entries.items():
        if not isinstance(body, dict) or not body.get("path"):
            raise PinsError(f"{p}: artifact {name!r} has no path; an artifact without one identifies "
                            "nothing")
        digest = body.get("digest") or ""
        if digest and not isinstance(digest, str):
            raise PinsError(f"{p}: artifact {name!r} digest must be a quoted string, got "
                            f"{type(digest).__name__}")
        out[str(name)] = Artifact(
            name=str(name), path=str(body["path"]), description=str(body.get("description") or ""),
            digest=str(digest), root_env=str(body.get("root_env") or ""),
            built_from=tuple(str(b) for b in (body.get("built_from") or ())),
            config=str(body.get("config") or ""), notes=str(body.get("notes") or ""))
    return out


def verify_artifact(name: str, *, path: "str | Path | None" = None) -> ArtifactCheck:
    """Compare a built artifact against its declaration.

    Reports rather than raises, and distinguishes the three states that matter: absent, present with a
    digest that disagrees, and present with nothing declared to compare against. The last is a GAP, not a
    pass -- an artifact that certifies itself is the failure this registry exists to prevent.
    """
    arts = load_artifacts(path)
    if name not in arts:
        raise PinsError(f"no artifact named {name!r}; declared: {sorted(arts)}")
    a = arts[name]
    target = a.resolve()
    gaps: list[str] = []
    if target is None:
        return ArtifactCheck(artifact=name, path=UNKNOWN, gaps=(
            f"${a.root_env} is unset, so {name!r} cannot be located",))
    if not target.is_file():
        return ArtifactCheck(artifact=name, path=str(target),
                             gaps=(f"no file at {target}",))
    got = file_digest(target)
    if not a.digest:
        gaps.append("no digest declared, so the file present cannot be confirmed to be the one meant")
        return ArtifactCheck(artifact=name, path=str(target), present=True, digest=got,
                             matches=None, gaps=tuple(gaps))
    if got != a.digest:
        gaps.append(f"digest is {got[:16]} but the registry declares {a.digest[:16]}")
    return ArtifactCheck(artifact=name, path=str(target), present=True, digest=got,
                         matches=(got == a.digest), gaps=tuple(gaps))


def load_pins(path: "str | Path | None" = None) -> dict[str, Pin]:
    """Every declared pin. Raises on a malformed file rather than returning a partial registry."""
    import yaml

    p = Path(path) if path is not None else pins_path()
    if not p.is_file():
        raise PinsError(f"no pin registry at {p}; hardware provenance cannot be verified without one")
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    entries = raw.get("pins") or {}
    if not isinstance(entries, dict):
        raise PinsError(f"{p}: 'pins' must be a mapping of name -> declaration")
    out: dict[str, Pin] = {}
    for name, body in entries.items():
        if not isinstance(body, dict) or body.get("commit") in (None, ""):
            raise PinsError(f"{p}: pin {name!r} has no commit; a pin without a revision pins nothing")
        commit = body["commit"]
        if not isinstance(commit, str):
            # An all-digit sha is valid hex and YAML reads it as a NUMBER, dropping leading zeros -- so a
            # pin would silently verify against a different revision than the one written down.
            raise PinsError(f"{p}: pin {name!r} commit must be quoted; YAML read {commit!r} as "
                            f"{type(commit).__name__}, which loses leading zeros")
        if len(commit) != 40 or any(c not in "0123456789abcdefABCDEF" for c in commit):
            raise PinsError(f"{p}: pin {name!r} commit {commit!r} is not a full 40-character hex sha; an "
                            "abbreviated revision can become ambiguous as history grows")
        out[name] = Pin(
            name=str(name), commit=commit,
            repo_canonical=str(body.get("repo_canonical") or ""),
            branch=(None if body.get("branch") in (None, "") else str(body["branch"])),
            root_env=str(body.get("root_env") or ""), path=str(body.get("path") or ""),
            requires_paths=tuple(body.get("requires_paths") or ()),
            forbids_paths=tuple(body.get("forbids_paths") or ()),
            description=str(body.get("description") or ""), notes=str(body.get("notes") or ""),
            used_by=tuple(body.get("used_by") or ()),
            repo_observed_note=str(body.get("repo_observed_note") or ""),
            local_edits=_local_edits(p, name, body.get("local_edits")),
            nested_in=str(body.get("nested_in") or ""),
            nested_path=str(body.get("nested_path") or ""),
            covers=tuple(str(c) for c in (body.get("covers") or ())),
            content_check=_tri(p, name, "content_check", body.get("content_check")))
    _check_references(p, out)
    return out


def _tri(src: Path, name: str, field_name: str, raw: Any) -> bool | None:
    """A three-valued flag. Absent means "let the pin's shape decide", which is NOT the same as false."""
    if raw is None:
        return None
    if not isinstance(raw, bool):
        raise PinsError(f"{src}: pin {name!r} {field_name} must be a boolean or absent, got "
                        f"{type(raw).__name__}; absent means 'decided by the pin's shape' and spelling "
                        "that as a string would silently pick one of the two")
    return raw


def _check_references(src: Path, pins: "Mapping[str, Pin]") -> None:
    """A pin that names another pin must name one that exists, and containment needs a path.

    Fail closed at LOAD time. A dangling ``nested_in`` would otherwise make the gitlink comparison
    unresolvable at verify time, and an unresolvable comparison is the state that reads as "nothing to
    report" — exactly how the nested headers stayed invisible.
    """
    for name, p in pins.items():
        if p.nested_in:
            if p.nested_in not in pins:
                raise PinsError(f"{src}: pin {name!r} declares nested_in {p.nested_in!r}, which is not a "
                                f"declared pin; declared: {sorted(pins)}")
            if p.nested_in == name:
                raise PinsError(f"{src}: pin {name!r} declares itself as its own container")
            if not p.nested_path:
                raise PinsError(f"{src}: pin {name!r} declares nested_in {p.nested_in!r} but no "
                                "nested_path; without the gitlink path the revision the container "
                                "records cannot be read, and the check would pass by being unable to run")
        elif p.nested_path:
            raise PinsError(f"{src}: pin {name!r} declares nested_path but no nested_in, so there is no "
                            "repository to read that gitlink out of")
        for child in p.covers:
            if child not in pins:
                raise PinsError(f"{src}: pin {name!r} covers {child!r}, which is not a declared pin; "
                                f"declared: {sorted(pins)}")
            if child == name:
                raise PinsError(f"{src}: pin {name!r} covers itself")


def _local_edits(src: Path, name: str, raw: Any) -> tuple[tuple[str, str], ...]:
    """Parse a pin's ``local_edits`` mapping of repo-relative path -> sha256 of the expected content."""
    if raw in (None, {}, ()):
        return ()
    if not isinstance(raw, dict):
        raise PinsError(f"{src}: pin {name!r} local_edits must be a mapping of path -> sha256")
    out = []
    for rel, digest in raw.items():
        if not isinstance(digest, str):
            # Same trap as the commit: an all-digit digest is valid hex and YAML reads it as a number.
            raise PinsError(f"{src}: pin {name!r} local_edits[{rel!r}] must be a quoted sha256 string; "
                            f"YAML read {type(digest).__name__}")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest.lower()):
            raise PinsError(f"{src}: pin {name!r} local_edits[{rel!r}] is not a 64-character sha256 "
                            f"({digest!r}); a partial digest does not identify content")
        out.append((str(rel), digest.lower()))
    return tuple(sorted(out))


def pin(name: str, path: "str | Path | None" = None) -> Pin:
    pins = load_pins(path)
    if name not in pins:
        raise PinsError(f"no pin named {name!r}; declared: {sorted(pins)}")
    return pins[name]


def _git(repo: Path, *args: str) -> str | None:
    try:
        got = subprocess.run(("git", "-C", str(repo)) + args, capture_output=True, text=True,
                             timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    return got.stdout.strip() if got.returncode == 0 else None


def observe(checkout: "str | Path") -> Observation:
    """Read a checkout's actual revision. Absent or non-git paths come back with UNKNOWN, not guesses."""
    p = Path(checkout)
    if not p.is_dir():
        return Observation(path=str(p), present=False)
    commit = _git(p, "rev-parse", "HEAD")
    branch = _git(p, "rev-parse", "--abbrev-ref", "HEAD")
    remote = _git(p, "remote", "get-url", "origin")
    status = _git(p, "status", "--porcelain")
    lines = [l for l in (status or "").splitlines() if l.strip()] if status is not None else []
    return Observation(
        path=str(p), present=True,
        commit=commit or UNKNOWN,
        branch=branch or UNKNOWN,
        remote=remote or UNKNOWN,
        dirty_files=(len(lines) if status is not None else -1),
        dirty_paths=tuple(sorted(_porcelain_paths(lines))))


def _porcelain_paths(lines: "Sequence[str]") -> set[str]:
    """The paths out of ``git status --porcelain`` lines.

    Split on the first whitespace run rather than at a fixed column. Porcelain v1 is ``XY <path>`` and the
    obvious ``line[3:]`` reads the path directly — but :func:`_git` STRIPS its output, so the leading space
    of a ``" M path"`` first line is already gone and the slice silently eats the path's first character
    (observed: ``kept.txt`` recorded as ``ept.txt``, which then matched nothing and reported a modified
    source as clean). Splitting cannot be wrong about where the path starts.

    ``maxsplit=1`` keeps a path containing spaces intact; a rename is spelled ``orig -> new`` and the
    destination is what matters, since that is the path the changed content now occupies.
    """
    out: set[str] = set()
    for line in lines:
        parts = line.split(maxsplit=1)
        if len(parts) < 2:
            continue
        out.add(parts[1].split(" -> ")[-1].strip().strip('"'))
    return out


def _touches(dirty_paths: "Sequence[str]", reads: "Sequence[str]") -> tuple[str, ...]:
    """Which of ``reads`` a dirty checkout actually affects.

    A read path matches a dirty entry when they are equal or when the dirty entry is a DIRECTORY prefix of
    it — git reports an untracked directory as a single entry with a trailing slash, so comparing only for
    equality would miss every file inside a newly-added tree.

    A GITLINK IS ALSO A DIRECTORY PREFIX and git spells it WITHOUT the trailing slash. Measured: the
    systolic generator reported ``software/gemmini-rocc-tests`` (one dirty submodule entry) while the ISA
    headers a claim was derived from sit at ``software/gemmini-rocc-tests/include/gemmini_params.h``. The
    trailing-slash-only rule matched nothing, so ``verify`` concluded "none of them a source this reads"
    and returned ok. Comparing path COMPONENTS — ``<entry>/`` as a prefix, whether or not git wrote the
    slash — is right for both shapes and cannot match a sibling whose name merely starts the same way
    (``foo`` never covers ``foobar/x``).
    """
    hit: set[str] = set()
    for rel in reads:
        norm = str(rel).lstrip("./")
        for d in dirty_paths:
            dn = str(d).lstrip("./")
            if norm == dn or norm.startswith(dn.rstrip("/") + "/"):
                hit.add(norm)
    return tuple(sorted(hit))


def _git_blob(repo: Path, commit: str, rel: str) -> bytes | None:
    """The raw bytes of ``rel`` at ``commit``, or None when they cannot be read.

    Read-only, out of the object store: never a checkout, never a fetch, never an index touch. Other
    sessions work in these trees and a provenance check that moved a HEAD to answer a question would be a
    worse failure than the one it is here to prevent.

    None covers every reason it could not be read — not a repo, commit absent (a shallow or partial
    clone), path absent at that revision — and the caller must render that as UNDETERMINABLE rather than
    as agreement or as drift.
    """
    try:
        got = subprocess.run(("git", "-C", str(repo), "cat-file", "blob", f"{commit}:{rel}"),
                             capture_output=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    return got.stdout if got.returncode == 0 else None


def source_status(pin_name: str, rel: str, *, checkout: "str | Path | None" = None,
                  path: "str | Path | None" = None) -> SourceStatus:
    """Does the file at ``rel`` belong to the revision pin ``pin_name`` declares?

    The question a header-derived claim has to answer before it may call itself pinned. Answered by
    content, in three states, because the two failing ones need different responses: OFF_PIN means
    somebody must reconcile the checkout (or the claim must stop saying "pinned"), UNDETERMINABLE means
    nobody can currently tell and a claim must not be published either way.
    """
    p = pin(pin_name, path)
    target = Path(checkout) if checkout is not None else p.checkout()
    norm = str(rel).lstrip("./")
    if target is None:
        return SourceStatus(pin=pin_name, rel=norm, status=UNDETERMINABLE,
                            reason=f"${p.root_env} is unset, so the file cannot be located")
    full = Path(target) / norm
    if not full.is_file():
        return SourceStatus(pin=pin_name, rel=norm, status=UNDETERMINABLE,
                            reason=f"no file at {full}")
    have = file_digest(full)
    declared = p.declared_edit(norm) or ""
    blob = _git_blob(Path(target), p.commit, norm)
    if blob is None:
        return SourceStatus(pin=pin_name, rel=norm, status=UNDETERMINABLE, digest=have,
                            declared_digest=declared,
                            reason=f"{norm} at {p.commit[:12]} could not be read from {target}'s object "
                                   "store, so whether these bytes are the pinned revision's is UNKNOWN "
                                   "— not 'no' and not 'yes'")
    want = hashlib.sha256(blob).hexdigest()
    if have == want:
        return SourceStatus(pin=pin_name, rel=norm, status=PINNED, digest=have, pinned_digest=want,
                            declared_digest=declared,
                            reason=f"bytes are byte-identical to {norm} at {p.commit[:12]}")
    if declared and have == declared:
        return SourceStatus(pin=pin_name, rel=norm, status=OFF_PIN, digest=have, pinned_digest=want,
                            declared_digest=declared,
                            reason="bytes differ from the pinned revision and match the local edit the "
                                   "pin declares; this is a REVIEWED off-pin file, and a claim citing it "
                                   "must say '<commit> plus these bytes', never 'pinned'")
    return SourceStatus(pin=pin_name, rel=norm, status=OFF_PIN, digest=have, pinned_digest=want,
                        declared_digest=declared,
                        reason=("bytes differ from the pinned revision AND from the local edit the pin "
                                "declares, so the reviewed content is not the content that would be read"
                                if declared else
                                "bytes differ from the pinned revision and the pin declares no local "
                                "edit for this path, so nothing describes what would be read"))


def verify(name: str, *, checkout: "str | Path | None" = None,
           path: "str | Path | None" = None,
           reads: "Sequence[str] | None" = None,
           _seen: "frozenset[str] | None" = None) -> Verification:
    """Compare a checkout against its pin and report every disagreement.

    ``reads`` is the set of repo-relative paths the caller will actually consume; it defaults to the pin's
    ``requires_paths``. A dirty tree is drift only when it touches one of them — an uncommitted edit to a
    source a derivation reads changes what gets emitted while leaving the commit looking correct, which is
    the failure this exists to catch, whereas a stray build artifact elsewhere in the tree changes nothing
    and reporting it as drift only teaches people to ignore the check. A dirty tree that touches nothing
    read is still recorded, in ``notes``.

    THREE THINGS THIS ALSO CHECKS, each added after a measured miss (see :class:`Pin`):

    * a nested checkout's revision against the gitlink its CONTAINER records (``nested_in``),
    * the BYTES of every read path against the same path at the pin's commit (``content_check``), which is
      the only comparison that is right when HEAD itself is off the pin,
    * the pins this one ``covers``, folded in, so verifying the coarse pin cannot miss a nested surface.
    """
    p = pin(name, path)
    target = Path(checkout) if checkout is not None else p.checkout()
    if target is None:
        return Verification(pin=name, observed=Observation(path=UNKNOWN),
                            drift=(f"${p.root_env} is unset, so {name!r} cannot be located",))
    got = observe(target)
    read_set = tuple(reads) if reads is not None else p.requires_paths
    drift: list[str] = []
    notes: list[str] = []
    statuses: tuple[SourceStatus, ...] = ()
    nested_recorded = ""
    if not got.present:
        drift.append(f"no checkout at {got.path}")
    else:
        if got.commit == UNKNOWN:
            drift.append(f"{got.path} is not a readable git checkout, so its revision is UNKNOWN")
        elif got.commit != p.commit:
            drift.append(f"commit is {got.commit[:12]} but the pin declares {p.commit[:12]}")
        if p.branch and got.branch not in (UNKNOWN, p.branch):
            drift.append(f"branch is {got.branch!r} but the pin declares {p.branch!r}")

        # A nested checkout has a THIRD revision beyond its HEAD and its pin: the gitlink its container
        # records. All three disagreed on the host this was written for, and the disagreement was
        # invisible because nothing read the gitlink.
        if p.nested_in:
            nested_recorded, nested_drift, nested_notes = _nested_gitlink(p, got, path)
            drift.extend(nested_drift)
            notes.extend(nested_notes)

        if p.checks_content:
            # CONTENT SUPERSEDES THE DIRTY HEURISTIC for the read set. `git status` answers "differs from
            # HEAD", and when HEAD is off the pin that question has the wrong subject: measured here, it
            # called an off-pin instruction header clean and a byte-identical-to-the-pin parameter header
            # modified. Comparing against the pinned revision's blob is right in both directions. The
            # dirty accounting still runs for dirt OUTSIDE the read set.
            statuses = tuple(source_status(name, rel, checkout=got.path, path=path)
                             for rel in read_set)
            off = [s for s in statuses if s.status == OFF_PIN]
            unknown = [s for s in statuses if s.status == UNDETERMINABLE]
            for s in off:
                drift.append(f"{s.rel} is OFF-PIN: {s.reason} (bytes {s.digest[:16]}, pinned revision "
                             f"has {s.pinned_digest[:16]})")
            for s in unknown:
                drift.append(f"{s.rel} is UNDETERMINABLE against this pin: {s.reason}")
            if statuses and not off and not unknown:
                notes.append(f"{len(statuses)} read path(s) are byte-identical to the pin's commit, so "
                             "claims derived from them are pinned claims regardless of what `git status` "
                             "says about the tree")
            if not read_set:
                drift.append("content_check is on but the pin declares no read set, so there is nothing "
                             "to verify by content and whether a claim would be pinned is UNKNOWN")
            outside = tuple(d for d in got.dirty_paths
                            if not _touches([d], read_set)) if got.dirty else ()
            if outside:
                notes.append(f"{len(outside)} uncommitted change(s) outside the read set; content "
                             "verified per-path above, so these do not bear on the claims")
        elif got.dirty:
            touched = _touches(got.dirty_paths, read_set)
            # A touched source whose CONTENT the pin declares (see Pin.local_edits) is accounted for:
            # the revision alone does not describe it, but "<commit> plus this digest" does, and that is
            # a nameable thing to attribute a result to. Anything else stays drift.
            accounted, unaccounted, stale = [], [], []
            for rel in touched:
                want = p.declared_edit(rel)
                if want is None:
                    unaccounted.append(rel)
                    continue
                full = Path(got.path) / rel
                have = file_digest(full) if full.is_file() else ""
                (accounted if have == want else stale).append(rel)
            if stale:
                drift.append(
                    f"declared local edit(s) {stale} no longer match the digest the pin records, so the "
                    "content that was reviewed is not the content that would be read")
            if unaccounted:
                drift.append(f"{len(unaccounted)} of the source(s) this reads carry uncommitted changes "
                             f"({unaccounted}), so the declared revision does not describe what would "
                             "be read")
            if accounted:
                notes.append(f"{len(accounted)} declared local edit(s) {accounted} match the digest the "
                             "pin records; this is the pinned commit PLUS those bytes, and a result "
                             "citing it must say so")
            if not touched and not read_set:
                # Nothing declared as read, so there is nothing to intersect and no basis for calling the
                # dirt harmless. Fail closed rather than silently downgrading an unknown to a note.
                drift.append(f"{got.dirty_files} uncommitted change(s) and no read set to check them "
                             "against, so whether the declared revision describes what would be read "
                             "is UNKNOWN")
            elif not touched:
                notes.append(f"{got.dirty_files} uncommitted change(s), none of them a source this reads")
        if (p.repo_canonical and got.remote not in (UNKNOWN, "")
                and got.remote != p.repo_canonical and not p.repo_observed_note):
            drift.append(f"origin is {got.remote} but the pin declares {p.repo_canonical}")

    missing = tuple(rel for rel in p.requires_paths
                    if got.present and not (Path(got.path) / rel).exists())
    forbidden = tuple(rel for rel in p.forbids_paths
                      if got.present and (Path(got.path) / rel).exists())

    # Covered pins. Their findings are FOLDED INTO this pin's drift, name-prefixed, rather than parked in
    # a side field: `ok` and `require`'s message are what callers act on, and a nested surface that only
    # showed up in a field nobody reads is the exact shape of the failure this closes.
    covered: list[Verification] = []
    seen = (_seen or frozenset()) | {name}
    for child in p.covers:
        if child in seen:
            notes.append(f"coverage cycle: {child!r} already verified in this chain, not re-entered")
            continue
        try:
            cv = verify(child, path=path, _seen=seen)
        except PinsError as e:
            drift.append(f"[{child}] could not be verified, so whether the surface it covers is pinned "
                         f"is UNKNOWN: {e}")
            continue
        covered.append(cv)
        for item in cv.drift:
            drift.append(f"[{child}] {item}")
        if cv.missing_paths:
            drift.append(f"[{child}] missing required path(s) {list(cv.missing_paths)}")
        if cv.forbidden_present:
            drift.append(f"[{child}] path(s) {list(cv.forbidden_present)} present but declared absent")
        for item in cv.notes:
            notes.append(f"[{child}] {item}")

    return Verification(pin=name, observed=got, drift=tuple(drift), missing_paths=missing,
                        forbidden_present=forbidden, notes=tuple(notes),
                        nested_recorded=nested_recorded, sources=statuses, covered=tuple(covered))


def _nested_gitlink(p: Pin, got: Observation,
                    path: "str | Path | None") -> tuple[str, list[str], list[str]]:
    """The revision the CONTAINER records for this nested checkout, and how it disagrees.

    Measured failure this exists for: the systolic generator (pinned, verified clean) records gitlink
    7c540b3a for ``software/gemmini-rocc-tests`` while that nested checkout's HEAD is e6df8b9f ("128x128
    fp6 sw"), and the ISA headers every int8 dtype claim is derived from are in it. Nothing read the
    gitlink, so the container's own statement about which revision belongs there was never compared to
    anything.
    """
    drift: list[str] = []
    notes: list[str] = []
    try:
        parent = pin(p.nested_in, path)
    except PinsError as e:
        return "", [f"nested_in {p.nested_in!r} is not readable, so the revision the container records "
                    f"for {p.nested_path!r} is UNDETERMINABLE: {e}"], notes
    parent_co = parent.checkout()
    if parent_co is None or not parent_co.is_dir():
        return "", [f"the containing checkout for {p.nested_in!r} is not present, so the gitlink it "
                    f"records for {p.nested_path!r} is UNDETERMINABLE — not absent, and not agreement"], notes
    entry = _git(parent_co, "ls-files", "-s", p.nested_path)
    fields = (entry or "").split()
    recorded = fields[1] if len(fields) > 1 else ""
    if not recorded:
        return "", [f"{p.nested_in!r} records no gitlink at {p.nested_path!r}, so which revision belongs "
                    "in this nested checkout is UNDETERMINABLE"], notes
    if recorded != p.commit:
        drift.append(f"the containing repo {p.nested_in!r} records {recorded[:12]} at {p.nested_path} but "
                     f"this pin declares {p.commit[:12]}: the pin and the superproject disagree about "
                     "which revision this nested checkout is")
    else:
        notes.append(f"{p.nested_in!r} records {recorded[:12]} at {p.nested_path}, agreeing with this "
                     "pin's declared commit")
    if got.commit not in (UNKNOWN, recorded):
        drift.append(f"the nested checkout is at {got.commit[:12]} but {p.nested_in!r} records "
                     f"{recorded[:12]} at {p.nested_path}: this working tree is OFF THE RECORDED GITLINK, "
                     "so anything read from it belongs to a revision the container does not claim")
    return recorded, drift, notes


def require(name: str, *, checkout: "str | Path | None" = None,
            path: "str | Path | None" = None,
            reads: "Sequence[str] | None" = None) -> Verification:
    """:func:`verify`, raising on any disagreement. Use before producing anything that claims a result.

    This is the call that must stand between a file-derived claim and a report that says "pinned": a read
    path that is off-pin or undeterminable raises here, and the three states stay distinguishable in the
    message rather than collapsing into one "does not match".
    """
    got = verify(name, checkout=checkout, path=path, reads=reads)
    if not got.ok:
        parts = list(got.drift)
        for s in got.sources:
            if s.status != PINNED:
                parts.append(f"{s.rel}: {s.status.upper()} — a claim derived from these bytes "
                             f"({s.digest[:16]}) is NOT a pinned claim")
        if got.missing_paths:
            parts.append(f"missing required path(s) {list(got.missing_paths)} — this checkout does not "
                         "contain what the work needs")
        if got.forbidden_present:
            parts.append(f"path(s) {list(got.forbidden_present)} are present but the pin declares them "
                         "absent, so this is not the revision it claims to be")
        raise PinsError(f"pin {name!r} does not match its checkout:\n  - " + "\n  - ".join(parts))
    return got


def source_digest(paths: Sequence["str | Path"]) -> str:
    """One digest over the exact bytes of the sources a derivation read.

    The commit says which revision was checked out; this says what was actually *read*, which differs
    whenever the tree is dirty. Recording both means a result produced from a modified source is
    identifiable instead of looking pinned.
    """
    h = hashlib.sha256()
    for item in sorted(str(p) for p in paths):
        h.update(item.encode("utf-8"))
        try:
            h.update(Path(item).read_bytes())
        except OSError:
            h.update(b"<unreadable>")
    return h.hexdigest()


def file_digest(path: "str | Path") -> str:
    """sha256 of one file, or UNKNOWN. Used to identify a prebuilt simulator binary."""
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return UNKNOWN


def record(*, pins: Mapping[str, Verification] | None = None,
           sources: Sequence["str | Path"] = (),
           artifacts: Mapping[str, "str | Path"] | None = None,
           extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The provenance block to embed in a manifest, run record or report.

    ``artifacts`` names binaries whose identity matters (a prebuilt simulator, a toolchain) and records a
    content digest for each, because a path alone does not say which build ran.
    """
    from .paths import repo_root

    merlin_commit = _git(Path(repo_root()), "rev-parse", "HEAD") or UNKNOWN
    merlin_dirty = _git(Path(repo_root()), "status", "--porcelain")
    out: dict[str, Any] = {
        "merlin": {
            "commit": merlin_commit,
            "dirty_files": (len([l for l in merlin_dirty.splitlines() if l.strip()])
                            if merlin_dirty is not None else -1),
        },
        "hardware_pins": {k: v.to_dict() for k, v in (pins or {}).items()},
        "all_pins_ok": all(v.ok for v in (pins or {}).values()) if pins else None,
    }
    if sources:
        out["source_digest"] = source_digest(sources)
        out["sources"] = sorted(str(p) for p in sources)
    if artifacts:
        out["artifact_digests"] = {k: file_digest(v) for k, v in artifacts.items()}
    if extra:
        out.update(dict(extra))
    return out
