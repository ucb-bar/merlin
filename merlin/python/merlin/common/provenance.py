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

__all__ = ["Artifact", "ArtifactCheck", "Observation", "Pin", "PinsError", "Verification",
           "load_artifacts", "load_pins", "observe", "pin", "pins_path", "record", "require",
           "source_digest", "verify", "verify_artifact"]

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

    @property
    def ok(self) -> bool:
        return not (self.drift or self.missing_paths or self.forbidden_present)

    def to_dict(self) -> dict[str, Any]:
        return {"pin": self.pin, "ok": self.ok, "observed": self.observed.to_dict(),
                "drift": list(self.drift), "missing_paths": list(self.missing_paths),
                "forbidden_present": list(self.forbidden_present), "notes": list(self.notes)}


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
            repo_observed_note=str(body.get("repo_observed_note") or ""))
    return out


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
    """
    hit: set[str] = set()
    for rel in reads:
        norm = str(rel).lstrip("./")
        for d in dirty_paths:
            dn = str(d).lstrip("./")
            if norm == dn or (dn.endswith("/") and norm.startswith(dn)):
                hit.add(norm)
    return tuple(sorted(hit))


def verify(name: str, *, checkout: "str | Path | None" = None,
           path: "str | Path | None" = None,
           reads: "Sequence[str] | None" = None) -> Verification:
    """Compare a checkout against its pin and report every disagreement.

    ``reads`` is the set of repo-relative paths the caller will actually consume; it defaults to the pin's
    ``requires_paths``. A dirty tree is drift only when it touches one of them — an uncommitted edit to a
    source a derivation reads changes what gets emitted while leaving the commit looking correct, which is
    the failure this exists to catch, whereas a stray build artifact elsewhere in the tree changes nothing
    and reporting it as drift only teaches people to ignore the check. A dirty tree that touches nothing
    read is still recorded, in ``notes``.
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
    if not got.present:
        drift.append(f"no checkout at {got.path}")
    else:
        if got.commit == UNKNOWN:
            drift.append(f"{got.path} is not a readable git checkout, so its revision is UNKNOWN")
        elif got.commit != p.commit:
            drift.append(f"commit is {got.commit[:12]} but the pin declares {p.commit[:12]}")
        if p.branch and got.branch not in (UNKNOWN, p.branch):
            drift.append(f"branch is {got.branch!r} but the pin declares {p.branch!r}")
        if got.dirty:
            touched = _touches(got.dirty_paths, read_set)
            if touched:
                drift.append(f"{len(touched)} of the source(s) this reads carry uncommitted changes "
                             f"({list(touched)}), so the declared revision does not describe what would "
                             "be read")
            elif not read_set:
                # Nothing declared as read, so there is nothing to intersect and no basis for calling the
                # dirt harmless. Fail closed rather than silently downgrading an unknown to a note.
                drift.append(f"{got.dirty_files} uncommitted change(s) and no read set to check them "
                             "against, so whether the declared revision describes what would be read "
                             "is UNKNOWN")
            else:
                notes.append(f"{got.dirty_files} uncommitted change(s), none of them a source this reads")
        if (p.repo_canonical and got.remote not in (UNKNOWN, "")
                and got.remote != p.repo_canonical and not p.repo_observed_note):
            drift.append(f"origin is {got.remote} but the pin declares {p.repo_canonical}")

    missing = tuple(rel for rel in p.requires_paths
                    if got.present and not (Path(got.path) / rel).exists())
    forbidden = tuple(rel for rel in p.forbids_paths
                      if got.present and (Path(got.path) / rel).exists())
    return Verification(pin=name, observed=got, drift=tuple(drift), missing_paths=missing,
                        forbidden_present=forbidden, notes=tuple(notes))


def require(name: str, *, checkout: "str | Path | None" = None,
            path: "str | Path | None" = None,
            reads: "Sequence[str] | None" = None) -> Verification:
    """:func:`verify`, raising on any disagreement. Use before producing anything that claims a result."""
    got = verify(name, checkout=checkout, path=path, reads=reads)
    if not got.ok:
        parts = list(got.drift)
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
