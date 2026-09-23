"""A CLEAN ROOM: a workspace holding only what an arm may see, built so the answer key is ABSENT.

WHY A SECOND MECHANISM. :mod:`merlin.targetgen.sandbox.bwrap` hides the answer key by MOUNT POLICY —
the key stays on disk, at its real path, and a kernel-enforced mount table denies the read. That is the
stronger guarantee and it should be used whenever it can be. It cannot always be used: on a host whose
policy forbids unprivileged user namespaces the sandbox binary cannot construct a namespace at all (see
:mod:`merlin.targetgen.sandbox.preflight`), and then the mount table is never built. Today the answer
key on such a host is unreachable only because the agent CLI's own sandbox is broken by the same policy
— an accident, not a guard, and one that disappears silently the moment the host policy is fixed.

So this module takes the other route: instead of making the answer key unreadable, it makes a workspace
in which the answer key IS NOT PRESENT. Nothing is denied because nothing denied was ever placed. The
allow set is the arm's own declared grants; the deny set is
:func:`merlin.targetgen.sandbox.answer_surfaces.answer_surfaces`, derived per target from that target's
descriptor and the declared oracle/grader registry — the same set the mount policy masks, so the two
mechanisms cannot disagree about what an answer is.

WHAT IT GUARANTEES, AND WHAT IT DOES NOT. It guarantees a property of the TREE: no byte of a derived
answer surface is reachable by walking down from the clean room, and no path in it walks back UP into
the checkout. It does NOT guarantee a property of the PROCESS: an agent that opens an absolute path
elsewhere on this filesystem is not stopped by a tidy workspace, and cannot be without kernel isolation
or a separate uid. That residual is why :mod:`merlin.targetgen.sandbox.read_audit` exists, and why
neither component is described here as sufficient on its own.

THREE WAYS A "CLEAN" TREE LEAKS, all of which this refuses:

* **A symlink that can be walked upward.** A link to an allowed FILE is harmless; a link to a
  DIRECTORY inside the checkout turns the clean room into a door onto the whole repo, and so does a
  link to ``..``. Materialisation therefore dereferences (via
  :mod:`merlin.common.content_store`, which also refuses cycles and non-regular files) and
  verification refuses any surviving link whose real path leaves the room.
* **A shared inode the agent can write.** Large read-only inputs are hard-linked rather than copied,
  which is what makes a clean room affordable — but a hard link into the CHECKOUT would let the agent
  edit the repo through it, and a ``chmod`` on it would reach back into the original. Links are
  therefore made from the content store's own copy of the bytes, never from the source file, and every
  input is verified unwritable.
* **The room sitting inside the checkout.** An agent CLI roots itself at the enclosing repository by
  walking up for a ``.git``; a room placed under the checkout hands it the checkout. Refused.

FAIL CLOSED. A grant that does not resolve, a bundle that is not well-formed, an answer-surface set
that cannot be derived, or a verification that cannot be completed all mean the room is NOT BUILT. A
partially built room is removed rather than handed over.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from merlin.common import content_store
from merlin.common.paths import repo_root
from merlin.targetgen.sandbox.answer_surfaces import AnswerSurface, answer_surfaces
from merlin.targetgen.sandbox.bwrap import _host_input_paths, path_kind, resolve_grant
from merlin.targetgen.target_experiment import TargetExperiment

#: Where the materialised allow set lands inside the room (read-only to the agent).
INPUTS_DIR = "inputs"
#: The agent's own writable area inside the room.
WORK_DIR = "work"
#: The host-only record of how the room was built. A SIBLING of the room, never inside it: it names
#: host source paths, and the room is exactly the place those names should not be readable from.
MANIFEST_NAME = "cleanroom.manifest.json"

#: Above this many answer-surface files the content index is declared incomplete rather than silently
#: truncated. An incomplete index makes the verdict incomplete, which is not a pass.
_MAX_INDEXED_SURFACE_FILES = 400_000

_DECISION_ALLOW = "allow"
_DECISION_DENY = "deny"
_DECISION_UNDECLARED = "undeclared"


class CleanRoomRefused(RuntimeError):
    """The clean room was not built, or was built and then refused. Never a warning."""


# --------------------------------------------------------------------------- the precedence rule
@dataclass(frozen=True)
class WithheldSet:
    """What is withheld, and the only declared ways in.

    ``surfaces`` are the derived answer surfaces (plus the bundle's own ``denied`` entries).
    ``exemptions`` are absolute sub-paths that a surface ITSELF declares grantable — today only a target
    package's contract sub-tree, because the package is denied as a blanket precisely so that an arm can
    still read the facts it is required to derive from.
    """

    surfaces: tuple[Path, ...] = ()
    exemptions: tuple[Path, ...] = ()

    def __len__(self) -> int:
        return len(self.surfaces)


def _deepest(path: Path, among: frozenset[Path]) -> Path | None:
    """The deepest ancestor-or-self of ``path`` in ``among``. Ancestors are visited deepest-first, so
    the first hit IS the longest prefix."""
    for ancestor in (path, *path.parents):
        if ancestor in among:
            return ancestor
    return None


def is_withheld(path: Path, withheld: WithheldSet) -> bool:
    """THE DEFINITION of withheld, and the only one. Deny wins over any grant, at any depth, unless a
    DECLARED exemption sits deeper than the surface that would otherwise cover this path.

    WHY NOT LONGEST-PREFIX-WINS. The mount policy resolves a grant against a mask by longest
    destination, so a deeper grant beats a broader deny there. That rule is fine when a kernel is
    enforcing the result and the mount table is replayed and proved empty of answer surfaces before the
    run. It is the wrong rule for THIS mechanism, whose entire job is containment on a host where
    nothing is enforcing anything: under it, the decision about what escapes a deny belongs to whoever
    writes the bundle, and an answer key gets re-admitted the first time somebody grants a subdirectory
    for a perfectly good reason. Nothing would record that it had happened.

    So the exemption is a property of the SURFACE, derived with it, and a bundle can only reach inside a
    surface where the surface says a way in exists. Granting a target package's ``backend/`` does
    nothing; granting its declared contract sub-tree works, because that sub-tree is declared grantable
    at derivation time. Widening containment therefore takes a reviewed edit to
    :data:`merlin.targetgen.sandbox.answer_surfaces.GRANTABLE_SUBPATHS_BY_ORIGIN`, not a manifest line.

    The consequence is that a clean room is STRICTER than the mount policy for every surface without a
    declared exemption. That asymmetry is deliberate and it is in the safe direction: the clean room is
    what is used when there is no kernel to fall back on.
    """
    path = path.absolute()
    surface = _deepest(path, frozenset(withheld.surfaces))
    if surface is None:
        return False
    exemption = _deepest(path, frozenset(withheld.exemptions))
    if exemption is not None and len(exemption.parts) > len(surface.parts):
        return False
    return True


def _crosses_withheld(outer: Path, inner: Path, withheld: WithheldSet) -> bool:
    """Whether walking from ``outer`` down to ``inner`` passes through a withheld directory."""
    outer, inner = outer.absolute(), inner.absolute()
    for ancestor in inner.parents:
        if ancestor == outer:
            break
        if is_withheld(ancestor, withheld):
            return True
    return False


class _Precedence:
    """The CONSTRUCTION-side decision: place this path, or excise it.

    It is a thin wrapper over :func:`is_withheld` plus the bundle's allow set — deliberately thin,
    because verification must not re-use it. Verification calls :func:`is_withheld` directly, so a
    mutation to this class (or to the builder that drives it) cannot also silence the check that is
    supposed to catch the mutation.
    """

    def __init__(self, allow: Iterable[Path], withheld: WithheldSet) -> None:
        self._allow = frozenset(p.absolute() for p in allow)
        self._withheld = withheld

    def decide(self, path: Path) -> str:
        path = path.absolute()
        if is_withheld(path, self._withheld):
            return _DECISION_DENY
        allow = _deepest(path, self._allow)
        if allow is None:
            return _DECISION_UNDECLARED
        # A declared exemption is a way IN, not an invitation. Reaching inside one takes a grant at or
        # below the exemption itself: a broad grant that merely happens to contain it does not pick it
        # up. Otherwise the bundle would be re-admitting the sub-tree by accident, which is the same
        # accident as depth-wins precedence wearing a different hat — and an explicit grant is the one
        # that shows up in the manifest when somebody later asks what this arm was allowed to read.
        exemption = _deepest(path, frozenset(self._withheld.exemptions))
        if exemption is not None and len(allow.parts) < len(exemption.parts):
            return _DECISION_UNDECLARED
        return _DECISION_ALLOW

    def blocks_descent(self, directory: Path) -> bool:
        """Whether a withheld DIRECTORY hides everything beneath it, or merely most of it.

        A surface that declares a grantable sub-tree still has to be walked into to reach it, so the
        excision is per-entry there rather than wholesale.
        """
        return not any(directory.absolute() in e.parents for e in self._withheld.exemptions)


# --------------------------------------------------------------------------- derivation, fail-closed
def _grant_paths(bundle: Mapping[str, object], repo: Path, key: str) -> list[tuple[str, Path]]:
    entries = bundle.get(key) or []
    if not isinstance(entries, list):
        raise CleanRoomRefused(f"bundle {key!r} is not a list; refusing to derive an allow set from it")
    out: list[tuple[str, Path]] = []
    for entry in entries:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("path"), str) or not entry["path"]:
            raise CleanRoomRefused(f"bundle {key!r} contains an entry without a path: {entry!r}")
        rel = str(entry["path"])
        out.append((rel, resolve_grant(rel, repo)))
    return out


def derive_allow_set(bundle: Mapping[str, object], *, repo: Path | None = None) -> list[tuple[str, Path]]:
    """The arm's declared inputs, resolved to host paths. Raises if any does not resolve.

    A missing grant is refused rather than skipped for the same reason the frozen-snapshot path refuses
    it: silently omitting a declared input makes the executed treatment smaller than the bundle claims,
    and the run then reports a capability failure that is really a setup failure.
    """
    repo = (repo or repo_root()).absolute()
    if not isinstance(bundle, Mapping):
        raise CleanRoomRefused("bundle is not a mapping; the allow set cannot be derived")
    grants = _grant_paths(bundle, repo, "allowed")
    missing = sorted(rel for rel, source in grants if path_kind(source) == "missing")
    if missing:
        raise CleanRoomRefused("bundle declares unresolvable allowed grant(s): " + ", ".join(missing))
    return grants


def derive_deny_set(
    te: TargetExperiment, bundle: Mapping[str, object] | None = None, *, repo: Path | None = None
) -> WithheldSet:
    """Everything that must not appear in the room, with the declared ways in.

    The surfaces are the derived answer surfaces plus the bundle's own ``denied`` entries; the
    exemptions are the sub-paths each surface DECLARES grantable, made absolute. Raises if the answer
    surfaces cannot be derived — an underivable deny set means no room, never an empty one.
    """
    repo = (repo or repo_root()).absolute()
    try:
        surfaces: list[AnswerSurface] = answer_surfaces(te)
    except Exception as exc:  # noqa: BLE001 — underivable deny set must refuse, not default to empty
        raise CleanRoomRefused(f"answer surfaces could not be derived for {te.target!r}: {exc}") from exc
    denied = [s.path.absolute() for s in surfaces]
    exemptions = [(s.path / sub).absolute() for s in surfaces for sub in getattr(s, "grantable", ()) if sub]
    if bundle:
        for _rel, source in _grant_paths(bundle, repo, "denied"):
            if path_kind(source) != "missing":
                denied.append(source.absolute())
        try:
            host_paths = [(repo / path).absolute() for path in _host_input_paths(bundle)]
        except RuntimeError as exc:
            raise CleanRoomRefused(str(exc)) from exc
        if any(path_kind(path) == "missing" for path in host_paths):
            raise CleanRoomRefused("host-only input is absent; private content cannot be verified")
        denied.extend(host_paths)
        # Private inputs are never grantable, even beneath an otherwise public
        # contract subtree. The verifier derives these same denies independently.
        exemptions = [
            path for path in exemptions if not any(path == host or host in path.parents for host in host_paths)
        ]
    return WithheldSet(surfaces=tuple(sorted(set(denied))), exemptions=tuple(sorted(set(exemptions))))


def withheld_files(withheld: WithheldSet) -> tuple[list[Path], bool, str]:
    """Every withheld FILE, expanded from the surfaces alone.

    No bundle, no allow set, no construction-side decision object takes part in this: it is what
    verification compares the room against, and it has to be derivable from the descriptor by itself or
    verification is merely repeating what the builder believed.

    Returns ``(files, complete, reason)``. A surface that cannot be walked or stat'd makes the answer
    INCOMPLETE rather than shorter — a surface we cannot rule out of the room is not one we have ruled
    out of the room.
    """
    files: list[Path] = []
    seen: set[Path] = set()
    for surface in withheld.surfaces:
        try:
            if surface.is_file():
                candidates: Iterable[Path] = (surface,)
            elif surface.is_dir():
                candidates = (p for p in surface.rglob("*") if p.is_file() and not p.is_symlink())
            else:
                continue
            for candidate in candidates:
                if not is_withheld(candidate, withheld):
                    continue  # inside a declared grantable sub-tree: withheld from nobody
                if candidate in seen:
                    continue
                seen.add(candidate)
                files.append(candidate)
                if len(files) > _MAX_INDEXED_SURFACE_FILES:
                    return files, False, f"withheld file count exceeded {_MAX_INDEXED_SURFACE_FILES}"
        except OSError as exc:
            return files, False, f"answer surface unreadable: {surface} ({exc})"
    return files, True, ""


# --------------------------------------------------------------------------- materialisation
def _room_path(inputs: Path, source: Path, repo: Path) -> Path:
    """Where an allowed source lands inside the room. Repo-relative sources keep their repo-relative
    spelling (so a task card's path references still read naturally); anything else lands under a
    private ``external/`` namespace with its anchor dropped."""
    source = source.absolute()
    try:
        return inputs / "repo" / source.relative_to(repo)
    except ValueError:
        return inputs / "external" / Path(*source.parts[1:])


@dataclass
class _Placement:
    placed: list[dict[str, object]] = field(default_factory=list)
    excised: list[dict[str, str]] = field(default_factory=list)
    n_files: int = 0
    n_bytes: int = 0


def _place(source: Path, dest: Path, rule: _Precedence, store: Path | None, out: _Placement) -> None:
    """Copy or link one allowed subtree into the room, excising anything the rule denies.

    A denied path is NOT placed and then deleted — it is never read. That distinction is the whole
    point: a deleted file has still been on the destination filesystem, and a build that crashes
    between the copy and the delete leaves the answer key in the room.
    """
    decision = rule.decide(source)
    if decision != _DECISION_ALLOW:
        # FAIL CLOSED: only an explicit ALLOW is placed. "Denied" and "not covered by any grant that
        # reaches here" are both reasons not to copy, and they are recorded distinctly so the manifest
        # says which one applied.
        if source.is_dir() and not rule.blocks_descent(source):
            # A withheld directory that declares a grantable sub-tree: walk in, and excise per entry.
            dest.mkdir(parents=True, exist_ok=True)
            for entry in sorted(source.iterdir()):
                _place(entry, dest / entry.name, rule, store, out)
            return
        out.excised.append({"source": str(source), "reason": decision})
        return
    if source.is_symlink():
        # Dereference: the room must not contain a link, and a link's TARGET is what the grant meant.
        resolved = Path(os.path.realpath(source))
        if path_kind(resolved) == "missing":
            raise CleanRoomRefused(f"granted path is a dangling symlink: {source} -> {resolved}")
        if rule.decide(resolved) == _DECISION_DENY:
            out.excised.append({"source": str(source), "reason": "symlink_target_denied"})
            return
        source = resolved
    if source.is_dir():
        dest.mkdir(parents=True, exist_ok=True)
        for entry in sorted(source.iterdir()):
            _place(entry, dest / entry.name, rule, store, out)
        return
    if not source.is_file():
        raise CleanRoomRefused(f"refusing to place {source}: not a regular file or directory")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return  # an overlapping grant already placed these bytes
    shared = content_store.place_file(source, dest, store)
    if not shared:
        # Our own copy: drop every write bit so the input cannot be mutated in place. A store-backed
        # link is already read-only AND shared, so its mode is not ours to change (chmod follows the
        # inode and would reach into every other consumer of the same object).
        dest.chmod(dest.stat().st_mode & ~0o222)
    size = dest.stat().st_size
    out.n_files += 1
    out.n_bytes += size
    out.placed.append({"source": str(source), "size": size, "shared": bool(shared)})


# --------------------------------------------------------------------------- verification
@dataclass(frozen=True)
class CleanRoomViolation:
    """One reason the room is not clean. ``kind`` is stable vocabulary; ``path`` is inside the room."""

    kind: str
    path: str
    detail: str = ""


@dataclass(frozen=True)
class CleanRoomVerdict:
    """Whether the room is clean, and whether that question could be answered COMPLETELY.

    ``ok`` is true only when there are no violations AND the check ran to completion. An incomplete
    check is not a pass: it is the same failure mode as an unauditable transcript.
    """

    ok: bool
    complete: bool
    violations: tuple[CleanRoomViolation, ...] = ()
    n_files: int = 0
    n_symlinks: int = 0
    incomplete_reason: str = ""

    def describe(self) -> str:
        if self.ok:
            return f"clean room OK: {self.n_files} files, no answer surface reachable"
        head = "clean room REFUSED"
        if not self.complete:
            head += f" (verification incomplete: {self.incomplete_reason})"
        rows = "; ".join(f"{v.kind}@{v.path}{(' ' + v.detail) if v.detail else ''}" for v in self.violations[:12])
        more = f" (+{len(self.violations) - 12} more)" if len(self.violations) > 12 else ""
        return f"{head}: {rows}{more}"


def _content_matches(
    room_by_size: Mapping[int, list[Path]], withheld: Iterable[Path]
) -> tuple[list[tuple[Path, Path]], bool, str]:
    """Room files that are BYTE-IDENTICAL to some withheld file.

    SIZE IS THE SIEVE, and it is what makes this affordable at the real scale. One target's derived
    withheld set is ~2,600 files and ~8 GB (the externalized model weights dominate it); digesting that
    on every verification would cost a minute and would be paid whether or not anything matched. A file
    cannot be identical to another of a different length, so only a withheld file whose size already
    occurs in the room is ever digested — normally none of them, and the check costs one ``stat`` per
    withheld file.

    Content, not provenance, is what is compared: a room contaminated by a stray copy, an operator's
    convenience ``cp``, or a builder bug is caught exactly as a mis-placed grant is. That is what makes
    this a second line of defence rather than a restatement of the builder's own bookkeeping.
    """
    matches: list[tuple[Path, Path]] = []
    digested: dict[Path, str] = {}

    def digest(path: Path) -> str:
        if path not in digested:
            digested[path] = _sha256(path)
        return digested[path]

    for withheld_file in withheld:
        try:
            size = withheld_file.stat().st_size
        except OSError as exc:
            return matches, False, f"withheld file unreadable: {withheld_file} ({exc})"
        candidates = room_by_size.get(size)
        if not candidates:
            continue
        try:
            wanted = digest(withheld_file)
            for candidate in candidates:
                if digest(candidate) == wanted:
                    matches.append((candidate, withheld_file))
        except OSError as exc:
            return matches, False, f"unreadable while comparing content: {exc}"
    return matches, True, ""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_clean_room(
    room: Path,
    te: TargetExperiment,
    *,
    bundle: Mapping[str, object] | None = None,
    repo: Path | None = None,
    check_content: bool = True,
) -> CleanRoomVerdict:
    """Walk a built room and decide whether it is clean — independently of how it was built.

    This is deliberately NOT a check of the builder's bookkeeping. It re-derives the answer surfaces
    and inspects the bytes actually present, so a room contaminated by a bug, by a stray copy, or by an
    operator's convenience ``cp`` is refused exactly as a mis-built one is. The three classes it
    decides are: a file whose CONTENT is an answer surface; a link or ``.git`` pointer that leaves the
    room; and an input the agent could write through a shared inode.
    """
    room = Path(room).absolute()
    repo = (repo or repo_root()).absolute()
    violations: list[CleanRoomViolation] = []
    complete = True
    incomplete_reason = ""

    if path_kind(room) != "dir":
        return CleanRoomVerdict(
            ok=False,
            complete=False,
            violations=(CleanRoomViolation("room_missing", str(room)),),
            incomplete_reason="room does not exist",
        )

    # The room must not sit inside the checkout: an agent CLI detects its project root by walking up
    # for a .git, and a room under the checkout hands it the checkout regardless of what is in the room.
    if room == repo or repo in room.parents:
        violations.append(CleanRoomViolation("room_inside_checkout", str(room), f"checkout at {repo}"))

    # VERIFICATION IS INDEPENDENT OF CONSTRUCTION, and that independence is the property this whole
    # module rests on. Nothing below consults the builder's manifest, its placement records, or the
    # `_Precedence` object it decided with; the withheld set is re-derived from the descriptor here and
    # the room is judged on the bytes actually present in it. A builder that has been broken — by a bug,
    # by an edit, by a mutation test — therefore cannot certify its own output as clean.
    withheld = derive_deny_set(te, bundle, repo=repo)
    surface_set = frozenset(withheld.surfaces)
    withheld_list, complete, incomplete_reason = withheld_files(withheld)

    n_files = 0
    n_symlinks = 0
    inputs = room / INPUTS_DIR
    by_size: dict[int, list[Path]] = {}
    for path in sorted(room.rglob("*")):
        if path.is_symlink():
            n_symlinks += 1
            target = Path(os.path.realpath(path))
            if not (target == room or room in target.parents):
                violations.append(CleanRoomViolation("symlink_escapes_room", str(path), f"-> {target}"))
            elif inputs in path.parents or path == inputs:
                violations.append(CleanRoomViolation("symlink_in_inputs", str(path), f"-> {target}"))
            continue
        if path.name == ".git" and path.is_file():
            # A ``.git`` FILE is a worktree pointer at another repository's object store. A ``.git``
            # DIRECTORY created inside the room by ``git init`` is the room's own and is fine.
            violations.append(CleanRoomViolation("git_pointer_out_of_room", str(path), "gitdir pointer file"))
            continue
        if not path.is_file():
            continue
        n_files += 1
        real = Path(os.path.realpath(path))
        if real in surface_set or is_withheld(real, withheld):
            # A hard link or a resolved link that lands on a withheld path: the bytes are in the room
            # whatever the directory entry says.
            violations.append(CleanRoomViolation("answer_surface_path", str(path), f"resolves to {real}"))
            continue
        try:
            stat = path.stat()
        except OSError as exc:
            complete = False
            incomplete_reason = incomplete_reason or f"unreadable room file: {path} ({exc})"
            continue
        if inputs in path.parents and (stat.st_mode & 0o222) and stat.st_nlink > 1:
            violations.append(CleanRoomViolation("writable_shared_input", str(path), f"nlink={stat.st_nlink}"))
        by_size.setdefault(stat.st_size, []).append(path)

    if check_content:
        matches, content_complete, reason = _content_matches(by_size, withheld_list)
        for room_file, surface_file in matches:
            violations.append(
                CleanRoomViolation(
                    "answer_surface_content",
                    str(room_file),
                    f"byte-identical to derived answer surface {surface_file}",
                )
            )
        if not content_complete:
            complete = False
            incomplete_reason = incomplete_reason or reason

    return CleanRoomVerdict(
        ok=complete and not violations,
        complete=complete,
        violations=tuple(violations),
        n_files=n_files,
        n_symlinks=n_symlinks,
        incomplete_reason=incomplete_reason,
    )


# --------------------------------------------------------------------------- construction
@dataclass(frozen=True)
class CleanRoom:
    """A built, verified clean room. ``root`` is what the agent is given; ``manifest`` is host-only."""

    root: Path
    inputs: Path
    work: Path
    manifest: dict
    verdict: CleanRoomVerdict


def build_clean_room(
    te: TargetExperiment,
    home: Path,
    bundle: Mapping[str, object] | None = None,
    *,
    seed: Mapping[str, Path] | None = None,
    repo: Path | None = None,
    check_content: bool = True,
) -> CleanRoom:
    """Materialise a clean room under ``home`` and REFUSE it unless verification says it is clean.

    ``home`` must not exist (a room built on top of another run's leftovers is not a clean room, and
    "it was already there" is how a stale answer file survives into a new run). ``seed`` maps a path
    relative to the room's writable area to a host source — the arm's task card, a starting submission
    — and is subject to the same deny rule as everything else.

    On any refusal the partially built room is removed, so there is no half-room for a later run to
    find and treat as usable.
    """
    repo = (repo or repo_root()).absolute()
    home = Path(home).absolute()
    if home.exists() or home.is_symlink():
        raise CleanRoomRefused(f"clean-room home is not fresh: {home}")

    bundle = bundle or {}
    grants = derive_allow_set(bundle, repo=repo)
    deny = derive_deny_set(te, bundle, repo=repo)
    rule = _Precedence(allow=[source for _rel, source in grants], withheld=deny)

    room = home / "room"
    inputs = room / INPUTS_DIR
    work = room / WORK_DIR
    placement = _Placement()
    store = content_store.store_root()
    try:
        inputs.mkdir(parents=True)
        work.mkdir(parents=True)

        # Copy the union of the declared grants, not every overlapping spelling: a broad contract grant
        # plus its children must produce one subtree, or the deeper spelling would re-place bytes the
        # shallower one already excised around.
        unique = sorted({source.absolute() for _rel, source in grants}, key=lambda p: (len(p.parts), str(p)))
        roots: list[Path] = []
        for source in unique:
            covering = next((r for r in roots if r == source or r in source.parents), None)
            # A nested grant is subsumed by an outer root ONLY if walking down from that root actually
            # reaches it. When a withheld directory sits in between — the ordinary shape of a target
            # package's declared contract sub-tree — the walk stops at that directory, so the grant has
            # to be placed from its own root or it would silently go missing.
            if covering is not None and not _crosses_withheld(covering, source, deny):
                continue
            roots.append(source)
        for source in roots:
            _place(source, _room_path(inputs, source, repo), rule, store, placement)

        for rel, source in (seed or {}).items():
            source = Path(source).absolute()
            if path_kind(source) == "missing":
                raise CleanRoomRefused(f"clean-room seed does not resolve: {rel} -> {source}")
            dest = work / str(rel)
            if not dest.absolute().is_relative_to(work):
                raise CleanRoomRefused(f"clean-room seed escapes the workspace: {rel}")
            _place(source, dest, rule, store, placement)
            # Seeds land in the agent's WRITABLE area; restore write for the owner.
            for path in [dest, *(dest.rglob("*") if dest.is_dir() else ())]:
                if not content_store.is_shared(path):
                    path.chmod(path.stat().st_mode | 0o200)

        manifest = {
            "version": 1,
            "target": te.target,
            "repo": str(repo),
            "room": str(room),
            "allowed": [rel for rel, _ in grants],
            "copied_roots": [str(p) for p in roots],
            "n_files": placement.n_files,
            "n_bytes": placement.n_bytes,
            "n_excised": len(placement.excised),
            "excised": placement.excised,
            "deny_set_size": len(deny.surfaces),
            "deny_exemptions": [str(p) for p in deny.exemptions],
        }
        verdict = verify_clean_room(room, te, bundle=bundle, repo=repo, check_content=check_content)
        manifest["verdict"] = {
            "ok": verdict.ok,
            "complete": verdict.complete,
            "n_files": verdict.n_files,
            "n_symlinks": verdict.n_symlinks,
            "incomplete_reason": verdict.incomplete_reason,
            "violations": [{"kind": v.kind, "path": v.path, "detail": v.detail} for v in verdict.violations],
        }
        # Host-only, beside the room rather than inside it: it names host source paths, and the room is
        # precisely where those names must not be readable.
        (home / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if not verdict.ok:
            raise CleanRoomRefused(verdict.describe())
        return CleanRoom(root=room, inputs=inputs, work=work, manifest=manifest, verdict=verdict)
    except BaseException:
        remove_clean_room(home)
        raise


def remove_clean_room(home: Path) -> None:
    """Remove a room, restoring owner write first (inputs are deliberately unwritable)."""
    home = Path(home)
    if home.is_symlink():
        raise CleanRoomRefused(f"refusing to remove a symlinked clean-room home: {home}")
    if not home.exists():
        return
    for path in sorted(home.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if path.is_symlink() or content_store.is_shared(path):
            continue
        try:
            path.chmod(0o700 if path.is_dir() else 0o600)
        except OSError:
            pass
    shutil.rmtree(home, ignore_errors=False)
