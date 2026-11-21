"""Byte claims of the pin registry: built artifacts, bytes recorded as lost, and the registry they share.

:mod:`merlin.common.provenance` answers "is this checkout the declared revision". The claims here have
no revision of their own: a BUILT artifact (a bitstream, a compiled simulator, a packaged image) is
identified by its bytes plus the pins it was elaborated from, and a loss record says which bytes a body
of results is attributed to that no longer exist. Both live in the same reviewed registry file as the
pins, so its location (:func:`pins_path`) and its error (:class:`PinsError`) are defined here, below the
pin verifier that uses them.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "Artifact",
    "ArtifactCheck",
    "LOST_REQUIRED_FIELDS",
    "LostArtifact",
    "PinsError",
    "UNKNOWN",
    "UNRECOVERABLE",
    "claimed_digests",
    "file_digest",
    "load_artifacts",
    "load_lost",
    "lost_for_digest",
    "parse_lost",
    "pins_path",
    "software_pins_path",
    "verify_artifact",
]

#: Recorded where a fact could not be read. Never compares equal to a real value, and callers must not
#: treat it as "unchanged".
UNKNOWN = "UNKNOWN"


class PinsError(RuntimeError):
    """A pin is missing, malformed, or the checkout disagrees with it."""


def pins_path() -> Path:
    from .paths import data_path

    return data_path("contract", "hardware_pins.yaml")


def software_pins_path() -> Path:
    """The registry of pinned toolchain/frontend revisions: same schema, same verifier, own file."""
    from .paths import data_path

    return data_path("contract", "software_pins.yaml")


def file_digest(path: str | Path) -> str:
    """sha256 of one file, or UNKNOWN. Used to identify a prebuilt simulator binary."""
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return UNKNOWN


# ---- lost bytes ----------------------------------------------------------------------------------
# Bytes a body of results is attributed to, which no longer exist and cannot be reproduced.
#
# A separate module from :mod:`merlin.common.provenance` because it answers a different question. That
# one asks "is this checkout the declared revision" and expects an answer either way. This one is for the
# case where there is no answer to be had and there never will be again, and the design follows from
# refusing to let that case borrow either of the ordinary answers.
#
# THE MEASURED CASE. A pinned simulator-compiler binary was rebuilt in place, and by the time anyone
# noticed, 244 artifacts recorded the old digest as the tool that produced them. Three things can be done
# with that, and two are worse than the loss:
#
# * repoint the declaration at the bytes now on disk -- every one of those artifacts keeps asserting a
#   digest nothing on the host has, while the registry says the pin is fine. The attribution is then not
#   broken, it is silently WRONG, which is the failure the pin registry exists to prevent;
# * delete the declaration -- the citations dangle with no account of what they point at;
# * record the loss: what was lost, when, why it cannot be rebuilt, and what DOES still verify.
#
# A loss record CLAIMS its digest: the pin and artifact loaders are cross-checked against it, so nothing in
# the registry can re-declare those bytes as its own. And :func:`lost_for_digest` makes a citation
# resolvable -- the artifacts citing lost bytes never re-hash them, so they do not fail and never will, and
# asking is the only way anyone finds out.

#: The state of a declared artifact whose bytes no longer exist anywhere and cannot be rebuilt. Not
#: spelled as a missing digest or an empty string: "gone, and here is the account" is a different fact
#: from "not yet recorded" (an empty ``Artifact.digest``) and from "present but wrong" (a digest mismatch).
UNRECOVERABLE = "UNRECOVERABLE"

#: Fields a loss record must state. Each is a question a reader of a dangling citation asks.
LOST_REQUIRED_FIELDS: tuple[str, ...] = ("digest", "what", "lost_on", "why_unrecoverable", "still_verifies")


@dataclass(frozen=True)
class LostArtifact:
    """One set of bytes recorded as gone, with the account of how.

    ``superseded_by`` records what occupies the path NOW. It is bookkeeping about the path, never a
    substitute for the lost bytes, and nothing reads it as one.
    """

    name: str
    digest: str
    what: str
    lost_on: str
    why_unrecoverable: str
    still_verifies: tuple[str, ...] = ()
    path: str = ""
    root_env: str = ""
    bytes_declared: int | None = None
    discovered_on: str = ""
    cause: str = ""
    superseded_by: str = ""
    cited_by: str = ""
    do_not: tuple[str, ...] = ()
    #: What now stands as the witness for the results that cited the lost bytes, and whether it is a
    #: WEAKER attestation than the one it replaces. A decision someone made, so it is a field a reader
    #: can find and disagree with, not a remark in ``notes``.
    attestation_decision: str = ""
    notes: str = ""
    state: str = UNRECOVERABLE

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact": self.name,
            "state": self.state,
            "digest": self.digest,
            "path": self.path,
            "what": self.what,
            "lost_on": self.lost_on,
            "discovered_on": self.discovered_on,
            "cause": self.cause,
            "why_unrecoverable": self.why_unrecoverable,
            "superseded_by": self.superseded_by,
            "cited_by": self.cited_by,
            "still_verifies": list(self.still_verifies),
            "do_not": list(self.do_not),
            "attestation_decision": self.attestation_decision,
            "bytes_declared": self.bytes_declared,
        }


#: Parsed at most once per (file, mtime, size), the same key as ``provenance.load_pins``: the loss
#: records are read on every pin and artifact load (the cross-check), and a grade performs thousands.
_LOST_MEMO: dict[tuple[str, int, int], dict[str, LostArtifact]] = {}


def load_lost(path: str | Path | None = None) -> dict[str, LostArtifact]:
    """Every declared UNRECOVERABLE artifact. An empty mapping when the registry declares none."""
    import yaml

    p = Path(path) if path is not None else pins_path()
    if not p.is_file():
        raise PinsError(f"no pin registry at {p}")
    try:
        st = p.stat()
        memo_key: tuple[str, int, int] | None = (str(p.resolve()), st.st_mtime_ns, st.st_size)
    except OSError:  # unstattable: parse it, and do not remember what we cannot key
        memo_key = None
    if memo_key is not None and memo_key in _LOST_MEMO:
        return dict(_LOST_MEMO[memo_key])
    out = parse_lost(yaml.safe_load(p.read_text(encoding="utf-8")) or {}, p)
    if memo_key is not None:
        _LOST_MEMO[memo_key] = dict(out)
    return out


def _digest(src: Path, name: str, field_name: str, raw: Any) -> str:

    if not isinstance(raw, str):
        raise PinsError(
            f"{src}: lost artifact {name!r} {field_name} must be a quoted sha256 string; YAML read "
            f"{type(raw).__name__}, which loses leading zeros"
        )
    digest = raw.lower()
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise PinsError(
            f"{src}: lost artifact {name!r} {field_name} {raw!r} is not a 64-character sha256; the digest "
            "IS the thing being recorded, so a partial one records nothing"
        )
    return digest


def parse_lost(document: Any, src: Path) -> dict[str, LostArtifact]:
    """The loss records out of an ALREADY-PARSED registry document.

    Split from :func:`load_lost` so the pin and artifact loaders cross-check against the document THEY
    parsed: re-reading would double the parses of a file loaded thousands of times per grade, and would
    let a loader and its own cross-check see two different versions of the file.
    """

    if not isinstance(document, dict):
        raise PinsError(f"{src}: loss registry must be a mapping")
    entries = document.get("lost_artifacts", {})
    if not isinstance(entries, dict):
        raise PinsError(f"{src}: 'lost_artifacts' must be a mapping of name -> declaration")
    out: dict[str, LostArtifact] = {}
    seen: dict[str, str] = {}
    for name, body in entries.items():
        if not isinstance(name, str) or not name.strip():
            raise PinsError(f"{src}: lost artifact name must be a nonempty string")
        if not isinstance(body, dict):
            raise PinsError(f"{src}: lost artifact {name!r} must be a mapping")
        state = str(body.get("state") or UNRECOVERABLE)
        if state != UNRECOVERABLE:
            raise PinsError(
                f"{src}: lost artifact {name!r} declares state {state!r}; this section records only "
                f"{UNRECOVERABLE!r}. Something recoverable belongs in `artifacts:` with its digest."
            )
        missing = [f for f in LOST_REQUIRED_FIELDS if not body.get(f)]
        if missing:
            raise PinsError(
                f"{src}: lost artifact {name!r} does not state {missing}; a loss record that does not say "
                "what was lost, when, why it cannot be rebuilt and what still verifies leaves a reader of "
                "a dangling citation exactly where they started"
            )
        for field in ("what", "lost_on", "why_unrecoverable"):
            if not isinstance(body[field], str) or not body[field].strip():
                raise PinsError(f"{src}: lost artifact {name!r} {field} must be a nonempty string")
        for field in ("still_verifies", "do_not"):
            values = body.get(field, [])
            if not isinstance(values, list) or any(not isinstance(value, str) or not value.strip() for value in values):
                raise PinsError(f"{src}: lost artifact {name!r} {field} must be a list of nonempty strings")
        digest = _digest(src, str(name), "digest", body["digest"])
        if digest in seen:
            raise PinsError(
                f"{src}: lost artifacts {seen[digest]!r} and {name!r} both claim digest {digest[:12]}; "
                "one set of bytes is lost once"
            )
        seen[digest] = str(name)
        superseded = (
            _digest(src, str(name), "superseded_by", body["superseded_by"]) if body.get("superseded_by") else ""
        )
        if superseded == digest:
            raise PinsError(
                f"{src}: lost artifact {name!r} declares superseded_by equal to its own lost digest, which "
                "would say the bytes replaced themselves"
            )
        declared = body.get("bytes_declared")
        if declared is not None and (isinstance(declared, bool) or not isinstance(declared, int) or declared < 0):
            raise PinsError(f"{src}: lost artifact {name!r} bytes_declared must be a nonnegative integer")
        out[str(name)] = LostArtifact(
            name=str(name),
            digest=digest,
            what=str(body["what"]),
            lost_on=str(body["lost_on"]),
            why_unrecoverable=str(body["why_unrecoverable"]),
            still_verifies=tuple(str(s) for s in body["still_verifies"]),
            path=str(body.get("path") or ""),
            root_env=str(body.get("root_env") or ""),
            bytes_declared=int(declared) if declared is not None else None,
            discovered_on=str(body.get("discovered_on") or ""),
            cause=str(body.get("cause") or ""),
            superseded_by=superseded,
            cited_by=str(body.get("cited_by") or ""),
            do_not=tuple(str(d) for d in (body.get("do_not") or ())),
            attestation_decision=str(body.get("attestation_decision") or ""),
            notes=str(body.get("notes") or ""),
        )
    return out


def claimed_digests(path: str | Path | None = None, *, document: Any = None) -> dict[str, LostArtifact]:
    """``{digest: record}`` for every set of bytes a loss record claims. Used by the loaders' checks.

    ``document`` is an already-parsed registry: pass it when the caller has one, so the cross-check
    reads the same bytes the caller did and the file is parsed once.
    """
    if document is not None:
        records = parse_lost(document, Path(path) if path is not None else pins_path())
    else:
        records = load_lost(path)
    return {rec.digest: rec for rec in records.values()}


def lost_for_digest(digest: str, *, path: str | Path | None = None) -> LostArtifact | None:
    """The loss record claiming ``digest``, or None.

    This turns a dangling citation into an answered one: an artifact recording lost bytes does not
    re-hash them at load, so it never fails, and asking is the only way a reader learns they are gone.
    """
    want = str(digest or "").strip().lower()
    if not want:
        return None
    return claimed_digests(path).get(want)


# ---- built artifacts -----------------------------------------------------------------------------


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
    path: str  # absolute, relative to root_env, or repo-relative
    description: str = ""
    digest: str = ""  # sha256 of the file/tree; empty = not yet recorded
    root_env: str = ""
    built_from: tuple[str, ...] = ()  # pin names this was elaborated from
    config: str = ""  # the elaborated configuration, when there is one
    notes: str = ""
    #: WHICH TARGET this artifact belongs to, and WHAT PART it plays for that target -- for example
    #: ``(target="…", role="verilator_binary")``. Both are declared, never parsed out of the artifact's
    #: NAME: a consumer that needs "this target's reference simulator" would otherwise have to encode a
    #: naming convention, and one target's elaboration is called after its own config
    #: (``…_gsim_model_serialclk``), which is not a convention any other target shares. Empty means the
    #: artifact declares no role, and a role-based lookup then does not see it at all -- absent, never
    #: guessed.
    target: str = ""
    role: str = ""
    #: The SIMULATOR/QUEUE CONFIGURATION NAMES that resolve to these bytes. Declared, never parsed
    #: out of the artifact's own name: a result records the hw-config it was submitted under, and
    #: nothing else connects that string to a device. Measured 2026-09-19, which is why this field
    #: exists: a whole week of whole-model cycle numbers was quoted under the hw-config
    #: ``alveo_u250_firesim_gemmini_rocket_30mhz`` while the registry knew only a DIFFERENT bitstream
    #: for the same ``config`` name, so a reader resolving those results by configuration name landed
    #: on the wrong silicon. A configuration name is not a device; this is the mapping that says
    #: which device a configuration name meant.
    hw_configs: tuple[str, ...] = ()
    #: sha256 of the HWDB CONFIG ENTRY that names these bytes -- the small YAML a submission points
    #: at -- which is NOT ``digest``, the sha256 of the built image itself. Both exist because a
    #: result records the former and the registry has always declared only the latter, so the one
    #: check meant to catch "a different device under a reused config name" was comparing a YAML's
    #: hash against a multi-gigabyte tarball's and could never agree. Empty means the hwdb entry's
    #: hash was never recorded, and a record stating one is then confirmed by NAME alone -- reported
    #: as such, never silently upgraded to a byte match and never held against the record.
    hwdb_digest: str = ""
    #: sha256 of the parameter header whose ABI this device ELABORATES, when it has been established.
    #: A program compiled against a different header runs and is wrong in whichever layer the two
    #: disagree about, which reads as a schedule regression; so a measurement compares a program's
    #: build header against this before a run is spent. Empty means UNKNOWN -- never assumed equal.
    abi_header_sha256: str = ""
    #: ``file`` (one built file) or ``tree`` (a built DIRECTORY whose identity is all of its bytes).
    #: A tree is hashed with the SAME hasher the host-lane pin uses, so a descriptor's
    #: ``package_sha256`` and this registry's ``digest`` are literally the same number -- two hashers
    #: for one question is the drift this registry exists to prevent.
    kind: str = "file"
    #: Resolve ``path`` against the repository root. Without this an artifact carrying no ``root_env``
    #: resolves relative to the CURRENT DIRECTORY, so the same declaration verifies from the repo root
    #: and reports "no file" from anywhere else -- a pin that passes in CI and fails on a laptop.
    repo_relative: bool = False

    KINDS = ("file", "tree")

    def resolve(self) -> Path | None:
        from .paths import env as _env

        if self.root_env:
            root = _env(self.root_env)
            return None if not root else Path(root) / self.path
        if self.repo_relative:
            from .paths import repo_root

            return repo_root() / self.path
        return Path(self.path)


@dataclass(frozen=True)
class ArtifactCheck:
    """Whether a built artifact is present and is the one that was declared."""

    artifact: str
    path: str
    present: bool = False
    digest: str = UNKNOWN
    matches: bool | None = None  # None when nothing was declared to match against
    gaps: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.present and self.matches is True

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact": self.artifact,
            "path": self.path,
            "present": self.present,
            "digest": self.digest,
            "matches": self.matches,
            "ok": self.ok,
            "gaps": list(self.gaps),
        }


def load_artifacts(path: str | Path | None = None) -> dict[str, Artifact]:
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
            raise PinsError(f"{p}: artifact {name!r} has no path; an artifact without one identifies nothing")
        digest = body.get("digest") or ""
        if digest and not isinstance(digest, str):
            raise PinsError(f"{p}: artifact {name!r} digest must be a quoted string, got {type(digest).__name__}")
        hwdb_digest = body.get("hwdb_digest") or ""
        if hwdb_digest and not isinstance(hwdb_digest, str):
            raise PinsError(
                f"{p}: artifact {name!r} hwdb_digest must be a quoted string, got {type(hwdb_digest).__name__}"
            )
        kind = str(body.get("kind") or "file")
        if kind not in Artifact.KINDS:
            raise PinsError(f"{p}: artifact {name!r} kind must be one of {list(Artifact.KINDS)}, got {kind!r}")
        out[str(name)] = Artifact(
            name=str(name),
            path=str(body["path"]),
            description=str(body.get("description") or ""),
            digest=str(digest),
            root_env=str(body.get("root_env") or ""),
            kind=kind,
            repo_relative=bool(body.get("repo_relative", False)),
            built_from=tuple(str(b) for b in (body.get("built_from") or ())),
            config=str(body.get("config") or ""),
            notes=str(body.get("notes") or ""),
            target=str(body.get("target") or ""),
            role=str(body.get("role") or ""),
            hw_configs=tuple(str(c) for c in (body.get("hw_configs") or ())),
            abi_header_sha256=str(body.get("abi_header_sha256") or ""),
            hwdb_digest=str(hwdb_digest),
        )
    # An artifact may not re-declare bytes the registry records as UNRECOVERABLE: repointing a live
    # declaration at dead bytes is exactly the move a loss record forbids.
    lost = claimed_digests(p, document=raw)
    for name, art in out.items():
        rec = lost.get(art.digest.lower())
        if rec is not None:
            raise PinsError(
                f"{p}: artifact {name!r} declares digest {art.digest[:12]}, which lost_artifacts"
                f"[{rec.name!r}] records as {UNRECOVERABLE}; an artifact declaration claims the bytes can "
                "be verified, and by the registry's own record these cannot"
            )
    return out


def verify_artifact(name: str, *, path: str | Path | None = None) -> ArtifactCheck:
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
        return ArtifactCheck(
            artifact=name, path=UNKNOWN, gaps=(f"${a.root_env} is unset, so {name!r} cannot be located",)
        )
    if a.kind == "tree":
        if not target.is_dir():
            return ArtifactCheck(artifact=name, path=str(target), gaps=(f"no directory at {target}",))
        from merlin.common.digest import hash_tree

        hashed = hash_tree(target)
        got = str(hashed.get("sha256") or "")
        if not hashed.get("present") or not got:
            return ArtifactCheck(artifact=name, path=str(target), gaps=(f"{target} has no hashable content",))
    else:
        if not target.is_file():
            return ArtifactCheck(artifact=name, path=str(target), gaps=(f"no file at {target}",))
        got = file_digest(target)
    if not a.digest:
        gaps.append("no digest declared, so the file present cannot be confirmed to be the one meant")
        return ArtifactCheck(artifact=name, path=str(target), present=True, digest=got, matches=None, gaps=tuple(gaps))
    if got != a.digest:
        gaps.append(f"digest is {got[:16]} but the registry declares {a.digest[:16]}")
    return ArtifactCheck(
        artifact=name, path=str(target), present=True, digest=got, matches=(got == a.digest), gaps=tuple(gaps)
    )
