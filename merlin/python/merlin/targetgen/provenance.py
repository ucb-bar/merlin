"""What a target-evolution result rests on, captured as data before the result is produced.

An evolution experiment claims "we reused N% of a CERTIFIED parent compiler". That claim is only
meaningful next to an exact statement of which parent, built from which hardware sources, with which
toolchain, graded on which simulator binary. Two runs a week apart can disagree entirely because a
sibling RTL checkout moved branch, and without this record the disagreement is unattributable.

Three rules this module exists to obey:

* **Ask, never assume.** Every field is read from the thing itself -- ``git rev-parse`` in the RTL
  checkout, ``--version`` from the compiler, ``stat`` on the simulator binary, the champion's own
  manifest. Nothing is a literal here.
* **Degrade honestly.** A source that cannot be resolved records ``{"available": False, "reason": ...}``
  and the run continues. It never raises and it never silently omits the key, because an absent key
  reads as "not applicable" when what happened was "we could not tell".
* **Name no target.** Which RTL trees, which simulator configs and which dtype strategy matter is the
  caller's (i.e. the experiment descriptor's) knowledge; this module takes them as parameters.

Paths to sibling checkouts are resolved from the environment (``MERLIN_CHIPYARD`` and friends) rather
than hardcoded, so a record made on one machine states where its facts came from without embedding
one developer's filesystem in the library.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..common.paths import env, repo_root

__all__ = [
    "content_sha",
    "declared_pins",
    "git_provenance",
    "parent_provenance",
    "repo_provenance",
    "record",
    "rtl_provenance",
    "simulator_provenance",
    "toolchain_provenance",
    "toolchain_shas",
]

_SHA_LEN = 12


def content_sha(obj: Any) -> str:
    """Stable content hash of a JSON-able record.

    Deliberately the same recipe as ``targetgen.rtl_check_runner._facts_sha`` -- sorted-keys JSON, so
    a dict that means the same thing hashes the same regardless of construction order. This is the
    hash an invalidation decision may rest on, so it hashes CONTENT, not identity (contrast
    ``publish._fingerprint``, which composes ids and therefore cannot notice that the facts moved
    under a stable package id).
    """
    payload = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:_SHA_LEN]


def _git(root: Path, *args: str) -> str | None:
    """One ``git`` query in ``root``, or None when it fails for any reason."""
    try:
        out = subprocess.run(("git", "-C", str(root), *args), capture_output=True, text=True,
                             timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def git_provenance(root: str | Path) -> dict[str, Any]:
    """``{available, path, head, branch, dirty}`` for any git checkout.

    ``dirty`` is load-bearing for an experiment record: a result produced from a tree with
    uncommitted changes is not reproducible from its sha, and saying so is cheaper than discovering
    it later.
    """
    p = Path(root)
    if not p.is_dir():
        return {"available": False, "reason": f"not a directory: {p}"}
    head = _git(p, "rev-parse", "HEAD")
    if head is None:
        return {"available": False, "reason": f"not a git checkout: {p}"}
    status = _git(p, "status", "--porcelain")
    return {
        "available": True,
        "path": str(p),
        "head": head,
        "branch": _git(p, "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status),
    }


def _submodule_pins(root: Path) -> dict[str, str]:
    """``{submodule path: sha}`` parsed structurally from ``git submodule status``.

    Lines are ``[+-U]?<sha> <path> (<describe>)``; split on whitespace rather than pattern-matching,
    and drop the leading state marker by character test so an out-of-date (``+``) or uninitialized
    (``-``) submodule is still recorded with its sha.
    """
    out = _git(root, "submodule", "status")
    pins: dict[str, str] = {}
    for line in (out or "").splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        sha = parts[0]
        if sha and not sha[0].isalnum():      # strip the +/-/U state marker
            sha = sha[1:]
        pins[parts[1]] = sha
    return pins


def repo_provenance() -> dict[str, Any]:
    """Merlin's own state: head, branch, dirty, and every submodule pin."""
    root = repo_root()
    rec = git_provenance(root)
    if rec.get("available"):
        rec["submodules"] = _submodule_pins(root)
    return rec


def _tool_version(exe: Path, flag: str = "--version") -> str | None:
    """First line of ``exe --version``, or None. Used for compilers, where the first line carries
    the version and the rest is build configuration."""
    try:
        out = subprocess.run((str(exe), flag), capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    first = out.stdout.strip().splitlines()
    return first[0].strip() if first else None


def toolchain_provenance() -> dict[str, Any]:
    """The compiler the binaries were built with, asked for its own version.

    Resolved through ``llvmlower.toolchain`` so the recorded path is the path that will actually be
    used (``MERLIN_CLANG`` overrides an in-repo install, and a stale override is exactly the kind of
    thing this record exists to expose).
    """
    rec: dict[str, Any] = {"python": sys.version.split()[0]}
    try:
        from ..llvmlower import toolchain as tc
    except Exception as exc:                                          # noqa: BLE001
        rec["clang"] = {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
        return rec
    for label, getter in (("clang", tc.clang), ("mlir_translate", tc.mlir_translate)):
        try:
            path = getter()
        except Exception as exc:                                      # noqa: BLE001
            rec[label] = {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
            continue
        if not Path(path).is_file():
            rec[label] = {"available": False, "reason": f"missing: {path}"}
            continue
        rec[label] = {"available": True, "path": str(path), "version": _tool_version(Path(path))}
    return rec


def rtl_provenance(sources: Mapping[str, str | Path]) -> dict[str, Any]:
    """Per-name git state of the hardware sources a target's facts were derived from.

    ``sources`` maps a caller-chosen label to a checkout path (typically assembled from an
    environment root plus a contract-declared subpath, so no tree name is baked in here). A hardware
    fact is only as pinned as the tree it was read from: an RTL checkout that switched branch is the
    single most common reason two "identical" runs disagree.
    """
    return {name: git_provenance(path) for name, path in sources.items()}


def simulator_provenance(paths: Mapping[str, str | Path]) -> dict[str, Any]:
    """``{label: {available, path, bytes, mtime, sha}}`` for grading binaries.

    A prebuilt simulator is an opaque artifact whose source config may no longer exist in the
    checkout that produced it, so identity has to come from the binary: size, mtime and a content
    hash of those two plus the name. Hashing the whole (tens of MB) binary is not worth the IO for a
    provenance line, and is noted here so the ``sha`` is not mistaken for a full digest.
    """
    out: dict[str, Any] = {}
    for label, path in paths.items():
        p = Path(path)
        if not p.is_file():
            out[label] = {"available": False, "reason": f"not built: {p}"}
            continue
        st = p.stat()
        stamp = {"name": p.name, "bytes": st.st_size, "mtime": int(st.st_mtime)}
        out[label] = {"available": True, "path": str(p), **stamp,
                      "sha": content_sha(stamp), "sha_covers": "name+size+mtime, not file contents"}
    return out


def parent_provenance(target: str, *, dtype_strategy: str | None = None) -> dict[str, Any]:
    """The certified parent package an evolution run claims to have reused.

    Resolved through ``publish.select_champion`` -- the same selector the compiler itself uses -- so
    the recorded parent cannot drift from the compiled one. Records the publish fingerprint AND a
    content hash of the package's own codegen artifacts, because the fingerprint is a composition of
    ids (package_id + merlin sha + cert run) and so cannot notice an edited schedule under a stable
    id.
    """
    try:
        from . import publish
    except Exception as exc:                                          # noqa: BLE001
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
    try:
        sel = publish.select_champion(target, dtype_strategy=dtype_strategy)
    except Exception as exc:                                          # noqa: BLE001
        return {"available": False, "reason": f"no champion for {target!r}: {exc}"}
    if sel is None:
        return {"available": False, "reason": f"no champion for {target!r}"}
    pub = (sel.manifest or {}).get("publication", {}) or {}
    files: dict[str, str] = {}
    pkg_dir = Path(sel.package_dir)
    for f in sorted(pkg_dir.rglob("*")):
        if f.is_file():
            files[str(f.relative_to(pkg_dir))] = hashlib.sha256(f.read_bytes()).hexdigest()[:_SHA_LEN]
    return {
        "available": True,
        "target": sel.target,
        "package_id": sel.package_id,
        "package_dir": str(pkg_dir),
        "family": sel.family,
        "layout_kind": sel.layout_kind,
        "status": sel.status,
        "version": sel.version,
        "fingerprint": pub.get("fingerprint"),
        "certification": pub.get("certification"),
        "certified_by_run": pub.get("certified_by_run"),
        "champion": pub.get("champion"),
        "artifact_shas": files,
        "content_sha": content_sha(files),
    }


def _env_root(var: str, subpath: str | None) -> str | None:
    """``$<var>/<subpath>`` when the variable is set, else None. Keeps sibling-checkout locations in
    the environment rather than in the library."""
    root = env(var)
    if not root:
        return None
    return str(Path(root) / subpath) if subpath else root


def record(target: str, *, dtype_strategy: str | None = None,
           rtl_sources: Mapping[str, str | Path] | None = None,
           simulators: Mapping[str, str | Path] | None = None,
           env_rtl_sources: Sequence[tuple[str, str, str | None]] = (),
           notes: str = "") -> dict[str, Any]:
    """The whole provenance record for one evolution run, with its own content hash.

    ``env_rtl_sources`` is a convenience for the common case: ``(label, ENV_VAR, subpath)`` triples
    resolved against the environment, so a descriptor can say "the hardware tree is
    ``$MERLIN_CHIPYARD/<its declared subpath>``" without the library knowing either half.

    The returned ``provenance_sha`` covers everything above it, so an experiment product can be
    compared to another by one field, and a mismatch points at the sub-record that moved.
    """
    sources: dict[str, str | Path] = dict(rtl_sources or {})
    for label, var, subpath in env_rtl_sources:
        resolved = _env_root(var, subpath)
        sources[label] = resolved if resolved else f"<unset: ${var}>"

    rec: dict[str, Any] = {
        "target": target,
        "merlin": repo_provenance(),
        "toolchain": toolchain_provenance(),
        "rtl": rtl_provenance(sources),
        "simulators": simulator_provenance(simulators or {}),
        "parent": parent_provenance(target, dtype_strategy=dtype_strategy),
        "notes": notes,
    }
    rec["provenance_sha"] = content_sha(rec)
    return rec


# --- which registry pins a target's results depend on ----------------------------------------------
#: The target-contract key naming the `merlin/contract/hardware_pins.yaml` entries a target's results
#: rest on. A target with no external RTL dependency legitimately declares none.
PINS_CONTRACT_KEY = "hardware_pins"


def declared_pins(target: str | None) -> tuple[str, ...]:
    """Pin names ``target``'s contract declares. Empty when it declares none or cannot be resolved.

    Deliberately not an error: a pure-simulation or host target has no external checkout to pin, and
    refusing would block provenance recording for exactly the targets that need it least.
    """
    if not target:
        return ()
    try:
        from .target_registry import resolve
        contract = resolve(target).load_contract() or {}
    except Exception:                       # noqa: BLE001 -- an unresolvable target has no declaration
        return ()
    return tuple(str(name) for name in (contract.get(PINS_CONTRACT_KEY) or ()))


def toolchain_shas(target: str | None = None) -> dict[str, str]:
    """``{name: sha}`` for merlin plus every hardware pin ``target`` declares.

    Replaces a helper that named one target's checkouts directly (a chipyard root and a generator
    subdirectory), which meant every OTHER target's run record carried merlin's sha and nothing else --
    unattributable, and silently so. The shape is unchanged so existing run records stay comparable;
    what changed is that the entries come from the target's own declaration.

    A pin whose checkout is absent or unreadable is recorded as UNKNOWN rather than omitted: a missing
    key reads as "this run had no such dependency" when it means "nobody could tell".

    THIS IS A SHA, NOT A CITATION. The value here is the revision the checkout is ON, and it says
    nothing about whether the bytes that were read are that revision's. A report that prints it beside
    a verdict is spelling "pinned" for a tree that may not be; pair it with :func:`toolchain_citations`,
    which is the form :mod:`merlin.common.provenance` licenses a claim to use.
    """
    from ..common import provenance as PROV

    shas = {"merlin": (git_provenance(repo_root()).get("head") or PROV.UNKNOWN)}
    for name in declared_pins(target):
        try:
            shas[name] = PROV.verify(name).observed.commit or PROV.UNKNOWN
        except Exception:                   # noqa: BLE001 -- unknown/unreadable pin is UNKNOWN, not absent
            shas[name] = PROV.UNKNOWN
    return shas


def toolchain_citations(target: str | None = None) -> dict[str, str]:
    """``{name: citation}`` for every hardware pin ``target`` declares — the spelling a claim may use.

    The sibling of :func:`toolchain_shas`, and the one a verdict-claiming report must carry. A bare sha
    cannot distinguish "measured on this commit" from "measured on this commit plus a handful of
    uncommitted lines", and both shapes have shipped in this repo's artifacts under 40 hex characters
    that look identical.

    Recorded as UNKNOWN, never omitted, for the same reason ``toolchain_shas`` does: an absent key reads
    as "no such dependency" when it means "nobody could tell".
    """
    from ..common import provenance as PROV

    out: dict[str, str] = {}
    for name in declared_pins(target):
        try:
            out[name] = PROV.citation(name)
        except Exception:                   # noqa: BLE001 -- unknown/unreadable pin is UNKNOWN, not absent
            out[name] = PROV.UNKNOWN
    return out
