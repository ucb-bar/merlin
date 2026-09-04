"""Where a target's GSIM emulator lives, and whether *these bytes* are allowed to certify.

GSIM is the fast elaborated-RTL engine (:mod:`rtl_engine_policy` ranks it above Verilator on cost at
equal fidelity), and every target that can be elaborated to FIRRTL can have one. It nonetheless kept
losing the selection, and the reason was never the policy: GSIM emits a **standalone C++ model built out
of tree**, so unlike Verilator there is no simulator rule inside the RTL checkout whose output path can be
derived. Each backend therefore resolved its emulator through a bare environment variable, that variable
was unset on every machine nobody had hand-exported it on, the availability probe answered False, and the
policy correctly — and silently — fell through to Verilator. A cert engine that only works when a human
remembers an env var is a cert engine that normally does not work.

Two things fix that, and this module is both of them.

**A DERIVED HOME.** The emulator is a build product, so it belongs under the build root the
generated-output convention already declares (``out/build/``, via :func:`merlin.common.paths.build_dir`)
— not under a scratch directory that is purgeable by design, and not behind an env var. The layout is
``<build>/rtl_engines/<target>/gsim/``:

    ``emulator``            the binary this target certifies on
    ``build_receipt.json``  the lineage receipt for those exact bytes (optional but expected)
    ``provenance.json``     how the binary got here, written by whoever adopted it

The environment override is KEPT — a caller pointing at a freshly built model must not have to install it
first — but it is now the exception rather than the only path. Resolution order is: the backend's own env
spelling, the derived per-target spelling, then the derived home.

**A PROVENANCE GATE.** The hazard this repo has already shipped once is a result attributed to the wrong
device. A binary sitting in the right directory proves nothing about which RTL revision it was elaborated
from, so when a receipt is present beside it, the receipt must BIND THESE BYTES (``artifacts.binary.sha256``
against the digest of the file actually resolved) or the emulator is **refused** — reported unavailable
with the reason, never quietly used. A receipt that is absent is not a pass either: it resolves with a
reason that says the provenance is unrecorded, so the sentence reaches the run record and the report
instead of nobody. ``MERLIN_GSIM_REQUIRE_RECEIPT=1`` turns that from loud into fatal for a run that must
not certify on unattributable bytes.

Nothing here knows a target name, a simulator flag, or an RTL fact: the target is a parameter, the
directory is derived from it, and the digests come from the bytes on disk.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: The canonical file names inside ``<build>/rtl_engines/<target>/gsim/``. Fixed names, not a glob: picking "the
#: newest matching binary" out of a directory is how a stale model gets certified against without anyone
#: choosing it. Installing a new emulator means replacing these, which is an act with an author.
BINARY_NAME = "emulator"
RECEIPT_NAME = "build_receipt.json"
ADOPTION_NAME = "provenance.json"

#: The WRAPPER flavour's entry point, derived from the engine name: ``<engine>_run.py``. :func:`engine_home`
#: has always documented two legitimate shapes under one home, and :func:`record_adoption` exists to install
#: the directory-shaped one — but :func:`resolve` looked only for :data:`BINARY_NAME`, so an engine that
#: ships a wrapper was reported ABSENT no matter how completely it was built. That is not a hypothetical:
#: a target on this host had a GSIM engine installed in its derived home, cycle-exact against Verilator on
#: 17/17 programs at 32x the speed, and every cert still ran on Verilator because the probe could not see
#: the shape it was in. A probe that cannot see a working engine is the same defect as an env var nobody
#: exported, which is the defect this module was written to remove.
WRAPPER_SUFFIX = "_run.py"

#: Receipts are written by the GSIM model builder (``merlin.gsim-model-build.v<N>``). The version is not
#: pinned here — a newer receipt schema still binds its binary the same way, and refusing an emulator
#: because its receipt is a version newer than this module would fail closed on the wrong thing.
RECEIPT_SCHEMA_PREFIX = "merlin.gsim-model-build."

#: Set to make an UNRECEIPTED emulator unusable rather than merely loudly unattributed. Off by default:
#: a developer who just built a model locally must be able to run it; a run that publishes a verdict
#: turns it on.
REQUIRE_RECEIPT_ENV = "MERLIN_GSIM_REQUIRE_RECEIPT"

_TRUTHY = ("1", "true", "yes", "on")


def _env(name: str) -> str:
    """An override read through :func:`merlin.common.paths.env`, so the gitignored ``.env`` counts.

    ``os.environ.get`` alone was the SECOND independent cause of the silent Verilator fallback, and the
    more embarrassing one: this repo's ``.env`` already declared a built GSIM emulator for the SIMT
    target, and the oracle never saw it because it read only the process environment. The machine had
    the fast engine, the configuration named it, and every cert still ran on Verilator. The process
    environment still WINS — ``.env`` is the fallback, exactly as `paths.env` defines it.
    """
    from merlin.common.paths import env as _paths_env
    return (_paths_env(name) or "").strip()


def engine_home(target: str, engine: str = "gsim") -> Path:
    """``<build>/rtl_engines/<target>/<engine>`` — the derived home for one target's build of one
    elaborated-RTL engine.

    ONE home per (target, engine), whatever that engine's build happens to produce. Two different
    shapes live under it today and both are legitimate: a self-contained emulator BINARY (the chipyard
    flavour, driven with an ELF) and a conventional ``<engine>_run.py`` WRAPPER beside its own binary
    (the program-oracle flavour, driven with assembled words). Giving each engine a different derived
    layout is how one of them ends up reachable only through an env var — which is the whole defect
    this module exists to remove.

    Target and engine are folder components (the generated-output convention), so everything for one
    target groups together and the inner file names are identical across targets.
    """
    from merlin.common.paths import build_dir
    return build_dir() / "rtl_engines" / str(target) / str(engine)


def gsim_home(target: str) -> Path:
    """This target's GSIM home — :func:`engine_home` with the engine fixed."""
    return engine_home(target, "gsim")


def wrapper_name(engine: str = "gsim") -> str:
    """``<engine>_run.py`` — the wrapper flavour's entry point, DERIVED from the engine name.

    Derived rather than listed so a new engine needs no edit here, and so the name cannot drift apart
    from the one :func:`engine_home` documents.
    """
    return f"{engine}{WRAPPER_SUFFIX}"


def wrapper_path(target: str, engine: str = "gsim") -> Path:
    """Where this target's wrapper-flavour engine entry point is, or would be installed."""
    return engine_home(target, engine) / wrapper_name(engine)


def derived_env_var(target: str) -> str:
    """The per-target env spelling this module honors, DERIVED from the target name rather than listed.

    A backend keeps its own historical spelling (passed as ``env_var``); this one exists so a target that
    never had a bespoke variable still has an override, without anyone editing shared code to add it.
    """
    ident = "".join(ch if ch.isalnum() else "_" for ch in str(target)).upper()
    return f"MERLIN_GSIM_EMU_{ident}"


def emulator_path(target: str, *, env_var: str | None = None) -> Path:
    """Where this target's GSIM emulator is, or where one would be installed.

    Never raises and never checks existence — callers that need availability use :func:`resolve` /
    :func:`probe`, which answer with a reason. This mirrors the shape of a backend's ``verilator_path``
    so the two engines resolve identically.
    """
    for name in _override_names(target, env_var):
        raw = _env(name)
        if raw:
            return Path(raw)
    return gsim_home(target) / BINARY_NAME


def receipt_path(target: str, *, env_var: str | None = None) -> Path:
    """The build receipt for the resolved binary: beside it, under :data:`RECEIPT_NAME`.

    Beside the BINARY, not in the derived home, so an env-pointed emulator carries its own lineage
    instead of borrowing the installed one's — which would attribute one build's bytes to another's RTL.
    """
    return emulator_path(target, env_var=env_var).parent / RECEIPT_NAME


def _override_names(target: str, env_var: str | None) -> tuple[str, ...]:
    names = [n for n in (env_var, derived_env_var(target)) if n]
    seen: list[str] = []
    for n in names:
        if n not in seen:
            seen.append(n)
    return tuple(seen)


@dataclass(frozen=True)
class Resolution:
    """The answer to "can this target certify on GSIM, and on what authority?"

    ``ok`` is availability; ``reason`` is the sentence the engine policy records and a report prints.
    ``refused`` distinguishes the two ways ``ok`` can be False, and they are not the same finding: an
    ABSENT emulator is work not done yet, a REFUSED one is bytes that exist and may not be trusted.
    """
    target: str
    path: Path
    source: str                       # "env:<VAR>" | "derived"
    ok: bool
    reason: str
    refused: bool = False
    receipt: dict[str, Any] | None = None      # the citable lineage block, when a receipt bound it
    receipt_status: str = "absent"             # absent | bound | invalid | adopted
    digest: str | None = None                  # sha256 of the bytes actually resolved
    #: Which shape answered: "binary" (a self-contained emulator driven with an ELF) or "wrapper"
    #: (an ``<engine>_run.py`` driven with assembled words). A caller INVOKES the two differently, so
    #: the shape has to survive resolution rather than be re-guessed from the path.
    flavour: str = "binary"


def _digest(path: Path) -> str:
    from merlin.common import provenance
    return provenance.file_digest(path)


def _receipt_block(doc: dict[str, Any], path: Path, digest: str) -> dict[str, Any]:
    """The part of a receipt a run record should CITE — the identity of the RTL and of the tools that
    turned it into this binary. Kept small on purpose: a cert cites what it was, not the whole transcript.
    """
    arts = doc.get("artifacts") or {}
    tools = doc.get("tools") or {}

    def _sha(entry: Any) -> str | None:
        return str(entry.get("sha256")) if isinstance(entry, dict) and entry.get("sha256") else None

    return {
        "receipt_path": str(path),
        "receipt_sha256": _digest(path),
        "schema_version": str(doc.get("schema_version") or ""),
        "status": str(doc.get("status") or ""),
        "binary_sha256": digest,
        "firrtl_sha256": doc.get("firrtl_sha256") or _sha(arts.get("firrtl")),
        "firrtl_path": (arts.get("firrtl") or {}).get("path") if isinstance(arts.get("firrtl"), dict) else None,
        "model_manifest_sha256": doc.get("model_manifest_sha256") or _sha(arts.get("model_manifest")),
        "inputs_sha256": doc.get("inputs_sha256"),
        "commands_sha256": doc.get("commands_sha256"),
        "tools": {k: _sha(v) for k, v in tools.items() if isinstance(v, dict)},
    }


def _validate_receipt(target: str, binary: Path, digest: str,
                      receipt: Path) -> tuple[str, str, dict[str, Any] | None]:
    """(status, reason_fragment, citable_block) for the receipt beside ``binary``.

    The one check that decides trust is the BINDING: does this receipt describe THESE bytes? Everything
    else it asserts (which FIRRTL, which emitter, which compiler) is a claim about a binary identified by
    digest, so a receipt whose digest is someone else's binary says nothing at all about this one — and
    reads, to anyone who opens it, as though it did.
    """
    if not receipt.is_file():
        return "absent", (f"no build receipt beside it ({receipt.name}) — provenance UNRECORDED: these "
                          f"bytes are not attributed to any RTL revision"), None
    try:
        doc = json.loads(receipt.read_text(encoding="utf-8"))
    except Exception as exc:                    # noqa: BLE001 — an unreadable receipt is not a pass
        return "invalid", f"build receipt {receipt.name} is unreadable ({type(exc).__name__}: {exc})", None
    if not isinstance(doc, dict):
        return "invalid", f"build receipt {receipt.name} is not a receipt document", None
    schema = str(doc.get("schema_version") or "")
    if not schema.startswith(RECEIPT_SCHEMA_PREFIX):
        return "invalid", (f"build receipt {receipt.name} declares schema {schema!r}, not a "
                           f"{RECEIPT_SCHEMA_PREFIX}* GSIM model-build receipt"), None
    status = str(doc.get("status") or "")
    if status != "complete":
        return "invalid", f"build receipt {receipt.name} status is {status!r}, not 'complete'", None
    declared = str(doc.get("binary_sha256")
                   or ((doc.get("artifacts") or {}).get("binary") or {}).get("sha256") or "")
    if not declared:
        return "invalid", f"build receipt {receipt.name} declares no binary digest", None
    if declared != digest:
        return "invalid", (f"build receipt {receipt.name} binds {declared[:12]} but {binary.name} is "
                           f"{digest[:12]} — the receipt describes a DIFFERENT binary, so its RTL and "
                           f"tool identity say nothing about these bytes"), None
    return "bound", f"lineage bound by {receipt.name}", _receipt_block(doc, receipt, digest)


def _resolve_wrapper(target: str, engine: str = "gsim") -> Resolution | None:
    """The wrapper flavour's resolution, or ``None`` if this home does not hold one.

    Answers the same question as the binary branch and to the same standard. The bytes identified are the
    WRAPPER's, because that is the file a caller executes; what the wrapper drives is named by the lineage
    record beside it. A home installed by :func:`record_adoption` carries no build receipt but does carry a
    digest of every file it installed, so an adoption record that COVERS these exact wrapper bytes is real
    evidence and is reported as its own status — weaker than a bound receipt, and not silently equated
    with one.
    """
    path = wrapper_path(target, engine)
    if not path.is_file():
        return None
    digest = _digest(path)
    home = engine_home(target, engine)

    # A receipt in a wrapper home normally binds the ENGINE BINARY the wrapper drives, not the wrapper
    # itself, so a non-binding receipt here is the expected case and must not be read as a refusal — that
    # would make installing a lineage record strictly worse than installing none. Honour it only when it
    # genuinely binds these bytes; otherwise the adoption record is the evidence.
    status, note, block = _validate_receipt(target, path, digest, home / RECEIPT_NAME)
    if status != "bound":
        status, note = _adoption_status(home, path, digest)
        block = None

    if status != "bound" and _env(REQUIRE_RECEIPT_ENV).lower() in _TRUTHY:
        return Resolution(target, path, "derived", False,
                          f"GSIM wrapper ({engine}) at {path} REFUSED: {note} and "
                          f"{REQUIRE_RECEIPT_ENV} is set",
                          refused=True, receipt_status=status, digest=digest, flavour="wrapper")
    return Resolution(target, path, "derived", True,
                      f"GSIM wrapper ({engine}) {path} (derived, {digest[:12]}); {note}",
                      receipt=block, receipt_status=status, digest=digest, flavour="wrapper")


def _adoption_status(home: Path, path: Path, digest: str) -> tuple[str, str]:
    """``(status, reason)`` from the adoption record, which must COVER these bytes to count.

    An adoption record that lists a different digest for this file describes an earlier install, and
    saying "provenance recorded" on the strength of it would attribute these bytes to bytes that are
    gone — the precise failure the registry exists to prevent.
    """
    record = home / ADOPTION_NAME
    absent = ("absent", f"no build receipt beside it ({RECEIPT_NAME}) — provenance UNRECORDED: these "
                        f"bytes are not attributed to any RTL revision")
    if not record.is_file():
        return absent
    try:
        doc = json.loads(record.read_text(encoding="utf-8"))
    except Exception:                           # noqa: BLE001 — an unreadable record is simply no record
        return absent
    entry = ((doc.get("files") or {}) if isinstance(doc, dict) else {}).get(path.name)
    declared = str(entry.get("sha256")) if isinstance(entry, dict) else ""
    if declared != digest:
        return absent
    return "adopted", (f"no build receipt ({RECEIPT_NAME}), but the adoption record {ADOPTION_NAME} "
                       f"covers these exact bytes — lineage ADOPTED, not built-and-bound")


def resolve(target: str, *, env_var: str | None = None) -> Resolution:
    """Resolve ``target``'s GSIM emulator and decide whether it may certify. Never raises."""
    path = emulator_path(target, env_var=env_var)
    source = "derived"
    for name in _override_names(target, env_var):
        if _env(name):
            source = f"env:{name}"
            break

    if not path.is_file():
        # The binary flavour is absent — but this home may hold the WRAPPER flavour instead, which is
        # just as much a built engine. Only when the caller did NOT point an override at a specific
        # binary: an env var names bytes, and answering with a different file than the one named would
        # be the wrong kind of helpful.
        if source == "derived":
            wrapped = _resolve_wrapper(target)
            if wrapped is not None:
                return wrapped
        where = f"{source} -> {path}" if source != "derived" else str(path)
        return Resolution(target, path, source, False,
                          f"no GSIM emulator at {where} (build one and install it as "
                          f"{gsim_home(target) / BINARY_NAME}, or a "
                          f"{wrapper_name('gsim')} wrapper in the same home)")
    # Existence is not enough: an artifact copied without its mode bit, or the emitted .cpp rather than
    # the built model, both exist. A cert tier reported available and then failing to exec is worse than
    # one reported absent.
    if not os.access(path, os.X_OK):
        return Resolution(target, path, source, False,
                          f"GSIM emulator at {path} is not executable (copied without its mode bit?)")

    digest = _digest(path)
    status, note, block = _validate_receipt(target, path, digest,
                                            path.parent / RECEIPT_NAME)
    if status == "invalid":
        return Resolution(target, path, source, False,
                          f"GSIM emulator at {path} REFUSED: {note}",
                          refused=True, receipt_status=status, digest=digest)
    if status == "absent" and _env(REQUIRE_RECEIPT_ENV).lower() in _TRUTHY:
        return Resolution(target, path, source, False,
                          f"GSIM emulator at {path} REFUSED: {note} and {REQUIRE_RECEIPT_ENV} is set",
                          refused=True, receipt_status=status, digest=digest)
    return Resolution(target, path, source, True,
                      f"GSIM emulator {path} ({source}, {digest[:12]}); {note}",
                      receipt=block, receipt_status=status, digest=digest)


def probe(target: str, *, env_var: str | None = None) -> tuple[bool, str]:
    """``(available, reason)`` in the shape :func:`rtl_engine_policy.select` consumes."""
    r = resolve(target, env_var=env_var)
    return r.ok, r.reason


def citation(target: str, *, env_var: str | None = None) -> dict[str, Any]:
    """The block a run record embeds so a GSIM verdict says which emulator produced it.

    Recorded whether or not the emulator is usable — "GSIM was refused because ..." is exactly as much a
    fact about a run as "GSIM certified these capsules", and only one of the two currently survives into
    any artifact.
    """
    r = resolve(target, env_var=env_var)
    return {"target": r.target, "engine": "gsim", "path": str(r.path), "source": r.source,
            "available": r.ok, "refused": r.refused, "reason": r.reason, "flavour": r.flavour,
            "binary_sha256": r.digest, "receipt_status": r.receipt_status, "receipt": r.receipt}


def install(target: str, binary: "str | Path", *, receipt: "str | Path | None" = None,
            note: str = "", extra: "dict[str, Any] | None" = None) -> Resolution:
    """Adopt ``binary`` as ``target``'s canonical GSIM emulator under the derived home.

    COPIES rather than links or moves: the sources are typically scratch build trees that other sessions
    are still running out of and that are purgeable by design, and a symlink into one of those makes the
    cert engine disappear the day the tree is cleaned. The receipt is copied beside it under the fixed
    name, so :func:`resolve` binds the lineage to the installed bytes rather than to the build tree's.

    Writes an adoption record (:data:`ADOPTION_NAME`) saying where these bytes came from and what was
    verified about them — refusing the install outright if the receipt does not bind the copied binary,
    since installing an emulator whose lineage describes a different one is precisely how a verdict gets
    attributed to the wrong device.
    """
    import datetime as _dt
    import shutil

    src = Path(binary)
    if not src.is_file():
        raise FileNotFoundError(f"no GSIM emulator to install at {src}")
    home = gsim_home(target)
    home.mkdir(parents=True, exist_ok=True)
    dest = home / BINARY_NAME
    shutil.copy2(src, dest)
    dest.chmod(dest.stat().st_mode | 0o111)
    digest = _digest(dest)

    dest_receipt = home / RECEIPT_NAME
    if receipt is not None:
        rsrc = Path(receipt)
        if not rsrc.is_file():
            raise FileNotFoundError(f"no GSIM build receipt to install at {rsrc}")
        shutil.copy2(rsrc, dest_receipt)
        status, why, _block = _validate_receipt(target, dest, digest, dest_receipt)
        if status != "bound":
            dest_receipt.unlink(missing_ok=True)
            dest.unlink(missing_ok=True)
            raise ValueError(f"refusing to install {src} for {target!r}: {why}")

    record = {
        "schema_version": "merlin.gsim-emulator-adoption.v1",
        "target": str(target),
        "installed_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "source_binary": str(src.resolve()),
        "source_receipt": str(Path(receipt).resolve()) if receipt is not None else None,
        "binary_sha256": digest,
        "receipt_sha256": _digest(dest_receipt) if dest_receipt.is_file() else None,
        "note": note,
    }
    if extra:
        record.update(extra)
    (home / ADOPTION_NAME).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n",
                                      encoding="utf-8")
    return resolve(target)


def record_adoption(target: str, engine: str = "gsim", *, sources: "dict[str, str] | None" = None,
                    note: str = "", extra: "dict[str, Any] | None" = None) -> Path:
    """Write the adoption record for an engine home whose files were installed by other means.

    :func:`install` covers the self-contained-binary flavour. An engine that ships a DIRECTORY (a
    conventional ``<engine>_run.py`` beside its own binary and inputs) is installed by copying that
    directory, and still owes the same answer: which bytes are these, where did they come from, and
    what is known about the revision they model. Every file present in the home is digested, so the
    record is a statement about what is actually there rather than about what the copier intended.

    ``note`` is where an UNRESOLVED provenance question belongs, spelled out. A home with no build
    receipt is not a scandal; a home with no build receipt and no sentence saying so is.
    """
    import datetime as _dt

    home = engine_home(target, engine)
    home.mkdir(parents=True, exist_ok=True)
    files = {}
    for f in sorted(home.rglob("*")):
        if f.is_file() and f.name != ADOPTION_NAME:
            files[str(f.relative_to(home))] = {"sha256": _digest(f), "n_bytes": f.stat().st_size}
    record: dict[str, Any] = {
        "schema_version": "merlin.gsim-emulator-adoption.v1",
        "target": str(target),
        "engine": str(engine),
        "installed_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "sources": dict(sources or {}),
        "files": files,
        "note": note,
    }
    if extra:
        record.update(extra)
    out = home / ADOPTION_NAME
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out
