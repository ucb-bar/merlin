#!/usr/bin/env python3
"""Read an EXTERNAL kernel bundle and derive everything the bench needs from its own bytes.

An external bundle is a directory carrying someone else's hand- or auto-tuned kernels for this
target, in the shape the bundles on this host use::

    descriptors/kernels.csv   one row per declared kernel: shape, call_sites, spad_rows_needed, status
    src/*.h                   the kernel definitions (+ whatever headers they include)
    MANIFEST.sha256           integrity over every file

Nothing here is typed as a literal. The kernel's shape comes from the descriptor, the *fit* predicate
comes from the guard the kernel source itself carries, and the design geometry comes from the
parameter header the target's own build recipe puts on the include path. Three traps this module
exists to make impossible:

* **Silent fallback.** Each kernel is wrapped in ``#if !<SOMETHING>_FITS`` with a stock-library call
  in the false branch. A bundle measured on a design where the guard fires measures the *library*
  while the row says the external kernel's name. :func:`fit_predicate` finds that guard by walking
  forward from the definition, reports the library symbol the fallback would call, and the caller
  turns the macro into a ``_Static_assert`` so the compiler refuses rather than substitutes.
* **Dead code.** A bundle may *declare* more kernels than it calls. ``call_sites`` is carried per row
  and a kernel with none is reported as refused-with-reason, never quietly benched as if it were live.
* **Undeclared footprint.** ``spad_rows_needed`` is checked against the footprint constants in the
  kernel source AND against the scratchpad the design actually has; a row whose two sources disagree
  is an error, not a shrug.
"""

from __future__ import annotations

import csv
import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

#: Geometry macros a kernel's footprint is expressed against. Every one is a plain integer literal in
#: the target's parameter header; they are READ from it, never assumed.
GEOMETRY_MACROS = ("DIM", "BANK_NUM", "BANK_ROWS", "ACC_ROWS", "MAX_BYTES")

#: Suffix of the compile-time guard that selects a kernel over its stock-library fallback.
FIT_SUFFIX = "_FITS"
#: Footprint macros named off the guard's prefix.
FOOTPRINT_SUFFIXES = ("_INPUT_ROWS", "_WEIGHT_ROWS")

_DEFINE = "#define"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def int_defines(text: str) -> dict[str, int]:
    """Every ``#define NAME <decimal>`` in ``text``, as a mapping.

    Structural on purpose: split the line, require exactly a directive, a name and a decimal token.
    A define whose body is an expression is NOT guessed at -- it is simply absent from the result, so
    a caller that needs it fails closed instead of reading a half-understood value.
    """
    out: dict[str, int] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 3 or parts[0] != _DEFINE:
            continue
        name, value = parts[1], parts[2]
        if not name.isidentifier():
            continue
        negative = value.startswith("-")
        digits = value[1:] if negative else value
        if digits.isdecimal():
            out[name] = -int(digits) if negative else int(digits)
    return out


def find_header(roots: Sequence[Path], required: Sequence[str]) -> tuple[Path, dict[str, int]]:
    """The header under ``roots`` that defines every name in ``required``, and its integer defines.

    Searched by CONTENT, not by file name: which header carries the design's geometry is a property
    of the target's include path, and a bundle that renamed it must still be readable.
    """
    seen: list[Path] = []
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.h")):
            seen.append(path)
            defines = int_defines(path.read_text(encoding="utf-8", errors="replace"))
            if all(name in defines for name in required):
                return path, defines
    raise LookupError(
        f"no header under {[str(r) for r in roots]} defines all of {list(required)} "
        f"as plain integers (looked at {len(seen)} headers)"
    )


@dataclass(frozen=True)
class KernelGuard:
    """The compile-time guard a kernel carries, and what the false branch would run instead."""

    macro: str
    fallback_symbol: str | None
    input_rows: int | None
    weight_rows: int | None

    @property
    def footprint_rows(self) -> int | None:
        if self.input_rows is None or self.weight_rows is None:
            return None
        return self.input_rows + self.weight_rows


def fit_predicate(source: str, symbol: str, defines: Mapping[str, int]) -> KernelGuard:
    """The ``#if !<NAME>_FITS`` guard opening ``symbol``'s body, and the symbol its fallback calls.

    Walks forward from the definition line to the first preprocessor conditional inside the body, so
    the guard is read off the kernel that carries it rather than assumed from a naming convention. The
    fallback symbol is the first call token in the false branch, which is what a run would silently
    measure if the guard fired.
    """
    lines = source.splitlines()
    start = None
    needle_prefix = "static void "
    needle = needle_prefix + symbol
    for i, line in enumerate(lines):
        if line.strip().startswith(needle):
            start = i
            break
    if start is None:
        raise LookupError(f"{symbol!r} is not defined in this source")
    macro = None
    body_at = None
    for i in range(start + 1, len(lines)):
        stripped = lines[i].strip()
        # BOUNDED at the next definition. An unbounded scan walks out of a kernel that has no guard
        # and adopts the NEXT kernel's -- which reads as "fit asserted" for a kernel nothing asserted.
        if stripped.startswith(needle_prefix) and not stripped.startswith(needle):
            break
        if stripped.startswith("#endif") or stripped.startswith("#else"):
            break
        if stripped.startswith("#if"):
            rest = stripped.partition(" ")[2].strip()
            if rest.startswith("!"):
                candidate = rest[1:].strip()
                if candidate.endswith(FIT_SUFFIX) and candidate.isidentifier():
                    macro, body_at = candidate, i
            break
    if macro is None or body_at is None:
        raise LookupError(f"{symbol!r} carries no `#if !<NAME>{FIT_SUFFIX}` guard; its fit cannot be asserted")
    fallback = None
    for i in range(body_at + 1, len(lines)):
        stripped = lines[i].strip()
        if stripped.startswith("#else") or stripped.startswith("#endif"):
            break
        if stripped.startswith("//") or stripped.startswith("/*") or stripped.startswith("*"):
            continue
        head, sep, _ = stripped.partition("(")
        token = head.strip()
        if sep and token.isidentifier():
            fallback = token
            break
    prefix = macro[: -len(FIT_SUFFIX)]
    return KernelGuard(
        macro=macro,
        fallback_symbol=fallback,
        input_rows=defines.get(prefix + FOOTPRINT_SUFFIXES[0]),
        weight_rows=defines.get(prefix + FOOTPRINT_SUFFIXES[1]),
    )


@dataclass(frozen=True)
class ExternalKernel:
    name: str
    op: str
    row: dict
    call_sites: int
    status: str
    declared_spad_rows: int | None
    header: Path
    #: ``None`` when the kernel carries no fit guard at all -- which is not a shrug: a kernel whose fit
    #: cannot be asserted cannot be benched, because nothing would stop a fallback from being measured
    #: under its name. Such a kernel is reported and skipped, never guessed at.
    guard: KernelGuard | None
    guard_error: str | None = None

    @property
    def live(self) -> bool:
        return self.call_sites > 0

    @property
    def benchable(self) -> bool:
        return self.live and self.guard is not None


@dataclass(frozen=True)
class ExternalBundle:
    root: Path
    descriptor: Path
    descriptor_sha256: str
    kernels: tuple[ExternalKernel, ...]
    manifest_verified: dict
    headers: dict
    #: The bundle's OWN parameter header, if it ships one: ``(path, sha256, geometry)`` or ``None``.
    #: The kernels are compiled against THIS design's header, not the bundle's, so whether the two
    #: describe the same accelerator is a fact the product has to carry rather than a thing to hope for.
    own_geometry: dict | None = None

    def live_kernels(self) -> tuple[ExternalKernel, ...]:
        return tuple(k for k in self.kernels if k.live)

    def dead_kernels(self) -> tuple[ExternalKernel, ...]:
        return tuple(k for k in self.kernels if not k.live)


def _verify_manifest(root: Path) -> dict:
    """Recompute every digest ``MANIFEST.sha256`` declares. A bundle whose bytes moved is not usable."""
    manifest = root / "MANIFEST.sha256"
    if not manifest.is_file():
        return {"present": False, "checked": 0, "mismatched": [], "missing": []}
    checked, bad, missing = 0, [], []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        declared, name = parts[0], parts[1].lstrip("*")
        path = root / name
        if not path.is_file():
            missing.append(name)
            continue
        checked += 1
        if sha256_file(path) != declared:
            bad.append(name)
    return {
        "present": True,
        "path": str(manifest),
        "sha256": sha256_file(manifest),
        "checked": checked,
        "mismatched": bad,
        "missing": missing,
    }


def _int_or_none(text: str) -> int | None:
    text = text.strip()
    return int(text) if text.isdecimal() else None


def load_bundle(root: Path) -> ExternalBundle:
    """Read the bundle: descriptor rows, the header defining each kernel, and each kernel's guard."""
    root = Path(root)
    descriptor = root / "descriptors" / "kernels.csv"
    if not descriptor.is_file():
        raise FileNotFoundError(f"{root} carries no descriptors/kernels.csv; it is not an external bundle")
    manifest = _verify_manifest(root)
    if manifest["mismatched"]:
        raise ValueError(f"{root}: bundle files do not match MANIFEST.sha256: {manifest['mismatched']}")
    sources = sorted((root / "src").glob("*.h")) if (root / "src").is_dir() else []
    texts = {p: p.read_text(encoding="utf-8", errors="replace") for p in sources}
    kernels: list[ExternalKernel] = []
    headers: dict[str, str] = {}
    with descriptor.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            name = (row.get("kernel") or "").strip()
            if not name:
                continue
            owning = [p for p, text in texts.items() if ("static void " + name) in text]
            if len(owning) != 1:
                raise LookupError(f"{name!r} is defined in {len(owning)} of the bundle's headers; expected exactly 1")
            header = owning[0]
            defines = int_defines(texts[header])
            try:
                guard, guard_error = fit_predicate(texts[header], name, defines), None
            except LookupError as exc:
                guard, guard_error = None, str(exc)
            headers[header.name] = sha256_file(header)
            kernels.append(
                ExternalKernel(
                    name=name,
                    op=(row.get("op") or "").strip(),
                    row=dict(row),
                    call_sites=_int_or_none(row.get("call_sites") or "") or 0,
                    status=(row.get("status") or "").strip(),
                    declared_spad_rows=_int_or_none(row.get("spad_rows_needed") or ""),
                    header=header,
                    guard=guard,
                    guard_error=guard_error,
                )
            )
    try:
        own_header, own_defines = find_header([root / "src"], GEOMETRY_MACROS)
        own_geometry = {
            "path": str(own_header),
            "sha256": sha256_file(own_header),
            "geometry": {name: own_defines[name] for name in GEOMETRY_MACROS},
        }
    except LookupError:
        own_geometry = None
    return ExternalBundle(
        root=root,
        descriptor=descriptor,
        descriptor_sha256=sha256_file(descriptor),
        kernels=tuple(kernels),
        manifest_verified=manifest,
        headers=headers,
        own_geometry=own_geometry,
    )


def geometry_agreement(bundle: ExternalBundle, design: Mapping[str, int]) -> dict:
    """Whether the bundle's own parameter header describes the same accelerator this build uses.

    The kernels here are compiled against THIS design's header, so a difference in the geometry macros
    would mean the bundle's absolute scratchpad addresses were searched against a different machine.
    A difference OUTSIDE those macros is reported too -- the headers are not interchangeable just
    because the numbers the kernels read happen to match.
    """
    own = bundle.own_geometry
    if own is None:
        return {"bundle_ships_a_parameter_header": False}
    differing = {k: (own["geometry"][k], design[k]) for k in design if own["geometry"].get(k) != design[k]}
    return {
        "bundle_ships_a_parameter_header": True,
        "bundle_header": own["path"],
        "bundle_header_sha256": own["sha256"],
        "geometry_matches": not differing,
        "differing_macros": differing,
        "compiled_against": "the design's own header, not the bundle's",
    }


def fit_evidence(kernel: ExternalKernel, geometry: Mapping[str, int]) -> dict:
    """Whether this kernel fits THIS design, from two independent sources, plus the disagreement if any.

    Source A is the bundle's descriptor (``spad_rows_needed``); source B is the footprint constants in
    the kernel's own source, summed through its guard's prefix. They must agree. The verdict compares
    the agreed figure with ``BANK_NUM * BANK_ROWS`` read from the design's parameter header -- which is
    exactly what the guard's own ``<PREFIX>_SPAD_ROWS`` means.
    """
    if kernel.guard is None:
        return {
            "spad_rows_available": geometry["BANK_NUM"] * geometry["BANK_ROWS"],
            "spad_rows_declared_by_descriptor": kernel.declared_spad_rows,
            "spad_rows_derived_from_source": None,
            "sources_agree": False,
            "fits": None,
            "asserted_at_compile_time": False,
            "why": kernel.guard_error,
        }
    available = geometry["BANK_NUM"] * geometry["BANK_ROWS"]
    declared = kernel.declared_spad_rows
    derived = kernel.guard.footprint_rows
    agree = declared is not None and derived is not None and declared == derived
    needed = declared if declared is not None else derived
    return {
        "spad_rows_available": available,
        "spad_rows_declared_by_descriptor": declared,
        "spad_rows_derived_from_source": derived,
        "sources_agree": agree,
        "acc_rows_available": geometry["ACC_ROWS"],
        "fit_macro": kernel.guard.macro,
        "fallback_symbol": kernel.guard.fallback_symbol,
        "fits": needed is not None and needed <= available,
        "asserted_at_compile_time": True,
    }
