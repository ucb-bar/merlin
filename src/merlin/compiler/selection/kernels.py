"""Select declared matrix kernels and enforce their shape/ABI constraints.

Ported from OPU-VME 80e00803d; hardware values remain caller-supplied.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REGISTRY_FILE = Path("kernels") / "kernels.yaml"


REGISTRY_SCHEMA = "merlin.matrix_kernels.v1"


ABI_SIGNATURE = (
    "void {function}(int32_t *c, const int8_t *at, const int8_t *b, const int32_t *bias, size_t m, size_t n, size_t k)"
)


REFERENCE_FUNCTION = "merlin_matrix_reference_gemm"


_TILE = "tile"


_CONSTRAINT_KEYS = ("m_multiple", "n_multiple", "k_multiple", "max_m", "max_n", "max_k", "bias")


_ENTRY_KEYS = ("unit", "file", "function", "generated", "description", "author", "constraints", "notes")


class KernelSelectionError(ValueError):
    """The requested kernel cannot be selected (unknown name, missing file, malformed registry)."""


class KernelConstraintError(ValueError):
    """A routed contraction falls outside what the selected kernel declares it handles."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _c_identifier(name: str, what: str) -> str:
    if not name or not (name[0].isalpha() or name[0] == "_") or not all(c.isalnum() or c == "_" for c in name):
        raise KernelSelectionError(f"{what} {name!r} is not a C identifier")
    return name


def _extent(value: Any, tile_edge: int | None, what: str) -> int | None:
    """An extent from an int or a ``tile``-relative expression (``tile``, ``2*tile``, ``tile*2``, ``tile/2``)."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise KernelSelectionError(f"{what}: {value!r} is not an extent")
    if isinstance(value, int):
        return int(value)
    text = "".join(str(value).split())
    if tile_edge is None:
        raise KernelSelectionError(f"{what}: {value!r} is relative to the tile edge, which is not known here")
    for sep in ("*", "/"):
        left, found, right = text.partition(sep)
        if not found:
            continue
        if left == _TILE and right.isdigit():
            return int(tile_edge) * int(right) if sep == "*" else max(1, int(tile_edge) // int(right))
        if sep == "*" and right == _TILE and left.isdigit():
            return int(left) * int(tile_edge)
        break
    if text == _TILE:
        return int(tile_edge)
    if text.isdigit():
        return int(text)
    raise KernelSelectionError(
        f"{what}: {value!r} is not an extent; write an integer or 'tile', 'N*tile', 'tile*N', 'tile/N'"
    )


@dataclass(frozen=True)
class Constraints:
    """What shapes a kernel takes. Every field is optional; the defaults accept any shape.

    ``*_multiple`` is a granularity (``m % m_multiple == 0``), ``max_*`` an upper bound; both may be written
    relative to the tile edge (``2*tile``). ``bias`` says whether the kernel honours a non-NULL ``bias``
    (the model's shim always passes NULL; kernel-test exercises it when declared).
    """

    m_multiple: Any = 1
    n_multiple: Any = 1
    k_multiple: Any = 1
    max_m: Any = None
    max_n: Any = None
    max_k: Any = None
    bias: bool = True

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None, *, where: str) -> Constraints:
        raw = dict(raw or {})
        unknown = sorted(set(raw) - set(_CONSTRAINT_KEYS))
        if unknown:
            raise KernelSelectionError(f"{where}: unknown constraint(s) {unknown}; known: {list(_CONSTRAINT_KEYS)}")
        got = cls(**raw)
        for key in _CONSTRAINT_KEYS[:-1]:
            _extent(getattr(got, key), 64, f"{where}: {key}")  # syntax only; the edge is supplied later
        if not isinstance(got.bias, bool):
            raise KernelSelectionError(f"{where}: bias must be true or false, got {got.bias!r}")
        return got

    def resolved(self, tile_edge: int | None) -> dict[str, int | None]:
        return {k: _extent(getattr(self, k), tile_edge, k) for k in _CONSTRAINT_KEYS[:-1]}

    def violations(self, m: int, n: int, k: int, *, tile_edge: int | None) -> list[str]:
        """Every rule ``(m, n, k)`` breaks, each as a sentence naming the rule and how it was declared."""
        r = self.resolved(tile_edge)
        out: list[str] = []
        for dim, val in (("m", m), ("n", n), ("k", k)):
            mult, top = r[f"{dim}_multiple"], r[f"max_{dim}"]
            written = getattr(self, f"{dim}_multiple")
            if mult and mult > 1 and val % mult:
                spelled = f"{written!r} = {mult}" if not isinstance(written, int) else str(mult)
                out.append(f"{dim.upper()}={val} is not a multiple of {dim}_multiple ({spelled})")
            if top is not None and val > top:
                out.append(f"{dim.upper()}={val} exceeds max_{dim} ({top})")
        return out

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in _CONSTRAINT_KEYS}

    @property
    def any_shape(self) -> bool:
        return self == Constraints()


@dataclass(frozen=True)
class KernelEntry:
    """One registered kernel."""

    name: str
    unit: str | None
    generated: bool = False
    file: Path | None = None
    function: str | None = None
    description: str = ""
    author: str = ""
    constraints: Constraints = field(default_factory=Constraints)
    registry: Path | None = None


@dataclass(frozen=True)
class Registry:
    path: Path
    header: str
    header_prefix: str
    defaults: dict[str, str]
    kernels: dict[str, KernelEntry]

    def default_for(self, unit: str) -> KernelEntry:
        name = self.defaults.get(unit)
        if name is None:
            raise KernelSelectionError(f"{self.path} declares no default kernel for unit {unit!r}")
        return self.kernels[name]

    def for_unit(self, unit: str) -> list[KernelEntry]:
        return [e for e in self.kernels.values() if e.unit in (None, unit)]


def load_registry(package_root: str | Path) -> Registry | None:
    """``<package_root>/kernels/kernels.yaml``, validated; None when the package registers no kernels."""
    import yaml

    path = Path(package_root) / REGISTRY_FILE
    if not path.is_file():
        return None
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if raw.get("schema") != REGISTRY_SCHEMA:
        raise KernelSelectionError(f"{path}: schema is {raw.get('schema')!r}, expected {REGISTRY_SCHEMA!r}")
    header = dict(raw.get("header") or {})
    for key in ("file", "prefix"):
        if not header.get(key):
            raise KernelSelectionError(
                f"{path}: header.{key} is required (the generated header's name and macro prefix)"
            )
    _c_identifier(str(header["prefix"]), f"{path}: header.prefix")
    kernels: dict[str, KernelEntry] = {}
    for name, body in (raw.get("kernels") or {}).items():
        where = f"{path}: kernel {name!r}"
        body = dict(body or {})
        unknown = sorted(set(body) - set(_ENTRY_KEYS))
        if unknown:
            raise KernelSelectionError(f"{where}: unknown key(s) {unknown}; known: {list(_ENTRY_KEYS)}")
        generated = bool(body.get("generated"))
        file = None
        if not generated:
            if not body.get("file") or not body.get("function"):
                raise KernelSelectionError(f"{where}: a hand-written kernel needs `file` and `function`")
            file = (path.parent / str(body["file"])).resolve()
            _c_identifier(str(body["function"]), f"{where}: function")
        kernels[str(name)] = KernelEntry(
            name=str(name),
            unit=str(body["unit"]) if body.get("unit") else None,
            generated=generated,
            file=file,
            function=str(body["function"]) if body.get("function") else None,
            description=" ".join(str(body.get("description") or "").split()),
            author=str(body.get("author") or ""),
            constraints=Constraints.from_mapping(body.get("constraints"), where=where),
            registry=path,
        )
    defaults = {str(u): str(n) for u, n in (raw.get("defaults") or {}).items()}
    for unit, name in defaults.items():
        if name not in kernels:
            raise KernelSelectionError(
                f"{path}: the default for {unit!r} is {name!r}, which is not a registered kernel"
            )
    return Registry(path, str(header["file"]), str(header["prefix"]), defaults, kernels)


@dataclass(frozen=True)
class SelectedKernel:
    """The kernel a build uses, and the identity it is recorded under."""

    entry: KernelEntry
    unit: str
    requested: str | None
    origin: str  # "default", "package" or "path"
    header: str
    header_prefix: str
    sha256: str | None = None

    @property
    def generated(self) -> bool:
        return self.entry.generated

    @property
    def name(self) -> str:
        return self.entry.name

    @property
    def function(self) -> str | None:
        return self.entry.function

    def record(self) -> dict[str, Any]:
        """What a build, a routing report and an image's provenance carry about the kernel."""
        out: dict[str, Any] = {
            "name": self.entry.name,
            "unit": self.unit,
            "requested": self.requested,
            "origin": self.origin,
            "generated": self.entry.generated,
        }
        if not self.entry.generated:
            out.update(
                file=str(self.entry.file),
                function=self.entry.function,
                sha256=self.sha256,
                constraints=self.entry.constraints.to_dict(),
                twin="merlin reference (merlin.kernels.matrix_kernels.reference_source)",
            )
        else:
            out["twin"] = "the package's generated kernel with its scalar tile body"
        return out


def _function_defined(text: str, function: str) -> bool:
    """Whether ``function`` appears as a whole identifier -- a cheap early check (``<P>_KERNEL(name)`` spells
    the definition through a macro, so the link is what finally proves it)."""
    start = 0
    while True:
        at = text.find(function, start)
        if at < 0:
            return False
        before = text[at - 1] if at else " "
        after = text[at + len(function)] if at + len(function) < len(text) else " "
        if not (before.isalnum() or before == "_") and not (after.isalnum() or after == "_"):
            return True
        start = at + len(function)


def resolve(spec: str | None, *, package_root: str | Path | None, unit: str) -> SelectedKernel:
    """The kernel ``spec`` names for ``unit``: None/empty is the package default, else a name, else a path.

    A path is ``file.c`` (function named like the file's stem) or ``file.c:function``; its constraints come
    from a ``kernels.yaml`` beside it that registers the same file, otherwise it is taken to accept any
    shape (and kernel-test will say whether it does).
    """
    reg = load_registry(package_root) if package_root is not None else None
    text = (spec or "").strip()
    if not text:
        if reg is None:
            return SelectedKernel(
                KernelEntry(name="generated", unit=unit, generated=True), unit, None, "default", "", ""
            )
        entry = reg.default_for(unit)
        return _selected(entry, unit, None, "default", reg)
    if reg is not None and text in reg.kernels:
        entry = reg.kernels[text]
        if entry.unit not in (None, unit):
            raise KernelSelectionError(f"kernel {text!r} targets unit {entry.unit!r}, not {unit!r} ({reg.path})")
        return _selected(entry, unit, text, "package", reg)
    file_part, colon, func = text.rpartition(":")
    if not colon or not file_part.endswith(".c"):
        file_part, func = text, ""
    path = Path(file_part).expanduser()
    if not path.suffix and reg is not None:
        known = ", ".join(sorted(e.name for e in reg.for_unit(unit)))
        raise KernelSelectionError(f"no kernel named {text!r} for {unit!r}; registered: {known} ({reg.path})")
    if not path.is_file():
        known = ", ".join(sorted(e.name for e in reg.for_unit(unit))) if reg else "none"
        raise KernelSelectionError(f"{text!r} is neither a registered kernel ({known}) nor an existing C file")
    path = path.resolve()
    sibling = load_registry(path.parent.parent) if (path.parent.name == REGISTRY_FILE.parent.name) else None
    for candidate in (sibling, reg):
        for entry in candidate.kernels.values() if candidate else ():
            if entry.file == path and (not func or func == entry.function):
                return _selected(entry, unit, text, "path", candidate)
    function = _c_identifier(func or path.stem, "kernel function (from the file name; use file.c:function)")
    entry = KernelEntry(name=path.stem, unit=unit, file=path, function=function, description=f"file {path}")
    if reg is None:
        raise KernelSelectionError(
            f"the support package registers no kernels ({REGISTRY_FILE} absent), so it has no header to compile "
            f"{path.name} against"
        )
    return _selected(entry, unit, text, "path", reg)


def _selected(entry: KernelEntry, unit: str, requested: str | None, origin: str, reg: Registry) -> SelectedKernel:
    sha = None
    if not entry.generated:
        if entry.file is None or not entry.file.is_file():
            raise KernelSelectionError(f"kernel {entry.name!r}: its file {entry.file} does not exist")
        text = entry.file.read_text(encoding="utf-8")
        if not _function_defined(text, str(entry.function)):
            raise KernelSelectionError(
                f"kernel {entry.name!r}: {entry.file} does not define {entry.function}(); the kernel must be "
                + ABI_SIGNATURE.format(function=entry.function)
            )
        if (entry.file.parent / reg.header).resolve() == entry.file:
            raise KernelSelectionError(f"kernel {entry.name!r}: the kernel file cannot be named {reg.header}")
        sha = _sha256(entry.file)
    return SelectedKernel(entry, unit, requested, origin, reg.header, reg.header_prefix, sha)


def check_signatures(kernel: SelectedKernel, signatures: Mapping[str, Sequence[int]], *, tile_edge: int | None) -> None:
    """Refuse when a routed contraction (``{symbol: (m, n, k)}`` or ``(b, m, n, k)``) is outside the kernel's
    declared constraints, naming every one and the rule it breaks."""
    if kernel.generated or kernel.entry.constraints.any_shape:
        return
    bad: list[str] = []
    for sym, sig in sorted(signatures.items()):
        m, n, k = (int(v) for v in tuple(sig)[-3:])
        why = kernel.entry.constraints.violations(m, n, k, tile_edge=tile_edge)
        if why:
            bad.append(f"  {sym} (M={m} N={n} K={k}): " + "; ".join(why))
    if bad:
        where = kernel.entry.registry or kernel.entry.file
        raise KernelConstraintError(
            f"matrix kernel {kernel.name!r} cannot take {len(bad)} of the {len(signatures)} routed contraction(s) "
            f"(tile edge {tile_edge}; constraints declared in {where}):\n"
            + "\n".join(bad)
            + "\nEither make the kernel handle these shapes and relax its constraints, or select a kernel that "
            "does (e.g. the package default, `--matrix-kernel` with no value)."
        )


def reference_source(function: str = REFERENCE_FUNCTION) -> str:
    """The scalar reference: the kernel ABI's definition as a plain loop. Wraps in int32 like the units do."""
    _c_identifier(function, "reference function")
    return f"""\
/* GENERATED by merlin.kernels.matrix_kernels -- the reference kernel, the ABI's definition as a loop.
 * C[i][j] = (bias ? bias[j] : 0) + sum_k at[k*m + i] * b[k*n + j]; both operands K-major; C is written. */
#include <stdint.h>
#include <stddef.h>

void {function}(int32_t *c, const int8_t *at, const int8_t *b, const int32_t *bias,
                size_t m, size_t n, size_t k)
{{
  for (size_t i = 0; i < m; ++i)
    for (size_t j = 0; j < n; ++j) {{
      uint32_t sum = bias ? (uint32_t)bias[j] : 0u;   /* unsigned: wraps without undefined behaviour */
      for (size_t kk = 0; kk < k; ++kk)
        sum += (uint32_t)((int32_t)at[kk * m + i] * (int32_t)b[kk * n + j]);
      c[i * n + j] = (int32_t)sum;
    }}
}}
"""
