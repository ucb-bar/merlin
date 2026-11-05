"""Hand-written matrix-unit kernels in a chip package: the registry, selection, header, reference and checks.

A chip package that routes int8 contractions to a matrix unit (``plugin.matrix_lowering``) supplies ONE
kernel with the fixed ABI of :mod:`merlin.kernels.matrix_shim`::

    void <function>(int32_t *c, const int8_t *at, const int8_t *b, const int32_t *bias,
                    size_t m, size_t n, size_t k);

    C[i][j] = (bias ? bias[j] : 0) + sum_k at[k*m + i] * b[k*n + j]     (int32, wrapping; C is WRITTEN)

By default that kernel is the one the package generates from its derived encodings. This module lets a
kernel author drop a plain C file into the package instead and have every routed contraction call it:

* **Registry.** ``<package>/kernels/kernels.yaml`` (schema ``merlin.matrix_kernels.v1``) names each kernel
  (its file, its function, the unit it targets, its constraints) and which one is each unit's default. The
  package's generated kernel is registered as ``generated: true`` and stays what an unselected build uses.
* **Selection.** ``--matrix-kernel <name|path>`` picks a registered kernel, or a C file anywhere
  (``path/to/k.c`` with the function named like the file, or ``path/to/k.c:function``).
  :func:`resolve` returns a :class:`SelectedKernel` carrying the file's sha256, which travels into the
  build record, the routing report and the image's provenance.
* **Header.** :func:`emit_header` writes ``<header>`` (named by the registry) from the DERIVED encoding
  table at build time: one ``.insn r`` string macro per instruction, a statement form taking register
  numbers, and the configuration facts the package hands it. A kernel names instructions; the words still
  come from the RTL. Nothing in it is typed by hand.
* **Reference twin.** For a hand-written kernel, the stand-in twin compiles :func:`reference_source` -- a
  plain scalar loop with the same ABI and the same function name -- rather than the author's file, so
  "shipped == twin, bit for bit" means "this kernel == the reference".
* **Constraints.** :func:`check_signatures` refuses a build whose routed contractions fall outside what
  the kernel declares (granularity, bounds), naming each contraction and the rule it breaks.

Nothing here names a target or an instruction: the encodings, facts and files are parameters.
"""

from __future__ import annotations

import hashlib
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "ABI_SIGNATURE",
    "Constraints",
    "KernelChoice",
    "KernelConstraintError",
    "KernelEntry",
    "KernelSelectionError",
    "PreparedKernel",
    "REFERENCE_FUNCTION",
    "REGISTRY_FILE",
    "REGISTRY_SCHEMA",
    "Registry",
    "SelectedKernel",
    "check_signatures",
    "emit_header",
    "load_registry",
    "prepare",
    "reference_source",
    "resolve",
    "unit_header",
]

#: Where a package registers its kernels, relative to the package root.
REGISTRY_FILE = Path("kernels") / "kernels.yaml"
REGISTRY_SCHEMA = "merlin.matrix_kernels.v1"

#: The one signature every matrix-unit kernel has (merlin.kernels.matrix_shim calls it).
ABI_SIGNATURE = (
    "void {function}(int32_t *c, const int8_t *at, const int8_t *b, const int32_t *bias, size_t m, size_t n, size_t k)"
)

#: Function name of the reference when it sits beside a kernel (kernel-test); the twin reuses the kernel's.
REFERENCE_FUNCTION = "merlin_matrix_reference_gemm"

#: The extents a constraint may be stated relative to: the unit's derived tile edge for the configuration.
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


# ---------------------------------------------------------------------------------------------------------
# The generated instruction header
# ---------------------------------------------------------------------------------------------------------

_OPERANDS = ("rd", "rs1", "rs2")


def _free_operands(enc: Any) -> tuple[str, ...]:
    """The register fields an instruction leaves to the caller (the ones its encoding does not fix)."""
    fixed = getattr(enc, "fixed", None)
    if not callable(fixed):
        return _OPERANDS
    return tuple(name for name in _OPERANDS if fixed(name) is None)


def _string_macro_body(enc: Any, free: tuple[str, ...]) -> str:
    """The ``.insn r`` text as C string-literal concatenation, with each free operand a macro argument."""
    marks = {name: f"\x00{name}\x00" for name in free}
    text = enc.insn_r(**marks)
    parts: list[str] = []
    rest = text
    while "\x00" in rest:
        lit, _, tail = rest.partition("\x00")
        name, _, rest = tail.partition("\x00")
        parts.append(f'"{lit}"' if lit else "")
        parts.append(name)
    if rest:
        parts.append(f'"{rest}"')
    return " ".join(p for p in parts if p)


def _macro_name(prefix: str, name: str) -> str:
    return f"{prefix}_" + "".join(c if c.isalnum() else "_" for c in str(name)).upper()


def emit_header(
    encodings: Mapping[str, Any],
    *,
    prefix: str,
    header: str,
    facts: Mapping[str, Any] | None = None,
    aliases: Mapping[str, str] | None = None,
    derivation_ok: bool = True,
    source_note: str = "",
) -> str:
    """The C header a hand-written kernel includes, written from the DERIVED encoding table.

    For each instruction ``NAME`` (the derivation's own name):

    * ``<P>_NAME(rd, rs1, rs2)`` -- the ``.insn r`` directive as a string literal, taking only the operands
      the encoding leaves free, each a string (``<P>_REG(5)``, ``"%[acc]"``). Paste it into ``asm volatile``.
    * ``<P>_NAME_R(rd, rs1, rs2)`` -- the whole ``asm volatile`` statement, operands as register NUMBERS.
    * ``<P>_NAME_MATCH`` / ``<P>_NAME_MASK`` when the derivation states them (the decoder's bits).

    ``facts`` are ``{NAME: value}`` or ``{NAME: (value, comment)}`` the package derives for the selected
    configuration (tile edge, reduction depth, operand-layout positions); each becomes ``#define <P>_NAME``.
    ``aliases`` maps an instruction to the name its author's own header gives the same word, recorded as a
    comment so a reader can find their way between the two. Refuses an unresolved derivation.
    """
    if not derivation_ok:
        raise ValueError("the encoding derivation did not cross-check; refusing to write a header from it")
    _c_identifier(prefix, "header prefix")
    guard = "".join(c if c.isalnum() else "_" for c in header).upper() + "_"
    lines = [
        f"/* {header} -- GENERATED by merlin.kernels.matrix_kernels at build time. Do not edit, do not copy.",
        " *",
        " * Every instruction word below comes from the unit's DERIVED encoding table (read from its RTL and",
        " * cross-checked), never from a hand-typed constant. Include it from a kernel in the package's kernels/",
        f' * directory: #include "{header}".',
    ]
    if source_note:
        lines += [" *", *(f" * {ln}" for ln in source_note.splitlines())]
    lines += [
        " *",
        f" * {prefix}_NAME(...)   the .insn directive as a string, free operands as string arguments",
        f" * {prefix}_NAME_R(...) the asm statement, free operands as register numbers",
        " */",
        f"#ifndef {guard}",
        f"#define {guard}",
        "",
        "#include <stddef.h>",
        "#include <stdint.h>",
        "",
        f"#define {prefix}_STR_(x) #x",
        f"#define {prefix}_STR(x) {prefix}_STR_(x)",
        "/* An operand slot: .insn r takes x-register names, and the field encodes the number (a vector, matrix",
        " * or tile register is spelled as the x-register with the same number). */",
        f'#define {prefix}_REG(n) "x" {prefix}_STR(n)',
        "",
    ]
    for key, value in sorted((facts or {}).items()):
        val, note = value if isinstance(value, tuple) else (value, "")
        spelled = f"{int(val):#x}" if int(val) >= 4096 else str(int(val))
        lines.append(f"#define {_macro_name(prefix, key)} {spelled}" + (f"  /* {note} */" if note else ""))
    if facts:
        lines.append("")
    for name in sorted(encodings):
        enc = encodings[name]
        free = _free_operands(enc)
        mac = _macro_name(prefix, name)
        args = ", ".join(free)
        mnemonic = str(getattr(enc, "mnemonic", name))
        alias = (aliases or {}).get(name)
        lines.append(
            f"/* {name}"
            + (f" ({mnemonic})" if mnemonic != name else "")
            + (f"; the same word as {alias} in the unit author's header" if alias else "")
            + " */"
        )
        lines.append(f"#define {mac}({args}) {_string_macro_body(enc, free)}")
        reg_args = ", ".join(f"{prefix}_REG({a})" for a in free)
        lines.append(f'#define {mac}_R({args}) __asm__ volatile({mac}({reg_args}) ::: "memory")')
        match, mask = getattr(enc, "match", None), getattr(enc, "mask", None)
        if isinstance(match, int) and isinstance(mask, int):
            lines.append(f"#define {mac}_MATCH {match:#010x}u")
            lines.append(f"#define {mac}_MASK {mask:#010x}u")
        lines.append("")
    lines += [
        "/* The kernel ABI (merlin.kernels.matrix_shim): define exactly this, under the name kernels.yaml gives. */",
        f"#define {prefix}_KERNEL(name) " + ABI_SIGNATURE.format(function="name"),
        "",
        f"#endif /* {guard} */",
        "",
    ]
    return "\n".join(lines)


def unit_header(provider: Any, *, unit: str, config: str, header: str, prefix: str) -> str:
    """The header for ``unit`` in ``config``, from what the package's ``matrix_lowering`` plugin derives.

    Uses only the plugin's contract: ``load_contract``/``derive_encodings`` (the encodings and their
    cross-check verdict), ``geometry`` (tile edge and operand alignment of this configuration), and the
    optional ``kernel_header_facts(unit=, config=)`` for anything else a kernel needs to know (a reduction
    depth, operand-layout positions), each a value the package derived, never one typed into Merlin.
    """
    contract = provider.load_contract(unit)
    derived = provider.derive_encodings(contract)
    tile_edge, align = provider.geometry(unit=unit, config=config)
    facts: dict[str, Any] = {
        "TILE_EDGE": (int(tile_edge), f"logical tile edge of {config}, derived from its configuration"),
        "ALIGN_BYTES": (int(align), "operand base alignment the shim guarantees (the datapath width)"),
    }
    extra = getattr(provider, "kernel_header_facts", None)
    if callable(extra):
        facts.update(dict(extra(unit=unit, config=config) or {}))
    declarations = getattr(contract, "declarations", None) or {}
    aliases = {str(k): str(v) for k, v in (declarations.get("crosscheck_pairs") or {}).items()}
    sources = getattr(derived, "sources", None) or {}
    note = f"unit {unit}, configuration {config}; derived from: " + ", ".join(
        f"{k}={Path(str(v)).name}" for k, v in sorted(sources.items())
    )
    return emit_header(
        derived.encodings,
        prefix=prefix,
        header=header,
        facts=facts,
        aliases=aliases,
        derivation_ok=bool(derived.ok),
        source_note=note,
    )


def package_root(support_target: str) -> Path:
    """The selected support package's root (where its ``kernels/`` live), resolved like its plugin."""
    from ..targetgen.plugins import resolve_support

    return Path(resolve_support(support_target).base)


def for_routing(routing: Any) -> SelectedKernel:
    """The kernel a ``MatrixRouting`` (``support_target``, ``unit``, ``kernel``) builds with, validated.

    With no kernel named, a package that cannot be resolved here still has its default -- its plugin's own
    build, which fails on its own terms if the package is missing -- but a NAMED kernel never falls back.
    """
    from ..targetgen.plugins import PluginError

    spec = getattr(routing, "kernel", None)
    target = getattr(routing, "support_target", None)
    try:
        root: Path | None = package_root(target) if target or spec else None
    except (PluginError, FileNotFoundError):
        if spec:
            raise
        root = None
    return resolve(spec, package_root=root, unit=routing.unit)


def build_kwargs(routing: Any, signatures: Mapping[str, Sequence[int]]) -> dict[str, Any]:
    """``{"kernel": KernelChoice}`` for the provider's ``build_object`` after the constraint check, or ``{}``
    when the package's generated kernel is selected -- so that build is exactly the one it always was."""
    chosen = for_routing(routing)
    if chosen.generated:
        return {}
    check_signatures(chosen, signatures, tile_edge=routing.tile_edge())
    provider = routing.provider()
    text = unit_header(
        provider, unit=routing.unit, config=routing.config, header=chosen.header, prefix=chosen.header_prefix
    )
    return {"kernel": KernelChoice(chosen, text)}


@dataclass(frozen=True)
class KernelChoice:
    """What a package's ``build_object(kernel=...)`` receives for a hand-written kernel: the selection and
    the header text written for this unit and configuration. The package calls :meth:`prepare`."""

    selected: SelectedKernel
    header_text: str

    def prepare(self, work: str | Path, *, scalar_tile: bool) -> PreparedKernel:
        return prepare(self.selected, work, header_text=self.header_text, scalar_tile=scalar_tile)

    def record(self) -> dict[str, Any]:
        return self.selected.record()


# ---------------------------------------------------------------------------------------------------------
# Preparing a build
# ---------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PreparedKernel:
    """What a package's ``build_object`` hands :func:`merlin.kernels.matrix_shim.emit_translation_unit`."""

    source: str
    function: str
    files: dict[str, str]
    record: dict[str, Any]


#: Where :func:`prepare` snapshots the kernel and its header, relative to the shim's source directory.
KERNEL_DIR = "kernel"


def prepare(kernel: SelectedKernel, work: str | Path, *, header_text: str, scalar_tile: bool) -> PreparedKernel:
    """Snapshot a hand-written kernel beside the shim source, with the generated header, and return the text
    the shim pastes as its kernel.

    The shipped build includes ``kernel/<file>`` (the snapshot), so the compiler's diagnostics name the
    kernel's own lines and the bytes compiled are the bytes hashed. Headers beside the kernel in its source
    directory are snapshotted too (a kernel may include its own helpers). The stand-in twin
    (``scalar_tile``) compiles :func:`reference_source` under the kernel's function name instead.
    """
    if kernel.generated:
        raise ValueError("prepare() is for a hand-written kernel; the generated one is the package's own path")
    work = Path(work)
    snap = work / KERNEL_DIR
    if snap.exists():
        shutil.rmtree(snap)
    snap.mkdir(parents=True)
    src = Path(str(kernel.entry.file))
    files: dict[str, str] = {}
    for helper in sorted(src.parent.glob("*.h")):
        if helper.name == kernel.header:
            raise KernelSelectionError(
                f"{helper} has the generated header's name {kernel.header!r}; remove it (the header is written at "
                "build time from the derived encodings)"
            )
        shutil.copy2(helper, snap / helper.name)
        files[helper.name] = _sha256(snap / helper.name)
    shutil.copy2(src, snap / src.name)
    files[src.name] = _sha256(snap / src.name)
    if files[src.name] != kernel.sha256:
        raise KernelSelectionError(f"{src} changed after it was selected (sha256 {kernel.sha256} -> {files[src.name]})")
    (snap / kernel.header).write_text(header_text, encoding="utf-8")
    files[kernel.header] = _sha256(snap / kernel.header)
    function = str(kernel.function)
    if scalar_tile:
        text = reference_source(function)
    else:
        text = (
            f"/* the selected matrix kernel {kernel.name!r}: {src} (sha256 {kernel.sha256}) */\n"
            f'#include "{KERNEL_DIR}/{src.name}"\n'
        )
    record = {
        **kernel.record(),
        "snapshot": {k: v for k, v in sorted(files.items())},
        "compiled": "reference" if scalar_tile else "kernel",
    }
    return PreparedKernel(source=text, function=function, files=files, record=record)
