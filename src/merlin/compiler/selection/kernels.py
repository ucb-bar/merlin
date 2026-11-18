"""Select declared matrix kernels and enforce their shape/ABI constraints.

Ported from OPU-VME 80e00803d; hardware values remain caller-supplied.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---- kernels -----------------------------------------------------------------------------------

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
                twin="merlin reference (merlin.analysis.kernels.matrix_kernels.reference_source)",
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
/* GENERATED by merlin.analysis.kernels.matrix_kernels -- the reference kernel, the ABI's definition as a loop.
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


# ---- capacity ----------------------------------------------------------------------------------
# Operand and accumulator capacity of a target's matrix unit, and the tile that fits it.
#
# Element widths come from the format registry (``_dtype_bits``); store sizes come from the target's RTL
# facts (``_operand_store_bytes``, ``_accumulator_capacity_elems``); ``capacity_fit`` is the obligation a
# mesh layer is checked against and ``_capacity_fit_tile`` the blocking that discharges it.


def _dtype_bits(tok: str | None) -> int:
    """Width in BITS of one stored element of an operand dtype token. Floats come from the quant-format
    registry's own layout; integer tokens (``i8``/``int8``/``i32``) from their trailing digit run.
    Defaults to 8 (int8-class) when the token says nothing.

    Not a digit scrape over the whole token: ``fp8_e4m3`` contains three digit runs, and concatenating
    them yields 843, which made an fp8 target's on-chip capacity read as 624 elements instead of 65536.
    """
    if not tok:
        return 8
    t = str(tok)
    try:
        from merlin.common.fp8_formats import storage_bits

        return storage_bits(t)
    except Exception:  # noqa: BLE001 — not a registered float format: try the integer spelling
        pass
    tail = ""
    for ch in reversed(t):  # trailing digit run, structurally (no regex)
        if not ch.isdigit():
            break
        tail = ch + tail
    return int(tail) if tail else 8


def _dtype_bytes(tok: str | None) -> int:
    """Byte width of an operand dtype token, rounded up (a sub-byte format still occupies a byte when
    something must address it individually). Prefer :func:`_dtype_bits` where packing matters."""
    return max(1, (_dtype_bits(tok) + 7) // 8)


def _operand_store_bytes(target: str) -> int | None:
    """On-chip OPERAND-store capacity in bytes for ``target``, derived from that target's own RTL. Tries
    three sources in descending order of independence, and returns None (never a default) when none of
    them can answer.

    1. mlc's discovered memory MAP — fully structural: mlc names the representative operand bank and this
       sums its sibling banks' ``depth x row_bytes``. Nothing is declared anywhere.
    2. The DESCRIPTOR's ``rtl.operand_store`` bank-name prefix, summed over mlc's discovered memory list.
       mlc discovers every SRAM on some targets and still refuses to classify them ("no memory map
       discovered"), which left the ``capacity_fit`` obligation decidable on one target and undecidable
       on another. Here the descriptor supplies only the LABEL; the bytes are still read out of the RTL.
       Deliberately not a guess: picking the largest discovered memory would select the INSTRUCTION
       memory on a device whose IMEM (128 KiB) is larger than its operand register file (64 KiB).
    3. The extracted RTL facts, through :func:`~merlin.target.capabilities.address_space.operand_store`: the
       narrowest-row store, accepted only when it was CLASSIFIED against a second store of a different
       width (``BY_WIDTH``). A sole store is not accepted here, for the same reason as tier 2 refuses to
       guess: a device's facts can name one extracted memory out of many, and ``capacity_fit`` would then
       become decidable against an operand store nothing validated. This tier used to look the store up
       by the name ``"scratchpad"``; every target whose facts use that name has two stores of distinct
       widths, so the answers are the same there and no longer depend on a spelling.
    """
    from merlin.target.rtl.extraction import mlc_discovery as _merlin_mlc_discovery_1
    from merlin.target.rtl.extraction import mlc_toolchain as _mb

    try:
        caps = _merlin_mlc_discovery_1.discovered_capacities(target)
        if caps and caps.get("operand_bytes"):
            return int(caps["operand_bytes"])
    except Exception:  # noqa: BLE001 — mlc unavailable → try the declared group next
        pass
    try:
        from merlin.target.registry.descriptor_loading import experiment_for

        prefix = getattr(experiment_for(target), "operand_store", None)
        if prefix:
            banks = _merlin_mlc_discovery_1.discovered_memories(target) or []
            total = sum(
                int(b["depth"]) * int(b["row_bytes"]) for b in banks if str(b.get("name", "")).startswith(prefix)
            )
            if total:
                return total
    except Exception:  # noqa: BLE001 — no descriptor / no discovery → fall through to facts
        pass
    try:
        from merlin.target.capabilities import address_space as _as

        resolved = _as.operand_store(_as.derive_address_space(target))
    except Exception:  # noqa: BLE001 — no facts bundle → capacity unknown, caller falls back
        return None
    if resolved.store is None or resolved.basis != _as.BY_WIDTH:
        return None
    nbytes = resolved.store.nbytes
    return int(nbytes) if nbytes else None


def _operand_store_capacity_elems(target: str, operand_dtype: str | None) -> int | None:
    """The operand store's capacity in ELEMENTS of ``operand_dtype`` — counted in bits, so a sub-byte
    packed format (fp4/fp6) is not silently rounded up to a byte apiece."""
    nbytes = _operand_store_bytes(target)
    return (int(nbytes) * 8) // max(1, _dtype_bits(operand_dtype)) if nbytes else None


def _accumulator_capacity_elems(target: str, accum_dtype: str | None) -> int | None:
    """The ACCUMULATOR's capacity in elements of ``accum_dtype``, or None when it cannot be derived.

    Separate from the operand store because it is a separate, much smaller SRAM, and it binds a
    different quantity: the operand store bounds the INPUT working set (weight tile + activation tile),
    the accumulator bounds the OUTPUT tile that the contraction accumulates into. A layer can sit
    comfortably inside a 256 KiB scratchpad and still overrun a 64 KiB accumulator, because the output
    grows as M*N while the operands grow as K*(M+N).

    Measured: the whole-model capsule routed four layers whose operands fit the scratchpad by a wide
    margin -- (345,32)@(32,256) and (96,64)@(64,512) -- so nothing tiled them, and every one aborted the
    simulator with ``vector::_M_range_check: __n (which is 1024) >= this->size() (which is 1024)``: the
    1024 accumulator rows (65536 B / (16 elems * 4 B)) addressed one past the end. The tiles that DID
    run all passed, so this cost four unmeasurable layers rather than four wrong ones -- the capsule
    failed for want of a measurement, not for a wrong answer.
    """
    from merlin.target.rtl.extraction import mlc_discovery as _merlin_mlc_discovery_2
    from merlin.target.rtl.extraction import mlc_toolchain as _mb

    try:
        caps = _merlin_mlc_discovery_2.discovered_capacities(target) or {}
    except Exception:  # noqa: BLE001 -- mlc unavailable => undecidable, never assumed
        return None
    nbytes = caps.get("accumulator_bytes")
    return (int(nbytes) * 8) // max(1, _dtype_bits(accum_dtype)) if nbytes else None


def declared_primitive_tile(package: str | Path | None) -> tuple[int, int, int] | None:
    """The (m, k, n) tile a backend DECLARES it implements, from its ``manifest.yaml``, or ``None``.

    Why this exists alongside the capacity derivation: they answer different questions and only one of
    them is always answerable. Capacity asks "how much fits on chip", which needs a classified operand
    store -- and on a device whose SRAMs cannot be classified there is nothing to compute, so the whole
    residency tiler becomes unreachable and an oversized layer is handed to a backend that cannot take
    it. The declared tile asks "what did you build", which the backend always knows.

    A backend that declares one is promising to lower EXACTLY that shape; the loop nest over larger
    extents is then the runtime's, and every result produced that way is attributed as such (see the
    ``discharged_by`` block in :func:`run_matmul_on_mesh`) -- never as evidence the backend generalizes.
    Declaring nothing is legal and keeps the previous behaviour.
    """
    if not package:
        return None
    try:
        import yaml as _yaml

        man = Path(package) / "manifest.yaml"
        if not man.is_file():
            return None
        t = (_yaml.safe_load(man.read_text(encoding="utf-8")) or {}).get("primitive_tile") or {}
        m, k, n = int(t.get("m") or 0), int(t.get("k") or 0), int(t.get("n") or 0)
        return (m, k, n) if m > 0 and k > 0 and n > 0 else None
    except Exception:  # noqa: BLE001 -- an unreadable/absent declaration is simply no declaration
        return None


def _capacity_fit_tile(
    M: int, K: int, N: int, D: int, cap_elems: int, acc_elems: int | None = None
) -> tuple[int, int, int, int]:
    """Shrink a matmul (M,K,N) to the largest D-aligned tile that fits on chip, and return
    (mt,kt,nt,n_tiles) where n_tiles is how many such tiles cover the layer.

    TWO stores bound the tile, and they bind different dimensions:
      * the OPERAND store holds the weight tile K·N plus the activation tile M·K (``cap_elems``);
      * the ACCUMULATOR holds the OUTPUT tile M·N (``acc_elems``), when it can be derived.

    Modelling only the first is what let four layers through whose operands fit the scratchpad easily
    while their output overran the accumulator. Output residency is the constraint that can force the
    ROW dim to split: with nt already at the tile edge, shrinking K does nothing for M·N, so M is the
    only dimension left to give. That is why M is no longer unconditionally whole -- but only when an
    accumulator bound is known AND the output actually exceeds it, so a caller that passes no
    ``acc_elems`` gets exactly the previous extents. Pure arithmetic, target-agnostic."""

    def _half(x):
        return max(D, (x // 2 // D) * D or D)

    def _operands_fit(mt, kt, nt):
        return kt * nt + mt * kt <= cap_elems

    def _output_fits(mt, nt):
        return (not acc_elems) or mt * nt <= acc_elems

    mt, kt, nt = M, K, N
    while not (_operands_fit(mt, kt, nt) and _output_fits(mt, nt)):
        if not _output_fits(mt, nt):
            # the OUTPUT is what does not fit: only M or N relieve it (K does not appear in M·N)
            if nt >= mt and nt > D:
                nt = _half(nt)
            elif mt > D:
                mt = _half(mt)
            elif nt > D:
                nt = _half(nt)
            else:
                break  # already a single tile edge: nothing left to give
        elif kt > D or nt > D:
            if nt >= kt and nt > D:
                nt = _half(nt)
            elif kt > D:
                kt = _half(kt)
            else:
                nt = _half(nt)
        else:
            break
    import math

    n_tiles = math.ceil(M / mt) * math.ceil(K / kt) * math.ceil(N / nt)
    return mt, kt, nt, n_tiles


def capacity_fit(
    target: str, m: int, k: int, n: int, operand_dtype: str | None, tile_dim: int, accum_dtype: str | None = None
) -> dict:
    """Evaluate the ``capacity_fit`` CONTRACT OBLIGATION for one contraction on ``target``.

    ``capacity_fit`` is not a heuristic of ours — it is a predicate the interface contract already
    names (``xdsl_dialects.contract.KNOWN_PREDICATES``) and already demands, alongside
    ``rhs_immutable``, on every ``resident_packed_tensor`` requirement. What was missing is anyone
    evaluating it. A backend that assumes unbounded on-chip storage therefore did not fail its
    contract; it segfaulted the simulator three layers away, and the result surfaced as a generic
    "the oracle returned nothing".

    Measured on the gemmini backend under grade: its lowering tiles the ITERATION space into DIM x DIM
    tiles correctly, but addresses every one of the ``kt*nt`` weight tiles as simultaneously resident
    (``b_spad = weight_base + (kk*nt + nn)*DIM``). At 512x512 that is 32*32*16 = 16384 scratchpad rows
    against a 16384-row scratchpad, and spike aborts with
    ``vector::_M_range_check: __n (which is 16384) >= this->size() (which is 16384)``. Compute tiling
    without a residency loop above it.

    WHAT ``required_elems`` MODELS, stated because it is a modelling choice and not a property of the
    layer: it is the working set of a lowering that keeps the contraction's OPERANDS RESIDENT -- the
    whole weight tile plus the activation tile at once. A backend that instead STREAMS, DMA-ing one
    mesh tile at a time from DRAM, has a resident set of a few tiles no matter how large the layer, and
    satisfies the obligation at every extent. Both kinds are under grade here: one target's generated
    backend addresses all kt*nt weight tiles as simultaneously resident, and the other's issues a DMA
    per 32x32 tile. So a ``holds: False`` predicts a decline only for the resident kind, which is why
    the backend is CHARGED only when the dispatch actually declines -- the predicate alone never
    convicts. Measured: an extent this predicate called too large for one target's 64 KiB register file
    ran unblocked on its cosim and returned output, because that backend streams.

    Returns ``{"holds", "required_elems", "capacity_elems", "obligation", "assumes"}``; ``holds`` is
    None when the target declares no capacity (unknown, never assumed true).
    """
    cap = _operand_store_capacity_elems(target, operand_dtype)
    need = int(k) * int(n) + int(m) * int(k)  # weight tile + activation tile, both resident
    # The OUTPUT must also be resident, in the accumulator -- a separate and much smaller store. It is
    # reported and conjoined separately because it fails on different layers: the output grows as M*N
    # while the operands grow as K*(M+N), so a wide-output layer overruns the accumulator while its
    # operands sit well inside the scratchpad. Undecidable (None) stays undecidable, never assumed true.
    acc = _accumulator_capacity_elems(target, accum_dtype)
    out_need = int(m) * int(n)
    op_holds = None if not cap else bool(need <= cap)
    acc_holds = None if not acc else bool(out_need <= acc)
    # Three-valued conjunction, fail-closed: a violated term convicts, but an UNEVALUATED one leaves the
    # obligation undecidable -- never "holds". Reporting True because the one term we could evaluate
    # passed would claim an obligation nobody checked.
    _terms = (op_holds, acc_holds)
    holds = False if False in _terms else None if None in _terms else True
    return {
        "obligation": "capacity_fit",
        "holds": holds,
        "required_elems": need,
        "capacity_elems": cap,
        "output_elems": out_need,
        "accumulator_capacity_elems": acc,
        "operands_hold": op_holds,
        "output_holds": acc_holds,
        "tile_dim": int(tile_dim or 0),
        "assumes": "operands resident (a streaming lowering needs far less and always fits) "
        "and the output tile resident in the accumulator",
    }


# ---- mesh --------------------------------------------------------------------------------------
# Existing package, numerical binding and tile-request selection for mesh consumers.


def _default_oot_package(target: str) -> str | None:
    """The conventional OOT backend package for ``target``, or None when the target ships no default (a
    bespoke-layout target must name it via ``--package``). Prefers the agent-submission package
    (``.../targets/<target>/agent_spec_v1_mlir_oot`` — the SAME default ``compile_oot`` resolves, so on-mesh
    tile verification reuses it verbatim) and falls back to a hand-curated REFERENCE backend package
    (``.../reference_v0``) when a target ships one instead of an agent submission (e.g. a SIMT core whose
    deterministic reference lowering is the package). Target-agnostic — both are directory conventions, not
    target literals."""
    from merlin.common.artifacts import artifacts_dir

    base = artifacts_dir() / "targets" / target
    # Selected by KIND, not merely by the presence of a manifest. ``out/artifacts/targets/<target>/`` also
    # holds CODEGEN packages (schedules/knobs/dialects, e.g. a hand-curated ``hand_v0``), whose manifest is
    # a different artifact entirely. Returning one of those as an OOT backend does not fail cleanly: it
    # gets as far as the package loader and dies on "manifest schema violation: 'artifact_type' is a
    # required property", which reads like a corrupt backend rather than a directory that was never one.
    for pkg_id in ("agent_spec_v1_mlir_oot", "reference_v0"):
        cand = base / pkg_id
        mf = cand / "manifest.yaml"
        if not mf.is_file():
            continue
        try:
            from merlin.common.record_formats import load_yaml

            if not str((load_yaml(mf) or {}).get("artifact_type", "")).strip():
                continue  # not an OOT backend manifest — keep looking
        except Exception:  # noqa: BLE001 — unreadable manifest: let the loader report it
            pass
        return str(cand)
    return None


def _mesh_tile_binding(
    target: str,
    operand_dtype: str | None,
    accum_dtype: str | None,
    requant_output_dtype: str | None = None,
    *,
    numeric_policy: dict | None = None,
):
    """A ``corpus_spec.CorpusBinding`` for synthesizing a single systolic mesh tile of ``target``, with the
    operand/accumulate datapath pinned to what the routed op actually needs (derived, never assumed). The
    tile dim and instruction-class deriver come from the selected target contract. The caller supplies
    the complete numerical policy as declared assumptions, not hardware-verified evidence.

    ``requant_output_dtype`` (the narrow dtype an ``acc_scale`` epilogue requants the i32 accumulator to,
    e.g. i8) is passed only when the caller needs a requant handoff — an int8 matmul chain commits each
    layer's accumulator back to i8 so it can feed the next mesh layer.

    Operation dtype pins overlay that policy without dropping subnormal handling, comparison, scaling,
    or future numerical fields. Corpus-generation controls are excluded; requantization is requested
    by this operation, never inherited from a profile's corpus defaults. No profile path is discovered
    here. A floating datapath without a declared policy refuses explicitly."""
    from types import SimpleNamespace

    from merlin.development.targetgen.corpus import binding as CS
    from merlin.target.simulators.oracle_policy import selected_sim_via as _bespoke_sim_via

    te = SimpleNamespace(target=target, sim_via=_bespoke_sim_via(target))
    datapath = CS.profile_datapath({"datapath": numeric_policy}, numeric_only=True)
    if operand_dtype:
        datapath["operand_dtype"] = operand_dtype
    if accum_dtype:
        datapath["accum_dtype"] = accum_dtype
    if requant_output_dtype:
        datapath["requant_output_dtype"] = requant_output_dtype
    binding = CS.derive_binding(te, datapath)
    if not binding.integer and numeric_policy is None:
        raise ValueError("floating mesh execution requires an explicit declared numeric_policy")
    return binding


def tile_builder_op(demand) -> tuple[str | None, str]:
    """``(builder name, how it was chosen)`` for a mesh demand, or ``(None, "unsynthesizable")``.

    A tile is synthesized from the demand's EXTENT, never from how the op is spelled -- the certifier
    below reads only ``m/k/n`` and the tile binding. So an op the router placed on the mesh AS a
    contraction, carrying a shape, has a matmul tile whatever it is called.

    Matching on the name alone is the name-vs-family mismatch :attr:`routing.Demand.family` exists for:
    torch-MLIR spells a convolution ``convolution_im2col_matmul`` while a contract spells the unit's ops
    ``[matmul]``. Measured: ``M2_microvit_gemmini`` routed 3 convolutions to the mesh under that
    spelling and ran the whole model bit-exact with 12 of 12 certifiable tiles passing on RTL; the three
    were counted ``n_unsynthesizable`` and failed the capsule at its cert tier, for a builder table the
    submission does not own.

    ``family`` is UNKNOWN-safe and never widens, so an op with no derivable family is still matched by
    name exactly as before, and a family match is recorded so the weaker provenance stays visible.
    """
    from merlin.development.targetgen.corpus import binding as CS
    from merlin.development.targetgen.corpus import model_builders as _merlin_model_builders_3

    name = "matmul" if demand.op in ("matmul", "linear") else demand.op
    if name in _merlin_model_builders_3.BUILDERS:
        return name, "name"
    if demand.family == "contraction" and demand.has_shape:
        return "matmul", "family"
    return None, "unsynthesizable"


for _legacy_function in (
    _default_oot_package,
    _mesh_tile_binding,
    tile_builder_op,
):
    _legacy_function.__module__ = "merlin.runtime.dispatch.mesh"
del _legacy_function
