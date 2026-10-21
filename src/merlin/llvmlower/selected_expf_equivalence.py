"""Explicit, implementation-pinned permission for one expf input equivalence class.

This helper selects no compiler feature. Its caller retains the live grant through
compilation and the actual link, uses the returned absolute-archive linker argv,
and checks the freshly generated GNU map afterwards. No saved report restores a
grant. The ordinary closed-region/errno/fenv guards remain separate obligations.
"""

from __future__ import annotations

import contextlib
import contextvars
import hashlib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

SELECTED_LIBM_SHA256 = "38965f2df6e94d3389596fcf414caf37040294fc0f43f3afa3e4369acb837934"
PROTOCOL = "merlin.selected_expf_finite_underflow_equivalence.v1"
_SECTIONS = {
    ".text.expf": "libm_a-wf_exp.o",
    ".text.finitef": "libm_a-sf_finite.o",
    ".text.__ieee754_expf": "libm_a-ef_exp.o",
    ".text.with_errnof": "libm_a-math_errf.o",
    ".text.xflowf": "libm_a-math_errf.o",
    ".text.__math_uflowf": "libm_a-math_errf.o",
}


class ExpfEquivalenceError(ValueError):
    """The explicitly selected callee is absent, changed, or unsupported."""


def _sha(path: Path, *, limit: int) -> str:
    if not path.is_file() or path.stat().st_size > limit:
        raise ExpfEquivalenceError(f"missing or oversized selected artifact: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class ExpfUnderflowEquivalence:
    """A live selected archive, with fixed binary32 and sticky-effect facts.

    The finite input words form an unsigned inclusive interval. The first class
    occurrence still invokes the original expf. These facts do not permit dropping
    its first UF/NX/errno effects, changing any source arithmetic, or merging -inf.
    """

    library_path: Path

    @property
    def minimum_input_bits(self) -> int:
        return 0xC2CFF1B6

    @property
    def maximum_input_bits(self) -> int:
        return 0xFF7FFFFF

    @property
    def result_bits(self) -> int:
        return 0

    @property
    def raised_fflags(self) -> int:
        return 3  # RISC-V UF | NX; all five legal FRM modes, unchanged FRM.

    @property
    def errno_value(self) -> int:
        return 34

    @property
    def library_sha256(self) -> str:
        return SELECTED_LIBM_SHA256

    def revalidate(self) -> None:
        if type(self) is not ExpfUnderflowEquivalence or not isinstance(self.library_path, Path):
            raise ExpfEquivalenceError("expected the original typed expf equivalence grant")
        if not self.library_path.is_absolute() or self.library_path.resolve() != self.library_path:
            raise ExpfEquivalenceError("selected archive path must remain canonical and absolute")
        if _sha(self.library_path, limit=64 << 20) != SELECTED_LIBM_SHA256:
            raise ExpfEquivalenceError("selected expf library bytes changed or are unsupported")


def select_expf_underflow_equivalence(library: Path) -> ExpfUnderflowEquivalence:
    """Explicitly select the reviewed archive; there is no library discovery/default."""
    selected = ExpfUnderflowEquivalence(Path(library).resolve())
    selected.revalidate()
    return selected


_CURRENT: contextvars.ContextVar[ExpfUnderflowEquivalence | None] = contextvars.ContextVar(
    "merlin_selected_expf_underflow_equivalence", default=None
)


def _checked(grant: ExpfUnderflowEquivalence) -> ExpfUnderflowEquivalence:
    if type(grant) is not ExpfUnderflowEquivalence:
        raise ExpfEquivalenceError("expected the original typed expf equivalence grant")
    grant.revalidate()
    return grant


@contextlib.contextmanager
def selected_expf_underflow_scope(grant: ExpfUnderflowEquivalence) -> Iterator[None]:
    """Compiler-only explicit scope; the caller must retain the grant through link."""
    token = _CURRENT.set(_checked(grant))
    try:
        yield
        _checked(grant)
    finally:
        _CURRENT.reset(token)


def current_expf_underflow_equivalence() -> ExpfUnderflowEquivalence | None:
    """Reopen the current archive at pipeline construction, or return None."""
    grant = _CURRENT.get()
    return _checked(grant) if grant is not None else None


def bind_expf_underflow_link(
    grant: ExpfUnderflowEquivalence, argv: Sequence[str], *, map_path: Path
) -> tuple[str, ...]:
    """Replace one ordinary -lm with the exact archive and request a fresh map.

    Caller executes this argv immediately. Library search paths may remain for
    unrelated libraries; alternate libm entries, symbol wrapping/definitions,
    response files, shared links and existing map overrides are unsupported.
    """
    _checked(grant)
    args = tuple(str(a) for a in argv)
    if not args or args.count("-lm") != 1:
        raise ExpfEquivalenceError("expected exactly one ordinary -lm in the actual link")
    forbidden = ("--wrap", "--defsym", "--allow-multiple-definition", "-Map", "--print-map")
    for arg in args:
        if arg.startswith("@") or arg in {"-shared", "-pie", "-M"}:
            raise ExpfEquivalenceError("unsupported indirect or dynamic selected-callee link")
        if any(part in arg for part in forbidden):
            raise ExpfEquivalenceError("unsupported symbol or linker-map override")
        if arg != "-lm" and (arg.startswith("-lm") or Path(arg).name in {"libm.a", "libm.so"}):
            raise ExpfEquivalenceError("duplicate or alternate libm selection")
    if [a for a in args if a.startswith("-march=")] != ["-march=rv64gc"] or [
        a for a in args if a.startswith("-mabi=")
    ] != ["-mabi=lp64d"]:
        raise ExpfEquivalenceError("selected expf equivalence requires rv64gc/lp64d")
    output_map = Path(map_path).resolve()
    if "," in str(output_map) or any(c.isspace() for c in str(output_map)):
        raise ExpfEquivalenceError("unsupported linker-map path")
    if output_map.exists():
        raise ExpfEquivalenceError("selected-callee link requires a fresh map path")
    return tuple(str(grant.library_path) if a == "-lm" else a for a in args) + ("-Wl,-Map=" + str(output_map),)


def verify_expf_underflow_link(grant: ExpfUnderflowEquivalence, *, elf: Path, map_path: Path) -> dict[str, object]:
    """Check fresh link evidence; returns diagnostics, never qualification authority.

    GNU map section ownership detects an object/library overriding any function
    in the reviewed path. The caller supplies the map produced by its actual
    returned link invocation, and retains its ordinary invocation/output custody.
    """
    _checked(grant)
    elf = Path(elf)
    with elf.open("rb") as stream:
        header = stream.read(20)
    if len(header) != 20 or header[:6] != b"\x7fELF\x02\x01" or header[18:20] != b"\xf3\x00":
        raise ExpfEquivalenceError("selected-callee output is not little-endian ELF64 RISC-V")
    map_path = Path(map_path)
    map_sha = _sha(map_path, limit=16 << 20)
    owners: dict[str, list[str]] = {section: [] for section in _SECTIONS}
    pending = None
    for line in map_path.read_text().splitlines():
        words = line.split()
        if not words:
            continue
        if words[0].startswith(".text"):
            pending = words[0] if words[0] in _SECTIONS else None
            words = words[1:]
        if pending and len(words) >= 3 and words[0].startswith("0x") and words[1].startswith("0x"):
            if int(words[1], 16):
                owners[pending].append(" ".join(words[2:]))
            pending = None
    for section, member in _SECTIONS.items():
        expected = f"{grant.library_path}({member})"
        if owners[section] != [expected]:
            raise ExpfEquivalenceError(f"selected expf path has missing/foreign owner for {section}")
    _checked(grant)
    return {
        "protocol": PROTOCOL,
        "qualification": "UNQUALIFIED",
        "library_path": str(grant.library_path),
        "library_sha256": SELECTED_LIBM_SHA256,
        "elf_sha256": _sha(elf, limit=512 << 20),
        "linker_map_sha256": map_sha,
        "sections": {key: value[0] for key, value in owners.items()},
    }
