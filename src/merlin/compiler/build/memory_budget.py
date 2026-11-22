"""Derive application space from physical regions and explicit reserved ranges.

The caller supplies linker-measured code/runtime/stack ranges and all persistent
allocations. Zero-byte entries explicitly record absent categories. This is a
capacity boundary for Merlin allocation/lifetimes, not a device allocator.
"""
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from merlin.target.soc import MemoryRegion, alignment, natural, text_fact
from merlin.target.system import MemoryDomain

REQUIRED_KINDS = frozenset(('code', 'runtime', 'stack', 'weights', 'state', 'kv_cache'))


@dataclass(frozen=True)
class Reservation:
    id: str
    kind: str
    region: str
    base: int
    size_bytes: int
    evidence: str

    def __post_init__(self):
        natural(self.base, 'reservation base')
        natural(self.size_bytes, 'reservation size')
        for name in ('id', 'kind', 'region', 'evidence'):
            text_fact(getattr(self, name), name)

    @property
    def end(self):
        return self.base + self.size_bytes


@dataclass(frozen=True)
class ApplicationBudget:
    region: MemoryRegion
    reservations: tuple[Reservation, ...]
    free_ranges: tuple[tuple[int, int], ...]
    alignment_bytes: int

    @property
    def capacity_bytes(self):
        return sum(end - base for base, end in self.free_ranges)

    def domains(self):
        """Separate capacity domains; bind their bases using ``free_ranges``.

        These records do not install physical placements in a selected topology.
        The application must retain each range/domain association.
        """
        return tuple(MemoryDomain(f'{self.region.id}.{i}', end - base, self.alignment_bytes)
                     for i, (base, end) in enumerate(self.free_ranges))


def derive_budget(region, reservations, *, alignment_bytes):
    """Subtract the occupied union, rounding free ranges inward.

    Linker categories may overlap or describe the same backing. Preserve their
    individual evidence, but charge each physical byte only once.
    """
    alignment(alignment_bytes)
    if type(region) is not MemoryRegion:
        raise ValueError('memory budgets require a typed immutable region')
    reservations = tuple(reservations)
    if any(type(r) is not Reservation for r in reservations):
        raise ValueError('memory budgets require typed immutable reservations')
    if len({r.id for r in reservations}) != len(reservations):
        raise ValueError('duplicate reservation identity')
    if REQUIRED_KINDS - {r.kind for r in reservations}:
        raise ValueError('missing explicit code/runtime/stack/weights/state/KV reservations')
    for row in reservations:
        if row.region != region.id or row.base < region.base or row.end > region.end:
            raise ValueError('reservation lies outside its memory region')
    used = []
    for row in sorted((r for r in reservations if r.size_bytes), key=lambda r: r.base):
        if used and row.base <= used[-1][1]:
            used[-1] = (used[-1][0], max(used[-1][1], row.end))
        else:
            used.append((row.base, row.end))
    free = []
    cursor = region.base
    for end, following in used + [(region.end, region.end)]:
        start = (cursor + alignment_bytes - 1) & -alignment_bytes
        stop = end & -alignment_bytes
        if start < stop:
            free.append((start, stop))
        cursor = following
    return ApplicationBudget(region, reservations, tuple(free), alignment_bytes)


def linked_reservation(region, symbols, *, identity, kind, start, end, evidence):
    """Use explicit retained linker bounds, including NOLOAD/symbol-only stack.

    No default size is substituted for missing symbols. Use final linked symbols
    rather than summing sections, which misses padding and symbolic reservations.
    """
    base, limit = symbols[start], symbols[end]
    return Reservation(identity, kind, region.id, base, limit - base, evidence)


@dataclass(frozen=True)
class LinkedReservation:
    """Caller-declared retained symbol pair, including symbol-only reservations."""

    id: str
    kind: str
    start: str
    end: str

    def __post_init__(self):
        for name in ('id', 'kind', 'start', 'end'):
            text_fact(getattr(self, name), name)


@dataclass(frozen=True)
class LinkedMemoryBudget:
    """Measured final ELF custody, not a loader or execution certificate.

    Re-read before publishing addresses. Any changed bytes require replanning,
    even if the occupied spans are unchanged. The application owns synchronization
    with linking/loading after this check; this record does not lock its file.
    """

    path: Path
    elf_sha256: str
    budget: ApplicationBudget
    bounds: tuple[LinkedReservation, ...]
    reservations: tuple[Reservation, ...]
    nm_tool: str
    timeout_s: int

    def current_budget(self):
        current = measure_linked_budget(
            self.budget.region, self.path, self.bounds, reservations=self.reservations,
            alignment_bytes=self.budget.alignment_bytes, nm_tool=self.nm_tool, timeout_s=self.timeout_s)
        if current.elf_sha256 != self.elf_sha256 or current.budget != self.budget:
            raise ValueError('linked ELF or reservations changed; replan')
        return current.budget


def measure_linked_budget(region, elf, bounds, *, reservations=(), alignment_bytes, nm_tool, timeout_s):
    """Derive from actual final-link bytes using caller-selected GNU/LLVM nm.

    Measure a private copy of the bytes hashed here, so a concurrent relink cannot
    mix one ELF identity with another symbol table. Only explicit symbol pairs
    are charged; caller declarations must cover every occupied range. Additional
    persistent/status/heap reservations use the existing occupied-union rule.
    Missing/ambiguous symbols, tool errors and timeouts refuse without defaults.
    """
    natural(timeout_s, 'symbol measurement timeout', positive=True)
    text_fact(nm_tool, 'nm tool')
    bounds, reservations = tuple(bounds), tuple(reservations)
    if not bounds or any(type(row) is not LinkedReservation for row in bounds):
        raise ValueError('link bounds require nonempty typed immutable symbol pairs')
    path = Path(elf).absolute()
    data = path.read_bytes()
    if not data.startswith(b'\x7fELF'):
        raise ValueError('linked memory measurement requires an ELF')
    # Symbol values are physical addresses only for a fixed-address final link.
    # Relocatable objects expose section offsets; ET_DYN needs a load bias that
    # this boundary deliberately does not infer from the supplied region.
    if len(data) < 16 or data[4] not in (1, 2) or data[5] not in (1, 2) or data[6] != 1:
        raise ValueError('linked memory measurement requires a valid ELF header')
    byteorder = 'little' if data[5] == 1 else 'big'
    if len(data) < (52 if data[4] == 1 else 64):
        raise ValueError('linked memory measurement requires a complete ELF header')
    if int.from_bytes(data[16:18], byteorder) != 2:
        raise ValueError('linked memory measurement requires a fixed-address final ELF (ET_EXEC)')
    digest = sha256(data).hexdigest()
    required = {name for row in bounds for name in (row.start, row.end)}
    with TemporaryDirectory(prefix='merlin-linked-memory-') as tmp:
        snapshot = Path(tmp) / 'image.elf'
        snapshot.write_bytes(data)
        output = subprocess.check_output(
            [nm_tool, '--format=posix', '--defined-only', str(snapshot)],
            text=True, timeout=timeout_s)
    symbols = {}
    for line in output.splitlines():
        fields = line.split()
        if not fields or fields[0] not in required:
            continue
        if len(fields) < 3 or fields[0] in symbols:
            raise ValueError('ambiguous linked reservation symbol')
        symbols[fields[0]] = int(fields[2], 16)
    if required - symbols.keys():
        raise ValueError(f'missing linked reservation symbols: {sorted(required - symbols.keys())}')
    linked = tuple(linked_reservation(
        region, symbols, identity=row.id, kind=row.kind, start=row.start, end=row.end,
        evidence=f'ELF sha256:{digest}; symbols {row.start}:{row.end}') for row in bounds)
    budget = derive_budget(region, linked + reservations, alignment_bytes=alignment_bytes)
    return LinkedMemoryBudget(path, digest, budget, bounds, reservations, nm_tool, timeout_s)
