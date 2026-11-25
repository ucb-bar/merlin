"""Physical SoC facts supplied by the outer application, without allocation policy.

Clock rates are selected configuration facts, not timing measurements. Device
ABI, loading and completion declarations do not grant execution admission.
"""
from dataclasses import dataclass


def text_fact(value, name):
    if type(value) is not str or not value:
        raise ValueError(f'{name} requires a nonempty immutable string')


def natural(value, name, *, positive=False):
    if type(value) is not int or value < int(positive):
        raise ValueError(f"{name} requires an explicit {'positive' if positive else 'nonnegative'} integer")


def alignment(value):
    natural(value, 'alignment', positive=True)
    if value & (value - 1):
        raise ValueError('alignment must be a power of two')


def _declared(rows, identity, kind):
    for row in rows:
        if row.id == identity:
            return row
    raise ValueError(f'SoC declares no {kind} {identity!r}')


@dataclass(frozen=True)
class ClockDomain:
    id: str
    frequency_hz: int | None
    evidence: str

    def __post_init__(self):
        text_fact(self.id, 'clock identity')
        text_fact(self.evidence, 'clock evidence')
        if self.frequency_hz is not None:
            natural(self.frequency_hz, 'frequency', positive=True)


@dataclass(frozen=True)
class SoCResource:
    id: str
    clock_domain: str | None
    capacity_bytes: int | None
    evidence: str
    address: int | None = None
    baud: int | None = None

    def __post_init__(self):
        text_fact(self.id, 'resource identity')
        text_fact(self.evidence, 'resource evidence')
        if self.clock_domain is not None:
            text_fact(self.clock_domain, 'clock domain')
        if self.capacity_bytes is not None:
            natural(self.capacity_bytes, 'resource capacity')
        if self.address is not None:
            natural(self.address, 'resource address')
        if self.baud is not None:
            natural(self.baud, 'serial baud', positive=True)


@dataclass(frozen=True)
class MemoryRegion:
    id: str
    base: int
    capacity_bytes: int
    owner: str
    evidence: str
    address_space: str = 'host'

    def __post_init__(self):
        natural(self.base, 'memory base')
        natural(self.capacity_bytes, 'memory capacity', positive=True)
        for name in ('id', 'owner', 'evidence', 'address_space'):
            text_fact(getattr(self, name), name)

    @property
    def end(self):
        return self.base + self.capacity_bytes


@dataclass(frozen=True)
class SoCDescription:
    resources: tuple[SoCResource, ...]
    clocks: tuple[ClockDomain, ...]
    memory_regions: tuple[MemoryRegion, ...]

    def __post_init__(self):
        for rows, record in ((self.resources, SoCResource), (self.clocks, ClockDomain),
                             (self.memory_regions, MemoryRegion)):
            if type(rows) is not tuple or any(type(r) is not record for r in rows):
                raise ValueError('SoC facts require immutable typed records')
            if type(rows) is not tuple or len({r.id for r in rows}) != len(rows):
                raise ValueError('SoC records must be immutable with unique identities')
        resource_ids = {r.id for r in self.resources}
        clock_ids = {c.id for c in self.clocks}
        if any(r.clock_domain is not None and r.clock_domain not in clock_ids for r in self.resources):
            raise ValueError('resource names an undeclared clock domain')
        if any(r.owner not in resource_ids for r in self.memory_regions):
            raise ValueError('memory names an undeclared owner')
        ordered = sorted(self.memory_regions, key=lambda r: (r.address_space, r.base))
        if any(a.address_space == b.address_space and a.end > b.base
               for a, b in zip(ordered, ordered[1:])):
            raise ValueError('physical memory regions overlap')

    def memory_region(self, identity):
        """Require a declared physical region, without a fallback map."""
        return _declared(self.memory_regions, identity, 'memory region')

    def resource(self, identity):
        """Require a declared resource, including optional MMIO addressing."""
        return _declared(self.resources, identity, 'resource')


def soc_description(document):
    """Validate a target-supplied serialized declaration into immutable SoC facts."""
    if document.get('schema') != 'merlin.soc_description.v1':
        raise ValueError('unsupported SoC description schema')
    return SoCDescription(
        resources=tuple(SoCResource(**row) for row in document['resources']),
        clocks=tuple(ClockDomain(**row) for row in document['clocks']),
        memory_regions=tuple(MemoryRegion(**row) for row in document['memory_regions']))
