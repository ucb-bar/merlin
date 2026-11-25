"""Bind selected logical capacity to one linker-derived physical free interval.

Translate offsets already allocated by Merlin; do not pack buffers or admit
device reachability, loading, visibility or completion. The outer application
supplies a freshly derived budget after each link/relink. Changed reservations
require selection and allocation to be repeated.
"""

from dataclasses import dataclass

from merlin.compiler.build.memory_budget import ApplicationBudget, derive_budget
from merlin.target.soc import alignment, natural
from merlin.target.system import SelectedTopology


def _derived(budget):
    if type(budget) is not ApplicationBudget or budget != derive_budget(
        budget.region, budget.reservations, alignment_bytes=budget.alignment_bytes
    ):
        raise ValueError("physical binding requires an intact derived memory budget")


@dataclass(frozen=True)
class SelectedMemoryRange:
    """One selected domain and its physical backing, including address space.

    Domain names may be chosen by the outer caller; capacity and alignment must
    exactly match one free interval. There is no implicit host/device address
    translation. Separate ranges cannot back one contiguous allocation.
    """

    budget: ApplicationBudget
    topology: SelectedTopology
    domain: str
    range_index: int

    def __post_init__(self):
        _derived(self.budget)
        if type(self.topology) is not SelectedTopology:
            raise ValueError("physical binding requires a typed original-caller selected topology")
        if type(self.range_index) is not int or not 0 <= self.range_index < len(self.budget.free_ranges):
            raise ValueError("physical memory range index is invalid")
        memory = self.topology.memory(self.domain)
        start, end = self.budget.free_ranges[self.range_index]
        if memory.capacity_bytes != end - start:
            raise ValueError("selected memory capacity differs from its physical free interval; replan")
        if memory.alignment_bytes != self.budget.alignment_bytes:
            raise ValueError("selected memory alignment differs from the derived budget; replan")

    @property
    def address_space(self):
        return self.budget.region.address_space

    def revalidate(self, current_budget):
        """Refuse changed topology or linker/physical facts before publication."""
        self.__post_init__()
        _derived(current_budget)
        if current_budget != self.budget:
            raise ValueError("linked memory reservations or physical map changed; replan")

    def address(self, offset, size_bytes, *, alignment_bytes, current_budget):
        """Resolve a range in ``address_space`` after current-link checks.

        The caller still checks allocation lifetimes/aliases with Merlin's
        existing allocation witness checker. A valid address grants no access
        permission. Keep this binding beside addresses in the deployment owner.
        """
        self.revalidate(current_budget)
        natural(offset, "allocation offset")
        natural(size_bytes, "allocation size")
        alignment(alignment_bytes)
        start, end = self.budget.free_ranges[self.range_index]
        if offset + size_bytes > end - start:
            raise ValueError("allocation lies outside its physical free interval")
        address = start + offset
        if address % alignment_bytes:
            raise ValueError("physical allocation violates required alignment")
        return address


def bind_memory_range(budget, topology, domain, *, range_index):
    """Bind an original caller-selected topology to a measured free interval."""
    return SelectedMemoryRange(budget, topology, domain, range_index)


@dataclass(frozen=True)
class SelectedMemoryMap:
    """Disjoint physical ownership for a set of logical allocation domains.

    A caller may bind a subset of selected domains. Two independent domains
    cannot own the same physical bytes. Aliases and lifetime reuse stay within
    Merlin's existing allocator for a single domain, rather than duplicating
    that arena under a second domain name.
    """

    ranges: tuple[SelectedMemoryRange, ...]

    def __post_init__(self):
        if type(self.ranges) is not tuple or not self.ranges or any(
                type(row) is not SelectedMemoryRange for row in self.ranges):
            raise ValueError('physical map requires nonempty immutable typed bindings')
        if len({row.domain for row in self.ranges}) != len(self.ranges):
            raise ValueError('duplicate physical memory domain binding')
        topology = self.ranges[0].topology
        regions = {}
        for row in self.ranges:
            row.__post_init__()
            if row.topology != topology:
                raise ValueError('physical map requires one original-caller selected topology')
            key = (row.address_space, row.budget.region.id)
            if key in regions and regions[key] != row.budget:
                raise ValueError('inconsistent budgets for one physical region')
            regions[key] = row.budget
        # Distinct region names cannot hide occupied backing from each other.
        # Check entire physical regions, not only the selected free intervals:
        # one region's free interval may contain another's reserved code/state.
        ordered_regions = sorted((budget.region for budget in regions.values()),
                                 key=lambda region: (region.address_space, region.base))
        if any(a.address_space == b.address_space and a.end > b.base
               for a, b in zip(ordered_regions, ordered_regions[1:])):
            raise ValueError('physical memory regions overlap; use one consistent region budget')
        intervals = sorted((row.address_space, *row.budget.free_ranges[row.range_index])
                           for row in self.ranges)
        if any(a[0] == b[0] and a[2] > b[1] for a, b in zip(intervals, intervals[1:])):
            raise ValueError('logical memory domains overlap in physical address space')

    def revalidate(self, current_budgets):
        """Check every bound region, including domains unused by this operation."""
        self.__post_init__()
        budgets = {}
        for budget in current_budgets:
            _derived(budget)
            key = (budget.region.address_space, budget.region.id)
            if key in budgets:
                raise ValueError('duplicate current physical memory budget')
            budgets[key] = budget
        for row in self.ranges:
            key = (row.address_space, row.budget.region.id)
            if key not in budgets:
                raise ValueError('missing current physical memory budget')
            row.revalidate(budgets[key])

    def address(self, domain, offset, size_bytes, *, alignment_bytes, current_budgets):
        self.revalidate(current_budgets)
        row = next((row for row in self.ranges if row.domain == domain), None)
        if row is None:
            raise ValueError('unbound physical memory domain')
        return row.address(offset, size_bytes, alignment_bytes=alignment_bytes, current_budget=row.budget)


def bind_memory_ranges(ranges):
    """Collect checked bindings without assigning the same physical bytes twice."""
    return SelectedMemoryMap(tuple(ranges))
