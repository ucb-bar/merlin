"""Outer application selection over a made-up SoC declaration.

This is an example, not an installed target provider: a real target supplies
its own declaration document, ABI, loading, scratch and completion requirements
from its own package, and Merlin's build/completion.py still requires
original-caller platform admission. No hart count or accelerator clock rate is
inferred from a host clock setting.
"""
from dataclasses import replace

from merlin.compiler.build.memory_budget import derive_budget, measure_linked_budget
from merlin.compiler.build.memory_placement import bind_memory_range
from merlin.target.soc import ClockDomain, soc_description

DECLARATION = {
    "schema": "merlin.soc_description.v1",
    "resources": [
        {"id": "example.host", "clock_domain": "host", "capacity_bytes": 65536, "evidence": "example declaration"},
        {"id": "example.accelerator", "clock_domain": "accelerator", "capacity_bytes": 262144,
         "evidence": "example declaration"},
        {"id": "example.ddr", "clock_domain": None, "capacity_bytes": None, "evidence": "example declaration"},
    ],
    "clocks": [
        {"id": "host", "frequency_hz": None, "evidence": "selected by the caller"},
        {"id": "accelerator", "frequency_hz": None, "evidence": "not established by the example"},
    ],
    "memory_regions": [
        {"id": "host_scratch", "base": 0x10000000, "capacity_bytes": 65536, "owner": "example.host",
         "evidence": "example declaration"},
        {"id": "accelerator_data", "base": 0x20000000, "capacity_bytes": 262144, "owner": "example.accelerator",
         "evidence": "example declaration"},
        {"id": "ddr", "base": 0x80000000, "capacity_bytes": 1 << 30, "owner": "example.ddr",
         "evidence": "example declaration"},
    ],
}


def describe(*, host_clock_hz, harts):
    """Caller selects the host clock and harts outside the device compiler."""
    declared = soc_description(DECLARATION)
    soc = replace(declared, clocks=tuple(
        ClockDomain('host', host_clock_hz, 'operator-stated, unmeasured') if clock.id == 'host' else clock
        for clock in declared.clocks))
    return dict(system='example', platform='example', harts=tuple(harts), soc=soc)


def bind_ddr(application, reservations, topology, domain, *, range_index, alignment_bytes):
    """Bind one measured DDR interval to caller-selected allocation capacity.

    Retain the returned budget and binding with the application plan. After
    linking, derive the final budget again and pass it to binding.address;
    changed reservations require replanning. This does not load or run a device.
    """
    region = application['soc'].memory_region('ddr')
    budget = derive_budget(region, reservations, alignment_bytes=alignment_bytes)
    return budget, bind_memory_range(budget, topology, domain, range_index=range_index)


def bind_linked_ddr(application, elf, bounds, topology, domain, *, reservations=(),
                    range_index, alignment_bytes, nm_tool, timeout_s):
    """Measure the final ELF and bind its budget without a fixed host prefix.

    Before address publication, pass measurement.current_budget() to
    binding.address. Keep the final ELF immutable through loading, or repeat the
    check after any relink. Atlas still owns ABI/loading/completion admission.
    """
    region = application['soc'].memory_region('ddr')
    measurement = measure_linked_budget(
        region, elf, bounds, reservations=reservations, alignment_bytes=alignment_bytes,
        nm_tool=nm_tool, timeout_s=timeout_s)
    return measurement, bind_memory_range(measurement.budget, topology, domain, range_index=range_index)
