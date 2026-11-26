"""Physical capacity must agree with a real link, including symbolic stack."""
import importlib.util
import json
import subprocess
from dataclasses import replace
from types import SimpleNamespace

import pytest
from corefixtures.row_table import run_rows

from merlin.common.paths import repo_root
from merlin.compiler.build.memory_budget import Reservation, derive_budget
from merlin.target.soc import MemoryRegion, SoCDescription, SoCResource

KINDS = ('code', 'runtime', 'stack', 'weights', 'state', 'kv_cache')


def _topology(path, domains, resource=('cpu', 'cpu')):
    from merlin.target.system import select_topology

    path.write_text(json.dumps(dict(schema='merlin.execution_topology.v1',
        resources=[dict(id=resource[0], kind=resource[1], features=[], shared_group=resource[0])],
        memory_domains=[dict(id=key, capacity_bytes=size, alignment_bytes=alignment) for key, size, alignment in domains],
        access=[])))
    return select_topology(path)


def _final_link(root, _):
    from merlin.compiler.build.memory_budget import LinkedReservation, measure_linked_budget
    from merlin.compiler.build.memory_placement import bind_memory_range

    region = MemoryRegion('ram', 0x80000000, 16384, 'controller', 'link')
    bounds = tuple(LinkedReservation(k, k, k + '_start', k + '_end') for k in KINDS)
    (root / 'input.s').write_text('.section .text\n.byte 1,2,3\n')
    (root / 'layout.ld').write_text('SECTIONS { . = 0x80000000; '
        'code_start = .; .text : { *(.text) } code_end = .; '
        'runtime_start = .; . += 13; runtime_end = .; '
        'stack_start = .; . += 1024; stack_end = .; '
        'weights_start = .; . += 64; weights_end = .; '
        'state_start = .; . += 16; state_end = .; '
        'kv_cache_start = .; . += 64; kv_cache_end = .; }')
    subprocess.run(['cc', '-c', str(root / 'input.s'), '-o', str(root / 'input.o')], check=True)
    elf = root / 'app.elf'
    subprocess.run(['ld', '-T', str(root / 'layout.ld'), str(root / 'input.o'), '-o', str(elf)], check=True)
    measured = measure_linked_budget(region, elf, bounds, alignment_bytes=64, nm_tool='nm', timeout_s=10)
    budget = measured.current_budget()
    with pytest.raises(ValueError, match='nonempty'):
        measure_linked_budget(region, elf, (), reservations=measured.budget.reservations,
                              alignment_bytes=64, nm_tool='nm', timeout_s=10)
    # Independent linker high-water mark, not a hand-selected host prefix.
    assert budget.free_ranges == ((region.base + 1216, region.end),)
    topology = _topology(root / 'topology.json', [('ram', budget.capacity_bytes, 64)])
    binding = bind_memory_range(budget, topology, 'ram', range_index=0)
    assert binding.address(0, 64, alignment_bytes=64, current_budget=measured.current_budget()) == region.base + 1216
    spec = importlib.util.spec_from_file_location('soc_link_example', repo_root() / 'examples/compilation/example_soc.py')
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    application = dict(soc=SoCDescription((SoCResource('controller', None, None, 'map'),), (),
                                          (replace(region, id='ddr'),)))
    example_measurement, example_binding = example.bind_linked_ddr(
        application, elf, bounds, topology, 'ram', range_index=0, alignment_bytes=64, nm_tool='nm', timeout_s=10)
    assert example_binding.address(0, 64, alignment_bytes=64,
        current_budget=example_measurement.current_budget()) == region.base + 1216
    with pytest.raises(ValueError, match='missing'):
        measure_linked_budget(region, elf, bounds[:-1] +
            (LinkedReservation('kv_cache', 'kv_cache', 'missing', 'kv_cache_end'),),
            alignment_bytes=64, nm_tool='nm', timeout_s=10)
    # Change an actual text byte, retaining every linker symbol and span.
    original = elf.read_bytes()
    position = original.index(bytes((1, 2, 3)))
    elf.write_bytes(original[:position] + bytes((4,)) + original[position + 1:])
    changed = measure_linked_budget(region, elf, bounds, alignment_bytes=64, nm_tool='nm', timeout_s=10)
    assert changed.budget.free_ranges == budget.free_ranges
    with pytest.raises(ValueError, match='changed.*replan'):
        measured.current_budget()
    with pytest.raises(ValueError, match='replan'):
        binding.address(0, 64, alignment_bytes=64, current_budget=changed.current_budget())
    elf.write_bytes(b'not an ELF')
    with pytest.raises(ValueError, match='ELF'):
        measure_linked_budget(region, elf, bounds, alignment_bytes=64, nm_tool='nm', timeout_s=10)


def _relocatable_or_shared(root, _):
    from merlin.compiler.build.memory_budget import LinkedReservation, measure_linked_budget

    region = MemoryRegion('ram', 0, 4096, 'controller', 'map')
    bounds = (LinkedReservation('code', 'code', 'code_start', 'code_end'),)
    absent = tuple(Reservation(k, k, 'ram', 0, 0, 'absent') for k in KINDS[1:])
    source = root / 'input.s'
    source.write_text('.section .text\n.globl code_start, code_end\ncode_start: .byte 1,2,3\ncode_end:\n')
    obj = root / 'input.o'
    subprocess.run(['cc', '-c', str(source), '-o', str(obj)], check=True)
    # nm reports section-relative offsets for ET_REL: they are not physical
    # addresses, even though they fit in this memory region.
    with pytest.raises(ValueError, match='fixed-address'):
        measure_linked_budget(region, obj, bounds, reservations=absent, alignment_bytes=1, nm_tool='nm', timeout_s=10)
    shared = root / 'shared.elf'
    subprocess.run(['ld', '-shared', str(obj), '-o', str(shared)], check=True)
    # ET_DYN also requires an explicit load-bias contract, absent here.
    with pytest.raises(ValueError, match='fixed-address'):
        measure_linked_budget(region, shared, bounds, reservations=absent, alignment_bytes=1, nm_tool='nm', timeout_s=10)


def _typed_records(root, _):
    region = MemoryRegion('ram', 0, 4096, 'controller', 'link')
    resource = SoCResource('controller', None, None, 'map')
    soc = SoCDescription((resource,), (), (region,))
    with pytest.raises(ValueError, match="no memory region 'absent'"):
        soc.memory_region('absent')
    with pytest.raises(ValueError, match="no resource 'absent'"):
        soc.resource('absent')
    assert SoCResource('uart', 'host', None, 'map', baud=115200).baud == 115200
    for baud in (0, -1, 115200.0, True):
        with pytest.raises(ValueError, match='serial baud'):
            SoCResource('uart', 'host', None, 'map', baud=baud)
    fake_region = SimpleNamespace(**vars(region))
    with pytest.raises(ValueError, match='typed'):
        SoCDescription((resource,), (), (fake_region,))
    with pytest.raises(ValueError, match='typed'):
        derive_budget(fake_region, (), alignment_bytes=64)
    rows = [Reservation(k, k, 'ram', 0, 0, 'absent') for k in KINDS]
    rows[0] = SimpleNamespace(**vars(rows[0]))
    with pytest.raises(ValueError, match='typed'):
        derive_budget(region, rows, alignment_bytes=64)
    with pytest.raises(ValueError, match='string'):
        MemoryRegion(['ram'], 0, 4096, 'controller', 'map')
    with pytest.raises(ValueError, match='string'):
        Reservation('code', 'code', 'ram', 0, 0, ['mutable evidence'])


def _reservations(root, _):
    region = MemoryRegion('ram', 0, 16384, 'controller', 'test physical map')
    rows = [Reservation(k, k, 'ram', 0, 0, 'explicitly absent') for k in KINDS]
    with pytest.raises(ValueError, match='missing explicit'):
        derive_budget(region, rows[:-1], alignment_bytes=16)
    with pytest.raises(ValueError, match='outside'):
        derive_budget(region, rows + [Reservation('bad', 'other', 'ram', 16384, 1, 'test')], alignment_bytes=16)
    # Overlapping linker categories and aliases have one occupied backing.
    union = derive_budget(region, rows + [Reservation('a', 'other', 'ram', 0, 32, 'test'),
                                          Reservation('b', 'other', 'ram', 16, 64, 'test'),
                                          Reservation('nested', 'other', 'ram', 32, 8, 'test')], alignment_bytes=16)
    assert union.free_ranges == ((80, 16384),)
    with pytest.raises(ValueError, match='duplicate'):
        derive_budget(region, rows + [rows[0]], alignment_bytes=16)
    # A hole cannot turn into a contiguous arena spanning reserved storage.
    budget = derive_budget(region, rows + [Reservation('hole', 'other', 'ram', 4096, 4096, 'test')], alignment_bytes=2048)
    assert budget.free_ranges == ((0, 4096), (8192, 16384))
    assert [d.capacity_bytes for d in budget.domains()] == [4096, 8192]


def _address_spaces(root, _):
    resource = SoCResource('controller', None, None, 'test')
    host = MemoryRegion('host', 0, 4096, 'controller', 'test')
    device = MemoryRegion('device', 0, 4096, 'controller', 'test', address_space='device')
    SoCDescription((resource,), (), (host, device))
    with pytest.raises(ValueError, match='overlap'):
        SoCDescription((resource,), (), (host, MemoryRegion('alias', 0, 8, 'controller', 'test')))


def _aggregate_overlap(root, _):
    from merlin.compiler.build.memory_placement import bind_memory_range, bind_memory_ranges

    region = MemoryRegion('ram', 0, 4096, 'controller', 'map')
    rows = tuple(Reservation(k, k, 'ram', 0, 0, 'absent') for k in KINDS)
    budget = derive_budget(region, rows, alignment_bytes=64)
    topology = _topology(root / 'topology.json', [(k, 4096, 64) for k in ('a', 'b')])
    first = bind_memory_range(budget, topology, 'a', range_index=0)
    duplicate = bind_memory_range(budget, topology, 'b', range_index=0)
    with pytest.raises(ValueError, match='overlap'):
        bind_memory_ranges((first, duplicate))
    with pytest.raises(ValueError, match='duplicate'):
        bind_memory_ranges((first, first))
    # Different named regions cannot conceal overlapping physical ownership.
    overlap = derive_budget(replace(region, id='alias'), tuple(replace(r, region='alias') for r in rows), alignment_bytes=64)
    with pytest.raises(ValueError, match='overlap'):
        bind_memory_ranges((first, bind_memory_range(overlap, topology, 'b', range_index=0)))
    separate = derive_budget(replace(region, address_space='device'), rows, alignment_bytes=64)
    second = bind_memory_range(separate, topology, 'b', range_index=0)
    mapping = bind_memory_ranges((first, second))
    assert mapping.address('b', 64, 64, alignment_bytes=64, current_budgets=(budget, separate)) == 64
    with pytest.raises(ValueError, match='missing'):
        mapping.address('a', 0, 64, alignment_bytes=64, current_budgets=(budget,))
    with pytest.raises(ValueError, match='duplicate'):
        mapping.revalidate((budget, budget, separate))
    moved = derive_budget(replace(region, base=4096), tuple(replace(r, base=4096) for r in rows), alignment_bytes=64)
    with pytest.raises(ValueError, match='replan'):
        mapping.address('a', 0, 64, alignment_bytes=64, current_budgets=(moved, separate))
    with pytest.raises(ValueError, match='inconsistent'):
        bind_memory_ranges((first, bind_memory_range(moved, topology, 'b', range_index=0)))


def _reservation_alias(root, _):
    from merlin.compiler.build.memory_placement import bind_memory_range, bind_memory_ranges

    region = MemoryRegion('ram', 0, 128, 'controller', 'map')
    rows = tuple(Reservation(k, k, 'ram', 0, 64 if k == 'code' else 0, 'link') for k in KINDS)
    first_budget = derive_budget(region, rows, alignment_bytes=64)
    alias_budget = derive_budget(replace(region, id='alias'),
        tuple(replace(r, region='alias', base=64) for r in rows), alignment_bytes=64)
    assert first_budget.free_ranges == ((64, 128),)
    assert alias_budget.free_ranges == ((0, 64),)
    topology = _topology(root / 'topology.json', [(k, 64, 64) for k in ('a', 'b')])
    first = bind_memory_range(first_budget, topology, 'a', range_index=0)
    second = bind_memory_range(alias_budget, topology, 'b', range_index=0)
    # Disjoint free intervals each overwrite the other's reserved code.
    with pytest.raises(ValueError, match='physical.*regions overlap'):
        bind_memory_ranges((first, second))
    # Adjacent regions and distinct address spaces remain independent.
    adjacent = derive_budget(replace(region, id='adjacent', base=128),
        tuple(replace(r, region='adjacent', base=r.base + 128) for r in rows), alignment_bytes=64)
    mapping = bind_memory_ranges((first, bind_memory_range(adjacent, topology, 'b', range_index=0)))
    assert mapping.address('b', 0, 64, alignment_bytes=64, current_budgets=(first_budget, adjacent)) == 192
    # Separate intervals in the same intact region budget remain legal.
    fragmented = derive_budget(replace(region, capacity_bytes=192), tuple(replace(r, base=64) for r in rows),
                               alignment_bytes=64)
    mapping = bind_memory_ranges(tuple(bind_memory_range(fragmented, topology, k, range_index=i)
                                       for i, k in enumerate(('a', 'b'))))
    assert mapping.address('b', 0, 64, alignment_bytes=64, current_budgets=(fragmented,)) == 128


def _absolute_alignment(root, _):
    from merlin.compiler.build.memory_placement import bind_memory_range

    region = MemoryRegion('device', 64, 128, 'controller', 'device map', address_space='device')
    rows = [Reservation(k, k, region.id, region.base, 0, 'absent') for k in KINDS]
    budget = derive_budget(region, rows, alignment_bytes=64)
    binding = bind_memory_range(budget, _topology(root / 'topology.json', [('local', 128, 64)],
                                                  ('device', 'accelerator')), 'local', range_index=0)
    assert binding.address_space == 'device'
    # Offset 0 is aligned to 128, but the physical address is not.
    with pytest.raises(ValueError, match='alignment'):
        binding.address(0, 64, alignment_bytes=128, current_budget=budget)
    assert binding.address(64, 64, alignment_bytes=128, current_budget=budget) == 128
    with pytest.raises(ValueError):
        binding.address(True, 1, alignment_bytes=1, current_budget=budget)
    weaker = _topology(root / 'weaker.json', [('local', 128, 32)], ('device', 'accelerator'))
    with pytest.raises(ValueError, match='alignment'):
        bind_memory_range(budget, weaker, 'local', range_index=0)


def _binding_bounds(root, _):
    from merlin.compiler.build.memory_placement import bind_memory_range

    region = MemoryRegion('ram', 0x80000000, 16384, 'controller', 'link')
    rows = [Reservation(k, k, 'ram', region.base, 0, 'absent') for k in KINDS]
    rows.append(Reservation('hole', 'other', 'ram', region.base + 4096, 4096, 'link'))
    budget = derive_budget(region, rows, alignment_bytes=64)
    path = root / 'topology.json'
    topology = _topology(path, [(d.id, d.capacity_bytes, d.alignment_bytes) for d in budget.domains()])
    first = bind_memory_range(budget, topology, 'ram.0', range_index=0)
    second = bind_memory_range(budget, topology, 'ram.1', range_index=1)
    fake = SimpleNamespace(memory=lambda _: budget.domains()[0])
    with pytest.raises(ValueError, match='typed original-caller'):
        bind_memory_range(budget, fake, 'ram.0', range_index=0)
    assert first.address(0, 4096, alignment_bytes=64, current_budget=budget) == region.base
    assert second.address(0, 8192, alignment_bytes=64, current_budget=budget) == region.base + 8192
    with pytest.raises(ValueError, match='outside'):
        first.address(0, 8192, alignment_bytes=64, current_budget=budget)
    with pytest.raises(ValueError, match='capacity'):
        bind_memory_range(budget, topology, 'ram.1', range_index=0)
    # Equal scalar capacity cannot hide a moved physical map after relinking.
    moved = replace(region, base=region.base + 0x10000)
    changed = derive_budget(moved, [replace(r, base=r.base + 0x10000) for r in rows], alignment_bytes=64)
    assert changed.capacity_bytes == budget.capacity_bytes
    with pytest.raises(ValueError, match='replan'):
        first.address(0, 64, alignment_bytes=64, current_budget=changed)
    with pytest.raises(ValueError, match='derived'):
        bind_memory_range(replace(budget, free_ranges=((region.base, region.end),)), topology, 'ram.0', range_index=0)
    with pytest.raises(ValueError, match='index'):
        bind_memory_range(budget, topology, 'ram.0', range_index=-1)
    path.write_text(path.read_text() + ' ')
    with pytest.raises(ValueError, match='changed'):
        first.address(0, 64, alignment_bytes=64, current_budget=budget)


def _budget_record(root, _):
    from argparse import Namespace
    from merlin.compiler.build.memory_budget import budget_record, budget_window
    from merlin.compiler.semantic.codegen import options_from_args

    region = MemoryRegion('ddr', 0x80000000, 1 << 20, 'controller', 'map')
    rows = [Reservation(k, k, 'ddr', region.base, 0x6a20 if k == 'code' else 0, 'link') for k in KINDS]
    budget = derive_budget(region, rows, alignment_bytes=4096)
    record = budget_record(budget, range_index=0)
    path = root / 'budget.json'
    path.write_text(json.dumps(record))
    window = (region.base + 0x7000, region.capacity_bytes - 0x7000)
    assert budget_window(json.loads(path.read_text())) == window
    # The compile's device window and capacity are the budget's selected free range, never a separate flag.
    options = options_from_args(Namespace(memory_budget=str(path)))
    assert (options.io_memory_base, options.io_memory_bytes,
            options.io_memory_capacity_base, options.io_memory_capacity_bytes) == window * 2
    with pytest.raises(ValueError, match='omit --io-memory-bytes'):
        options_from_args(Namespace(memory_budget=str(path), io_memory_bytes=2048))
    for change, match in ((dict(free_ranges=[[region.base, region.end]]), 'differ from the derived'),
                          (dict(range_index=1), 'no free range'), (dict(schema='other'), 'unsupported')):
        with pytest.raises(ValueError, match=match):
            budget_window(dict(record, **change))
    with pytest.raises(ValueError, match='intact'):
        budget_record(replace(budget, free_ranges=((region.base, region.end),)), range_index=0)
    with pytest.raises(ValueError, match='no free range'):
        budget_record(budget, range_index=1)


def test_memory_claims_follow_the_final_link_and_refuse_unsound_bindings(tmp_path, monkeypatch):
    # The first two rows link with the native toolchain (cc, ld, nm); the others need none.
    run_rows(tmp_path, monkeypatch, {
        'final_link': _final_link,
        'relocatable_or_shared': _relocatable_or_shared,
        'typed_records': _typed_records,
        'reservations': _reservations,
        'address_spaces': _address_spaces,
        'aggregate_overlap': _aggregate_overlap,
        'reservation_alias': _reservation_alias,
        'absolute_alignment': _absolute_alignment,
        'binding_bounds': _binding_bounds,
        'budget_record': _budget_record,
    })
