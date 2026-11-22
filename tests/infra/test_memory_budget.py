"""Physical capacity must agree with a real link, including symbolic stack."""
import importlib.util
import json
import subprocess
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from merlin.common.paths import repo_root
from merlin.compiler.build.memory_budget import Reservation, derive_budget, linked_reservation
from merlin.target.soc import MemoryRegion, SoCDescription, SoCResource


class MemoryBudgetTest(unittest.TestCase):
    def test_linked_application_budget(self):
        spec = importlib.util.spec_from_file_location(
            'soc_example', repo_root() / 'examples/compilation/example_soc.py')
        example = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(example)
        application = example.describe(host_clock_hz=100_000_000, harts=(0,))
        region = application['soc'].memory_region('ddr')
        with self.assertRaisesRegex(ValueError, "no memory region 'absent'"):
            application['soc'].memory_region('absent')
        with self.assertRaisesRegex(ValueError, "no resource 'absent'"):
            application['soc'].resource('absent')
        # Native link only: no target toolchain, board, or numerical claim.
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'input.s').write_text('.section .text\n.byte 1,2,3\n')
            (root / 'layout.ld').write_text('''SECTIONS {
              . = 0x80000000;
              code_start = .; .text : { *(.text) } code_end = .;
              runtime_start = .; . += 13; runtime_end = .;
              stack_start = .; . += 8192; stack_end = .;
              . = ALIGN(2048);
              weights_start = .; . += 4096; weights_end = .;
              state_start = .; . += 256; state_end = .;
              kv_cache_start = .; . += 512; kv_cache_end = .;
            }''')
            subprocess.run(['cc', '-c', str(root / 'input.s'), '-o', str(root / 'input.o')], check=True)
            subprocess.run(['ld', '-T', str(root / 'layout.ld'), str(root / 'input.o'),
                            '-o', str(root / 'app.elf')], check=True)
            rows = subprocess.check_output(['nm', '--defined-only', str(root / 'app.elf')], text=True)
            symbols = {name: int(address, 16) for address, _, name in (line.split() for line in rows.splitlines())}
        kinds = ('code', 'runtime', 'stack', 'weights', 'state', 'kv_cache')
        reserved = [linked_reservation(region, symbols, identity=k, kind=k, start=k+'_start',
                                      end=k+'_end', evidence='native test link: nm') for k in kinds]
        budget = derive_budget(region, reserved, alignment_bytes=2048)
        # Independent linker high-water mark, not a hand-selected host prefix.
        expected_base = (symbols['kv_cache_end'] + 2047) // 2048 * 2048
        self.assertEqual(budget.free_ranges, ((expected_base, 0xc0000000),))
        self.assertEqual(budget.capacity_bytes, (1 << 30) - (expected_base - region.base))
        self.assertEqual(budget.domains()[0].capacity_bytes, budget.capacity_bytes)
        self.assertEqual(reserved[2].size_bytes, 8192)
        self.assertEqual(application['soc'].resources[0].scratchpad_bytes, 65536)
        self.assertIsNone(application['soc'].clocks[1].frequency_hz)
        from merlin.target.system import select_topology

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'topology.json'
            path.write_text(json.dumps(dict(schema='merlin.execution_topology.v1',
                resources=[dict(id='host', kind='cpu', features=[], shared_group='host')],
                memory_domains=[dict(id='application', capacity_bytes=budget.capacity_bytes,
                                     alignment_bytes=2048)], access=[])))
            bound_budget, binding = example.bind_ddr(application, reserved, select_topology(path),
                'application', range_index=0, alignment_bytes=2048)
            self.assertEqual(binding.address(0, 2048, alignment_bytes=2048,
                                            current_budget=bound_budget), expected_base)

    def test_memory_facts_reject_mutable_substitutes(self):
        region = MemoryRegion('ram', 0, 4096, 'controller', 'link')
        resource = SoCResource('controller', None, None, 'map')
        self.assertEqual(SoCResource('uart', 'host', None, 'map', baud=115200).baud, 115200)
        for baud in (0, -1, 115200.0, True):
            with self.subTest(baud=baud), self.assertRaisesRegex(ValueError, 'serial baud'):
                SoCResource('uart', 'host', None, 'map', baud=baud)
        fake_region = SimpleNamespace(**vars(region))
        with self.assertRaisesRegex(ValueError, 'typed'):
            SoCDescription((resource,), (), (fake_region,))
        with self.assertRaisesRegex(ValueError, 'typed'):
            derive_budget(fake_region, (), alignment_bytes=64)
        rows = [Reservation(k, k, 'ram', 0, 0, 'absent')
                for k in ('code', 'runtime', 'stack', 'weights', 'state', 'kv_cache')]
        rows[0] = SimpleNamespace(**vars(rows[0]))
        with self.assertRaisesRegex(ValueError, 'typed'):
            derive_budget(region, rows, alignment_bytes=64)
        with self.assertRaisesRegex(ValueError, 'string'):
            MemoryRegion(['ram'], 0, 4096, 'controller', 'map')
        with self.assertRaisesRegex(ValueError, 'string'):
            Reservation('code', 'code', 'ram', 0, 0, ['mutable evidence'])

    def test_final_elf_budget_refresh(self):
        from merlin.compiler.build.memory_budget import LinkedReservation, measure_linked_budget
        from merlin.compiler.build.memory_placement import bind_memory_range
        from merlin.target.system import select_topology

        region = MemoryRegion('ram', 0x80000000, 16384, 'controller', 'link')
        kinds = ('code', 'runtime', 'stack', 'weights', 'state', 'kv_cache')
        bounds = tuple(LinkedReservation(k, k, k + '_start', k + '_end') for k in kinds)
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
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
            subprocess.run(['ld', '-T', str(root / 'layout.ld'), str(root / 'input.o'),
                            '-o', str(elf)], check=True)
            measured = measure_linked_budget(region, elf, bounds, alignment_bytes=64,
                                             nm_tool='nm', timeout_s=10)
            budget = measured.current_budget()
            with self.assertRaisesRegex(ValueError, 'nonempty'):
                measure_linked_budget(region, elf, (), reservations=measured.budget.reservations,
                                      alignment_bytes=64, nm_tool='nm', timeout_s=10)
            self.assertEqual(budget.free_ranges, ((region.base + 1216, region.end),))
            path = root / 'topology.json'
            path.write_text(json.dumps(dict(schema='merlin.execution_topology.v1',
                resources=[dict(id='cpu', kind='cpu', features=[], shared_group='cpu')],
                memory_domains=[dict(id='ram', capacity_bytes=budget.capacity_bytes,
                                     alignment_bytes=64)], access=[])))
            binding = bind_memory_range(budget, select_topology(path), 'ram', range_index=0)
            self.assertEqual(binding.address(0, 64, alignment_bytes=64,
                current_budget=measured.current_budget()), region.base + 1216)
            spec = importlib.util.spec_from_file_location(
                'soc_link_example', repo_root() / 'examples/compilation/example_soc.py')
            example = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(example)
            application = dict(soc=SoCDescription(
                (SoCResource('controller', None, None, 'map'),), (),
                (replace(region, id='ddr'),)))
            example_measurement, example_binding = example.bind_linked_ddr(
                application, elf, bounds, select_topology(path), 'ram', range_index=0,
                alignment_bytes=64, nm_tool='nm', timeout_s=10)
            self.assertEqual(example_binding.address(0, 64, alignment_bytes=64,
                current_budget=example_measurement.current_budget()), region.base + 1216)
            with self.assertRaisesRegex(ValueError, 'missing'):
                measure_linked_budget(region, elf, bounds[:-1] +
                    (LinkedReservation('kv_cache', 'kv_cache', 'missing', 'kv_cache_end'),),
                    alignment_bytes=64, nm_tool='nm', timeout_s=10)
            # Change an actual text byte, retaining every linker symbol and span.
            original = elf.read_bytes()
            position = original.index(bytes((1, 2, 3)))
            elf.write_bytes(original[:position] + bytes((4,)) + original[position + 1:])
            changed = measure_linked_budget(region, elf, bounds, alignment_bytes=64,
                                            nm_tool='nm', timeout_s=10)
            self.assertEqual(changed.budget.free_ranges, budget.free_ranges)
            with self.assertRaisesRegex(ValueError, 'changed.*replan'):
                measured.current_budget()
            with self.assertRaisesRegex(ValueError, 'replan'):
                binding.address(0, 64, alignment_bytes=64, current_budget=changed.current_budget())
            elf.write_bytes(b'not an ELF')
            with self.assertRaisesRegex(ValueError, 'ELF'):
                measure_linked_budget(region, elf, bounds, alignment_bytes=64,
                                      nm_tool='nm', timeout_s=10)

    def test_measurement_requires_fixed_address_final_link(self):
        from merlin.compiler.build.memory_budget import LinkedReservation, measure_linked_budget

        region = MemoryRegion('ram', 0, 4096, 'controller', 'map')
        kinds = ('code', 'runtime', 'stack', 'weights', 'state', 'kv_cache')
        bounds = (LinkedReservation('code', 'code', 'code_start', 'code_end'),)
        absent = tuple(Reservation(k, k, 'ram', 0, 0, 'absent') for k in kinds[1:])
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'input.s'
            source.write_text('.section .text\n.globl code_start, code_end\n'
                              'code_start: .byte 1,2,3\ncode_end:\n')
            obj = root / 'input.o'
            subprocess.run(['cc', '-c', str(source), '-o', str(obj)], check=True)
            # nm reports section-relative offsets for ET_REL: they are not
            # physical addresses, even though they fit in this memory region.
            with self.assertRaisesRegex(ValueError, 'fixed-address'):
                measure_linked_budget(region, obj, bounds, reservations=absent,
                                      alignment_bytes=1, nm_tool='nm', timeout_s=10)
            shared = root / 'shared.elf'
            subprocess.run(['ld', '-shared', str(obj), '-o', str(shared)], check=True)
            # ET_DYN also requires an explicit load-bias contract, absent here.
            with self.assertRaisesRegex(ValueError, 'fixed-address'):
                measure_linked_budget(region, shared, bounds, reservations=absent,
                                      alignment_bytes=1, nm_tool='nm', timeout_s=10)

    def test_aggregate_physical_binding_ownership(self):
        from merlin.compiler.build.memory_placement import bind_memory_range, bind_memory_ranges
        from merlin.target.system import select_topology

        region = MemoryRegion('ram', 0, 4096, 'controller', 'map')
        rows = tuple(Reservation(k, k, 'ram', 0, 0, 'absent')
                     for k in ('code', 'runtime', 'stack', 'weights', 'state', 'kv_cache'))
        budget = derive_budget(region, rows, alignment_bytes=64)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'topology.json'
            path.write_text(json.dumps(dict(schema='merlin.execution_topology.v1',
                resources=[dict(id='cpu', kind='cpu', features=[], shared_group='cpu')],
                memory_domains=[dict(id=k, capacity_bytes=4096, alignment_bytes=64)
                                for k in ('a', 'b')], access=[])))
            topology = select_topology(path)
            first = bind_memory_range(budget, topology, 'a', range_index=0)
            duplicate = bind_memory_range(budget, topology, 'b', range_index=0)
            with self.assertRaisesRegex(ValueError, 'overlap'):
                bind_memory_ranges((first, duplicate))
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                bind_memory_ranges((first, first))
            # Different named regions cannot conceal overlapping physical ownership.
            overlap = derive_budget(replace(region, id='alias'),
                                    tuple(replace(r, region='alias') for r in rows), alignment_bytes=64)
            with self.assertRaisesRegex(ValueError, 'overlap'):
                bind_memory_ranges((first, bind_memory_range(overlap, topology, 'b', range_index=0)))
            separate = derive_budget(replace(region, address_space='device'), rows, alignment_bytes=64)
            second = bind_memory_range(separate, topology, 'b', range_index=0)
            mapping = bind_memory_ranges((first, second))
            self.assertEqual(mapping.address('b', 64, 64, alignment_bytes=64,
                                            current_budgets=(budget, separate)), 64)
            with self.assertRaisesRegex(ValueError, 'missing'):
                mapping.address('a', 0, 64, alignment_bytes=64, current_budgets=(budget,))
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                mapping.revalidate((budget, budget, separate))
            moved = derive_budget(replace(region, base=4096),
                                  tuple(replace(r, base=4096) for r in rows), alignment_bytes=64)
            with self.assertRaisesRegex(ValueError, 'replan'):
                mapping.address('a', 0, 64, alignment_bytes=64, current_budgets=(moved, separate))
            with self.assertRaisesRegex(ValueError, 'inconsistent'):
                bind_memory_ranges((first, bind_memory_range(moved, topology, 'b', range_index=0)))

    def test_aggregate_rejects_free_ranges_aliasing_other_region_reservations(self):
        from merlin.compiler.build.memory_placement import bind_memory_range, bind_memory_ranges
        from merlin.target.system import select_topology

        region = MemoryRegion('ram', 0, 128, 'controller', 'map')
        kinds = ('code', 'runtime', 'stack', 'weights', 'state', 'kv_cache')
        rows = tuple(Reservation(k, k, 'ram', 0, 64 if k == 'code' else 0, 'link')
                     for k in kinds)
        first_budget = derive_budget(region, rows, alignment_bytes=64)
        alias_budget = derive_budget(replace(region, id='alias'),
            tuple(replace(r, region='alias', base=64) for r in rows), alignment_bytes=64)
        self.assertEqual(first_budget.free_ranges, ((64, 128),))
        self.assertEqual(alias_budget.free_ranges, ((0, 64),))
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'topology.json'
            path.write_text(json.dumps(dict(schema='merlin.execution_topology.v1',
                resources=[dict(id='cpu', kind='cpu', features=[], shared_group='cpu')],
                memory_domains=[dict(id=k, capacity_bytes=64, alignment_bytes=64)
                                for k in ('a', 'b')], access=[])))
            topology = select_topology(path)
            first = bind_memory_range(first_budget, topology, 'a', range_index=0)
            second = bind_memory_range(alias_budget, topology, 'b', range_index=0)
            # Disjoint free intervals each overwrite the other's reserved code.
            with self.assertRaisesRegex(ValueError, 'physical.*regions overlap'):
                bind_memory_ranges((first, second))
            # Adjacent regions and distinct address spaces remain independent.
            adjacent = derive_budget(replace(region, id='adjacent', base=128),
                tuple(replace(r, region='adjacent', base=r.base + 128) for r in rows),
                alignment_bytes=64)
            mapping = bind_memory_ranges((first,
                bind_memory_range(adjacent, topology, 'b', range_index=0)))
            self.assertEqual(mapping.address('b', 0, 64, alignment_bytes=64,
                current_budgets=(first_budget, adjacent)), 192)
            # Separate intervals in the same intact region budget remain legal.
            large = replace(region, capacity_bytes=192)
            middle_rows = tuple(replace(r, base=64) for r in rows)
            fragmented = derive_budget(large, middle_rows, alignment_bytes=64)
            mapping = bind_memory_ranges(tuple(bind_memory_range(fragmented, topology, k,
                range_index=i) for i, k in enumerate(('a', 'b'))))
            self.assertEqual(mapping.address('b', 0, 64, alignment_bytes=64,
                current_budgets=(fragmented,)), 128)

    def test_invalid_reservations(self):
        region = MemoryRegion('ram', 0, 16384, 'controller', 'test physical map')
        rows = [Reservation(k, k, 'ram', 0, 0, 'explicitly absent')
                for k in ('code', 'runtime', 'stack', 'weights', 'state', 'kv_cache')]
        with self.assertRaisesRegex(ValueError, 'missing explicit'):
            derive_budget(region, rows[:-1], alignment_bytes=16)
        with self.assertRaisesRegex(ValueError, 'outside'):
            derive_budget(region, rows + [Reservation('bad', 'other', 'ram', 16384, 1, 'test')], alignment_bytes=16)
        # Overlapping linker categories and aliases have one occupied backing.
        union = derive_budget(region, rows + [Reservation('a', 'other', 'ram', 0, 32, 'test'),
                                              Reservation('b', 'other', 'ram', 16, 64, 'test'),
                                              Reservation('nested', 'other', 'ram', 32, 8, 'test')],
                              alignment_bytes=16)
        self.assertEqual(union.free_ranges, ((80, 16384),))
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            derive_budget(region, rows + [rows[0]], alignment_bytes=16)
        # A hole cannot turn into a contiguous arena spanning reserved storage.
        budget = derive_budget(region, rows + [Reservation('hole', 'other', 'ram', 4096, 4096, 'test')],
                               alignment_bytes=2048)
        self.assertEqual(budget.free_ranges, ((0, 4096), (8192, 16384)))
        self.assertEqual([d.capacity_bytes for d in budget.domains()], [4096, 8192])

    def test_physical_binding_preserves_holes_and_allocator_lifetimes(self):
        from merlin.compiler.build.memory_placement import bind_memory_range
        from merlin.compiler.semantic.search.allocation import StorageBank
        from merlin.compiler.semantic.search.buffers import Buffer, allocate_buffers, first_fit_buffer_assignment
        from merlin.target.system import select_topology

        region = MemoryRegion('ram', 0x80000000, 16384, 'controller', 'link')
        rows = [Reservation(k, k, 'ram', region.base, 0, 'absent')
                for k in ('code', 'runtime', 'stack', 'weights', 'state', 'kv_cache')]
        rows.append(Reservation('hole', 'other', 'ram', region.base + 4096, 4096, 'link'))
        budget = derive_budget(region, rows, alignment_bytes=64)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'topology.json'
            path.write_text(json.dumps(dict(schema='merlin.execution_topology.v1',
                resources=[dict(id='cpu', kind='cpu', features=[], shared_group='cpu')],
                memory_domains=[dict(id=d.id, capacity_bytes=d.capacity_bytes,
                                     alignment_bytes=d.alignment_bytes) for d in budget.domains()], access=[])))
            topology = select_topology(path)
            first = bind_memory_range(budget, topology, 'ram.0', range_index=0)
            second = bind_memory_range(budget, topology, 'ram.1', range_index=1)
            fake = SimpleNamespace(memory=lambda _: budget.domains()[0])
            with self.assertRaisesRegex(ValueError, 'typed original-caller'):
                bind_memory_range(budget, fake, 'ram.0', range_index=0)
            # Existing Merlin allocator owns lifetimes and may reuse completed storage.
            buffers = (Buffer('early', 'arena', 4096, 0, 1, alignment=64),
                       Buffer('later', 'arena', 4096, 2, 3, alignment=64))
            offsets = first_fit_buffer_assignment(buffers, (StorageBank('arena', 'arena', 4096, 'byte'),))
            self.assertEqual(offsets, {'early': 0, 'later': 0})
            banks = (StorageBank('arena', 'arena', 4096, 'byte'),)
            # The native provider can use deterministic work instead of elapsed
            # time, while tiny limits refuse inconclusively rather than reroute.
            solved = allocate_buffers(buffers, banks, timeout_ms=None, rlimit=100_000_000)
            self.assertEqual(solved.status, 'sat')
            self.assertEqual(dict(solved.addresses), offsets)
            self.assertEqual(allocate_buffers(buffers, banks, timeout_ms=None,
                                             rlimit=100_000_000), solved)
            limited = allocate_buffers(buffers, banks, timeout_ms=None, rlimit=1)
            self.assertEqual(limited.status, 'unknown')
            self.assertTrue(limited.reason)
            for bad in (0, -1, True):
                with self.assertRaisesRegex(ValueError, 'positive solver resource limit'):
                    allocate_buffers(buffers, banks, timeout_ms=None, rlimit=bad)
            with self.assertRaisesRegex(ValueError, 'timeout or resource limit'):
                allocate_buffers(buffers, banks, timeout_ms=None)

            self.assertEqual(first.address(offsets['later'], 4096, alignment_bytes=64,
                                           current_budget=budget), region.base)
            self.assertEqual(second.address(0, 8192, alignment_bytes=64,
                                            current_budget=budget), region.base + 8192)
            with self.assertRaisesRegex(ValueError, 'outside'):
                first.address(0, 8192, alignment_bytes=64, current_budget=budget)
            with self.assertRaisesRegex(ValueError, 'capacity'):
                bind_memory_range(budget, topology, 'ram.1', range_index=0)
            # Equal scalar capacity cannot hide a moved physical map after relinking.
            moved = replace(region, base=region.base + 0x10000)
            moved_rows = [replace(r, base=r.base + 0x10000) for r in rows]
            changed = derive_budget(moved, moved_rows, alignment_bytes=64)
            self.assertEqual(changed.capacity_bytes, budget.capacity_bytes)
            with self.assertRaisesRegex(ValueError, 'replan'):
                first.address(0, 64, alignment_bytes=64, current_budget=changed)
            with self.assertRaisesRegex(ValueError, 'derived'):
                bind_memory_range(replace(budget, free_ranges=((region.base, region.end),)),
                                  topology, 'ram.0', range_index=0)
            with self.assertRaisesRegex(ValueError, 'index'):
                bind_memory_range(budget, topology, 'ram.0', range_index=-1)
            path.write_text(path.read_text() + ' ')
            with self.assertRaisesRegex(ValueError, 'changed'):
                first.address(0, 64, alignment_bytes=64, current_budget=budget)

    def test_physical_binding_uses_address_space_and_absolute_alignment(self):
        from merlin.compiler.build.memory_placement import bind_memory_range
        from merlin.target.system import select_topology

        region = MemoryRegion('device', 64, 128, 'controller', 'device map', address_space='device')
        rows = [Reservation(k, k, region.id, region.base, 0, 'absent')
                for k in ('code', 'runtime', 'stack', 'weights', 'state', 'kv_cache')]
        budget = derive_budget(region, rows, alignment_bytes=64)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'topology.json'
            document = dict(schema='merlin.execution_topology.v1',
                resources=[dict(id='device', kind='accelerator', features=[], shared_group='device')],
                memory_domains=[dict(id='local', capacity_bytes=128, alignment_bytes=64)], access=[])
            path.write_text(json.dumps(document))
            binding = bind_memory_range(budget, select_topology(path), 'local', range_index=0)
            self.assertEqual(binding.address_space, 'device')
            # Offset 0 is aligned to 128, but the physical address is not.
            with self.assertRaisesRegex(ValueError, 'alignment'):
                binding.address(0, 64, alignment_bytes=128, current_budget=budget)
            self.assertEqual(binding.address(64, 64, alignment_bytes=128, current_budget=budget), 128)
            with self.assertRaises(ValueError):
                binding.address(True, 1, alignment_bytes=1, current_budget=budget)
            document['memory_domains'][0]['alignment_bytes'] = 32
            path.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, 'alignment'):
                bind_memory_range(budget, select_topology(path), 'local', range_index=0)

    def test_address_spaces_and_saved_deploy_link(self):
        resource = SoCResource('controller', None, None, 'test')
        host = MemoryRegion('host', 0, 4096, 'controller', 'test')
        device = MemoryRegion('device', 0, 4096, 'controller', 'test', address_space='device')
        SoCDescription((resource,), (), (host, device))
        with self.assertRaisesRegex(ValueError, 'overlap'):
            SoCDescription((resource,), (), (host, MemoryRegion('alias', 0, 8, 'controller', 'test')))

        # Retained readelf/link.json bounds from a qualified DDR deployment.
        # Source chip.elf SHA256: 6cfba0b4fbdc31cbc49fcc5c820dbefb82008149a6640642b735dbdacf046502
        # This is linked occupancy only, not complete application allocation or
        # an attribution of weights/state/KV inside shared .window backing.
        symbols = dict(__text_start=0x80000000, __data_start=0x800044e0,
                       __stack_start=0x80004a20, __stack_end=0x80006a20,
                       atlas_window_start=0x80007000, atlas_window_end=0x80010900,
                       __atlas_images_start=0x80010900, __atlas_images_end=0x8001157c)
        region = MemoryRegion('ddr', 0x80000000, 1 << 30, 'controller', 'saved link')
        bounds = [('host_code', 'code', '__text_start', '__data_start'),
                  ('host_runtime', 'runtime', '__data_start', '__stack_start'),
                  ('stack', 'stack', '__stack_start', '__stack_end'),
                  ('images', 'code', '__atlas_images_start', '__atlas_images_end')]
        # Conservatively charge each unknown persistent category the entire
        # same window; unioning retains one backing without inventing mappings.
        bounds += [(kind, kind, 'atlas_window_start', 'atlas_window_end')
                   for kind in ('weights', 'state', 'kv_cache')]
        rows = [linked_reservation(region, symbols, identity=i, kind=k, start=s, end=e,
                                   evidence='saved final ELF symbols; opaque shared window')
                for i, k, s, e in bounds]
        budget = derive_budget(region, rows, alignment_bytes=2048)
        self.assertEqual(symbols['__stack_start'] - region.base, 18976)
        self.assertEqual(rows[2].size_bytes, 8192)
        host_prefix = (symbols['__stack_end'] - region.base + 2047) // 2048 * 2048
        self.assertEqual(host_prefix, 28672)
        self.assertEqual(budget.free_ranges, ((0x80011800, 0xc0000000),))
        self.assertEqual(budget.capacity_bytes, (1 << 30) - 71680)


if __name__ == '__main__':
    unittest.main()
