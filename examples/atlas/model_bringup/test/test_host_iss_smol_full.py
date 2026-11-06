"""Full Smol admission binds selected code to captured data and bounded DDR."""

import copy
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

TOOLS = Path(__file__).absolute().parents[1] / "tools"
sys.path[:0] = [str(TOOLS), str(TOOLS / "host_iss")]
import smol_full_contract as contract
import kernel_library  # noqa: E402,F401
from atlas_program import assembly, dma_load_bytes
from chipyard.arena import relocate, relocate_assembly
from smol_full_phases import V1, V2
from smol_full_plan import panels


class FullSmolTest(unittest.TestCase):
    def fixture(self, reuse=False, arena=0x100000000):
        m, k, n = 32, 32, 960
        base = 0x100000000
        captured = dict(index=0, physical_mkn=[m, k, n], output_offset=m*k+k*n+256,
                        total_dram_bytes=m*k+k*n+2*m*n+512, output_rescale_exponent=-15,
                        source="baseline.S", original="baseline.bin", activation="activation.bin",
                        weights="weights.bin", expected="expected.bin", dram_base=base,
                        requested_dma=dict(load_bytes=m*k+k*n, store_bytes=2*m*n))
        source = "li x30, 1\ndma.config x30, ch0\ndma.load x1, x2, x3, ch0\necall\n"
        raw = struct.pack("<4I", 0x00100f13, 30 << 15 | 0x7f, 3 << 20 | 2 << 15 | 1 << 7 | 0x7b, 0x73)
        selected = dict(captured, source="selected.S", original="selected.bin", dram_base=arena,
                        program_policy="traffic-reuse-v1" if reuse else "baseline-v1")
        if reuse:
            source = assembly(m, k, n, captured["output_offset"], -15, traffic_reuse=True)
            selected["requested_dma"] = dict(load_bytes=dma_load_bytes(m, k, n, traffic_reuse=True), store_bytes=2*m*n)
        payloads = {"baseline.S": b"li x30, 1\ndma.config x30, ch0\ndma.load x1, x2, x3, ch0\necall\n",
                    "baseline.bin": raw,
                    "selected.S": relocate_assembly(source, base, arena).encode(),
                    "selected.bin": struct.pack("<2I", 0x13, 0x73) if reuse else relocate(raw, base, arena)}
        physical = dict(arena_base=arena, dram_base=0x80000000, dram_bytes=64 << 30)
        policy = "qualified-reuse-v1" if reuse else "captured-baseline-v1"
        bundle = SimpleNamespace(provenance=dict(placement=physical, program_policy=policy), read=payloads.__getitem__)
        data = dict(placement=physical, program_policy=policy, kernels=[selected])
        return bundle, data, [captured], payloads

    def test_captured_baseline_and_relocated_code_are_bound(self):
        for arena in (0x100000000, 0x90000000):
            bundle, data, captured, _ = self.fixture(arena=arena)
            contract.selected_kernels(bundle, data, captured)

    def test_public_reuse_code_and_dma_are_bound(self):
        for arena in (0x100000000, 0x90000000):
            bundle, data, captured, payloads = self.fixture(reuse=True, arena=arena)
            program = SimpleNamespace(assemble=lambda: [0x13, 0x73])
            contract.selected_kernels(bundle, data, captured, lambda _: program)
            contract.selected_kernels(bundle, data, captured)
            payloads["selected.S"] += b"nop\n"
            with self.assertRaisesRegex(ValueError, "reuse source"):
                contract.selected_kernels(bundle, data, captured)

    def test_wrong_reuse_binary_and_missing_assembler_fail_static_admission(self):
        bundle, data, captured, payloads = self.fixture(reuse=True)
        with self.assertRaisesRegex(ValueError, "pinned assembly"):
            contract.selected_kernels(bundle, data, captured)
        payloads["selected.bin"] = payloads["baseline.bin"]
        program = SimpleNamespace(assemble=lambda: [0x13, 0x73])
        with self.assertRaisesRegex(ValueError, "reuse binary"):
            contract.selected_kernels(bundle, data, captured, lambda _: program)

    def test_immutable_pointers_must_be_readonly_and_fully_file_backed(self):
        image = SimpleNamespace(segments=[dict(address=0x80000000, file_bytes=64,
                                               memory_bytes=128, flags=4)])
        contract.readonly_span(image, 0x80000020, 32)
        with self.assertRaisesRegex(ValueError, "read-only"):
            contract.readonly_span(image, 0x80000020, 33)
        image.segments[0]["flags"] = 6
        with self.assertRaisesRegex(ValueError, "read-only"):
            contract.readonly_span(image, 0x80000000, 1)

    def test_changed_inputs_layout_policy_and_dma_fail_before_execution(self):
        bundle, data, captured, _ = self.fixture(reuse=True)
        for change in (dict(activation="replaced.bin"), dict(dram_base=True),
                       dict(output_offset=data["kernels"][0]["output_offset"]+64),
                       dict(program_policy="unqualified"), dict(requested_dma=dict(load_bytes=32, store_bytes=61440))):
            altered = copy.deepcopy(data)
            altered["kernels"][0].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                contract.selected_kernels(bundle, altered, captured)
        data["placement"]["dram_bytes"] = 1 << 31
        with self.assertRaises(ValueError):
            contract.selected_kernels(bundle, data, captured)

    def test_finite_reference_rejects_bf16_nan_and_inf(self):
        self.assertTrue(contract.finite(struct.pack("<2H", 0x3f80, 0), "uint16"))
        for word in (0x7f80, 0x7fc0, 0xff80):
            self.assertFalse(contract.finite(struct.pack("<H", word), "uint16"))
        self.assertFalse(contract.finite(struct.pack("<f", float("nan")), "float32"))

    def test_fusion_registry_uses_nobits_zero_fill_without_file_read(self):
        def segment(address,flags,offset,file_bytes,memory_bytes):
            return dict(address=address,physical=address,flags=flags,offset=offset,
                        file_bytes=file_bytes,memory_bytes=memory_bytes)
        segments=[segment(0x80000000,5,0,64,64),segment(0x80001000,4,64,64,64),
                  segment(0x80002000,6,128,64,0x20000),segment(0x90000000,6,192,0,1024)]
        values=dict(__atlas_dram_start=0x90000000,__atlas_dram_end=0x90000400,
                    __bss_start=0x80002040,__bss_end=0x80002100,
                    __stack_bottom=0x80003000,__stack_top=0x80013000)
        symbols={name:dict(address=value) for name,value in values.items()}
        symbols['state']=dict(address=0x80002040,size=104)
        def bytes_at(address,size):
            if address>=0x80002040:raise ValueError('NOBITS is not file backed')
            return bytes(size)
        image=SimpleNamespace(entry=0x80000000,raw=bytes(192),segments=segments,symbols=symbols,bytes_at=bytes_at)
        physical=dict(arena_base=0x90000000,dram_base=0x80000000,dram_bytes=0x80000000)
        with patch.object(contract,'stop_sites',return_value={}):
            result=contract.audit_image(image,None,1024,physical,fusion=True)
            self.assertTrue(result['host_and_arena_disjoint'])
            symbols['state']['address']=0x80002038
            with self.assertRaisesRegex(ValueError,'registry'):
                contract.audit_image(image,None,1024,physical,fusion=True)
            symbols['state'].update(address=0x80002040,size=112)
            with self.assertRaisesRegex(ValueError,'registry'):
                contract.audit_image(image,None,1024,physical,fusion=True)
            # The production rodata-after-arena layout places writable BSS above the arena.
            segments[1:3]=[segment(0x90001000,4,64,64,64),segment(0x90002000,6,128,64,0x20000)]
            values.update(__bss_start=0x90002040,__bss_end=0x90002100,__stack_bottom=0x90003000,__stack_top=0x90013000)
            symbols.update({name:dict(address=value) for name,value in values.items()})
            symbols['state']=dict(address=0x90002040,size=104)
            self.assertTrue(contract.audit_image(image,None,1024,physical,fusion=True)['host_and_arena_disjoint'])

    def test_completion_uses_selected_profile_counts_and_cycle_extent(self):
        for profile in (V1,V2):
            data=dict(kernel_count=2715,kernels=panels(),reference_buffers=[])
            data['kernels'][-1]['total_dram_bytes']=1024
            if profile==V2:
                data.update(attention_count=200,device_event_count=2915,device_events=profile.events(),
                    attention_kernels=[dict(index=e['kernel'],phase=e['kernel'],before_call=e['before_call'])
                                       for e in profile.events() if e['kind']=='attention'])
            names=('atlas_workload_mailbox','tohost','fromhost','smol_full_graph_state',
                   'smol_full_vision_states','smol_full_raw','smol_full_host','smol_full_cycles')
            symbols={name:dict(address=i*65536) for i,name in enumerate(names)}
            context=[1,2,3,2715,1421,0,0,0x100000000,4,5,12706304,786432,3145728,100000000,
                     symbols['smol_full_cycles']['address'],bytes(24),0,0,0,bytes(4),
                     6 if profile.attention_count else 0,profile.attention_count,0]
            initial=struct.pack('<15Q24sIIi4s3Q',*context)
            context[5:7]=[2715,1421]
            context[15]=struct.pack('<QQI4s',0x100000000,1024,2,bytes(4))
            context[16]=1;context[22]=profile.attention_count
            host=bytearray(112);struct.pack_into('<I',host,108,profile.launch_count)
            payloads=dict(zip(names,(
                struct.pack('<16Q',contract.MAGIC,2,profile.launch_count,1421,1,10,1,0,0,2715,1421,profile.launch_count,0,0,0,0),
                struct.pack('<Q',1),bytes(8),struct.pack('<4I',1,10,1273,0),struct.pack('<8I',*((1,74,1,0)*2)),
                struct.pack('<15Q24sIIi4s3Q',*context),bytes(host),struct.pack('<I',7)*profile.launch_count)))
            memory={symbols[name]['address']:raw for name,raw in payloads.items()}
            image=SimpleNamespace(symbols=symbols,bytes_at=lambda address,size:initial)
            remote=SimpleNamespace(reg=lambda index:42,read=lambda address,size:memory[address])
            device=SimpleNamespace(completed=[{}]*profile.launch_count)
            bundle=SimpleNamespace(kernel_paths=lambda index:dict(original='program',expected='expected'),
                                   read=lambda member:b'',provenance={})
            with tempfile.TemporaryDirectory() as directory,patch.object(contract,'contract',return_value=(data,{})), \
                 patch.object(contract,'stop_sites',return_value={42:4}),patch.object(contract,'verify_kernel_receipt') as verify:
                result=contract.verify_completion(remote,image,device,bundle,Path(directory))
                self.assertEqual(result['expected_starts'],profile.launch_count)
                self.assertEqual(verify.call_count,profile.launch_count)
                self.assertEqual((Path(directory)/'cycles.bin').stat().st_size,profile.cycle_bytes)


if __name__ == "__main__":
    unittest.main()
