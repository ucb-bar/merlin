"""Arena ownership, ordered fusion, and legacy admission for disabled Smol fusion."""
from copy import deepcopy
import hashlib
from pathlib import Path
import struct
import sys
import unittest

sys.path[:0] = [str(Path(__file__).absolute().parents[1]/"tools"),
                str(Path(__file__).absolute().parents[1]/"tools/host_iss")]
import kernel_library  # noqa: E402,F401
from atlas_program import assembly, dma_load_bytes
from chipyard.arena import relocate_assembly
from smol_fusion_program import stage_panel, channel_tiles


def fixture(dims=(128, 768, 768), module="vlm_with_expert.vlm.model.vision_model.embeddings.patch_embedding",
            recipe="traffic-reuse-v1"):
    m, k, n = dims
    offset = m*k+k*n+256
    flags = dict(traffic_reuse=True) if recipe == "traffic-reuse-v1" else dict(
        resident_b=k*n<=262144, resident_a=k*n>262144 and m*k<=262144)
    row = dict(index=0, call=0, module=module, physical_mkn=list(dims), logical_mkn=list(dims),
               logical_row_start=0, output_offset=offset, dram_bytes=offset+2*m*n+256,
               recipe=recipe, program_policy=recipe, output_rescale_exponent=-19, dram_base=0x90000000,
               requested_dma=dict(load_bytes=dma_load_bytes(m,k,n,**flags), store_bytes=2*m*n))
    arena = b"A"*(m*k)+b"W"*(k*n)+b"\xa5"*256+b"\xcd"*(2*m*n)+b"Z"*256
    return row, arena


class SmolFusionProgramTests(unittest.TestCase):
    def test_channel_tiles_are_pair_half_and_immutable(self):
        bias = struct.pack('<64H', *range(64))
        tiled = channel_tiles(bias, 64)
        self.assertEqual(len(tiled), 4096)
        words = struct.unpack('<2048H', tiled)
        for tile in range(2):
            for half in range(2):
                start = tile*1024+half*512
                for row in range(32):
                    self.assertEqual(words[start+row*16:start+(row+1)*16],
                                     tuple(range(tile*32+half*16, tile*32+(half+1)*16)))
        self.assertEqual(bias, struct.pack('<64H', *range(64)))

    def test_disabled_legacy_sources_and_arenas_are_exact(self):
        for dims, recipe in [((128,768,768),'traffic-reuse-v1'), ((128,768,3072),'traffic-reuse-v1'),
                             ((128,3072,768),'traffic-reuse-v1'), ((64,12288,960),'traffic-reuse-v1'),
                             ((32,32,960),'baseline-v1'), ((64,736,736),'baseline-v1')]:
            with self.subTest(dims=dims, recipe=recipe):
                row, arena = fixture(dims, recipe=recipe)
                frozen = deepcopy(row)
                result = stage_panel(row, arena)
                m,k,n = dims
                flags = dict(traffic_reuse=True) if recipe=='traffic-reuse-v1' else dict(
                    resident_b=k*n<=262144, resident_a=k*n>262144 and m*k<=262144,
                    pretranspose_resident_b=k*n<=262144)
                canonical = assembly(m,k,n,row['output_offset'],-19,**flags)
                self.assertEqual(result.source,relocate_assembly(canonical,0x100000000,0x90000000))
                self.assertIs(result.arena,arena)
                self.assertEqual(result.metadata['requested_dma'],row['requested_dma'])
                self.assertEqual(result.metadata['operations'],[])
                self.assertEqual(result.metadata['extra_bytes'],0)
                self.assertEqual(row,frozen)

    def test_bias_staging_preserves_arena_and_declares_actual_dma(self):
        row, arena = fixture()
        bias = struct.pack('<768H',*range(768))
        result = stage_panel(row, arena, operations=('bias',), bias_bf16=bias)
        meta = result.metadata
        extra = channel_tiles(bias,768)
        self.assertEqual(result.arena,arena+extra+b'Z'*256)
        self.assertEqual(meta['extra_offset'],len(arena))
        self.assertEqual(meta['extra_bytes'],768*64)
        self.assertEqual(meta['extra_sha256'],hashlib.sha256(extra).hexdigest())
        self.assertEqual(meta['total_dram_bytes'],len(result.arena))
        self.assertEqual(meta['input_regions'][0]['sha256'],meta['extra_sha256'])
        self.assertEqual(meta['requested_dma']['load_bytes'],row['requested_dma']['load_bytes']+128*768*2)
        self.assertEqual(meta['requested_dma']['store_bytes'],row['requested_dma']['store_bytes'])
        self.assertEqual(meta['scratch_registers'],[27,28])
        self.assertIn('vadd.bf16 m2, m2, m6',result.source)
        self.assertEqual(meta['host_ops_on_success'],['bias'])

    def test_fc1_bias_precedes_gelu_with_blocked_scratch(self):
        row,arena = fixture((128,768,3072),'vlm_with_expert.vlm.model.vision_model.encoder.layers.0.mlp.fc1')
        result = stage_panel(row,arena,operations=('bias','gelu_tanh'),bias_bf16=b'\0'*(3072*2))
        self.assertLess(result.source.index('vadd.bf16 m2, m2, m6'),result.source.index('vsquare.bf16 m6, m2'))
        self.assertEqual(result.metadata['scratch_registers'],[10,11])
        self.assertEqual(result.metadata['output_bytes'],128*3072*2)
        self.assertNotIn('vmatpop.fp8.acc.mxu1',result.source)

    def test_connector_has_no_extra_dma_or_bias(self):
        row,arena = fixture((64,12288,960),'vlm_with_expert.vlm.model.connector.modality_projection.proj')
        result = stage_panel(row,arena,operations=('connector_scale',))
        self.assertIs(result.arena,arena)
        self.assertIn('vli.all m8, 16888',result.source)
        self.assertEqual(result.metadata['requested_dma'],row['requested_dma'])
        self.assertEqual(result.metadata['extra_bytes'],0)

    def test_refuse_invalid_or_unsupported_sites_and_operands(self):
        row,arena = fixture()
        for operations,bias in [(('gelu_tanh',),None),(('gelu_tanh','bias'),b'\0'*1536),
                                (('bias',),None),((),b'\0'*1536),(('bias',),b'\0'*1535),
                                (('bias',),bytearray(1536)),(('bias',),b'\x80\x7f'*768),
                                (('bias','gelu_tanh'),b'\0'*1536),(('connector_scale',),None)]:
            with self.subTest(operations=operations,bias_type=type(bias)), self.assertRaises(ValueError):
                stage_panel(row,arena,operations=operations,bias_bf16=bias)
        for changed in [dict(module='action_in_proj'),dict(recipe='baseline-v1',program_policy='baseline-v1'),
                        dict(output_rescale_exponent=True),dict(dram_base=0x100000000),
                        dict(output_offset=row['output_offset']+32),dict(physical_mkn=[True,768,768]),
                        dict(requested_dma=dict(load_bytes=0,store_bytes=128*768*2))]:
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                stage_panel({**row,**changed},arena,operations=('bias',),bias_bf16=b'\0'*1536)
        for offset in (row['output_offset']-1,row['output_offset'],len(arena)-1):
            changed = bytearray(arena);changed[offset] ^= 1
            with self.subTest(offset=offset), self.assertRaises(ValueError):
                stage_panel(row,bytes(changed))


if __name__=='__main__':
    unittest.main()
