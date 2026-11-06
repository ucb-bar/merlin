"""Shared immutable weights and exact per-launch images for full-model hosts."""

import hashlib
import json
import re
import struct

from atlas_host_layout import fp8_output_offset, ATLAS_DRAM_BASE, ATLAS_IMEM_BYTES
from block_bundle import integer
from bundle import ChecksumBundle
from placement import arena_extent, placement


class KernelAssets(ChecksumBundle):
    """Checksum-bound kernel sources only; never host/live admission."""

    def __init__(self, root, workload):
        self._metadata(root, workload, "kernels.json", "atlas.full_kernel_assets.v1")
        self.preparation_only = True
        self._validate_kernels()

    def _metadata(self, root, workload, filename, schema):
        super().__init__(root)
        self.metadata_name = filename
        self.provenance = json.loads(self.read(filename))
        if (workload not in ("tiny-full", "smol-full")
                or self.provenance.get("schema") != schema
                or self.provenance.get("workload") != workload):
            raise ValueError("unsupported full-model bundle")
        self.attention_kernels = self.provenance.get("attention_kernels", [])
        self.device_events = self.provenance.get("device_events")
        self.workload = workload
        self.placement = placement(self.provenance)
        self.projection_kernel_count = integer(self.provenance.get("kernel_count"), 1)
        self.kernel_count = integer(self.provenance.get("device_event_count", self.projection_kernel_count), 1)
        if self.kernel_count > 20000:
            raise ValueError("full-model launch count exceeds bound")
        self.kernels = self.provenance.get("kernels")
        if not isinstance(self.kernels, list) or len(self.kernels) != self.projection_kernel_count:
            raise ValueError("full-model launch count differs")

    def _validate_kernels(self):
        fusion = self.workload == 'smol-full' and (self.provenance.get('fusion_set','none') != 'none'
            or any(isinstance(row,dict) and row.get('fusion') for row in self.kernels) or bool(self.provenance.get('call_views')))
        if fusion:
            from smol_fusion_assets import validate_profile
            validate_profile(self.provenance)
        for index, descriptor in enumerate(self.kernels):
            if (not isinstance(descriptor, dict) or type(descriptor.get("index")) is not int
                    or descriptor["index"] != index):
                raise ValueError("full-model launch descriptor order differs")
            self.validate_kernel(descriptor)
            self.projection_initial_dram(index)
        if fusion:
            from smol_fusion_assets import validate_views
            validate_views(self.provenance,self.read)

        if self.attention_kernels or self.device_events is not None or self.kernel_count != self.projection_kernel_count:
            from smol_attention_assets import event_plan, validate_attention
            if self.workload != 'smol-full' or self.device_events != event_plan() or len(self.attention_kernels) != 200 or self.kernel_count != 2915:
                raise ValueError('Incomplete typed Smol device inventory')
            for index, item in enumerate(self.attention_kernels):
                if item.get('index') != index or item.get('phase') != index:
                    raise ValueError('Attention descriptor order differs')
                validate_attention(item, self.read)

    def validate_kernel(self, item):
        if not isinstance(item, dict):
            raise ValueError("invalid full-model launch descriptor")
        for key in ("source", "original", "activation", "weights", "expected"):
            if not isinstance(item.get(key), str) or item[key] not in self.sums:
                raise ValueError("unlisted full-model kernel payload: " + key)
        dimensions = item.get("physical_mkn")
        if (not isinstance(dimensions, list) or len(dimensions) != 3
                or any(type(value) is not int or value <= 0 or value % 32 for value in dimensions)):
            raise ValueError("full-model physical dimensions must be positive tile multiples")
        m, k, n = dimensions
        output_size = 2*m*n
        fused_total = None
        if item.get('fusion'):
            if self.workload != 'smol-full' or self.provenance.get('fusion_set','none') == 'none':
                raise ValueError('Fused kernel requires an explicit Smol fusion profile')
            from smol_fusion_assets import geometry
            output_size,_,fused_total = geometry(item)
        offset = integer(item.get("output_offset"))
        total = integer(item.get("total_dram_bytes"), 1)
        if (item.get("dram_base") != self.placement["arena_base"]
                or type(item.get("dram_base")) is not int
                or offset != fp8_output_offset(m, k, n)
                or total != (fused_total if fused_total is not None else offset + 2 * m * n + 256)
                or total > 64 * 1024 * 1024):
            raise ValueError("full-model staging extent differs")
        arena_extent(self.provenance, total)
        if not isinstance(item.get("initial_sha256"), str) or not re.fullmatch(
                r"[0-9a-f]{64}", item["initial_sha256"]):
            raise ValueError("missing full-model initial image identity")
        original = self.read(item["original"])
        if (not original or len(original) % 4 or len(original) + 20 > ATLAS_IMEM_BYTES
                or original[-4:] != struct.pack("<I", 0x73)):
            raise ValueError("full-model program must fit IMEM and end in ECALL")
        for key, size in (("activation", m * k), ("weights", k * n), ("expected", output_size)):
            if len(self.read(item[key])) != size:
                raise ValueError("full-model payload extent differs: " + key)
        requested = item.get("requested_dma")
        if (not isinstance(requested, dict) or not {"load_bytes", "store_bytes"} <= requested.keys()
                or requested.keys() - {"load_bytes", "store_bytes", "response_beats"}):
            raise ValueError("invalid full-model requested DMA counters")
        for value in requested.values():
            integer(value)
        if requested["store_bytes"] != output_size:
            raise ValueError("full-model DMA store extent differs")

    def kernel_paths(self, index):
        integer(index)
        if index >= self.kernel_count:
            raise ValueError("full-model launch index exceeds contract")
        if self.device_events is not None:
            event = self.device_events[index]
            item = (self.attention_kernels if event['kind'] == 'attention' else self.kernels)[event['kernel']]
        else:
            item = self.kernels[index]
        result = dict(item, requested_dma=dict(item['requested_dma']))
        if 'physical_mkn' in item: result['physical_mkn'] = list(item['physical_mkn'])
        return result

    def projection_initial_dram(self, index):
        item = self.kernels[index]
        if item.get('fusion'):
            from smol_fusion_assets import validate_program
            return validate_program(item,self.read).arena
        m, _, n = item["physical_mkn"]
        initial = (self.read(item["activation"]) + self.read(item["weights"])
                   + b"\xa5" * 256 + b"\xcd" * (2 * m * n) + b"\x5a" * 256)
        if (len(initial) != item["total_dram_bytes"]
                or hashlib.sha256(initial).hexdigest() != item["initial_sha256"]):
            raise ValueError("full-model reconstructed initial image differs")
        return initial

    def initial_dram(self, index):
        item = self.kernel_paths(index)
        if item.get('kind') == 'attention':
            return self.read(item['initial'])
        projection = self.device_events[index]['kernel'] if self.device_events is not None else index
        return self.projection_initial_dram(projection)


class FullBundle(KernelAssets):
    def __init__(self, root, workload):
        self._metadata(root, workload, "provenance.json", "atlas.full_bundle.v1")
        self.build = json.loads(self.read("host/build.json"))
        for name in ("host/host.elf", self.provenance.get("contract")):
            if not isinstance(name, str) or name not in self.sums:
                raise ValueError("unlisted full-model ELF or contract")
        contract = self.provenance["contract"]
        if self.sums[contract] != self.provenance.get("contract_sha256"):
            raise ValueError("full-model contract identity differs")
        self._validate_kernels()
