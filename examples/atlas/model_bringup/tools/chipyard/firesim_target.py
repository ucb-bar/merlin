"""Static ELF admission and packaging for a proposed EE290 FireSim target.

This software contract does not identify an existing bitstream or build hardware.
Keep HTIF and require the owner to match the selected physical DRAM contract.
"""
import json
import struct

import kernel_library  # noqa: E402,F401
from atlas_tooling import sha
from elf_image import Image


def package(out, name, base, dram_bytes, *, uart=None):
    raw = (out / 'host.elf').read_bytes()
    image = Image(raw)
    if image.entry != base:
        raise ValueError('FireSim target requires entry at DRAM base')
    if struct.unpack_from('<I', raw, 48)[0] & 6 != 4:
        raise ValueError('FireSim target requires the RV64 double-float ABI')
    addresses = []
    for symbol in ('tohost', 'fromhost'):
        address = image.symbols[symbol]['address']
        if address % 8 or not any(s['flags'] & 2 and s['physical'] <= address
                                 and address + 8 <= s['physical'] + s['memory_bytes']
                                 for s in image.segments):
            raise ValueError('HTIF symbol is not aligned writable DRAM')
        addresses.append(address)
    if addresses[0] == addresses[1]:
        raise ValueError('aliased HTIF mailboxes')
    contract = dict(
        schema='atlas.firesim_elf_contract.v1', status='PASS_STATIC_ONLY',
        hardware_config_status='PROPOSED_NOT_ELABORATED_OR_BUILT',
        required_soc='EE290 Shuttle/Saturn/Atlas with FireSim TSI and FASED bridges',
        dram_base=base, dram_bytes=dram_bytes,
        console='HTIF write syscall via TSI/FESVR; manager captures stdout in uartlog',
        completion='HTIF tohost=(exit_code<<1)|1 plus exact workload PASS/check lines',
        loader='ELF PT_LOAD via TSI; fast loadmem when FASED MainMemory is present',
        elf_sha256=sha(raw), entry=image.entry,
        tohost=addresses[0], fromhost=addresses[1], segments=image.segments,
        host_elf_executed=False, hardware_execution=False)
    if uart is not None:
        contract['console'] = 'physical UART MMIO; manager captures uartlog'
        contract['uart'] = uart
    (out / 'firesim-contract.json').write_text(json.dumps(contract, indent=2) + '\n')
    # No host-init hook: the binary already exists. Marshal install is an owner action.
    workload = dict(name=name, distro=dict(name='bare', opts={}), workdir='.',
                    bin=str(out / 'host.elf'), firesim=dict(simulation_outputs=['uartlog']))
    (out / 'marshal.json').write_text(json.dumps(workload, indent=2) + '\n')
    return contract
