"""Functional SiFive UART bridge for exact board ELFs in generic QEMU.

Only the admitted TX data/control/divisor operations are modeled. This is not
UART hardware execution, baud timing or an FPGA target configuration.
"""
from elf_image import memory_instruction, signed

MASK = (1 << 64)-1
INIT_NAMES = ('main','atlas_workload_main','atlas_block_trap_report','tiny_full_trap_report','report_init','fail','atlas_result_uart_init')


def constants(known, pc, length, word):
    """Conservative straight-line integer constants for UART init discovery."""
    if length == 4:
        op, rd, rs1, rs2, kind = word & 127, word >> 7 & 31, word >> 15 & 31, word >> 20 & 31, word >> 12 & 7
        value = None
        if op == 0x37: value = signed(word & 0xfffff000,32) & MASK
        elif op == 0x17: value = (pc+signed(word & 0xfffff000,32)) & MASK
        elif op in (0x13,0x1b) and kind == 0 and rs1 in known:
            value = (known[rs1]+signed(word >> 20,12)) & MASK
            if op == 0x1b: value = signed(value & 0xffffffff,32) & MASK
        elif op == 0x33 and kind == 0 and word >> 25 == 0 and rs1 in known and rs2 in known:
            value = (known[rs1]+known[rs2]) & MASK
        elif op in (0x63,0x67,0x6f):
            known.clear();known[0] = 0;return
        elif op in (0x23,0x27,0x0f): return
        known.pop(rd,None)
        if rd and value is not None: known[rd] = value
    else:
        quad, kind, rd, rs2 = word & 3, word >> 13, word >> 7 & 31, word >> 2 & 31
        immediate = signed(((word >> 12 & 1) << 5) | rs2,6)
        value = None
        if quad == 1 and kind == 2: value = immediate & MASK
        elif quad == 1 and kind == 3 and rd not in (0,2): value = (immediate << 12) & MASK
        elif quad == 1 and kind in (0,1) and rd in known:
            value = (known[rd]+immediate) & MASK
            if kind == 1: value = signed(value & 0xffffffff,32) & MASK
        elif quad == 2 and kind == 4 and rs2:
            if not word >> 12 & 1: value = known.get(rs2)
            elif rd in known and rs2 in known: value = (known[rd]+known[rs2]) & MASK
        elif quad in (0,2) and kind in (5,6,7): return
        else:
            known.clear();known[0] = 0;return
        known.pop(rd,None)
        if rd and value is not None: known[rd] = value
    known[0] = 0


def validate_profile(profile):
    if (not isinstance(profile,dict) or profile.get('kind') != 'sifive'
            or type(profile.get('base')) is not int or not 0 < profile['base'] <= MASK-24
            or profile['base'] % 4):
        raise ValueError('explicit aligned bounded SiFive UART profile required')
    divisor=profile.get('divisor')
    if divisor is not None and (type(divisor) is not int or not 0 < divisor < 1 << 16):
        raise ValueError('explicit nonzero 16-bit UART divisor required')
    return dict(profile)


def sites(image, profile):
    profile=validate_profile(profile)
    base = profile['base']
    result = {}
    name = 'atlas_result_uart_putchar'
    if name not in image.symbols:
        raise ValueError('named generated UART callback required')
    for pc,length,word in image.instructions(name):
        access = memory_instruction(word,length)
        if access:
            result[pc] = dict(access,length=length,uart_register=base,uart_role='tx-data')
    if not any(not x['store'] for x in result.values()) or not any(x['store'] for x in result.values()):
        raise ValueError('UART callback requires actual TX poll and store instructions')
    initialized = set()
    for name in INIT_NAMES:
        if name not in image.symbols: continue
        known = {0:0}
        for pc,length,word in image.instructions(name):
            access = memory_instruction(word,length)
            if access and access['store'] and access['base'] in known:
                address = (known[access['base']]+access['offset']) & MASK
                if address in (base+8,base+24):
                    result[pc] = dict(access,length=length,uart_register=address,uart_role='initialize')
                    initialized.add(address)
            constants(known,pc,length,word)
    if base+8 not in initialized:
        raise ValueError('cannot admit actual UART TX control initialization instruction')
    if (base+24 in initialized) != (profile.get('divisor') is not None):
        raise ValueError('UART divisor instruction/profile binding differs')
    return result


class SifiveUART:
    def __init__(self, profile, stream):
        self.profile, self.stream = validate_profile(profile), stream
        self.enabled, self.characters, self.polls, self.initializations = False, 0, 0, 0
        self.divisor = None

    def access(self, remote, site):
        role=site.get('uart_role')
        expected=site.get('uart_register')
        base=self.profile['base']
        if (role not in ('tx-data','initialize')
                or (role == 'tx-data' and expected != base)
                or (role == 'initialize' and (not site['store'] or expected not in (base+8,base+24)))):
            raise ValueError('unadmitted UART site role/register')
        address = (remote.reg(site['base'])+site['offset']) & MASK
        if address != site['uart_register']:
            raise ValueError('actual UART instruction address differs from admitted register')
        offset = address-self.profile['base']
        if site['store']:
            value = remote.reg(site['reg']) & 0xffffffff
            if offset == 8 and value == 1:
                if self.profile.get('divisor') is not None and self.divisor != self.profile['divisor']:
                    raise ValueError('UART TX enable before admitted divisor initialization')
                self.enabled = True;self.initializations += 1
            elif offset == 24 and self.profile.get('divisor') is not None and value == self.profile['divisor']:
                self.divisor = value
            elif offset == 0 and self.enabled and value <= 255:
                self.stream.write(bytes([value]));self.stream.flush();self.characters += 1
            else:
                raise ValueError('unadmitted UART store or TX before enable')
        elif offset == 0 and self.enabled:
            self.polls += 1
            remote.set_reg(site['reg'],0)
        else:
            raise ValueError('unadmitted UART read or TX poll before enable')

    def summary(self):
        return dict(backend='modeled-sifive-uart-mmio',profile=self.profile,tx_enabled=self.enabled,
                    tx_control_writes=self.initializations,characters=self.characters,polls=self.polls,
                    divisor=self.divisor,hardware_execution=False,baud_timing=False)
