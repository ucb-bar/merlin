"""Build-time DMA relocation and physical DRAM bounds for bare ELF packages."""
import re
import struct
from elf_image import Image

DEFAULT_ARENA = 0x100000000


def add_arguments(parser):
    number = lambda s: int(s, 0)
    parser.add_argument("--arena-base", type=number,
        help="DMA arena physical base (default: retained base, normally 0x100000000)")
    parser.add_argument("--dram-base", type=number, default=0x80000000)
    parser.add_argument("--dram-bytes", type=number, default=64 << 30)


def window(base, size):
    if (type(base) is not int or type(size) is not int or base < 0 or base % 64
            or size <= 0 or base + size > 1 << 64 or (base & 0xffffffff) + size > 1 << 32):
        raise ValueError("DMA arena must be aligned and fit one low32 window")


def _dram_register(load, rd, rs1, rs2):
    reg, others = (rs1, (rd, rs2)) if load else (rd, (rs1, rs2))
    if reg == 0 or reg in others:
        raise ValueError("DMA relocation requires a writable, distinct DRAM register")
    return reg


def relocate_assembly(source, old_base, new_base, *, force=False):
    """Keep local offsets/VMEM operands, change high32 and displace DMA bytes."""
    if old_base == new_base and not force:
        return source
    if old_base != DEFAULT_ARENA or re.search(r"\bx31\b", source):
        raise ValueError("relocation requires the retained base and unused x31")
    setup = re.compile(r"^li x30, 1$", re.MULTILINE)
    if len(setup.findall(source)) != 1:
        raise ValueError("expected one retained DMA high-half setup")
    source = setup.sub(f"li x30, {new_base >> 32}", source)
    lines = [f"li x31, {new_base & 0xffffffff}"]
    for line in source.splitlines():
        m = re.fullmatch(r"dma\.(load|store) x(\d+), x(\d+), x(\d+), ch[0-7]", line)
        if re.search(r"\bdma\.(load|store)\b", line) and m is None:
            raise ValueError("unsupported DMA transfer syntax")
        if re.search(r"\bdma\.config\b", line) and not re.fullmatch(r"dma\.config x30, ch[0-7]", line):
            raise ValueError("unexpected DMA configuration source or syntax")
        reg = _dram_register(m[1] == "load", *map(int, m.group(2, 3, 4))) if m else None
        lines.extend([f"add x{reg}, x{reg}, x31", line, f"sub x{reg}, x{reg}, x31"] if m else [line])
    return "\n".join(lines) + "\n"


def _li(reg, value):
    low = ((value + 2048) & 4095) - 2048
    high = ((value - low) >> 12) & 0xfffff
    return ([high << 12 | reg << 7 | 0x37] if high else []) + (
        [((low & 4095) << 20) | (reg if high else 0) << 15 | reg << 7 | 0x13] if low or not high else [])


def relocate(raw, old_base, new_base):
    """Relocate retained raw bytes, including Atlas word-relative branches."""
    if old_base == new_base:
        return raw
    if old_base != DEFAULT_ARENA:
        raise ValueError("raw relocation requires retained 4 GiB base")
    words = list(struct.unpack("<" + "I" * (len(raw) // 4), raw))
    chunks, starts = [], []
    cursor = len(_li(31, new_base & 0xffffffff))
    setups = transfers = 0
    for w in words:
        op, rd, rs1, rs2 = w & 127, (w >> 7) & 31, (w >> 15) & 31, (w >> 20) & 31
        if op in (0x17, 0x67) and (op != 0x67 or (w >> 12) & 7 == 0):
            raise ValueError("unsupported PC-relative instruction")
        # Decode scalar fields for every admitted format. Unknown formats use
        # all possible register fields conservatively; tensor-only formats do
        # not access x31 even when their tensor/immediate bits happen to be 31.
        if op in (0x03, 0x13, 0x67, 0x73):
            scalar_regs = (rd, rs1)
        elif op in (0x23, 0x63):
            scalar_regs = (rs1, rs2)
        elif op in (0x07, 0x7f):
            scalar_regs = (rs1,)
        elif op in (0x37, 0x6f):
            scalar_regs = (rd,)
        elif op in (0x0f, 0x57, 0x5f, 0x6b, 0x77):
            scalar_regs = ()
        else:
            scalar_regs = (rd, rs1, rs2)
        if 31 in scalar_regs:
            raise ValueError("raw relocation requires unused x31")
        starts.append(cursor)
        chunk = [w]
        if w == 0x00100f13:  # addi x30,x0,1, the retained DMA high32.
            chunk = _li(30, new_base >> 32); setups += 1
        if op == 0x7f and w >> 25 == 0 and rs1 != 30:
            raise ValueError("unexpected DMA configuration source")
        if op == 0x7b:
            if w >> 25 not in (0, 1): raise ValueError("unknown DMA transfer")
            reg = _dram_register(w >> 25 == 0, rd, rs1, rs2)
            chunk = [31 << 20 | reg << 15 | reg << 7 | 0x33, w,
                     0x40000000 | 31 << 20 | reg << 15 | reg << 7 | 0x33]
            transfers += 1
        chunks.append(chunk); cursor += len(chunk)
    starts.append(cursor)
    if setups != 1 or not transfers:
        raise ValueError("expected retained DMA setup and transfers")
    for i, w in enumerate(words):
        op = w & 127
        if op not in (0x63, 0x6f): continue
        if op == 0x63:
            imm = ((w >> 31) << 12) | (((w >> 7) & 1) << 11) | (((w >> 25) & 63) << 5) | (((w >> 8) & 15) << 1)
            bits = 13
        else:
            imm = ((w >> 31) << 20) | (((w >> 12) & 255) << 12) | (((w >> 20) & 1) << 11) | (((w >> 21) & 1023) << 1)
            bits = 21
        target = i + (imm - (1 << bits) if imm & (1 << (bits - 1)) else imm) // 2
        if not 0 <= target < len(words): raise ValueError("branch outside program")
        delta = (starts[target] - starts[i]) * 2
        if not -(1 << (bits - 1)) <= delta < 1 << (bits - 1): raise ValueError("relocated branch overflow")
        if op == 0x63:
            chunks[i] = [(w & 0x01fff07f) | ((delta >> 12) & 1) << 31 | ((delta >> 11) & 1) << 7
                         | ((delta >> 5) & 63) << 25 | ((delta >> 1) & 15) << 8]
        else:
            chunks[i] = [(w & 0xfff) | ((delta >> 20) & 1) << 31 | ((delta >> 12) & 255) << 12
                         | ((delta >> 11) & 1) << 20 | ((delta >> 1) & 1023) << 21]
    result = _li(31, new_base & 0xffffffff) + [w for chunk in chunks for w in chunk]
    return struct.pack("<" + "I" * len(result), *result)


def audit(out, dram_base, dram_bytes):
    """Check p_memsz and every SHF_ALLOC section, including NOLOAD/stack/BSS."""
    raw = (out / "host.elf").read_bytes(); image = Image(raw)
    limit = dram_base + dram_bytes
    if dram_base < 0 or dram_bytes <= 0 or limit > 1 << 64:
        raise ValueError("invalid physical DRAM range")
    spans = []
    for s in image.segments:
        start, size = s["physical"], s["memory_bytes"]
        if (s["address"] != start or not dram_base <= start <= limit - size
                or s["file_bytes"] > size or s["offset"] + s["file_bytes"] > len(raw)):
            raise ValueError("PT_LOAD outside physical DRAM range")
        if size and any(start < end and begin < start + size for begin, end in spans):
            raise ValueError("overlapping physical PT_LOAD ranges")
        if size: spans.append((start, start + size))
    shoff = struct.unpack_from("<Q", raw, 40)[0]
    shsize, count = struct.unpack_from("<HH", raw, 58)
    allocated = 0
    for i in range(count):
        s = struct.unpack_from("<IIQQQQIIQQ", raw, shoff + i * shsize)
        if s[2] & 2 and s[5]:
            start, size = s[3], s[5]; allocated += 1
            if (not dram_base <= start <= limit - size
                    or not any(begin <= start and start + size <= end for begin, end in spans)):
                raise ValueError("allocated section outside physical DRAM/PT_LOAD range")
    # HTIF CRT reserves heap and one boot-hart stack only through symbols,
    # outside PT_LOAD p_memsz. The freestanding layer/block use alloc sections.
    symbols = {name: row["address"] for name, row in image.symbols.items()}
    reserved = []
    if "__stack_start" in symbols:
        start = symbols["__stack_start"]
        reserved.append(("stack", start, start + symbols["__stack_size"]))
    if "__heap_end" in symbols:
        reserved.append(("heap", symbols["_end"], symbols["__heap_end"]))
    for name, start, end in reserved:
        if not dram_base <= start <= end <= limit:
            raise ValueError(name + " outside physical DRAM range")
        if any(start < hi and lo < end for lo, hi in spans):
            raise ValueError(name + " overlaps physical PT_LOAD")
    if not any(begin <= image.entry < end for begin, end in spans):
        raise ValueError("entry outside physical DRAM")
    return dict(dram_base=dram_base, dram_bytes=dram_bytes, segments=image.segments, allocated_sections=allocated,
        reserved=reserved, boot_harts=1)
