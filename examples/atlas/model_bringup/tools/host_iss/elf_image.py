"""Read the admitted ELF64 image and locate exact MMIO memory instructions."""

import struct


class Image:
    def __init__(self, raw):
        self.raw = raw
        h = struct.unpack_from("<16sHHIQQQIHHHHHH", raw)
        assert h[0][:7] == b"\x7fELF\x02\x01\x01" and h[1] == 2 and h[2] == 243
        self.entry = h[4]
        self.segments = []
        self.symbols = {}
        assert h[9] == 56 and h[11] == 64
        for index in range(h[10]):
            p = struct.unpack_from("<IIQQQQQQ", raw, h[5] + index * h[9])
            if p[0] == 1:
                self.segments.append(dict(address=p[3], physical=p[4], offset=p[2], file_bytes=p[5], memory_bytes=p[6],
                    flags=p[1]))
        sections = [struct.unpack_from("<IIQQQQIIQQ", raw, h[6] + i * h[11]) for i in range(h[12])]
        for section in sections:
            if section[1] != 2:
                continue
            assert section[9] == 24
            strings = sections[section[6]]
            table = raw[strings[4] : strings[4] + strings[5]]
            for offset in range(section[4], section[4] + section[5], 24):
                no, info, other, si, value, size = struct.unpack_from("<IBBHQQ", raw, offset)
                if not no:
                    continue
                name = table[no : table.index(b"\0", no)].decode()
                self.symbols[name] = dict(address=value, size=size, section=si)

    def bytes_at(self, address, size):
        for segment in self.segments:
            if segment["address"] <= address and address + size <= segment["address"] + segment["file_bytes"]:
                offset = segment["offset"] + address - segment["address"]
                result = self.raw[offset : offset + size]
                assert len(result) == size
                return result
        raise ValueError("requested bytes are not in a file-backed load segment")

    def instructions(self, name):
        symbol = self.symbols[name]
        raw = self.bytes_at(symbol["address"], symbol["size"])
        offset = 0
        result = []
        while offset < len(raw):
            half = struct.unpack_from("<H", raw, offset)[0]
            length = 2 if half & 3 != 3 else 4
            assert offset + length <= len(raw)
            word = int.from_bytes(raw[offset : offset + length], "little")
            result.append((symbol["address"] + offset, length, word))
            offset += length
        return result


def signed(value, bits):
    return value - (1 << bits) if value & (1 << (bits - 1)) else value


def memory_instruction(word, length):
    if length == 4:
        opcode = word & 127
        kind = (word >> 12) & 7
        base = (word >> 15) & 31
        if opcode == 3 and kind in (2, 6):
            return dict(store=False, base=base, reg=(word >> 7) & 31, offset=signed(word >> 20, 12), signed=kind == 2)
        if opcode == 0x23 and kind == 2:
            offset = (word >> 25) << 5 | ((word >> 7) & 31)
            return dict(store=True, base=base, reg=(word >> 20) & 31, offset=signed(offset, 12), signed=False)
    elif length == 2 and word & 3 == 0 and word >> 13 in (2, 6):
        offset = ((word >> 10) & 7) << 3 | ((word >> 6) & 1) << 2 | ((word >> 5) & 1) << 6
        return dict(store=word >> 13 == 6, base=8 + ((word >> 7) & 7), reg=8 + ((word >> 2) & 7), offset=offset,
            signed=word >> 13 == 2)
    return None


def mmio_sites(image):
    result = {}
    for name, store in [("atlas_host_direct_read32", False), ("atlas_host_direct_write32", True)]:
        sites = []
        for pc, length, word in image.instructions(name):
            access = memory_instruction(word, length)
            if access:
                assert access["base"] == 11 and access["offset"] == 0 and access["store"] == store
                if store:
                    assert access["reg"] == 12
                else:
                    assert access["reg"] == 10
                sites.append(dict(pc=pc, length=length, word=word, **access))
        assert len(sites) == 1, "expected exactly one real MMIO load/store instruction per accessor"
        result[sites[0]["pc"]] = sites[0]
    return result
