#!/usr/bin/env python3
"""Recreate the binary blobs that source/assets.S includes, by copying each labelled object out of the ELF.

usage: python3 extract_blobs.py ../smolvla-full-f2.elf ../source/assets.S OUT_DIR

Every blob is content addressed: the file name is the SHA-256 of its bytes, and the script checks that.
The blobs total about 534 MB (Atlas weights, host parameters, expected outputs, programs, and the phase archive).
Only the Python standard library is used.
"""
import argparse
import hashlib
import mmap
import os
import re
import struct
import sys
from pathlib import Path


def symbols(raw):
    if len(raw) < 64 or raw[:4] != b'\x7fELF' or raw[4] != 2 or raw[5] != 1:
        raise SystemExit('expected a little-endian ELF64 file')
    shoff, = struct.unpack_from('<Q', raw, 0x28)
    shentsize, shnum = struct.unpack_from('<HH', raw, 0x3a)
    sh = [struct.unpack_from('<IIQQQQIIQQ', raw, shoff + i * shentsize) for i in range(shnum)]
    result = {}
    for _, kind, _, _, off, size, link, _, _, _ in sh:
        if kind != 2:
            continue
        strs = raw[sh[link][4]:sh[link][4] + sh[link][5]]
        for j in range(size // 24):
            name, _, _, _, value, length = struct.unpack_from('<IBBHQQ', raw, off + 24 * j)
            result[strs[name:strs.index(b'\0', name)].decode()] = (value, length)
    segments = []
    phoff, = struct.unpack_from('<Q', raw, 0x20)
    phentsize, phnum = struct.unpack_from('<HH', raw, 0x36)
    for i in range(phnum):
        kind, _, off, vaddr, _, filesz, _, _ = struct.unpack_from('<IIQQQQQQ', raw, phoff + i * phentsize)
        if kind == 1:
            segments.append((vaddr, off, filesz))
    return result, segments


def checked(value):
    path=Path(value).absolute()
    if '..' in path.parts:
        raise ValueError('restricted or ambiguous path')
    for prefix in (*reversed(path.parents),path):
        if prefix.is_symlink():raise ValueError('symlink paths are unsupported')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('elf',type=checked, help='original, unstripped package ELF')
    parser.add_argument('assets',type=checked, help='generated source/assets.S')
    parser.add_argument('out',type=checked, help='output blob directory')
    args = parser.parse_args()
    with open(args.assets, encoding='utf-8') as stream:
        assembly = stream.read()
    pairs = re.findall(r'^(\w+):\n\.incbin "([^"]+)"', assembly, re.M)
    if not pairs or len(pairs) != len(re.findall(r'^\.incbin ', assembly, re.M)):
        raise SystemExit('assets.S must contain labelled .incbin entries')
    if any(not re.fullmatch(r'[0-9a-f]{64}\.bin', name) for _, name in pairs):
        raise SystemExit('blob names must be SHA-256 hashes followed by .bin')
    with open(args.elf, 'rb') as stream:
        if os.fstat(stream.fileno()).st_size < 64:
            raise SystemExit('expected a little-endian ELF64 file')
        with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as raw:
            try:
                extract(raw, pairs, args.out)
            except (KeyError, StopIteration, IndexError, ValueError, struct.error) as error:
                raise SystemExit('cannot extract blobs from ELF: ' + str(error)) from error


def extract(raw, pairs, out):
    table, segments = symbols(raw)
    os.makedirs(out, exist_ok=True)
    total = 0
    for label, path in pairs:
        address, size = table[label]
        vaddr, off, filesz = next(s for s in segments if s[0] <= address and address + size <= s[0] + s[2])
        if off + filesz > len(raw):
            raise ValueError('load segment extends past the end of the file')
        data = raw[off + address - vaddr: off + address - vaddr + size]
        name = os.path.basename(path)
        if hashlib.sha256(data).hexdigest() + '.bin' != name:
            raise SystemExit('content hash differs for ' + label)
        with open(os.path.join(out, name), 'wb') as stream:
            stream.write(data)
        total += size
    print(f'wrote {len(pairs)} blobs, {total} bytes, to {out}')


if __name__ == '__main__':
    main()
