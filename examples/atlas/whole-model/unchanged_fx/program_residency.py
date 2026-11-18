"""Qualification-only native residency bridge; see the example README recipe."""
from __future__ import annotations

import ctypes as ct
from hashlib import sha256
import json
from pathlib import Path
import subprocess

from model_runtime import READ, WRITE


class NativeProgramResidency:
    """Run the real generic C session against persistent, observed target IMEM.

    Legacy application IMEM writes are authenticated separately by the caller.
    They never select a predecoded body here: returned bytes contain the actual
    trampoline, all loaded words and retained historical IMEM tail.
    """
    COUNTERS = ("body_loads", "hits", "evictions", "fallbacks", "imem_words_written")

    def __init__(self, atlas_root, images, output, *, busy=lambda: False, capacity=None):
        self.images = {sha256(raw).hexdigest(): raw for raw in images}
        if not self.images or any(not raw or len(raw) % 4 for raw in self.images.values()):
            raise ValueError("residency requires nonempty complete program words")
        self.capacity = len(self.images) if capacity is None else capacity
        self.busy = busy
        self.words, self.writes, self.errors, self.payloads = {}, [], [], {}
        self.trampoline_words = self.body_words = self.fallback_words = 0
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        merlin = Path(__file__).resolve().parents[4]
        atlas_root = Path(atlas_root)
        source = Path(__file__).with_name("program_residency_bridge.c")
        files = [source, atlas_root / "runtime/atlas_host.c",
                 atlas_root / "runtime/atlas_host.h", atlas_root / "runtime/atlas_host_layout.h",
                 atlas_root / "runtime/atlas_program_slots.c", atlas_root / "runtime/atlas_program_slots.h",
                 merlin / "merlin/runtime/c/program_residency.h",
                 merlin / "examples/atlas/program_session.h"]
        self.sources = {str(path): sha256(path.read_bytes()).hexdigest() for path in files}
        identity = sha256(json.dumps(self.sources, sort_keys=True).encode()).hexdigest()
        library = output / ("residency-" + identity + ".so")
        # Compile the requested current sources explicitly; never trust a stale .so.
        subprocess.run(["cc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
                        "-shared", "-fPIC", "-I" + str(atlas_root / "runtime"),
                        "-I" + str(merlin / "merlin/runtime/c"),
                        "-I" + str(merlin / "examples/atlas"), str(source),
                        str(atlas_root / "runtime/atlas_host.c"),
                        str(atlas_root / "runtime/atlas_program_slots.c"),
                        "-o", str(library)], check=True, capture_output=True)
        self.library_sha256 = sha256(library.read_bytes()).hexdigest()
        self.lib = ct.CDLL(str(library))
        self.callbacks = READ(self._read), WRITE(self._write)
        self.lib.residency_open.argtypes = [READ, WRITE, ct.c_size_t]
        self.lib.residency_open.restype = ct.c_void_p
        self.lib.residency_prepare.argtypes = [ct.c_void_p, ct.POINTER(ct.c_uint32), ct.c_size_t]
        self.lib.residency_counter.argtypes = [ct.c_void_p, ct.c_uint]
        self.lib.residency_counter.restype = ct.c_uint64
        self.lib.residency_slot_count.argtypes = [ct.c_void_p]
        self.lib.residency_slot_count.restype = ct.c_size_t
        self.lib.residency_slot_value.argtypes = [ct.c_void_p, ct.c_size_t, ct.c_uint]
        self.lib.residency_slot_value.restype = ct.c_size_t
        self.lib.residency_close.argtypes = [ct.c_void_p]
        self.handle = self.lib.residency_open(*self.callbacks, self.capacity)
        if not self.handle:
            raise ValueError("generic residency session refused configuration")

    def _read(self, _, address):
        if address != 0x40008:
            self.errors.append("unexpected residency CSR read")
            return 0
        return 0 if self.busy() else 5

    def _write(self, _, address, word):
        if self.busy() or not 0x20000 <= address < 0x40000 or address % 4:
            self.errors.append("invalid or active residency IMEM write")
            return
        self.words[(address - 0x20000) // 4] = word
        self.writes.append((address, word))

    def counters(self):
        if not self.handle:
            raise ValueError("residency session is closed")
        return {key: int(self.lib.residency_counter(self.handle, i))
                for i, key in enumerate(self.COUNTERS)}

    def prepare(self, raw):
        digest = sha256(raw).hexdigest()
        if digest not in self.images or self.images[digest] != raw:
            raise AssertionError("residency payload differs from authenticated publication")
        if digest not in self.payloads:
            self.payloads[digest] = (ct.c_uint32 * (len(raw) // 4)).from_buffer_copy(raw)
        before = self.counters()
        self.writes.clear()
        status = self.lib.residency_prepare(self.handle, self.payloads[digest], len(raw) // 4)
        if status or self.errors:
            raise AssertionError("generic residency preparation failed: " + repr((status, self.errors)))
        after = self.counters()
        raw_words = [int.from_bytes(raw[i:i + 4], "little") for i in range(0, len(raw), 4)]
        fallback = after["fallbacks"] != before["fallbacks"]
        if fallback:
            entry = 0
            expected = [(0x20000 + i * 4, word) for i, word in enumerate(raw_words)]
            self.fallback_words += len(raw_words)
        else:
            jump = self.words.get(0, 0)
            immediate = ((jump >> 31) << 20) | (((jump >> 21) & 1023) << 1) | \
                        (((jump >> 20) & 1) << 11) | (((jump >> 12) & 255) << 12)
            entry = (immediate - (0x200000 if immediate & 0x100000 else 0)) // 2
            if jump & 0xfff != 0x6f or entry < 2 or self.words.get(1) != 0x13:
                raise AssertionError("invalid physical reset-entry JAL/NOP trampoline")
            loaded = after["body_loads"] != before["body_loads"]
            expected = ([(0x20000 + (entry + i) * 4, word) for i, word in enumerate(raw_words)]
                        if loaded else []) + [(0x20004, 0x13), (0x20000, jump)]
            self.body_words += len(raw_words) if loaded else 0
            self.trampoline_words += 2
        if self.writes != expected:
            raise AssertionError("physical writes differ from requested body and trampoline")
        extent = max(self.words) + 1
        if set(self.words) != set(range(extent)):
            raise AssertionError("persistent physical IMEM has an unauthenticated gap")
        combined = b"".join(self.words[i].to_bytes(4, "little") for i in range(extent))
        if combined[entry * 4:entry * 4 + len(raw)] != raw:
            raise AssertionError("physical selected body differs from original words")
        ranges = []
        for i in range(self.lib.residency_slot_count(self.handle)):
            offset = self.lib.residency_slot_value(self.handle, i, 0)
            count = self.lib.residency_slot_value(self.handle, i, 1)
            payload = combined[offset:offset + count]
            key = sha256(payload).hexdigest()
            if offset < 8 or key not in self.images or self.images[key] != payload:
                raise AssertionError("physical retained slot differs from authenticated publication")
            ranges.append((offset, offset + count))
        ordered = sorted(ranges)
        if any(left[1] > right[0] for left, right in zip(ordered, ordered[1:])):
            raise AssertionError("physical retained slots overlap")
        if not fallback and (entry * 4, entry * 4 + len(raw)) not in ranges:
            raise AssertionError("trampoline selected an unretained slot")
        return combined, dict(entry_word=entry, fallback=fallback,
                              mapped_image_sha256=sha256(combined).hexdigest(), **after)

    def finish(self):
        counters = self.counters()
        if self.lib.residency_close(self.handle):
            raise AssertionError("generic residency close refused")
        self.handle = None
        return dict(policy="Merlin generic caller-owned C session", default_enabled=False,
                    library_sha256=self.library_sha256, source_sha256=self.sources,
                    physical_imem_extent_words=len(self.words), trampoline_words_written=self.trampoline_words,
                    body_words_written=self.body_words, fallback_words_written=self.fallback_words,
                    **counters)
