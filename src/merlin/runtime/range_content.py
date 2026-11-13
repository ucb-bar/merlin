"""Caller-owned exact-byte retention policy for synchronous target write hooks."""
from __future__ import annotations


class RangeContentSession:
    """One exclusive endpoint owner supplies all writes and mutable footprints.

    The target validates its address space and generation. No identity/hash or
    host pointer substitutes for complete immutable bytes. Overlapping writes
    invalidate first, including failed/partial writes. Unknown mutations and
    reset must invalidate the session; target launch errors clear every entry.
    """
    def __init__(self, *, max_bytes, max_entries=4096):
        if type(max_bytes) is not int or max_bytes <= 0 or type(max_entries) is not int or max_entries <= 0:
            raise ValueError("content retention requires positive bounds")
        self.max_bytes, self.max_entries = max_bytes, max_entries
        self.entries = {}
        self.active = self.closed = False
        self.generation = None
        self.hits = self.skipped_bytes = self.writes = self.written_bytes = 0
        self.evictions = self.failures = self.invalidations = 0

    def _idle(self):
        if self.closed or self.active:
            raise ValueError("content retention session is closed or active")

    @staticmethod
    def _range(address, count):
        if type(address) is not int or address < 0 or type(count) is not int or count <= 0:
            raise ValueError("invalid content range")
        return address, address + count

    def invalidate_all(self):
        self.entries.clear()
        self.invalidations += 1

    def observe_generation(self, generation):
        self._idle()
        if generation != self.generation:
            self.invalidate_all()
            self.generation = generation

    def invalidate(self, address, count):
        first, last = self._range(address, count)
        for key in list(self.entries):
            if first < key[0] + key[1] and key[0] < last:
                del self.entries[key]

    def write(self, address, payload, writer):
        self._idle()
        snapshot = bytes(payload)
        self._range(address, len(snapshot))
        key = address, len(snapshot)
        if self.entries.get(key) == snapshot:
            self.hits += 1
            self.skipped_bytes += len(snapshot)
            return 0
        self.invalidate(address, len(snapshot))
        try:
            status = writer(snapshot)
        except BaseException:
            self.failures += 1
            self.invalidate_all()
            raise
        if status:
            self.failures += 1
            self.invalidate_all()
            return status
        self.writes += 1
        self.written_bytes += len(snapshot)
        if len(snapshot) <= self.max_bytes:
            if len(self.entries) == self.max_entries or sum(len(value) for value in self.entries.values()) + len(snapshot) > self.max_bytes:
                self.entries.clear()
                self.evictions += 1
            self.entries[key] = snapshot
        return 0

    def begin_launch(self, mutable_ranges):
        self._idle()
        try:
            ranges = [self._range(address, count) for address, count in mutable_ranges]
        except (TypeError, ValueError):
            self.invalidate_all()
            raise
        for first, last in ranges:
            self.invalidate(first, last - first)
        self.active = True

    def end_launch(self, *, success):
        if self.closed or not self.active:
            raise ValueError("content retention launch is not active")
        self.active = False
        if not success:
            self.failures += 1
            self.invalidate_all()

    def close(self):
        self._idle()
        record = dict(hits=self.hits, skipped_bytes=self.skipped_bytes, writes=self.writes,
                      written_bytes=self.written_bytes, evictions=self.evictions,
                      failures=self.failures, invalidations=self.invalidations)
        self.entries.clear()
        self.closed = True
        return record
