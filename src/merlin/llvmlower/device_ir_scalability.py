"""Cheap static device-build cost observations and explicitly selected budgets.

Counts describe canonical textual LLVM, not instruction effects, hardware limits
or numerical correctness. The original build caller selects and freezes budgets;
emitted programs and environment variables never select or relax them. Counted
device loops should keep these static costs bounded across larger logical shapes.
"""

from __future__ import annotations

import hashlib
import os
import stat
import time
from dataclasses import asdict, dataclass
from pathlib import Path

SCHEMA = "merlin.device_ir_scalability.v1"
_IDENTIFIER_BYTES = frozenset(b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.$-")
_LABEL_BYTES = _IDENTIFIER_BYTES | frozenset(b" \t:")
_BEFORE_WORD = _IDENTIFIER_BYTES | frozenset(b"%@")
_AFTER_WORD = _IDENTIFIER_BYTES | frozenset(b":")
_LIMITS = {
    "bytes": "max_bytes",
    "function_code_bytes": "max_code_bytes",
    "lines": "max_lines",
    "inline_asm_sites": "max_inline_asm_sites",
    "call_sites": "max_call_sites",
    "functions": "max_functions",
    "basic_blocks": "max_basic_blocks",
}


def _word_spans(data):
    """Observe exact unquoted keywords, retaining LLVM identifier boundaries."""
    for prefix in (b"call", b"invoke", b"asm"):
        pos = 0
        while (start := data.find(prefix, pos)) >= 0:
            word = b"callbr" if prefix == b"call" and data.startswith(b"callbr", start) else prefix
            end = start + len(word)
            if (start == 0 or data[start - 1] not in _BEFORE_WORD) and (
                end == len(data) or data[end] not in _AFTER_WORD
            ):
                yield start, end, word
            pos = end


def _define_prefix(prefix):
    return prefix.startswith(b"define") and len(prefix) > 6 and prefix[6] in b" \t\r\f\v\n"


def _next_special(chunk, positions, start, selected):
    """Advance each fixed-character search monotonically within this chunk."""
    result = len(chunk)
    for char in selected:
        found = positions[char]
        if 0 <= found < start:
            found = positions[char] = chunk.find(bytes((char,)), start)
        if found >= 0:
            result = min(result, found)
    return result


@dataclass(frozen=True)
class DeviceIRBudget:
    mode: str = "report_only"
    max_bytes: int | None = None
    max_code_bytes: int | None = None
    max_lines: int | None = None
    max_inline_asm_sites: int | None = None
    max_call_sites: int | None = None
    max_functions: int | None = None
    max_basic_blocks: int | None = None

    def __post_init__(self):
        if self.mode not in {"report_only", "enforce"}:
            raise ValueError("device IR budget mode must be report_only or enforce")
        values = [getattr(self, key) for key in _LIMITS.values()]
        if any(value is not None and (type(value) is not int or value <= 0) for value in values):
            raise ValueError("device IR limits must be explicit positive integers")
        if self.mode == "enforce" and all(value is None for value in values):
            raise ValueError("enforced device IR budget requires at least one explicit limit")

    def record(self):
        return asdict(self)

    def evaluate(self, analysis):
        if type(analysis) is not dict or analysis.get("schema") != SCHEMA:
            raise ValueError("device IR budget requires its actual source analysis")
        exceeded = [
            {"metric": key, "observed": analysis[key], "limit": getattr(self, field)}
            for key, field in _LIMITS.items()
            if getattr(self, field) is not None
            and analysis.get(key) is not None
            and analysis[key] > getattr(self, field)
        ]
        accepted = self.mode == "report_only" or (analysis["lexical_complete"] and not exceeded)
        reason = "device_ir_static_expansion_budget" if exceeded else "device_ir_lexical_observation_incomplete"
        return {
            "status": "accepted" if accepted else "refused",
            "policy": self.record(),
            "exceeded": exceeded,
            "reason": None if accepted else reason,
            "action": None if accepted else "Retain counted loops or reduce static expansion; raw LLVM is retained.",
            "scope": "operator-selected build resource budget; not hardware legality or correctness",
        }


class _Counts:
    def __init__(self):
        self.functions = self.basic_blocks = self.call_sites = self.inline_asm_sites = 0
        self.function_code_bytes = self.module_bytes = self.global_definition_bytes = 0
        self.in_function = self.pending = self.first_block = False
        self.reset()

    def reset(self):
        self.prefix = self.suffix = self.tail = b""
        self.prior = b" "
        self.calls = self.asms = self.colons = 0
        self.raw_bytes = 0
        self.label_chars = True

    def feed(self, data, final=False):
        self.prefix = (self.prefix + data)[:512]
        self.suffix = (self.suffix + data)[-512:]
        self.colons += data.count(b":")
        if self.label_chars:
            self.label_chars = all(char in _LABEL_BYTES for char in data)
        joined = self.prior + self.tail + data
        cut = len(joined) if final else max(1, len(joined) - 32)
        matches = list(_word_spans(joined))
        for start, end, _word in matches:
            if start >= 1 and start < cut < end:
                cut = start
        for start, end, word in matches:
            if 1 <= start < cut and end <= cut:
                if word == b"asm":
                    self.asms += 1
                else:
                    self.calls += 1
        self.prior, self.tail = joined[cut - 1 : cut], joined[cut:]
        if not final:
            return
        prefix, suffix = self.prefix.strip(), self.suffix.strip()
        if self.in_function or self.pending or _define_prefix(prefix):
            self.function_code_bytes += self.raw_bytes
        else:
            self.module_bytes += self.raw_bytes
            if prefix.startswith(b"@") and b"=" in prefix:
                self.global_definition_bytes += self.raw_bytes
        label = self.label_chars and self.colons == 1 and suffix.endswith(b":")
        if not label:
            self.call_sites += self.calls
            if self.calls:
                self.inline_asm_sites += self.asms
        if _define_prefix(prefix):
            self.functions += 1
            self.pending = True
        if self.pending and suffix.endswith(b"{"):
            self.in_function, self.pending, self.first_block = True, False, True
            self.basic_blocks += 1  # explicit or implicit function entry
        elif self.in_function and label:
            if not self.first_block:
                self.basic_blocks += 1
            self.first_block = False
        elif self.in_function and prefix and prefix != b"}":
            self.first_block = False
        if prefix == b"}":
            self.in_function = False
        self.reset()


def analyze_device_ir(path):
    """One bounded-buffer byte/lexical pass; quotes/comments do not contribute ops.

    The digest, physical lines and keyword counts are gathered from the same
    actual read. No LLVM parser, model data, subprocess or dependency scan is used.
    This analysis can be shared with an already deduplicated identical artifact;
    admission decisions are freshly bound to caller shape/source/tool/policy.
    """
    started = time.monotonic()
    path = Path(path)
    digest, counts = hashlib.sha256(), _Counts()
    total = newlines = quoted_bytes = 0
    state, escaped, last = "normal", False, None
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_mode")
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("device IR is not an ordinary file")
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            total += len(chunk)
            newlines += chunk.count(b"\n")
            last = chunk[-1]
            pos = 0
            positions = {char: chunk.find(bytes((char,))) for char in (10, 34, 59, 92)}
            while pos < len(chunk):
                if state == "comment":
                    end = _next_special(chunk, positions, pos, (10,))
                    if end == len(chunk):
                        counts.raw_bytes += len(chunk) - pos
                        break
                    counts.raw_bytes += end + 1 - pos
                    counts.feed(b"", final=True)
                    state, pos = "normal", end + 1
                elif state == "string":
                    if escaped:
                        counts.raw_bytes += 1
                        quoted_bytes += 1
                        escaped = False
                        pos += 1
                        continue
                    end = _next_special(chunk, positions, pos, (10, 34, 92))
                    if end == len(chunk):
                        counts.raw_bytes += len(chunk) - pos
                        quoted_bytes += len(chunk) - pos
                        break
                    delta = end + 1 - pos
                    counts.raw_bytes += delta
                    quoted_bytes += delta
                    pos = end
                    char = chunk[pos]
                    pos += 1
                    if char == 34:
                        quoted_bytes -= 1  # contents only; delimiters are not literal payload
                    if char == 92:
                        escaped = True
                    elif char == 34:
                        state = "normal"
                    else:
                        counts.feed(b"", final=True)
                else:
                    end = _next_special(chunk, positions, pos, (10, 34, 59))
                    counts.feed(chunk[pos:end])
                    counts.raw_bytes += end - pos
                    if end == len(chunk):
                        break
                    char, pos = chunk[end], end + 1
                    counts.raw_bytes += 1
                    if char == 10:
                        counts.feed(b"", final=True)
                    elif char == 59:
                        state = "comment"
                    else:
                        counts.feed(b"quoted")  # preserve quoted identifiers for block-label observation
                        state = "string"
        if total and last != 10:
            counts.feed(b"", final=True)
        after = os.fstat(stream.fileno())
    current = path.stat()
    if any(
        getattr(before, key) != getattr(after, key) or getattr(after, key) != getattr(current, key) for key in fields
    ):
        raise ValueError("device IR changed during static build analysis")
    return {
        "schema": SCHEMA,
        "sha256": digest.hexdigest(),
        "bytes": total,
        "lines": newlines + int(bool(total and last != 10)),
        "functions": counts.functions,
        "basic_blocks": counts.basic_blocks,
        "call_sites": counts.call_sites,
        "inline_asm_sites": counts.inline_asm_sites,
        "lexical_complete": state != "string",
        "function_code_bytes": counts.function_code_bytes,
        "module_bytes": counts.module_bytes,
        "global_definition_bytes": counts.global_definition_bytes,
        "quoted_bytes": quoted_bytes,
        "elapsed_s": time.monotonic() - started,
        "scope": "canonical textual LLVM static cost observations",
    }


def inspect_device_ir(path, budget):
    """Refuse an explicit total-byte cap before any scan; never cache that as PASS."""
    if type(budget) is not DeviceIRBudget:
        raise ValueError("device build requires an explicitly selected DeviceIRBudget")
    size = Path(path).stat().st_size
    if budget.mode == "enforce" and budget.max_bytes is not None and size > budget.max_bytes:
        analysis = {
            "schema": SCHEMA,
            "sha256": None,
            "bytes": size,
            "lexical_complete": False,
            "elapsed_s": 0.0,
            "scope": "early total-file resource refusal; contents were not scanned",
        }
    else:
        analysis = analyze_device_ir(path)
    return {"analysis": analysis, "decision": budget.evaluate(analysis)}
