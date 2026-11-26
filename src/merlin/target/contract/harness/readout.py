"""Solution-neutral runner-owned harness renderer (the logical kernel ABI, version 2).

A graded package emits a kernel FUNCTION; the runner owns the C caller that embeds the leaf tensors,
calls that function once per measured invocation, and prints the ``OUT``/``METRIC``/``DONE`` protocol
the console parser reads. This module writes that caller for ANY target, from declared data only:

* ``logical_kernel_abi`` in ``src/merlin/resources/contracts/mlir_oot_backend_contract.yaml`` -- the pointer order,
  the pointee layout (dense row-major LOGICAL tensors), the output poison, the readout format;
* the target contract's ``harness_abi.entry_symbol`` / ``cycle_window_metric`` and its
  ``logical_harness`` block -- the host instructions that read the cycle counter and wait for
  outstanding accelerator work (conventions of the host ISA, not of any compiler);
* the command buffer's own tensor declarations, and Merlin's reference semantics for which
  destinations are results (:mod:`merlin.runtime.engine.reference`).

The caller does not pad, gather, repack or reorder anything, and includes no target library header.
Tiling, padding, packing and layout transforms are the candidate compiler's job; the harness hands it
the tensors exactly as the capsule states them, in one declared pointer order.

Every output buffer is filled with a declared poison byte before EVERY invocation (outside the cycle
window), so a kernel that does not write an output -- or wrote it only during the warm-up -- cannot
pass on stale or zero-initialised memory.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from math import prod
from pathlib import Path
from typing import Any

from merlin.compiler.ir.logical_abi import (
    Buffer,
    HarnessRenderError,
    LogicalAbi,
    _ordered,
    _output_buffer,
    _produced,
    _shape,
    kernel_abi_from_commands,
    kernel_arg_order,
    logical_abi,
    logical_inputs,
    logical_interface,
    logical_outputs,
)
from merlin.target.contract.harness.harness_render import _COHERENT_OUTPUT_DTYPES, _HOST_STORAGE_BYTES, container_for

# ---- harness_render ----------------------------------------------------------------------------


class _Readout:
    """The output transport: ``OUT`` lines (default), an ``OUTSUM`` digest above the console cap, or an
    explicitly selected full-value policy. Every buffer is dense, so element ``i*cols+j`` is simply
    index ``i*cols+j`` unless the candidate declared an explicit storage encoding."""

    def __init__(self, cb, buffers, storage, policy, value_cap):
        from merlin.target.contract.harness.readback_policy import (
            COHERENT_DUMP_V1,
            COHERENT_PACKET_V1,
            COHERENT_PACKET_V2,
            FULL_VALUES_B64,
            FULL_VALUES_BIN,
            OUT_DIGEST_V1,
            selected,
        )

        policy = selected(policy)
        self.digest = policy is not None and policy.transport == OUT_DIGEST_V1
        self.cb, self.storage, self.value_cap = cb, storage, value_cap
        self.outputs = [b for b in buffers if b.kind == "output"]
        self.b64 = policy is not None and policy.transport == FULL_VALUES_B64
        self.binary = policy is not None and policy.transport == FULL_VALUES_BIN
        self.packet_v2 = policy is not None and policy.transport == COHERENT_PACKET_V2
        self.packet = policy is not None and policy.transport in (COHERENT_PACKET_V1, COHERENT_PACKET_V2)
        self.coherent = policy is not None and policy.transport in (
            COHERENT_DUMP_V1,
            COHERENT_PACKET_V1,
            COHERENT_PACKET_V2,
        )
        self.digested = False
        self.packet_expected: list = []
        self.packet_capacity = None
        if self.coherent:
            self._check_coherent(buffers)
        self._lines = []
        for buf in self.outputs:
            rows, cols = buf.matrix
            if storage is not None:
                binding = storage[buf.name]
                shape = binding.encoding.logical_shape
                rows, cols = (prod(shape[:-1]), shape[-1]) if shape else (1, 1)
                terms = [str(binding.encoding.offset_elements)]
                for divisor, extent, stride in binding.logical_offset_terms():
                    terms.append(f"(((i * {cols} + j) / {divisor}) % {extent}) * {stride}")
                element = f"T_{buf.name}[{' + '.join(terms)}]"
                container = container_for(binding.encoding.dtype)
                contiguous = _contiguous_output_pointer(buf.name, binding)
            else:
                element = f"T_{buf.name}[i * {cols} + j]"
                container = container_for(buf.dtype)
                contiguous = f"T_{buf.name}"
            self._lines += self._one(buf, rows, cols, container, element, contiguous)
        if self.packet:
            from merlin.runtime.harness.output_decoding import out_bin_packet_capacity, out_packet_v2_capacity

            try:
                self.packet_capacity = (
                    out_packet_v2_capacity(self.packet_expected)
                    if self.packet_v2
                    else out_bin_packet_capacity(self.packet_expected)
                )
            except ValueError as exc:
                raise HarnessRenderError(str(exc)) from exc

    def _check_coherent(self, buffers):
        kabi = self.cb.get("kernel_abi") or {}
        names = [b.name for b in self.outputs]
        tensors = self.cb.get("tensors") or {}
        if (
            kabi.get("kind") != "whole_program"
            or not names
            or len(names) > 1024
            or any(not n.isascii() or not all(c.isalnum() or c == "_" for c in n) for n in names)
        ):
            raise HarnessRenderError("coherent dump requires a closed whole-program output roster")
        if any(
            tensors[n].get("role") != "output" or tensors[n].get("dtype") not in _COHERENT_OUTPUT_DTYPES for n in names
        ):
            raise HarnessRenderError("coherent dump requires declared writable outputs of supported physical dtype")
        if set(names) != {n for n, s in tensors.items() if isinstance(s, dict) and s.get("role") == "output"}:
            raise HarnessRenderError("coherent dump output roster differs from declared output tensors")
        total = 0
        for buf in self.outputs:
            elements = self.storage[buf.name].encoding.storage_elements if self.storage is not None else buf.elements
            total += elements * container_for(buf.dtype).word_bytes
            if total > _HOST_STORAGE_BYTES:
                raise HarnessRenderError("coherent dump exceeds the bounded output region total")

    def _one(self, buf, rows, cols, container, element, contiguous):
        name = buf.name
        loop = f"  for (long i = 0; i < {rows}; i++) for (long j = 0; j < {cols}; j++)"
        sign = 1 if container.signed else 0
        scan = "signed" if container.signed else "unsigned"
        scan_type = "int64_t" if container.signed else "uint64_t"
        count = rows * cols
        if self.coherent and not self.packet:
            return []
        if self.digest:
            # One XXH64 of the output's dense bytes, computed by this grader-rendered harness over the
            # buffer the candidate wrote (src/merlin/runtime/native/baremetal/out_digest.h).
            if contiguous is None or not name.isascii() or not all(c.isalnum() or c == "_" for c in name):
                raise HarnessRenderError("an output digest needs a dense, closed-identifier output buffer")
            nbytes = count * container.word_bytes
            return [
                f'  printf("OUT_DIGEST {name} %lu %016lx\\n", (unsigned long){nbytes}UL, '
                f"(unsigned long)merlin_out_digest({contiguous}, {nbytes}ULL));"
            ]
        if (self.b64 or self.binary or self.packet) and (
            not name.isascii() or not all(c.isalnum() or c == "_" for c in name)
        ):
            raise HarnessRenderError("packed output name is not a closed ASCII identifier")
        if self.packet:
            from merlin.runtime.harness.output_decoding import ExpectedOutput

            self.packet_expected.append(
                ExpectedOutput(
                    name=name,
                    rows=rows,
                    cols=cols,
                    logical_dtype=buf.dtype,
                    container_signed=container.signed,
                    container_max_bytes=container.word_bytes,
                    logical_shape=tuple(buf.shape),
                )
            )
            if self.packet_v2:
                mode = "e" if container.word_bytes > 1 and buf.dtype != "f32" else "r"
                sign_char = "s" if container.signed else "u"
                encode = (
                    f"  if (!merlin_out_packet_v2_words_i32(&merlin_out_v2, {contiguous}, {count}ULL)) return 2;"
                    if container.ctype == "int32_t" and contiguous is not None and mode == "e"
                    else f"{loop} if (!merlin_out_packet_v2_word(&merlin_out_v2, (uint64_t)({element}))) return 2;"
                )
                return [
                    "  {",
                    f"  if (!merlin_out_bin_memory_cstr(&merlin_packet_sink, "
                    f'"OUT_PKT_BEGIN v2 {name} {rows} {cols} {buf.dtype} {container.word_bytes} {sign_char} {mode}\\n")) return 2;',
                    f"  merlin_out_packet_v2_init(&merlin_out_v2, {count}ULL, {container.word_bytes}, "
                    f"{sign}, {int(mode == 'e')}, merlin_packet_write);",
                    encode,
                    "  if (!merlin_out_packet_v2_finish(&merlin_out_v2)) return 2;",
                    '  if (!merlin_out_bin_memory_cstr(&merlin_packet_sink, "OUT_PKT_END v2 ")) return 2;',
                    "  if (!merlin_out_bin_memory_hex16(&merlin_packet_sink, merlin_out_v2.checksum)) return 2;",
                    '  if (!merlin_out_bin_memory_cstr(&merlin_packet_sink, "\\n")) return 2;',
                    "  }",
                ]
            pack = (
                f"  if (!merlin_out_bin_words_i32(&merlin_out, {contiguous}, {count}ULL)) return 2;"
                if container.ctype == "int32_t" and contiguous is not None
                else f"{loop} if (!merlin_out_bin_word(&merlin_out, (uint64_t)({element}))) return 2;"
            )
            return [
                "  {",
                f"  merlin_out_b64_range_init(&merlin_out_range, {count}ULL, {container.word_bytes}u, {sign});",
                f"{loop} if (!merlin_out_b64_range_{scan}(&merlin_out_range, ({scan_type})({element}))) return 2;",
                "  unsigned merlin_wire_width = merlin_out_b64_range_width(&merlin_out_range);",
                "  int merlin_wire_signed = merlin_out_b64_range_wire_signed(&merlin_out_range);",
                "  if (!merlin_wire_width || merlin_wire_signed < 0) return 2;",
                f'  if (!merlin_out_bin_memory_cstr(&merlin_packet_sink, "OUT_BIN_BEGIN v1 {name} ")) return 2;',
                f"  if (!merlin_out_bin_memory_decimal(&merlin_packet_sink, {rows}ULL)) return 2;",
                '  if (!merlin_out_bin_memory_cstr(&merlin_packet_sink, " ")) return 2;',
                f"  if (!merlin_out_bin_memory_decimal(&merlin_packet_sink, {cols}ULL)) return 2;",
                '  if (!merlin_out_bin_memory_cstr(&merlin_packet_sink, " ")) return 2;',
                "  if (!merlin_out_bin_memory_decimal(&merlin_packet_sink, merlin_wire_width)) return 2;",
                '  if (!merlin_out_bin_memory_cstr(&merlin_packet_sink, merlin_wire_signed ? " s " : " u ")) return 2;',
                f"  if (!merlin_out_bin_memory_decimal(&merlin_packet_sink, {count}ULL * merlin_wire_width)) return 2;",
                '  if (!merlin_out_bin_memory_cstr(&merlin_packet_sink, "\\n")) return 2;',
                f"  merlin_out_bin_init(&merlin_out, {count}ULL, merlin_wire_width, "
                + "merlin_wire_signed, merlin_packet_write);",
                pack,
                "  if (!merlin_out_bin_finish(&merlin_out)) return 2;",
                '  if (!merlin_out_bin_memory_cstr(&merlin_packet_sink, "OUT_BIN_END v1 ")) return 2;',
                "  if (!merlin_out_bin_memory_hex16(&merlin_packet_sink, merlin_out.checksum)) return 2;",
                '  if (!merlin_out_bin_memory_cstr(&merlin_packet_sink, "\\n")) return 2;',
                "  }",
            ]
        if self.b64:
            literal = f"OUT_B64_BEGIN v1 {name} {rows} {cols} 8 s\\n"
            header = f"OUT_B64_BEGIN v1 {name} {rows} {cols} %u %c\\n"
            pack = (
                f"  if (!merlin_out_b64_words_i32(&merlin_out, {contiguous}, {count}ULL)) return 2;"
                if container.ctype == "int32_t" and contiguous is not None
                else f"{loop} if (!merlin_out_b64_word(&merlin_out, (uint64_t)({element}))) return 2;"
            )
            return [
                "  {",
                f"  merlin_out_b64_range_init(&merlin_out_range, {count}ULL, {container.word_bytes}u, {sign});",
                f"{loop} if (!merlin_out_b64_range_{scan}(&merlin_out_range, ({scan_type})({element}))) return 2;",
                "  unsigned merlin_wire_width = merlin_out_b64_range_width(&merlin_out_range);",
                "  int merlin_wire_signed = merlin_out_b64_range_wire_signed(&merlin_out_range);",
                "  if (!merlin_wire_width || merlin_wire_signed < 0) return 2;",
                f'  char merlin_begin[sizeof("{literal}")];',
                f"  sprintf(merlin_begin, \"{header}\", merlin_wire_width, merlin_wire_signed ? 's' : 'u');",
                "  printstr(merlin_begin);",
                f"  merlin_out_b64_init(&merlin_out, {count}ULL, merlin_wire_width, merlin_wire_signed, printstr);",
                pack,
                "  if (!merlin_out_b64_finish(&merlin_out)) return 2;",
                '  printstr("OUT_B64_END\\n");',
                "  }",
            ]
        if self.binary:
            biggest = f"OUT_BIN_BEGIN v1 {name} {rows} {cols} 8 s {count * 8}\\n"
            header = f"OUT_BIN_BEGIN v1 {name} {rows} {cols} %u %c %llu\\n"
            return [
                "  {",
                f"  merlin_out_b64_range_init(&merlin_out_range, {count}ULL, {container.word_bytes}u, {sign});",
                f"{loop} if (!merlin_out_b64_range_{scan}(&merlin_out_range, ({scan_type})({element}))) return 2;",
                "  unsigned merlin_wire_width = merlin_out_b64_range_width(&merlin_out_range);",
                "  int merlin_wire_signed = merlin_out_b64_range_wire_signed(&merlin_out_range);",
                "  if (!merlin_wire_width || merlin_wire_signed < 0) return 2;",
                f'  char merlin_begin[sizeof("{biggest}")];',
                f'  sprintf(merlin_begin, "{header}", merlin_wire_width, '
                f"merlin_wire_signed ? 's' : 'u', (unsigned long long)({count}ULL * merlin_wire_width));",
                "  printstr(merlin_begin);",
                f"  merlin_out_bin_init(&merlin_out, {count}ULL, merlin_wire_width, merlin_wire_signed, printbuf);",
                f"{loop} if (!merlin_out_bin_word(&merlin_out, (uint64_t)({element}))) return 2;",
                "  if (!merlin_out_bin_finish(&merlin_out)) return 2;",
                '  char merlin_end[sizeof("OUT_BIN_END v1 0123456789abcdef\\n")];',
                '  sprintf(merlin_end, "OUT_BIN_END v1 %016llx\\n", (unsigned long long)merlin_out.checksum);',
                "  printstr(merlin_end);",
                "  }",
            ]
        from merlin.runtime.engine.command_buffer import OUTPUT_DIGEST_LINE

        print_element = container.printf_element(element)
        if self.value_cap is None or count <= self.value_cap:
            return [f'  printf("OUT {name} {rows} {cols}");', f"{loop} {print_element}", '  printf("\\n");']
        self.digested = True
        return [
            "  merlin_outsum = 1469598103934665603ULL;",
            f"{loop} MERLIN_OUTSUM_ADD({print_element[len('printf(') :]}",
            f'  printf("{OUTPUT_DIGEST_LINE} {name} {rows} {cols} %016llx\\n", merlin_outsum);',
        ]

    def lines(self) -> list[str]:
        return list(self._lines)

    def includes(self) -> str:
        if self.digest:
            return '#include "out_digest.h"\n'
        if self.b64:
            return '#include "out_b64.h"\nextern void printstr(const char *);\n'
        if self.binary:
            return (
                '#include "out_b64.h"\n#include "out_bin.h"\n'
                "extern void printstr(const char *);\nextern int printbuf(const void *, size_t);\n"
            )
        if self.packet:
            includes = '#include "out_b64.h"\n#include "out_bin.h"\n#include "out_bin_memory.h"\n'
            return includes + ('#include "out_packet_v2.h"\n' if self.packet_v2 else "")
        return ""

    def declarations(self) -> list[str]:
        decls: list[str] = []
        if self.coherent:
            # Propose (never assert) the span the outputs occupy; the trusted post-link reader must
            # prove the output symbols tile it exactly, or refuse.
            names = [b.name for b in self.outputs]
            first, last = names[-1], names[0]
            buf = self.outputs[0]
            elements = self.storage[last].encoding.storage_elements if self.storage is not None else buf.elements
            decls.append(
                '__asm__(".globl begin_signature\\n"\n'
                f'        ".set begin_signature, T_{first}\\n"\n'
                '        ".globl end_signature\\n"\n'
                f'        ".set end_signature, T_{last}+{elements * container_for(buf.dtype).word_bytes}\\n");'
            )
        if self.b64:
            decls += ["static merlin_out_b64 merlin_out;", "static merlin_out_b64_range merlin_out_range;"]
        if self.binary:
            decls += ["static merlin_out_bin merlin_out;", "static merlin_out_b64_range merlin_out_range;"]
        if self.packet:
            decls += [
                f"unsigned char merlin_readback_packet[{self.packet_capacity}] __attribute__((used));",
                "volatile uint64_t merlin_readback_packet_used __attribute__((used));",
                "static merlin_out_bin_memory merlin_packet_sink;",
                "static merlin_out_packet_v2 merlin_out_v2;" if self.packet_v2 else "static merlin_out_bin merlin_out;",
                *([] if self.packet_v2 else ["static merlin_out_b64_range merlin_out_range;"]),
                "static int merlin_packet_write(const void *bytes, size_t count) {",
                "  return merlin_out_bin_memory_append(&merlin_packet_sink, bytes, count);",
                "}",
            ]
        if self.digested:
            decls.append(
                "static unsigned long long merlin_outsum;\n"
                "#define MERLIN_OUTSUM_ADD(...) do { char b_[48]; int n_ = sprintf(b_, __VA_ARGS__); "
                "for (int k_ = 0; k_ < n_; k_++) merlin_outsum = (merlin_outsum ^ (unsigned char)b_[k_]) "
                "* 1099511628211ULL; } while (0)"
            )
        return decls

    def setup(self) -> list[str]:
        if not self.packet:
            return []
        return [
            f"  merlin_out_bin_memory_init(&merlin_packet_sink, merlin_readback_packet, {self.packet_capacity}u, "
            "&merlin_readback_packet_used);",
            "  if (!merlin_packet_sink.valid) return 2;",
        ]

    def finish(self) -> list[str]:
        if not self.packet:
            return []
        return [
            '  if (!merlin_out_bin_memory_cstr(&merlin_packet_sink, "DONE\\n")) return 2;',
            '  __asm__ __volatile__("fence rw,rw" ::: "memory");',
            "  if (!merlin_out_bin_memory_finish(&merlin_packet_sink)) return 2;",
            '  __asm__ __volatile__("fence rw,rw" ::: "memory");',
        ]


def _contiguous_output_pointer(name: str, binding) -> str | None:
    encoding = binding.encoding
    shape = tuple(encoding.logical_shape)
    offset = encoding.offset_elements
    if (
        any(type(d) is not int or d <= 0 for d in shape)
        or any(extent > 1 and stride != divisor for divisor, extent, stride in binding.logical_offset_terms())
        or type(offset) is not int
        or offset < 0
        or prod(shape) <= 0
        or prod(shape) > encoding.storage_elements - offset
    ):
        return None
    return f"T_{name} + {offset}"


def _identifier(name) -> bool:
    return (
        isinstance(name, str)
        and bool(name)
        and name.isascii()
        and (name[0].isalpha() or name[0] == "_")
        and all(c.isalnum() or c == "_" for c in name)
    )


def _compact_caller(cb, prepared, *, abi: LogicalAbi, hooks: HostHooks, warm_profile) -> str:
    """One warm call, restore mutable initial bytes, one measured call, typed readback.

    ``prepared`` is the selected backend's trusted host preparation (``prepare_compact_caller``); it
    re-verifies its identity against ``cb`` and is read only through ``compact_contract_json``,
    ``binding.base_views()`` and ``derived.target_abi.base_access`` / ``.base_alignments``.
    """
    from merlin.analysis.programs.storage_encoding import GroupedAxesStorage
    from merlin.target.contract.harness.harness_render import HostHooks, _preamble, container_for

    verify = getattr(prepared, "verify_identity", None)
    if not callable(verify) or not hasattr(prepared, "compact_contract_json"):
        raise HarnessRenderError("exact host-prepared compact caller required")
    verify(cb)
    contract = json.loads(prepared.compact_contract_json)
    symbol = contract["compact_symbol"]
    if not _identifier(symbol):
        raise HarnessRenderError("compact symbol is not a C identifier")
    declarations, restore = [], []
    arenas = prepared.binding.base_views()
    target_abi = prepared.derived.target_abi
    for i, (arena, access, alignment) in enumerate(
        zip(arenas, target_abi.base_access, target_abi.base_alignments, strict=True)
    ):
        name = f"merlin_compact_base_{i}"
        initializer = ",".join(str(v) for v in arena) if access != "write" else "0"
        declarations.append(
            f"{'const ' if access == 'read' else ''}unsigned char {name}[{len(arena)}] "
            f"__attribute__((aligned({alignment}), used)) = {{{initializer}}};"
        )
        if access == "readwrite":
            declarations.append(
                f"static const unsigned char merlin_compact_initial_{i}[{len(arena)}] = {{{initializer}}};"
            )
            restore.append(f"  __builtin_memcpy({name}, merlin_compact_initial_{i}, {len(arena)});")
        elif access == "write":
            restore.append(f"  __builtin_memset({name}, {abi.poison_byte}, {len(arena)});")
    bindings = sorted(contract["bindings"], key=lambda row: row["argument_index"])
    outputs = set(cb["kernel_abi"]["outputs"])
    prints = []
    for i, arg in enumerate(cb["kernel_abi"]["args"]):
        name = arg["tensor"]
        if not _identifier(name):
            raise HarnessRenderError("compact output names must be safe identifiers")
        if name not in outputs:
            continue
        encoding = GroupedAxesStorage.from_dict(cb["params"]["storage_encodings"][name])
        container = container_for(encoding.dtype)
        rows, cols = (
            (prod(encoding.logical_shape[:-1]), encoding.logical_shape[-1]) if encoding.logical_shape else (1, 1)
        )
        terms = [str(encoding.offset_elements)]
        divisor = prod(encoding.logical_shape)
        for extent, stride in zip(encoding.logical_shape, encoding.logical_strides_elements, strict=True):
            divisor //= extent
            terms.append(f"(((i * {cols} + j) / {divisor}) % {extent}) * {stride}")
        row = bindings[i]
        address = (
            f"merlin_compact_base_{row['base_index']} + {row['byte_offset']} + "
            f"({' + '.join(terms)}) * sizeof({container.ctype})"
        )
        prints += [
            f'  printf("OUT {name} {rows} {cols}");',
            f"  for (long i = 0; i < {rows}; i++) for (long j = 0; j < {cols}; j++) {{",
            f"    {container.ctype} value;",
            f"    __builtin_memcpy(&value, {address}, sizeof(value));",
            "    " + container.printf_element("value"),
            "  }",
            '  printf("\\n");',
        ]
    arguments = ", ".join(f"(void*)merlin_compact_base_{i}" for i in range(len(arenas)))
    preamble = _preamble(hooks, len(arenas), symbol=symbol)
    invoke = f"{symbol}({arguments});"
    if warm_profile is not None:
        from merlin.analysis.perf.measurement.warm_profile_harness import (
            TargetInvocationHooks,
            render_warm_then_measure_main,
            require_strict_final_warm_profile,
        )

        main = render_warm_then_measure_main(
            prepare_input="/* Compact arenas are initialized before main. */",
            invocation=TargetInvocationHooks(invoke=invoke, complete=hooks.completion),
            reset_after_warm=("\n".join(r.strip() for r in restore) if restore else None),
            validate_outputs="0",
            contract=require_strict_final_warm_profile(warm_profile),
            cycle_reader="merlin_read_cycles",
            success_body="\n".join(prints) + ("\n" if prints else "") + 'printf("DONE\\n");',
        )
        return preamble + "\n".join(declarations) + "\n" + main
    body = [
        "int main(void) {",
        f"  {invoke}",
        f"  {hooks.completion}",
        "  // Restore caller initial bytes after the unmeasured warm invocation.",
        *restore,
        "  uint64_t c0 = merlin_read_cycles();",
        f"  {invoke}",
        f"  {hooks.completion}",
        "  uint64_t c1 = merlin_read_cycles();",
        '  printf("METRIC cycles %lu\\n", (unsigned long)(c1 - c0));',
    ]
    if hooks.cycle_window_metric:
        body.append(f'  printf("METRIC {hooks.cycle_window_metric} 1\\n");')
    body += prints + ['  printf("DONE\\n");', "  return 0;", "}"]
    return preamble + "\n".join(declarations) + "\n" + "\n".join(body) + "\n"


# ---- harness_blobs -----------------------------------------------------------------------------
# Stage renderer-supplied constant operands beside a runner-owned C harness.
#
# The target renderer owns tensor layout and bytes. The runner owns filenames,
# assembly, and linking so no target-specific source is needed in the core.


def _symbol_name(value: object) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and ("A" <= value[0] <= "Z" or "a" <= value[0] <= "z" or value[0] == "_")
        and all("A" <= char <= "Z" or "a" <= char <= "z" or "0" <= char <= "9" or char == "_" for char in value[1:])
    )


def _payload_name(value: object) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and ("A" <= value[0] <= "Z" or "a" <= value[0] <= "z" or "0" <= value[0] <= "9" or value[0] == "_")
        and all("A" <= char <= "Z" or "a" <= char <= "z" or "0" <= char <= "9" or char in "_.-" for char in value[1:])
        and value.endswith(".bin")
        and Path(value).name == value
    )


def render_blob_asm(symbol: str, payload_name: str, *, align: int, elems: int, section: str = ".rodata") -> str:
    """Define an aligned C-visible symbol from exact binary input bytes."""
    if not _symbol_name(symbol):
        raise ValueError(f"invalid harness blob symbol {symbol!r}")
    if not _payload_name(payload_name):
        raise ValueError(f"invalid harness blob payload name {payload_name!r}")
    if type(align) is not int or type(elems) is not int or align < 1 or align & (align - 1) or elems < 1:
        raise ValueError("harness blob alignment must be a power of two and element count positive")
    if section not in (".data", ".rodata"):
        raise ValueError("harness blob needs an explicit initialized data section")
    return (
        f"  .section {section}\n"
        f"  .balign {align}\n"
        f"  .globl {symbol}\n"
        f"{symbol}:\n"
        f'  .incbin "{payload_name}"\n'
        f"  .size {symbol}, . - {symbol}\n"
        f"  /* {elems} elements */\n"
    )


def stage_harness_blobs(workdir: Path, blobs: Mapping[str, Mapping]) -> tuple[Path, ...]:
    """Validate and materialize a renderer's sidecars for the current build."""
    sources = []
    records = []
    extended = False
    for symbol, spec in sorted(blobs.items()):
        from merlin.compiler.lower.c.weights import WeightPayload

        if not isinstance(spec, Mapping) or set(spec) not in (
            {"bytes", "align", "elems"},
            {"bytes", "align", "elems", "section"},
        ):
            raise ValueError(f"invalid harness blob declaration for {symbol!r}")
        payload = spec["bytes"]
        align, elems = spec["align"], spec["elems"]
        lazy = type(payload) is WeightPayload
        if (
            not (type(payload) is bytes or lazy)
            or not len(payload)
            or type(align) is not int
            or type(elems) is not int
            or elems < 1
            or len(payload) % elems
        ):
            raise ValueError(f"invalid harness blob bytes/extent for {symbol!r}")
        name = f"harness_blob_{symbol}.bin"
        section = spec.get("section", ".rodata")
        asm = render_blob_asm(symbol, name, align=align, elems=elems, section=section)
        blob_path = workdir / name
        source = workdir / f"harness_blob_{symbol}.S"
        if blob_path.is_symlink() or source.is_symlink():
            raise ValueError("harness blob output path is a symlink")
        if lazy:
            expected = hashlib.sha256()
            with tempfile.TemporaryDirectory(dir=workdir, prefix="pending-blob-") as temporary:
                pending = Path(temporary) / name
                with pending.open("wb") as stream:
                    for chunk in payload.chunks():
                        expected.update(chunk)
                        stream.write(chunk)
                with pending.open("rb") as stream:
                    actual = hashlib.file_digest(stream, "sha256").hexdigest()
                if pending.stat().st_size != len(payload) or actual != expected.hexdigest():
                    raise ValueError("realized harness blob differs from complete source bytes")
                pending.replace(blob_path)
            digest = actual
        else:
            blob_path.write_bytes(payload)
            digest = hashlib.sha256(payload).hexdigest()
        source.write_text(asm, encoding="utf-8")
        sources.append(source)
        records.append(
            {
                "symbol": symbol,
                "file": name,
                "assembly": source.name,
                "object": source.with_suffix(".o").name,
                "sha256": digest,
                "bytes": len(payload),
                "alignment": align,
                "elements": elems,
                **(
                    {"storage": "streamed" if lazy else "bytes", "section": section}
                    if lazy or "section" in spec
                    else {}
                ),
            }
        )
        extended |= lazy or "section" in spec
    receipt = workdir / "harness_blobs.json"
    if receipt.is_symlink():
        raise ValueError("harness blob receipt path is a symlink")
    if records:
        receipt.write_text(
            json.dumps(
                {"schema": "merlin.harness_blobs.v2" if extended else "merlin.harness_blobs.v1", "blobs": records},
                sort_keys=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    else:
        receipt.unlink(missing_ok=True)
    return tuple(sources)
