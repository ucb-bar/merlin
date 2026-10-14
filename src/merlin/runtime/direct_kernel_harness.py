"""Pure full-value harness for a declared pointer entry ABI.

The host supplies the entry/completion convention and canonical operand bytes.
This owner allocates storage, calls the emitted kernel and publishes every output
container word. It contains no target instruction, tiling or reference compiler.
It does not establish source effects, synchronization semantics or target timing.
"""

from __future__ import annotations

import base64
import math
from dataclasses import dataclass

from merlin.common.quant_formats import get
from merlin.targetgen.contract.readback_policy import (
    COHERENT_DUMP_V1,
    COHERENT_PACKET_V1,
    COHERENT_PACKET_V2,
    FULL_VALUES_B64,
    ReadbackPolicy,
)

# Transport blocks are independent of logical tensor geometry and actual values.
# The original complete roster determines every block and its exact last extent.
PACKET_BLOCK_WORDS = 256
PACKET_ARENA_SYMBOL = "merlin_readback_packet"
PACKET_USED_SYMBOL = "merlin_readback_packet_used"


def packet_output_blocks(cb):
    """Bind closed OUT_BIN frames to every dense original output in order."""
    from .out_packet import ExpectedOutput, out_bin_packet_capacity

    frames = []
    for output_index, name in enumerate(cb["kernel_abi"]["outputs"]):
        spec = cb["tensors"][name]
        count, width, dtype = _layout(spec)
        if not spec["shape"] or spec["dtype"] not in {"i8", "i16", "i32", "i64", "f32"}:
            raise ValueError("neutral packet requires supported dense non-scalar outputs")
        for start in range(0, count, PACKET_BLOCK_WORDS):
            words = min(PACKET_BLOCK_WORDS, count - start)
            frame = ExpectedOutput(
                name=f"merlin_block_{output_index}_{start // PACKET_BLOCK_WORDS}",
                rows=1, cols=words, logical_dtype=spec["dtype"],
                container_signed=bool(dtype.signed and not dtype.is_float),
                container_max_bytes=width, logical_shape=(words,),
            )
            frames.append((name, start, frame))
            if len(frames) > 65536:
                raise ValueError("neutral packet exceeds its bounded frame roster")
    out_bin_packet_capacity(tuple(frame for _, _, frame in frames))
    return tuple(frames)


def packet_output_roster_v2(cb):
    """One exact frame per declared output, without a shape-specific block size."""
    from .out_packet import ExpectedOutput
    from .out_packet_v2 import out_packet_v2_capacity

    frames = []
    for name in cb["kernel_abi"]["outputs"]:
        spec = cb["tensors"][name]
        count, width, dtype = _layout(spec)
        if not spec["shape"] or spec["dtype"] not in {"i8", "i16", "i32", "i64", "f32"}:
            raise ValueError("neutral packet v2 requires supported dense non-scalar outputs")
        frames.append(ExpectedOutput(
            name=name, rows=1, cols=count, logical_dtype=spec["dtype"],
            container_signed=bool(dtype.signed and not dtype.is_float),
            container_max_bytes=width, logical_shape=(count,),
        ))
    out_packet_v2_capacity(tuple(frames))
    return tuple(frames)


_PACKET_LOAD_C = """static uint64_t merlin_packet_load(const unsigned char *source, unsigned width, int sign){
#if (defined(__GNUC__) || defined(__clang__)) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  /* Character arrays are original storage; may_alias permits aligned word reads. */
  typedef uint16_t u16 __attribute__((__may_alias__));
  typedef uint32_t u32 __attribute__((__may_alias__));
  typedef uint64_t u64 __attribute__((__may_alias__));
  uint64_t word=width==1 ? *source : width==2 ? *(const u16*)source :
                width==4 ? *(const u32*)source : *(const u64*)source;
#else
  uint64_t word=0;
  for(unsigned byte=0;byte<width;byte++) word|=(uint64_t)source[byte]<<(byte*8);
#endif
  if(sign && width<8 && (word & (UINT64_C(1)<<(width*8-1))))
    word|=~((UINT64_C(1)<<(width*8))-1);
  return word;
}
"""


def _packet_publication(cb, slots):
    """Encode actual completed output words, never input/reference values."""
    lines = []
    for output_index, name in enumerate(cb["kernel_abi"]["outputs"]):
        index, _, (count, width, dtype) = slots[name]
        signed = int(dtype.signed and not dtype.is_float)
        i32 = cb["tensors"][name]["dtype"] == "i32"
        lines.extend([
            "  {",
            f"    {'int32_t' if i32 else 'uint64_t'} words[{PACKET_BLOCK_WORDS}];",
            f"    for(uint64_t start=0;start<UINT64_C({count});start+={PACKET_BLOCK_WORDS}){{",
            f"      uint64_t count=UINT64_C({count})-start;",
            f"      if(count>{PACKET_BLOCK_WORDS}) count={PACKET_BLOCK_WORDS};",
            "      merlin_out_b64_range range;",
            f"      merlin_out_b64_range_init(&range,count,{width},{signed});",
            "      for(uint64_t i=0;i<count;i++){",
            f"        uint64_t word=merlin_packet_load(tensor_{index}+(start+i)*{width},{width},{signed});",
            f"        words[i]=({'int32_t' if i32 else 'uint64_t'})word;",
            f"        if(!merlin_out_b64_range_{'signed' if signed else 'unsigned'}"
            f"(&range,({'int64_t' if signed else 'uint64_t'})word)) htif_exit(2);",
            "      }",
            "      unsigned width=merlin_out_b64_range_width(&range);",
            "      int sign=merlin_out_b64_range_wire_signed(&range);",
            "      if(!width || sign<0) htif_exit(2);",
            '      if(!merlin_out_bin_memory_cstr(&merlin_packet_sink,'
            f'"OUT_BIN_BEGIN v1 merlin_block_{output_index}_") ||',
            f"         !merlin_out_bin_memory_decimal(&merlin_packet_sink,start/{PACKET_BLOCK_WORDS}) ||",
            '         !merlin_out_bin_memory_cstr(&merlin_packet_sink," 1 ") ||',
            "         !merlin_out_bin_memory_decimal(&merlin_packet_sink,count) ||",
            '         !merlin_out_bin_memory_cstr(&merlin_packet_sink," ") ||',
            "         !merlin_out_bin_memory_decimal(&merlin_packet_sink,width) ||",
            '         !merlin_out_bin_memory_cstr(&merlin_packet_sink,sign ? " s " : " u ") ||',
            "         !merlin_out_bin_memory_decimal(&merlin_packet_sink,count*width) ||",
            '         !merlin_out_bin_memory_cstr(&merlin_packet_sink,"\\n")) htif_exit(2);',
            "      merlin_out_bin encoder;",
            "      merlin_out_bin_init(&encoder,count,width,sign,merlin_packet_write);",
        ])
        lines.append("      if(!merlin_out_bin_words_i32(&encoder,words,count)) htif_exit(2);" if i32 else
                     "      for(uint64_t i=0;i<count;i++) if(!merlin_out_bin_word(&encoder,words[i])) htif_exit(2);")
        lines.extend([
            "      if(!merlin_out_bin_finish(&encoder) ||",
            '         !merlin_out_bin_memory_cstr(&merlin_packet_sink,"OUT_BIN_END v1 ") ||',
            "         !merlin_out_bin_memory_hex16(&merlin_packet_sink,encoder.checksum) ||",
            '         !merlin_out_bin_memory_cstr(&merlin_packet_sink,"\\n")) htif_exit(2);',
            "    }", "  }",
        ])
    lines.extend([
        '  if(!merlin_out_bin_memory_cstr(&merlin_packet_sink,"DONE\\n")) htif_exit(2);',
        '#if defined(__riscv)', '  __asm__ volatile("fence rw,rw" ::: "memory");', '#endif',
        "  if(!merlin_out_bin_memory_finish(&merlin_packet_sink)) htif_exit(2);",
        '#if defined(__riscv)', '  __asm__ volatile("fence rw,rw" ::: "memory");', '#endif',
    ])
    return lines


def _packet_publication_v2(cb, slots):
    """Stream actual logical words once; no range scan or per-block framing."""
    lines = []
    for name in cb["kernel_abi"]["outputs"]:
        index, _, (count, width, dtype) = slots[name]
        signed = int(dtype.signed and not dtype.is_float)
        escaped = int(width > 1 and not dtype.is_float)
        mode = "e" if escaped else "r"
        sign = "s" if signed else "u"
        c_header = f"OUT_PKT_BEGIN v2 {name} 1 {count} {cb['tensors'][name]['dtype']} {width} {sign} {mode}\\n"
        lines.extend([
            "  {",
            f'    if(!merlin_out_bin_memory_cstr(&merlin_packet_sink,"{c_header}")) htif_exit(2);',
            "    merlin_out_packet_v2 encoder;",
            f"    merlin_out_packet_v2_init(&encoder,UINT64_C({count}),{width},{signed},{escaped},merlin_packet_write);",
            f"    for(uint64_t i=0;i<UINT64_C({count});i++){{",
            f"      uint64_t word=merlin_packet_load(tensor_{index}+i*{width},{width},{signed});",
            "      if(!merlin_out_packet_v2_word(&encoder,word)) htif_exit(2);",
            "    }",
            "    if(!merlin_out_packet_v2_finish(&encoder) ||",
            '       !merlin_out_bin_memory_cstr(&merlin_packet_sink,"OUT_PKT_END v2 ") ||',
            "       !merlin_out_bin_memory_hex16(&merlin_packet_sink,encoder.checksum) ||",
            '       !merlin_out_bin_memory_cstr(&merlin_packet_sink,"\\n")) htif_exit(2);',
            "  }",
        ])
    lines.extend([
        '  if(!merlin_out_bin_memory_cstr(&merlin_packet_sink,"DONE\\n")) htif_exit(2);',
        '#if defined(__riscv)', '  __asm__ volatile("fence rw,rw" ::: "memory");', '#endif',
        "  if(!merlin_out_bin_memory_finish(&merlin_packet_sink)) htif_exit(2);",
        '#if defined(__riscv)', '  __asm__ volatile("fence rw,rw" ::: "memory");', '#endif',
    ])
    return lines

_OUTPUT_POISON_C = """static void merlin_poison_bytes(void *storage, uint64_t bytes, unsigned char byte){
  volatile unsigned char *cursor=(volatile unsigned char*)storage;
  if(!bytes) return;
#if defined(__GNUC__) || defined(__clang__)
  /* may_alias preserves character-storage aliasing; all stores remain volatile. */
  typedef uintptr_t merlin_poison_word __attribute__((__may_alias__));
  while(bytes && (uintptr_t)cursor % sizeof(merlin_poison_word)){
    *cursor++=byte;
    bytes--;
  }
  const merlin_poison_word word=((merlin_poison_word)~(merlin_poison_word)0 / UINT8_MAX) * byte;
  volatile merlin_poison_word *words=(volatile merlin_poison_word*)cursor;
  while(bytes>=sizeof(merlin_poison_word)){
    *words++=word;
    bytes-=sizeof(merlin_poison_word);
  }
  cursor=(volatile unsigned char*)words;
#endif
  while(bytes){
    *cursor++=byte;
    bytes--;
  }
}
"""


def _identifier(value):
    if (
        not isinstance(value, str)
        or not value.isascii()
        or not value
        or not (value[0].isalpha() or value[0] == "_")
        or any(not (character.isalnum() or character == "_") for character in value)
    ):
        raise ValueError("direct kernel harness needs a plain C identifier")
    return value


@dataclass(frozen=True)
class DirectKernelAbi:
    """Explicit software calling convention, independently selected by the host."""

    entry_symbol: str
    completion_symbol: str | None
    tensor_alignment: int
    byte_order: str
    main_convention: str

    def verify(self):
        _identifier(self.entry_symbol)
        if self.completion_symbol is not None:
            _identifier(self.completion_symbol)
            if self.completion_symbol == self.entry_symbol:
                raise ValueError("completion and kernel entries must be distinct")
        if (
            type(self.tensor_alignment) is not int
            or self.tensor_alignment <= 0
            or self.tensor_alignment & (self.tensor_alignment - 1)
            or self.byte_order not in ("little", "big")
            or self.main_convention not in ("void", "primary_context_id")
        ):
            raise ValueError("direct kernel ABI needs explicit alignment, byte order and entry convention")


def _layout(spec):
    shape = spec.get("shape")
    if not isinstance(shape, list) or any(type(value) is not int or value <= 0 for value in shape):
        raise ValueError("direct kernel buffer requires a concrete nonempty shape")
    dtype = get(spec.get("dtype"))
    if dtype.element_bits not in (8, 16, 32, 64) or dtype.is_block_scaled:
        raise ValueError("direct kernel harness needs explicit byte-aligned scalar storage")
    width = dtype.element_bits // 8
    return math.prod(shape), width, dtype


def _raw(spec, values, *, count, width, dtype, byte_order):
    encoded = spec.get("preload_b64")
    if encoded is not None:
        if not isinstance(encoded, str):
            raise ValueError("direct kernel preload must contain canonical base64 bytes")
        raw = base64.b64decode(encoded, validate=True)
        if base64.b64encode(raw).decode() != encoded:
            raise ValueError("direct kernel preload is not canonical base64")
    else:
        # Floating point inputs require their exact source bits, including NaNs
        # and signed zero; Python numeric values cannot substitute that authority.
        if dtype.is_float or not isinstance(values, dict):
            raise ValueError("direct floating input has no complete source byte projection")
        data = values.get("values")
        if values.get("shape") != spec["shape"] or not isinstance(data, list) or len(data) != count:
            raise ValueError("direct integer input has incomplete canonical values")
        words = []
        for value in data:
            if type(value) is not int:
                raise ValueError("direct integer values must be exact integers")
            words.append(value.to_bytes(width, "little", signed=dtype.signed))
        raw = b"".join(words)
    if len(raw) != count * width:
        raise ValueError("direct input byte extent differs from its full logical shape")
    if byte_order == "big":
        raw = b"".join(raw[index : index + width][::-1] for index in range(0, len(raw), width))
    return raw


def render_direct_kernel(
    cb,
    *,
    inputs,
    readback_policy,
    abi: DirectKernelAbi,
    original_storage=None,
    invocation_plan=None,
    counter_plan=None,
    phase_plan=None,
    output_poison=None,
    prelude_symbol=None,
    complete_call_timing=None,
    source_owned_mutables=(),
):
    """Render complete pointer-call storage with explicitly selected readback.

    This accepts an explicit command-buffer argument roster. The ordinary source
    binder separately verifies its correspondence to the original input/output
    interface. Neither a candidate-selected ABI nor a correct console grants
    numerical, placement, lifetime or hardware authority.
    """
    if type(abi) is not DirectKernelAbi:
        raise ValueError("direct kernel harness requires a typed host ABI")
    if complete_call_timing is not None:
        from .complete_call_timing import CompleteCallTiming

        if type(complete_call_timing) is not CompleteCallTiming or any(
            p is not None for p in (invocation_plan, counter_plan, phase_plan)
        ):
            raise ValueError("complete-call timing requires one original invocation and its exact normal timer")
        complete_call_timing.verify()
    abi.verify()
    if output_poison is not None and (type(output_poison) is not int or not 0 <= output_poison <= 255):
        raise ValueError("direct output poison must be an explicit byte")
    if prelude_symbol is not None:
        _identifier(prelude_symbol)
    if original_storage is not None:
        from merlin.targetgen.contract.pointer_storage import OriginalPointerStorageContract

        if type(original_storage) is not OriginalPointerStorageContract:
            raise ValueError("direct kernel original storage requires the exact independent typed declaration")
        original_storage.bind_candidate(cb)
        if (
            original_storage.policy.byte_order != abi.byte_order
            or original_storage.policy.tensor_alignment != abi.tensor_alignment
        ):
            raise ValueError("direct kernel storage differs from its selected byte order/alignment")
    if type(readback_policy) is not ReadbackPolicy or readback_policy.transport not in (
        FULL_VALUES_B64,
        COHERENT_DUMP_V1,
        COHERENT_PACKET_V1,
        COHERENT_PACKET_V2,
    ):
        raise ValueError("direct kernel harness requires complete B64 or coherent memory readback")
    memory = readback_policy.transport == COHERENT_DUMP_V1
    packet = readback_policy.transport in (COHERENT_PACKET_V1, COHERENT_PACKET_V2)
    packet_v2 = readback_policy.transport == COHERENT_PACKET_V2
    if packet and abi.byte_order != "little":
        raise ValueError("neutral coherent packet requires explicit little-endian storage")
    histories = ()
    if invocation_plan is not None:
        from .direct_kernel_invocation import DirectKernelInvocationPlan

        if type(invocation_plan) is not DirectKernelInvocationPlan or not memory:
            raise ValueError("repeated invocation evaluation requires an explicit plan and coherent history reader")
        histories = invocation_plan.bind(cb, entry_symbol=abi.entry_symbol, completion_symbol=abi.completion_symbol)
    tensors, kernel = cb.get("tensors"), cb.get("kernel_abi")
    if not isinstance(tensors, dict) or not isinstance(kernel, dict):
        raise ValueError("direct kernel harness has no explicit tensor/argument declaration")
    args, outputs = kernel.get("args"), kernel.get("outputs")
    if (
        not isinstance(args, list)
        or not args
        or not isinstance(outputs, list)
        or not outputs
        or len(set(outputs)) != len(outputs)
        or not isinstance(inputs, dict)
    ):
        raise ValueError("direct kernel harness needs a complete unique input/output roster")
    slots = {}
    if type(source_owned_mutables) is not tuple or len(set(source_owned_mutables)) != len(source_owned_mutables):
        raise ValueError("direct source-owned scratch needs the original complete unique tuple")
    mutable = set(source_owned_mutables)
    for index, arg in enumerate(args):
        if (
            not isinstance(arg, dict)
            or set(arg) != {"tensor", "access"}
            or arg["access"] not in {"read", "write", "readwrite"}
        ):
            raise ValueError("direct kernel argument requires a tensor and explicit access")
        name = _identifier(arg["tensor"])
        if name in slots or name not in tensors:
            raise ValueError("direct kernel argument is duplicated or undeclared")
        slots[name] = (index, arg["access"], _layout(tensors[name]))
    if not mutable <= set(slots) or any(
        slots[name][1] == "read" or tensors[name].get("role") != "intermediate" for name in mutable
    ):
        raise ValueError("direct source-owned scratch differs from its original writable pointer roster")
    expected_inputs = {name for name, (_, access, _) in slots.items() if access != "write" and name not in mutable}
    expected_outputs = {name for name, (_, access, _) in slots.items() if access != "read" and name not in mutable}
    if set(inputs) != expected_inputs or set(outputs) != expected_outputs:
        raise ValueError("direct kernel ABI omits or adds an input/output pointer")
    counter_roster = None
    if counter_plan is not None:
        from .direct_kernel_counter import DirectKernelCounterPlan

        if type(counter_plan) is not DirectKernelCounterPlan or not memory:
            raise ValueError("counter capture requires an explicit typed plan and complete coherent object reader")
        counter_roster = counter_plan.bind(cb, abi=abi, invocation_plan=invocation_plan)
    phase_roster = None
    if phase_plan is not None:
        from .direct_kernel_phases import DirectKernelPhasePlan

        if type(phase_plan) is not DirectKernelPhasePlan or not memory or phase_plan.counter_plan is not counter_plan:
            raise ValueError("harness phases require coherent readback and the same explicit counter plan")
        phase_roster = phase_plan.bind(cb, abi=abi, invocation_plan=invocation_plan)
    if prelude_symbol is not None:
        reserved = {
            abi.entry_symbol,
            abi.completion_symbol,
            "main",
            "console_init",
            "htif_puts",
            "htif_exit",
            "invocation",
            "invocation_completed",
            "byte",
            "merlin_poison_index",
            *("tensor_" + str(index) for index in range(len(args))),
            *(history.symbol for history in histories),
            *(counter_roster or ()),
            *(phase_roster or ()),
        }
        if invocation_plan is not None:
            reserved.add(invocation_plan.count_symbol)
        if counter_plan is not None:
            reserved.update((counter_plan.counter_symbol, counter_plan.state_symbol))
        if prelude_symbol in reserved:
            raise ValueError("direct prelude overlaps a harness storage or entry symbol")
    declarations = [
        "#include <stdint.h>",
        '#include "htif.h"',
        f"extern void {abi.entry_symbol}({', '.join('void *' for _ in args)});",
    ]
    if not memory:
        declarations.insert(2, '#include "out_b64.h"')
    if packet:
        if packet_v2:
            from .out_packet_v2 import out_packet_v2_capacity

            capacity = out_packet_v2_capacity(packet_output_roster_v2(cb))
        else:
            from .out_packet import out_bin_packet_capacity

            capacity = out_bin_packet_capacity(tuple(frame for _, _, frame in packet_output_blocks(cb)))
        if {abi.entry_symbol, abi.completion_symbol, prelude_symbol} & {
            PACKET_ARENA_SYMBOL, PACKET_USED_SYMBOL, "merlin_packet_sink", "merlin_packet_write", "merlin_packet_load"
        }:
            raise ValueError("neutral packet storage overlaps a selected entry symbol")
        declarations.extend([
            '#include "out_bin.h"', '#include "out_bin_memory.h"',
            *(['#include "out_packet_v2.h"'] if packet_v2 else []),
            # The arena is scratch: only the complete published prefix is read.
            # A nonzero initializer puts it in the original ELF data segment,
            # so selected direct ELF preload avoids clearing unused capacity in
            # the guest CRT. No initial arena byte is an output value.
            f"unsigned char {PACKET_ARENA_SYMBOL}[{capacity}] __attribute__((used,aligned(8)))={{1}};",
            f"volatile uint64_t {PACKET_USED_SYMBOL} __attribute__((used,aligned(8)));",
            "static merlin_out_bin_memory merlin_packet_sink;",
            "static int merlin_packet_write(const void *data,size_t size){",
            "  return merlin_out_bin_memory_append(&merlin_packet_sink,data,size);", "}",
            *_PACKET_LOAD_C.splitlines(),
        ])
    if abi.completion_symbol:
        declarations.append(f"extern void {abi.completion_symbol}(void);")
    if prelude_symbol is not None:
        declarations.append(f"extern void {prelude_symbol}(void);")
    poison_symbol = "merlin_poison_bytes"
    while poison_symbol in {abi.entry_symbol, abi.completion_symbol, prelude_symbol}:
        poison_symbol += "_"
    if output_poison is not None:
        declarations.extend(_OUTPUT_POISON_C.replace("merlin_poison_bytes", poison_symbol).splitlines())
    if complete_call_timing is not None:
        declarations.extend(complete_call_timing.declarations())
    for name, (index, access, (count, width, dtype)) in slots.items():
        if width > abi.tensor_alignment:
            raise ValueError("direct kernel software alignment is smaller than a container word")
        payload = "0"
        if access != "write" and name not in mutable:
            raw = _raw(tensors[name], inputs[name], count=count, width=width, dtype=dtype, byte_order=abi.byte_order)
            payload = ",".join(f"0x{value:02x}" for value in raw)
        declarations.append(
            f"static unsigned char tensor_{index}[{count * width}] "
            f"__attribute__((aligned({abi.tensor_alignment})))={{{payload}}};"
        )
    if invocation_plan is not None:
        declarations.append(f"volatile unsigned char {invocation_plan.count_symbol}[8]={{0}};")
        declarations.extend(f"unsigned char {history.symbol}[{history.byte_extent}]={{0}};" for history in histories)
    if counter_roster is not None:
        declarations.extend(counter_plan.declarations(counter_roster))
    if phase_roster is not None:
        declarations.extend(phase_plan.declarations(phase_roster))
    body = (
        ["int main(unsigned long context_id){", "  if(context_id) return 0;"]
        if abi.main_convention == "primary_context_id"
        else ["int main(void){"]
    )
    if complete_call_timing is not None:
        body.extend(complete_call_timing.begin())
    if phase_roster is not None:
        body.extend(
            [
                "  uint64_t phase_main_start,phase_main_end,phase_main_state_before,phase_main_state_after;",
                "  uint64_t phase_value_start,phase_value_end,phase_value_state_before,phase_value_state_after;",
                *phase_plan.begin("main_body", "  "),
                *phase_plan.begin("console_setup", "  "),
            ]
        )
    body.append("  console_init();")
    if packet:
        body.extend([
            f"  merlin_out_bin_memory_init(&merlin_packet_sink,{PACKET_ARENA_SYMBOL},"
            f"sizeof({PACKET_ARENA_SYMBOL}),&{PACKET_USED_SYMBOL});",
            "  if(!merlin_packet_sink.valid) htif_exit(2);",
        ])
    if phase_roster is not None:
        body.extend(phase_plan.end("console_setup", "  "))
        body.extend(phase_plan.store("console_setup", index="0", byte_order=abi.byte_order, indent="  "))
    if prelude_symbol is not None:
        body.append(f"  {prelude_symbol}();")
    if phase_roster is not None:
        body.extend(phase_plan.begin("calibration_loop", "  "))
    if counter_roster is not None:
        body.extend(
            [
                "  uint64_t counter_start,counter_end,counter_state_before,counter_state_after;",
                "  for(uint64_t counter_index=0;"
                f"counter_index<UINT64_C({counter_plan.calibration_count});counter_index++){{",
                *counter_plan.begin("    "),
                *counter_plan.end("    "),
                *counter_plan.store(
                    kind="calibration", index="counter_index", byte_order=abi.byte_order, indent="    "
                ),
                "  }",
            ]
        )
    if phase_roster is not None:
        body.extend(phase_plan.end("calibration_loop", "  "))
        body.extend(phase_plan.store("calibration_loop", index="0", byte_order=abi.byte_order, indent="  "))
    call = f"{abi.entry_symbol}({', '.join('tensor_' + str(index) for index in range(len(args)))});"

    def poison(indent):
        if output_poison is None:
            return []
        lines = []
        for index, access, (count, width, _) in slots.values():
            if access == "write":
                lines.append(indent + f"{poison_symbol}(tensor_{index},UINT64_C({count * width}),"
                             f"{output_poison});")
        return lines

    if invocation_plan is None:
        body.extend(poison("  "))
        if counter_roster is not None:
            body.extend(counter_plan.begin("  "))
        body.append("  " + call)
        if abi.completion_symbol:
            body.append(f"  {abi.completion_symbol}();")
        if counter_roster is not None:
            body.extend(counter_plan.end("  "))
            body.extend(counter_plan.store(kind="call", index="0", byte_order=abi.byte_order, indent="  "))
            body.extend(counter_plan.count("1", byte_order=abi.byte_order, indent="  "))
    else:
        body.extend(
            [
                "  uint64_t invocation_completed=0;",
                f"  for(uint64_t invocation=0;invocation<UINT64_C({invocation_plan.count});invocation++){{",
            ]
        )
        body.extend(poison("    "))
        if counter_roster is not None:
            body.extend(counter_plan.begin("    "))
        body.append("    " + call)
        if abi.completion_symbol:
            body.append(f"    {abi.completion_symbol}();")
        if counter_roster is not None:
            body.extend(counter_plan.end("    "))
            body.extend(counter_plan.store(kind="call", index="invocation", byte_order=abi.byte_order, indent="    "))
            body.extend(counter_plan.count("invocation+1", byte_order=abi.byte_order, indent="    "))
        if phase_roster is not None:
            body.extend(phase_plan.begin("history_copy", "    "))
        for history in histories:
            index = slots[history.emitted_tensor][0]
            width = history.bytes_per_invocation
            body.extend(
                [
                    f"    for(uint64_t byte=0;byte<UINT64_C({width});byte++)",
                    f"      {history.symbol}[invocation*UINT64_C({width})+byte]=tensor_{index}[byte];",
                ]
            )
        count_offset = "byte" if abi.byte_order == "little" else "(7-byte)"
        body.extend(
            [
                "    invocation_completed++;",
                "    for(unsigned byte=0;byte<8;byte++)",
                f"      {invocation_plan.count_symbol}[{count_offset}]="
                "(unsigned char)(invocation_completed>>(byte*8));",
            ]
        )
        if phase_roster is not None:
            body.extend(phase_plan.end("history_copy", "    "))
            body.extend(phase_plan.store("history_copy", index="invocation", byte_order=abi.byte_order, indent="    "))
        body.append("  }")
    # The selected coherent reader resolves these actual linked static objects.
    # It owns full-value admission; DONE alone supplies no output or effect proof.
    if packet:
        body.extend(_packet_publication_v2(cb, slots) if packet_v2 else _packet_publication(cb, slots))
    for name in () if memory or packet else outputs:
        index, _, (count, width, dtype) = slots[name]
        rows = math.prod(tensors[name]["shape"][:-1])
        cols = tensors[name]["shape"][-1] if tensors[name]["shape"] else 1
        signed = int(dtype.signed and not dtype.is_float)
        body.extend(
            [
                f'  htif_puts("OUT_B64_BEGIN v1 {name} {rows} {cols} {width} {"s" if signed else "u"}\\n");',
                "  { merlin_out_b64 packet;",
                f"    merlin_out_b64_init(&packet,{count},{width},{signed},htif_puts);",
                f"    for(unsigned long i=0;i<{count};i++){{",
                "      uint64_t word=0;",
                f"      for(unsigned j=0;j<{width};j++){{",
            ]
        )
        offset = "j" if abi.byte_order == "little" else f"({width}-1-j)"
        body.extend([f"        word|=((uint64_t)tensor_{index}[i*{width}+{offset}])<<(j*8);", "      }"])
        if signed and width < 8:
            body.append(f"      if(word & (UINT64_C(1)<<{width * 8 - 1})) word|=~((UINT64_C(1)<<{width * 8})-1);")
        body.extend(
            [
                "      if(!merlin_out_b64_word(&packet,word)) htif_exit(2);",
                "    }",
                "    if(!merlin_out_b64_finish(&packet)) htif_exit(3);",
                '    htif_puts("OUT_B64_END\\n");',
                "  }",
            ]
        )
    if phase_roster is not None:
        body.extend(phase_plan.begin("done_publication", "  "))
    if complete_call_timing is not None:
        body.extend(complete_call_timing.end())
    body.append('  htif_puts("DONE\\n");')
    if phase_roster is not None:
        body.extend(phase_plan.end("done_publication", "  "))
        body.extend(phase_plan.store("done_publication", index="0", byte_order=abi.byte_order, indent="  "))
        body.extend(phase_plan.end("main_body", "  "))
        # These writes and the selected exit/parent readback are outside the
        # main bracket. Their unknown cost must not be represented as zero.
        body.extend(phase_plan.store("main_body", index="0", byte_order=abi.byte_order, indent="  "))
        body.extend(phase_plan.count(str(invocation_plan.count), byte_order=abi.byte_order, indent="  "))
    body.extend(["  htif_exit(0);", "  return 0;", "}"])
    return "\n".join((*declarations, "", *body)) + "\n"
