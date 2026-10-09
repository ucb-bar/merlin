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
from merlin.targetgen.contract.readback_policy import COHERENT_DUMP_V1, FULL_VALUES_B64, ReadbackPolicy


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
    if not isinstance(shape, list) or not shape or any(type(value) is not int or value <= 0 for value in shape):
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


def render_direct_kernel(cb, *, inputs, readback_policy, abi: DirectKernelAbi, original_storage=None):
    """Render complete pointer-call storage with explicitly selected readback.

    This accepts an explicit command-buffer argument roster. The ordinary source
    binder separately verifies its correspondence to the original input/output
    interface. Neither a candidate-selected ABI nor a correct console grants
    numerical, placement, lifetime or hardware authority.
    """
    if type(abi) is not DirectKernelAbi:
        raise ValueError("direct kernel harness requires a typed host ABI")
    abi.verify()
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
    ):
        raise ValueError("direct kernel harness requires complete B64 or coherent memory readback")
    memory = readback_policy.transport == COHERENT_DUMP_V1
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
    expected_inputs = {name for name, (_, access, _) in slots.items() if access != "write"}
    expected_outputs = {name for name, (_, access, _) in slots.items() if access != "read"}
    if set(inputs) != expected_inputs or set(outputs) != expected_outputs:
        raise ValueError("direct kernel ABI omits or adds an input/output pointer")
    declarations = [
        "#include <stdint.h>",
        '#include "htif.h"',
        f"extern void {abi.entry_symbol}({', '.join('void *' for _ in args)});",
    ]
    if not memory:
        declarations.insert(2, '#include "out_b64.h"')
    if abi.completion_symbol:
        declarations.append(f"extern void {abi.completion_symbol}(void);")
    for name, (index, access, (count, width, dtype)) in slots.items():
        if width > abi.tensor_alignment:
            raise ValueError("direct kernel software alignment is smaller than a container word")
        payload = "0"
        if access != "write":
            raw = _raw(tensors[name], inputs[name], count=count, width=width, dtype=dtype, byte_order=abi.byte_order)
            payload = ",".join(f"0x{value:02x}" for value in raw)
        declarations.append(
            f"static unsigned char tensor_{index}[{count * width}] "
            f"__attribute__((aligned({abi.tensor_alignment})))={{{payload}}};"
        )
    body = (
        ["int main(unsigned long context_id){", "  if(context_id) return 0;"]
        if abi.main_convention == "primary_context_id"
        else ["int main(void){"]
    )
    body.extend(
        [
            "  console_init();",
            f"  {abi.entry_symbol}({', '.join('tensor_' + str(index) for index in range(len(args)))});",
        ]
    )
    if abi.completion_symbol:
        body.append(f"  {abi.completion_symbol}();")
    # The selected coherent reader resolves these actual linked static objects.
    # It owns full-value admission; DONE alone supplies no output or effect proof.
    for name in () if memory else outputs:
        index, _, (count, width, dtype) = slots[name]
        rows = math.prod(tensors[name]["shape"][:-1])
        cols = tensors[name]["shape"][-1]
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
    body.extend(['  htif_puts("DONE\\n");', "  htif_exit(0);", "  return 0;", "}"])
    return "\n".join((*declarations, "", *body)) + "\n"
