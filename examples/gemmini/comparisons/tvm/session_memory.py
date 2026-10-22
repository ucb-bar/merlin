"""Explicit near-code or absolute-address placement for the TVM session bridge."""
from __future__ import annotations

from package_session import aligned


GUEST_BASE = 0x80000000
CONSTANT_BINDING = {"default": "link_symbol", "symbol": "model_constants",
                    "absolute_address_macro": "TVM_MODEL_CONSTANTS_ADDRESS"}


def plan_memory(plan, stages, memory_bytes, stack_bytes, *, weights_base=None, arena_base=None):
    if (type(memory_bytes) is not int or memory_bytes <= 0 or memory_bytes % 4096
            or GUEST_BASE + memory_bytes > 1 << 64):
        raise ValueError("Require a page-aligned RV64 DRAM span")
    if type(stack_bytes) is not int or stack_bytes < 16384 or stack_bytes % 16:
        raise ValueError("Stack reserve must be aligned and at least the startup's one-hart stack")
    if (weights_base is None) != (arena_base is None):
        raise ValueError("Absolute placement requires both weights-base and arena-base")
    if weights_base is None:
        if memory_bytes > 1 << 31:
            raise ValueError("Maps above 2 GiB require explicit weights-base and arena-base")
        required = plan["storage_bytes"] + plan["constants_bytes"] + plan["reference_bytes"] + stack_bytes
        if required >= memory_bytes:
            raise ValueError("Session tensors/references/constants/stack exceed DRAM before code/runtime")
        return {"kind": "near_code", "dram_base": GUEST_BASE, "dram_bytes": memory_bytes,
                "near_limit": GUEST_BASE + memory_bytes, "stack_bytes": stack_bytes}
    for base in (weights_base, arena_base):
        if type(base) is not int or base % 4096 or not GUEST_BASE < base < GUEST_BASE + memory_bytes:
            raise ValueError("Absolute regions require page-aligned bases inside DRAM above code")
    constants, references, cursor = {}, {}, weights_base
    for name, stage in stages.items():
        if stage.get("constants_binding") != CONSTANT_BINDING:
            raise ValueError("Package lacks the explicit absolute-constant binding contract: " + name)
        constants[name] = cursor
        cursor += aligned(stage["constants_bytes"])
    for kind in ("input", "output"):
        for index, tensor in enumerate(plan[kind + "s"]):
            references[f"{kind}_{index}"] = cursor
            cursor += aligned(tensor["bytes"])
    weights_end, arena_end = cursor, arena_base + plan["storage_bytes"]
    if max(weights_end, arena_end) > GUEST_BASE + memory_bytes:
        raise ValueError("Absolute session regions exceed declared DRAM")
    if weights_base < arena_end and arena_base < weights_end:
        raise ValueError("Absolute weights/reference and arena regions overlap")
    near_limit = min(weights_base, arena_base, GUEST_BASE + (1 << 31))
    if near_limit - GUEST_BASE <= stack_bytes:
        raise ValueError("Absolute placement leaves no near-code space beyond the stack")
    return {"kind": "absolute", "dram_base": GUEST_BASE, "dram_bytes": memory_bytes,
            "near_limit": near_limit, "stack_bytes": stack_bytes,
            "weights_base": weights_base, "weights_bytes": weights_end - weights_base,
            "arena_base": arena_base, "arena_bytes": plan["storage_bytes"],
            "stage_constants": constants, "references": references}


def linker_suffix(layout):
    lines = [f'ASSERT(_stack_top <= {layout["near_limit"]}, "session code/stack exceeds near region")']
    if layout["kind"] == "absolute":
        # Reuse model_link.ld's weights_blob.o owner. The explicit NOLOAD arena
        # participates in the ELF span audit without allocating a huge file.
        lines += [f'ASSERT(ADDR(.weights) == {layout["weights_base"]}, "wrong weights base")',
                  f'ASSERT(SIZEOF(.weights) == {layout["weights_bytes"]}, "wrong weights extent")',
                  'SECTIONS {',
                  f' .session_arena {layout["arena_base"]} (NOLOAD) : {{ . += {layout["arena_bytes"]}; }}',
                  '}']
    return '\n' + '\n'.join(lines) + '\n'


def blob_assembly(layout, stages, arrays):
    """Place constants and oracle/input bytes at their planned absolute offsets."""
    lines = ['.section .rodata.session_weights,"a",@progbits', '.balign 64']
    for index, (name, stage) in enumerate(stages.items()):
        lines.append(f'.org {layout["stage_constants"][name] - layout["weights_base"]}')
        if stage["constants_bytes"]:
            lines.append(f'.incbin "stage_{index}.constants.bin"')
    for name in arrays:
        lines += [f'.org {layout["references"][name] - layout["weights_base"]}',
                  f'.incbin "{name}.bin"']
    lines += [f'.org {layout["weights_bytes"]}', '.section .note.GNU-stack,"",@progbits']
    return '\n'.join(lines) + '\n'
