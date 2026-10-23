"""Observe compiled memrefs without imposing the private answer's layout or aliases."""

from __future__ import annotations

import json
from pathlib import Path


def generate_observer(contract_path: Path, output: Path, out_specs, input_arrays, dtype_bytes, input_layouts=None):
    """Emit snapshots and observations of the compiled function's result descriptors.

    The sidecar selects input and post-call result ordinals only. Its expected tensor
    metadata and alias relationships never construct storage or observations. Packed
    byte transport is separate from the original result descriptors. Autograd state
    is absent from the compiled memref ABI and is explicitly marked unobserved.
    """
    contract = json.loads(contract_path.read_text())
    count = len(out_specs)
    inputs, posts = contract["input_indices"], contract["post_indices"]
    if len(inputs) != len(posts) or any(type(i) is not int or not 0 <= i < count for i in posts):
        raise ValueError("invalid post-call input result ordinals")
    lines = [
        "/* Observed compiled memrefs; requires_grad is imposed_by_harness. */",
        "#define MERLIN_SEMANTIC_IO 1",
        "extern const void *merlin_result_descriptor(int output);",
        "static void merlin_semantic_before(void) {",
    ]
    for i in inputs:
        array, dt = input_arrays[i]
        nbytes = array.size * dtype_bytes[dt]
        layout = input_layouts[i] if input_layouts else None
        if layout:
            lines += [
                f'htif_puts("PRE_BYTES {i} {nbytes}");',
                f"for(size_t n=0;n<{array.size};n++) {{",
                f"size_t linear=n, at={layout['storage_offset']};",
            ]
            for axis in reversed(range(len(layout["shape"]))):
                size = layout["shape"][axis]
                lines += [f"at += (linear % {max(1, size)}) * {layout['stride'][axis]};", f"linear /= {max(1, size)};"]
            lines += [
                f"for(size_t b=0;b<{dtype_bytes[dt]};b++) {{ htif_putc(' '); htif_putd(((unsigned char*)MERLIN_INPUT_PTR[{i}])[at*{dtype_bytes[dt]}+b]); }}",
                "}",
                "htif_putc('\\n');",
            ]
        else:
            lines += [
                f'htif_puts("PRE_BYTES {i} {nbytes}");',
                f"for(size_t b=0;b<{nbytes};b++) {{ htif_putc(' '); htif_putd(((unsigned char*)MERLIN_INPUT_PTR[{i}])[b]); }}",
                "htif_putc('\\n');",
            ]
    lines += ["}", "static void merlin_semantic_after(void) {"]
    for i, (shape, _) in enumerate(out_specs):
        rank = len(shape)
        lines += [
            f"const int64_t *words_{i}=(const int64_t*)merlin_result_descriptor({i});",
            f"void *const *storage_{i}=(void *const *)merlin_result_descriptor({i});",
            f'htif_puts("OUT_META {i} ");',
            f"htif_putd(words_{i}[2]);",
            f'htif_puts(" -1 {rank}");',
        ]
        for axis in range(rank):
            lines += [
                f"htif_putc(' '); htif_putd(words_{i}[3+{axis}]);",
                f"htif_putc(' '); htif_putd(words_{i}[3+{rank}+{axis}]);",
            ]
        lines += ["htif_putc('\\n');", f'htif_puts("OUT_ALIAS {i}");']
        for j in posts:
            lines += [
                f"void *const *post_{i}_{j}=(void *const *)merlin_result_descriptor({j});",
                f"if(storage_{i}[0] && storage_{i}[0]==post_{i}_{j}[0]) {{ htif_putc(' '); htif_putd({j}); }}",
            ]
        lines += ["htif_putc('\\n');"]
    lines += ["}"]
    output.write_text("\n".join(lines) + "\n")
