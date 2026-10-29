"""Fresh actual buffer inputs for source/binding controls, without Lean execution."""

import json
import os
from pathlib import Path

from buffer_reuse_qualification_cases import BufferCase

from merlin.transforms import loops, normalization, planning
from merlin.transforms.certificates import ir_text, parse_ir
from merlin.verify.buffer_reuse_bridge import reconstruct_chain


def case_root(tmp_path_factory):
    selected = os.environ.get("MERLIN_BUFFER_REUSE_CASE_ROOT")
    if selected:
        return Path(selected)
    root = tmp_path_factory.mktemp("fresh-buffer-inputs")
    base = root / "epoch-cases-v1/two_epochs"
    module = parse_ir(BufferCase("unit", 3, 2, 2).source)
    buffers = [ir_text(module)]
    normalization.normalize_shapes(module)
    buffers.append(ir_text(module))
    planning.reuse_storage(module, planning.plan_storage(module))
    buffers.append(ir_text(module))
    loops.segment_loops(module, (2,), index_bits=64)
    buffers.append(ir_text(module))
    for name, text in zip(
        ("source.mlir", "shape/after.mlir", "memory/middle.mlir", "memory/after.mlir"), buffers, strict=True
    ):
        path = base / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    hint = reconstruct_chain(tuple(buffers), index_bits=64)
    path = root / "after-binding-deadline-v1/tests-v9/pytest/test_actual_middle_is_checked_0/PREPARATION.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(hint))
    return root
