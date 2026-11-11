"""Recover native implicit named regions when a partial Python reader misbuilds them.

For example, xDSL 0.68's MatmulOp.get_hidden_region multiplies BF16 operands
without the accumulator-type casts required by native linalg.matmul. Custom
syntax omits that region; MLIR's own parser/printer is its authoritative builder.
Generic syntax already carrying a valid region takes the unchanged fast path.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory


def restore_named_regions(module, source: str, context):
    from xdsl.dialects.linalg.abstract_ops import NamedOperation
    from xdsl.utils.exceptions import VerifyException

    try:
        for op in module.walk():
            if isinstance(op, NamedOperation):
                op.verify()
    except VerifyException:
        pass
    else:
        return module

    from merlin.common.dense_float_ir import DenseFloatParser
    from merlin.common.paths import build_dir
    from merlin.llvmlower.generic_form import to_generic_form

    # Ask the real parser to validate and print the original source. Never
    # repair an explicitly invalid region, invent casts, generalize named ops,
    # or suppress verification. Keep compiler scratch inside the project.
    scratch = build_dir() / "named-region-parser"
    scratch.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="source-", dir=scratch) as directory:
        selected = Path(directory) / "source.mlir"
        selected.write_text(source, encoding="utf-8")
        selected.chmod(0o400)
        generic = to_generic_form(selected, work=directory, timeout=30).read_text(encoding="utf-8")
    restored = DenseFloatParser(context, generic).parse_module()
    restored.verify()
    return restored
