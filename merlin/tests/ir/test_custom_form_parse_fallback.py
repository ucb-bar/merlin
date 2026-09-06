"""``parse_mlir_file`` must read a module whichever form MLIR printed it in.

xDSL's dialect coverage is partial by design and leans on modules arriving in MLIR's GENERIC form.
A capture is printed that way, but two preparation-layer rewrites -- ``perop_blocks.tag_prepared_mlir``
and ``prov_cse.rewrite_prepared_file`` -- hand the module back through the MLIR printer, which emits
CUSTOM form. Every consumer downstream of them then meets text xDSL refuses:

    ParseError: Operation tensor.extract_slice does not have a custom format.

That single cause produced BOTH failure shapes. It took every ``tiny_llama`` int8 build down (that
capture has a ``tensor.extract_slice``; ``lstmnetvit`` under the identical feature set does not,
which is why it read as model-specific rather than structural). And where a consumer catches the
error instead, it degrades SILENTLY -- the block-table derivation reports "no contractions observed"
and drops the register block, so a lever is reported as applied while doing nothing.
"""
from __future__ import annotations

from pathlib import Path

import pytest

#: Custom-form MLIR: `tensor.extract_slice` printed the way the MLIR printer prints it. xDSL knows
#: the op but implements no custom parser for it, so a direct parse of this text raises.
_CUSTOM_FORM = """\
module {
  func.func @forward(%arg0: tensor<1x8xi64>) -> tensor<1x1xi64> {
    %extracted_slice = tensor.extract_slice %arg0[0, 0] [1, 1] [1, 1] : tensor<1x8xi64> to tensor<1x1xi64>
    return %extracted_slice : tensor<1x1xi64>
  }
}
"""


def test_custom_form_is_what_actually_breaks_the_direct_parse() -> None:
    """Pin the PREMISE. If xDSL ever grows a custom parser for this op the fallback test below
    would still pass while measuring nothing, so the thing being worked around is asserted first."""
    from xdsl.utils.exceptions import ParseError

    from merlin.frontends.linalg_mlir import parse_mlir_text

    with pytest.raises(ParseError):
        parse_mlir_text(_CUSTOM_FORM)


def test_parse_mlir_file_reads_custom_form(tmp_path: Path) -> None:
    """The same text, through the file door, must parse -- via the generic re-print."""
    from merlin.llvmlower.toolchain import m2m_python

    if not Path(m2m_python()).is_file():
        pytest.skip("no MLIR-capable m2m interpreter; the generic re-print cannot run here")

    from merlin.frontends.linalg_mlir import parse_mlir_file

    src = tmp_path / "model.mlir"
    src.write_text(_CUSTOM_FORM, encoding="utf-8")
    module = parse_mlir_file(src)
    names = [getattr(op, "name", "") for op in module.walk()]
    assert any("extract_slice" in n for n in names), names


def test_generic_form_modules_take_the_unchanged_path(tmp_path: Path) -> None:
    """A module that already parses must not gain a subprocess -- same bytes, same path.

    Asserted by the ABSENCE of the re-printed sibling: the fallback writes ``*.generic.mlir`` next
    to the source, so if one appears for a module that parsed directly, the fast path was skipped.
    """
    from merlin.frontends.linalg_mlir import parse_mlir_file
    from merlin.llvmlower.generic_form import GENERIC_SUFFIX

    src = tmp_path / "plain.mlir"
    src.write_text("module {\n  func.func @forward() {\n    return\n  }\n}\n", encoding="utf-8")
    parse_mlir_file(src)
    assert not list(tmp_path.glob(f"*{GENERIC_SUFFIX}"))
