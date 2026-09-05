"""Re-print an MLIR module in GENERIC form so a partial dialect implementation can read it.

WHY THIS EXISTS. Several derivations in this repo read a module through xDSL, whose dialect
coverage is deliberately partial: it implements the ops the derivations reason about, and for the
rest it relies on the module arriving in MLIR's *generic* form (``"tensor.extract_slice"(%x) <{...}>
: (...) -> ...``), which needs no per-op parser at all. A model2MLIR capture is printed that way, so
that has held.

It stops holding the moment a module has been round-tripped through the MLIR printer -- which is
what ``perop_blocks.tag_prepared_mlir`` and ``prov_cse.rewrite_prepared_file`` do. MLIR prints in
*custom* form by default, and xDSL then refuses ops whose custom syntax it does not implement::

    ParseError: Operation tensor.extract_slice does not have a custom format.

MEASURED: that refusal blocked every ``tiny_llama`` int8 build the moment the post-codegen census
(:mod:`merlin.llvmlower.codegen_census`) began reading the module lowering actually receives -- a
module which, with ``perop_register_block`` or ``cse_through_provenance`` enabled, has been through
the MLIR printer. ``tensor.extract_slice`` is not special and is not the last such op; the fix
therefore cannot be to teach xDSL one more spelling.

WHAT THIS DOES. Hands the text back through the SAME interpreter that printed it (the model2MLIR
venv, the only one with the MLIR bindings -- exactly as ``prov_cse`` and ``perop_blocks`` already
do) and asks for ``print_generic_op_form=True``. Same interpreter, so there is no version skew
between the printer that produced the file and the parser that re-reads it. Nothing is rewritten:
generic form is a printing mode, not a transformation, so the module a caller then parses is the
module on disk.

FAIL CLOSED. Every failure here raises :class:`GenericFormError`. A caller must not treat an
un-normalizable module as an empty one: the gates that read these modules exist because a check
that silently could not run reported success.
"""
from __future__ import annotations

import subprocess
from pathlib import Path


class GenericFormError(RuntimeError):
    """A module could not be re-printed in generic form."""


#: The re-print, as it runs in the m2m venv. ``assume_verified`` because the module is mid-pipeline
#: and already verified by whatever produced it; ``large_elements_limit=None`` so no dense attribute
#: is elided into ``...``, which would make the result unparseable in a different way.
_PRINT_GENERIC_SRC = (
    "import sys\n"
    "from torch_mlir import ir\n"
    "src, dst = sys.argv[1], sys.argv[2]\n"
    "ctx = ir.Context()\n"
    "ctx.allow_unregistered_dialects = True\n"
    "mod = ir.Module.parse(open(src).read(), ctx)\n"
    "with ctx:\n"
    "    txt = mod.operation.get_asm(print_generic_op_form=True, large_elements_limit=None,\n"
    "                                assume_verified=True)\n"
    "open(dst, 'w').write(txt)\n"
    "print('OK generic-form reprint', len(txt), 'bytes')\n"
)

#: Suffix of the re-printed sibling file. Written next to the source (a build work dir), never over it.
GENERIC_SUFFIX = ".generic.mlir"


def to_generic_form(mlir_path: "str | Path", work: "str | Path | None" = None,
                    timeout: int = 3600) -> Path:
    """Re-print ``mlir_path`` in generic form; return the path of the re-printed file.

    Raises :class:`GenericFormError` if the m2m interpreter is unavailable or the re-print fails.
    """
    from .toolchain import m2m_python

    src = Path(mlir_path)
    out_dir = Path(work) if work is not None else src.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / (src.stem + GENERIC_SUFFIX)
    interp = m2m_python()
    if not Path(interp).is_file():
        raise GenericFormError(
            f"cannot re-print {src} in generic form: no MLIR-capable interpreter at {interp} "
            "(set MERLIN_M2M_VENV / MERLIN_M2M_DIR). Refusing rather than reading a module "
            "nothing could parse.")
    script = out_dir / "_print_generic.py"
    script.write_text(_PRINT_GENERIC_SRC, encoding="utf-8")
    proc = subprocess.run([str(interp), str(script), str(src), str(out)],
                          capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0 or not out.is_file():
        raise GenericFormError(
            f"generic-form re-print of {src} failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
    return out


def parse_mlir_file_any_form(mlir_path: "str | Path", work: "str | Path | None" = None):
    """Parse ``mlir_path`` into an xDSL module whichever form MLIR printed it in.

    The generic re-print is only attempted when the direct parse fails, so a module that already
    parses takes exactly the path it took before this function existed -- byte-identical behaviour
    and no extra subprocess for every module that never had the problem.
    """
    from ..frontends.linalg_mlir import parse_mlir_file
    from xdsl.utils.exceptions import ParseError

    try:
        return parse_mlir_file(mlir_path)
    except ParseError as first:
        try:
            generic = to_generic_form(mlir_path, work)
        except GenericFormError as exc:
            raise GenericFormError(
                f"{mlir_path} is printed in MLIR custom form that xDSL cannot read "
                f"({str(first).splitlines()[0] if str(first) else first}) and it could not be "
                f"re-printed in generic form: {exc}") from first
        return parse_mlir_file(generic)
