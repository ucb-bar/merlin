"""Authenticate known numeric boundaries through their ordinary generic lowering.

This grants only payload provenance. Numeric admission and exactness still need
all ordinary source, dtype, quantization and provider proofs.
"""


def checked_boundary(op):
    from xdsl.dialects.builtin import ModuleOp
    from merlin.llvmlower.passes_xdsl import _DEQUANT_KINDS, _QUANT_KINDS, lower_quant_ext
    from merlin.targetgen.native_payload_carriers import op_name
    from merlin.targetgen.core_aten_spec_int8 import validate_quantization_attributes

    if op_name(op) not in _DEQUANT_KINDS | _QUANT_KINDS:
        return False
    if len(op.operands) != 3 or len(op.results) != 1:
        return False
    copy = op.clone()
    scratch = ModuleOp([copy])
    rewrites = []
    try:
        validate_quantization_attributes(op)
        lower_quant_ext(scratch, on_rewrite=lambda old, replacements, new: rewrites.append(new))
        if len(rewrites) != 1 or copy.parent is not None:
            return False
        for replacement in rewrites:
            replacement.verify()
        return True
    except Exception:  # Unsupported boundary semantics remain UNKNOWN, never a carrier grant.
        return False
    finally:
        scratch.drop_all_references()
        copy.drop_all_references()
