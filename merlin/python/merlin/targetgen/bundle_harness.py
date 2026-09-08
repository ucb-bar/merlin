"""Render a whole-model bundle's harness from a pack plan and a declared gate.

This is the last ResNet-specific piece of the bundle path. The shipped harness was produced by
text-splicing the previous one (``template.index("extern void gemmini_kernel(\\n")`` plus three
regex substitutions), which is why it carries ResNet-50's ``1000``, its ``expected_top1=258`` and its
single ``float *output`` in the source. Here the pointer list comes from
:func:`merlin.targetgen.bundle_pack.plan` and the check comes from
:class:`merlin.targetgen.bundle_gate.CorrectnessGate`, so a second model needs neither a template nor
an edit.

WHAT THIS OWNS AND WHAT IT DOES NOT. The measurement ORDERING is
:func:`merlin.perf.warm_profile_harness.render_warm_then_measure_main`'s -- one completed warm
invocation, the post-warm reset, then exactly one counter-bracketed measured invocation, with
validation strictly after the closing cycle read. That is not re-implemented here, because the
ordering is the property the whole profile depends on and two copies of it would drift. This module
owns only the parts that are per-bundle: the blob externs, the kernel prototype and its pointer
arguments, and the gate's check.

⚠️ THE OUTPUT POINTER IS READ FROM THE PLAN, NOT ASSUMED TO BE FIRST. The shipped harness reads its
result at ``merlin_mutable_blob + 0``, which is true for ResNet-50 and is a property of that layout
rather than of the ABI. A model whose graded output is not the first write argument would read
whichever tensor happened to be there -- correct arithmetic on the wrong bytes, which is this
repo's most expensive failure shape.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from merlin.targetgen.bundle_gate import CONSOLE_DUMP_CAP, CorrectnessGate
from merlin.targetgen.bundle_pack import PackPlan

__all__ = ["render_bundle_harness", "BundleHarnessError", "pointer_expression",
           "CONST_SYMBOL", "MUTABLE_SYMBOL"]

#: The linker symbols the blob objects expose. Named here once so the renderer and the packaging
#: step cannot disagree about them.
CONST_SYMBOL = "merlin_const_blob_start"
MUTABLE_SYMBOL = "merlin_mutable_blob"


class BundleHarnessError(ValueError):
    """The harness cannot be rendered, and the message says what is missing."""


def pointer_expression(storage: str, offset: int) -> str:
    """The C expression for one argument's pointer. Offsets come from the plan, never from order."""
    if storage == "const":
        return f"(void *)({CONST_SYMBOL} + {int(offset)})"
    if storage == "mutable":
        return f"(void *)({MUTABLE_SYMBOL} + {int(offset)})"
    raise BundleHarnessError(f"storage {storage!r} is neither 'const' nor 'mutable'")


def _gate_check(gate: CorrectnessGate, *, output_offset: int, output_ctype: str) -> str:
    """The C for one declared gate. Every branch prints what it checked, not just a verdict."""
    out = f"({output_ctype} *)({MUTABLE_SYMBOL} + {int(output_offset)})"
    n = int(gate.output_elements)
    lines = [
        f"const {output_ctype} *merlin_out = {out};",
        f'printf("MERLIN_GATE reference=%s comparison=%s atol=%.9g rtol=%.9g elements=%d\\n",',
        f'       "{gate.reference_file}", "{gate.comparison}", {gate.atol!r}, {gate.rtol!r}, {n});',
        "int merlin_bad = 0, merlin_nonfinite = 0;",
        "unsigned long long merlin_digest = 1469598103934665603ULL;",
        f"int merlin_argmax = 0;",
        f"for (int i = 0; i < {n}; ++i) {{",
        "  const double got = (double)merlin_out[i];",
        "  if (!(got == got)) { ++merlin_nonfinite; continue; }",
        "  if (merlin_out[i] > merlin_out[merlin_argmax]) merlin_argmax = i;",
    ]
    if gate.comparison in ("exact_elementwise", "tolerance_and_topk", "trajectory"):
        lines += [
            "  const double want = (double)merlin_reference[i];",
            "  const double diff = got > want ? got - want : want - got;",
            "  const double mag  = want < 0.0 ? -want : want;",
            f"  if (diff > {gate.atol!r} + {gate.rtol!r} * mag) ++merlin_bad;",
        ]
    lines += [
        "  const unsigned char *merlin_bytes = (const unsigned char *)&merlin_out[i];",
        f"  for (unsigned b = 0; b < sizeof(merlin_out[0]); ++b) {{",
        "    merlin_digest ^= (unsigned long long)merlin_bytes[b];",
        "    merlin_digest *= 1099511628211ULL;",
        "  }",
        "}",
    ]
    # WHAT MAY BE PRINTED. Above the cap the console is the binding constraint on gradeability, so
    # the digest and the argmax stand in for the values -- and the harness SAYS it did that.
    if gate.prints_values:
        lines += [
            f'printf("MERLIN_GATE_MODE values elements=%d\\n", {n});',
            f"for (int i = 0; i < {n}; ++i) printf(\"MERLIN_OUT %d %.9g\\n\", i, (double)merlin_out[i]);",
        ]
    else:
        lines += [
            f'printf("MERLIN_GATE_MODE digest_and_argmax elements=%d cap=%d\\n",',
            f"       {n}, {CONSOLE_DUMP_CAP});",
        ]
    lines += [
        'printf("MERLIN_GATE_RESULT bad=%d nonfinite=%d argmax=%d digest=%016llx\\n",',
        "       merlin_bad, merlin_nonfinite, merlin_argmax, merlin_digest);",
    ]
    if gate.expected_argmax is not None:
        lines += [
            f'printf("MERLIN_GATE_EXPECT argmax=%d\\n", {int(gate.expected_argmax)});',
            f"const int merlin_argmax_ok = (merlin_argmax == {int(gate.expected_argmax)});",
        ]
    else:
        lines += [
            "/* No literal argmax: the gate checks AGREEMENT with the reference's own argmax, which",
            "   is the right shape for a language model -- the ranking is the result. */",
            "int merlin_ref_argmax = 0;",
            f"for (int i = 0; i < {n}; ++i)",
            "  if (merlin_reference[i] > merlin_reference[merlin_ref_argmax]) merlin_ref_argmax = i;",
            'printf("MERLIN_GATE_EXPECT argmax=%d\\n", merlin_ref_argmax);',
            "const int merlin_argmax_ok = (merlin_argmax == merlin_ref_argmax);",
        ]
    # RETURNED AS A FUNCTION, because the profile harness takes its validation as a single C
    # EXPRESSION evaluated immediately after the closing cycle read and BEFORE the metric. An
    # earlier shape emitted the gate as a post-metric block and referenced its variables from the
    # validation expression; clang rejected it outright, which is the good outcome -- had the
    # ordering merely been wrong rather than unbuildable, the metric would have been published
    # before anything checked the output.
    lines += [
        "  return (merlin_bad != 0 || merlin_nonfinite != 0 || !merlin_argmax_ok) ? 1 : 0;",
    ]
    body = "\n".join("  " + line if line else line for line in lines[:-1])
    return ("static int merlin_gate_check(void) {\n" + body + "\n" + lines[-1] + "\n}")


def render_bundle_harness(plan: PackPlan, gate: CorrectnessGate, *, entry_symbol: str,
                          output_tensor: str, output_ctype: str = "float",
                          counter_bracket: object = None,
                          reset_after_warm: str | None = None) -> dict[str, Any]:
    """``{"declarations", "call", "validate", "gate"}`` C fragments for one bundle's harness.

    ``declarations`` already contains the gate as ``merlin_gate_check()`` and ``validate`` is a call
    to it, because the profile harness takes its validation as one C expression.

    Fragments rather than a whole file, because the measurement ordering belongs to
    :func:`merlin.perf.warm_profile_harness.render_warm_then_measure_main` and is threaded through
    it -- ``call`` as the invocation, ``gate`` as the post-profile body, ``validate`` as the
    expression that gates the metric.

    ``output_tensor`` names the tensor the gate grades and is REQUIRED: its offset is read from the
    plan rather than assumed to be the first write argument, which is true of ResNet-50 and is a
    property of that layout rather than of the ABI.
    """
    if not entry_symbol or not entry_symbol.isidentifier():
        raise BundleHarnessError("the kernel entry symbol must be one plain C identifier")
    args = [*plan.const, *plan.mutable]
    if not args:
        raise BundleHarnessError("the pack plan lays out no arguments, so there is nothing to call")

    graded = next((t for t in plan.mutable if t.tensor == output_tensor), None)
    if graded is None:
        available = [t.tensor for t in plan.mutable]
        raise BundleHarnessError(
            f"the graded output {output_tensor!r} is not a write argument in this plan "
            f"(have {available[:8]}); reading the result from a guessed offset is correct "
            f"arithmetic on the wrong bytes")
    graded_elements = 1
    for extent in graded.shape:
        graded_elements *= int(extent)
    if gate.comparison != "trajectory" and graded_elements != gate.output_elements:
        raise BundleHarnessError(
            f"the gate grades {gate.output_elements} element(s) but {output_tensor!r} holds "
            f"{graded_elements}; a gate over a different count than the tensor has is checking "
            f"either padding or someone else's tensor")

    gate_fn = _gate_check(gate, output_offset=graded.offset, output_ctype=output_ctype)
    declarations = "\n".join([
        f"extern const unsigned char {CONST_SYMBOL}[];",
        f"extern unsigned char {MUTABLE_SYMBOL}[];",
        f"extern const {output_ctype} merlin_reference[];",
        f"/* {len(args)} pointer arguments, in the kernel ABI's own declared order. */",
        f"extern void {entry_symbol}(" + ", ".join(["void *"] * len(args)) + ");",
        "",
        gate_fn,
    ])

    pointers = ",\n    ".join(pointer_expression(t.storage, t.offset) for t in args)
    call = f"{entry_symbol}(\n    {pointers});"

    return {
        "declarations": declarations,
        "call": call,
        # The metric is published only on a clean gate. The check runs after the closing cycle
        # read, so its own cost is outside the measured window.
        "validate": "merlin_gate_check()",
        "gate": gate_fn,
        "n_arguments": len(args),
        "graded_output": {"tensor": graded.tensor, "offset": graded.offset,
                          "elements": graded_elements},
        "const_bytes": plan.const_bytes,
        "mutable_bytes": plan.mutable_bytes,
        "gate_declaration": gate.to_dict(),
    }
