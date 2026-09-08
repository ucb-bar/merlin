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

from collections.abc import Mapping, Sequence
from typing import Any

from merlin.targetgen.bundle_gate import CONSOLE_DUMP_CAP, CorrectnessGate
from merlin.targetgen.bundle_pack import PackPlan

__all__ = ["render_bundle_harness", "BundleHarnessError", "pointer_expression",
           "CONST_SYMBOL", "MUTABLE_SYMBOL", "TRAJECTORY_SYMBOL", "CONST_BASE_MACRO",
           "PC_RELATIVE_REACH_BYTES", "render_reseed", "render_session_loop",
           "render_freestanding_support", "unresolved_symbols", "FREESTANDING_SHIMS"]

#: The linker symbols the blob objects expose. Named here once so the renderer and the packaging
#: step cannot disagree about them.
#: Bytes per C type this module may emit as an output element. Used only to REPORT the retain
#: cost; the layout itself never depends on it.
_CTYPE_BYTES: dict[str, int] = {"float": 4, "double": 8, "int": 4, "short": 2, "char": 1}

CONST_SYMBOL = "merlin_const_blob_start"
MUTABLE_SYMBOL = "merlin_mutable_blob"

#: The preprocessor name a far const blob's base address arrives under. It must be a COMPILE-TIME
#: LITERAL, not a linker symbol: taking a symbol's address emits a relocation, and a relocation is
#: exactly what a PC-relative code model cannot satisfy across a multi-gigabyte image. A literal
#: compiles to `li` and has no reach at all.
CONST_BASE_MACRO = "MERLIN_CONST_BLOB_BASE"

#: The reach of a PC-relative reference on this ISA: +/-2 GiB. A property of the RISC-V code model,
#: not of any target -- the same constant `liveness.preconditions.medany_span` uses, and the reason
#: a large blob must be addressed absolutely rather than linked beside the code.
PC_RELATIVE_REACH_BYTES = 1 << 31

#: Where a session's per-step graded outputs are retained so the gate can run AFTER the measured
#: window. The alternative -- grading inside the loop -- puts a printf per step inside a
#: counter-bracketed region and charges it to the program.
TRAJECTORY_SYMBOL = "merlin_trajectory"


class BundleHarnessError(ValueError):
    """The harness cannot be rendered, and the message says what is missing."""


#: DEFAULT total value lines a run may print. A DEFAULT, because console throughput is a property
#: of the SUBSTRATE, not of the bundle: a FireSim console was measured at single-digit characters
#: per microsecond, while spike's is a host pipe. So a caller running a cheap simulator for a
#: diagnostic raises it, and the same bundle built for hardware does not -- which is why it is a
#: parameter of :func:`render_bundle_harness` rather than a constant it reads.
#:
#: Total value lines a run may print. `CorrectnessGate.prints_values` decides on the PER-STEP
#: element count, which is the right question for a one-shot program and the wrong one for a
#: session: SmolVLA's 1,600 elements are under the 4,096 cap, but ten steps is 16,000 lines. The
#: console is the binding constraint on what is gradeable at all -- a FireSim console measured
#: single-digit characters per microsecond -- so the budget is checked against steps x elements.
CONSOLE_LINE_BUDGET = 4096

#: Float conversion specifiers this module must NEVER emit. A baremetal console's printf is a few
#: hundred lines of `vprintfmt`, and the gemmini target's implements exactly `c s d u x l` -- no
#: float conversions at all. A `%.9g` there is not a formatting nicety that degrades gracefully: it
#: prints the SPECIFIER LITERALLY and then mis-consumes the varargs, so every later field on the
#: line is garbage. Measured on a real 10-step run: 16,000 value lines each reading
#: `MERLIN_OUT 0 %.9g`, and a header claiming `elements=-350469331`. A value dump that prints no
#: values still looks like a value dump, so this is enforced rather than remembered.
_FORBIDDEN_CONVERSIONS: frozenset[str] = frozenset("fFeEgGaA")

#: Characters that may appear between the ``%`` and its conversion: flags, width, precision, and
#: length modifiers. Scanned rather than substring-matched, because ``"%.9g"`` does not contain
#: ``"%g"`` -- a substring check passes it, which is exactly the specifier that was measured
#: printing itself literally.
_CONVERSION_PREFIX = frozenset("-+ #0123456789.*hlLqjzZt'")


def _conversions_in(fragment: str) -> set[str]:
    """Every conversion character a printf format in ``fragment`` reaches, parsed structurally."""
    found: set[str] = set()
    index = 0
    length = len(fragment)
    while index < length:
        if fragment[index] != "%":
            index += 1
            continue
        index += 1
        if index < length and fragment[index] == "%":      # an escaped percent converts nothing
            index += 1
            continue
        while index < length and fragment[index] in _CONVERSION_PREFIX:
            index += 1
        if index < length:
            found.add(fragment[index])
            index += 1
    return found


def assert_console_portable(fragment: str) -> None:
    """Refuse C that asks a baremetal printf for a float. See :data:`_FORBIDDEN_CONVERSIONS`."""
    found = sorted(_conversions_in(fragment) & _FORBIDDEN_CONVERSIONS)
    if found:
        raise BundleHarnessError(
            f"the rendered harness asks printf for float conversion(s) {found}, which a baremetal "
            f"console's vprintfmt does not implement: it prints the specifier literally and "
            f"mis-consumes every later vararg on the line. Emit the IEEE bit pattern with an "
            f"integer conversion and decode it off-target instead")


def pointer_expression(storage: str, offset: int, *, const_is_far: bool = False) -> str:
    """The C expression for one argument's pointer. Offsets come from the plan, never from order.

    ``const_is_far`` addresses the const blob from :data:`CONST_BASE_MACRO` instead of its linker
    symbol. That is the difference between a relocation and an `li`, and for an image past the
    PC-relative reach it is the difference between a program that runs and one that links and reads
    the wrong bytes.
    """
    if storage == "const":
        if const_is_far:
            return f"(void *)((unsigned char *){CONST_BASE_MACRO} + {int(offset)})"
        return f"(void *)({CONST_SYMBOL} + {int(offset)})"
    if storage == "mutable":
        return f"(void *)({MUTABLE_SYMBOL} + {int(offset)})"
    raise BundleHarnessError(f"storage {storage!r} is neither 'const' nor 'mutable'")


def render_reseed(plan: PackPlan, *, const_is_far: bool = False) -> str:
    """Copy every carried state's SEED from the const blob into its mutable working copy.

    WHY THIS IS NOT OPTIONAL. A recurrent session overwrites its carried state every step, so after
    one session the working copies hold step-N state. A warm-then-measure profile invokes the
    program twice; without a re-seed the measured invocation starts from the warm one's final state
    and is therefore a DIFFERENT program from the one that was warmed and graded. The cycle count
    would still be published, and would not be a count of the graded program -- a plausible number
    for the wrong thing.

    Emitted from the plan's declared carries, so a feed-forward plan yields an empty fragment and
    the ordering is unchanged.
    """
    if not plan.carried:
        return "/* no carried session state: nothing to re-seed */"
    lines = ["/* Re-seed every carried state so this invocation runs the graded program. */"]
    for row in plan.carried:
        seed = pointer_expression("const", int(row["seed_offset"]), const_is_far=const_is_far)
        lines.append(
            f"memcpy((void *)({MUTABLE_SYMBOL} + {int(row['working_offset'])}),"
            f" (const void *){seed},"
            f" {int(row['bytes'])}u);  /* {row['state']} */")
    return "\n".join(lines)


def render_session_loop(plan: PackPlan, *, steps: int, call: str,
                        record: Mapping[str, Any] | None = None) -> str:
    """The declared step loop: invoke, RECORD the graded output, then carry each output to its input.

    WHY IT RECORDS RATHER THAN GRADES. The gate prints a line per step, and console output inside a
    counter-bracketed window is charged to the program. But the warm and measured invocations MUST
    be the same body -- a warm run that grades and a measured run that does not are two different
    programs, and the cycle count would belong to the one that was never checked. So every step's
    graded output is copied into :data:`TRAJECTORY_SYMBOL` and the gate runs once, after the closing
    cycle read. ``record`` says how many bytes per step that costs, and the caller reports it: it is
    real work inside the measured window, small but not zero, and an unstated overhead is one that
    gets discovered as a discrepancy later.

    The carry happens AFTER the record, so a step is recorded from the output it produced rather
    than from state the next step has already overwritten.
    """
    if steps < 1:
        raise BundleHarnessError("a session loop must run at least one step")
    if steps > 1 and not plan.carried:
        raise BundleHarnessError(
            f"a {steps}-step session was asked for but the plan declares no carried state; every "
            f"step would re-run the identical computation and the loop would report a trajectory "
            f"that never advanced")
    body = [f"for (int merlin_step_index = 0; merlin_step_index < {int(steps)}; "
            f"++merlin_step_index) {{"]
    body += ["  " + line for line in call.splitlines()]
    if record is not None:
        elements, ctype = int(record["elements"]), str(record["ctype"])
        body.append(
            f"  memcpy((void *)({TRAJECTORY_SYMBOL} + (long)merlin_step_index * {elements}),"
            f" (const void *)({MUTABLE_SYMBOL} + {int(record['offset'])}),"
            f" {elements}u * sizeof({ctype}));  /* retain step for the post-window gate */")
    for row in plan.carried:
        body.append(
            f"  memcpy((void *)({MUTABLE_SYMBOL} + {int(row['working_offset'])}),"
            f" (const void *)({MUTABLE_SYMBOL} + {int(row['output_offset'])}),"
            f" {int(row['bytes'])}u);  /* carry {row['state']} */")
    body.append("}")
    return "\n".join(body)


def _gate_check(gate: CorrectnessGate, *, output_offset: int, output_ctype: str,
                step_expression: str = "0", from_trajectory: bool = False,
                dump_values: bool | None = None,
                line_budget: int = CONSOLE_LINE_BUDGET) -> str:
    """The C for one declared gate. Every branch prints what it checked, not just a verdict.

    ``step_expression`` selects this invocation's slice of the reference. For a trajectory the
    reference is ``[steps][elements]`` and grading step 3 against ``merlin_reference[0..n)`` would
    compare the right count of the wrong step -- and would PASS on any model whose trajectory barely
    moves, which is the same invisibility as the discarded activation.
    """
    n = int(gate.output_elements)
    out = (f"{TRAJECTORY_SYMBOL} + (long)({step_expression}) * {n}" if from_trajectory
           else f"({output_ctype} *)({MUTABLE_SYMBOL} + {int(output_offset)})")
    lines = [
        f"const {output_ctype} *merlin_out = {out};",
        f"const int merlin_step = (int)({step_expression});",
        f"const {output_ctype} *merlin_want = merlin_reference + (long)merlin_step * {n};",
        # The tolerances travel as IEEE bit patterns: exact, and printable by a printf with no
        # float support. Their decimal values are in the recorded gate declaration.
        f"const double merlin_atol = {gate.atol!r}, merlin_rtol = {gate.rtol!r};",
        "unsigned long long merlin_atol_bits = 0, merlin_rtol_bits = 0;",
        "for (unsigned k = 0; k < sizeof(double); ++k) {",
        "  merlin_atol_bits |= (unsigned long long)((const unsigned char *)&merlin_atol)[k] << (8u * k);",
        "  merlin_rtol_bits |= (unsigned long long)((const unsigned char *)&merlin_rtol)[k] << (8u * k);",
        "}",
        f'printf("MERLIN_GATE reference=%s comparison=%s atol_bits=%016llx rtol_bits=%016llx '
        f'elements=%d step=%d/%d\\n",',
        f'       "{gate.reference_file}", "{gate.comparison}", merlin_atol_bits, merlin_rtol_bits,',
        f"       {n}, merlin_step, {int(gate.steps)});",
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
            "  const double want = (double)merlin_want[i];",
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
    if dump_values if dump_values is not None else gate.prints_values:
        # BIT PATTERNS, not formatted floats: exact, and printable by a printf with no float
        # support. The step index is on every line, because a dump that cannot say which step it
        # belongs to is not usable for a trajectory.
        lines += [
            f'printf("MERLIN_GATE_MODE value_bits elements=%d width=%u\\n", {n},',
            "       (unsigned)sizeof(merlin_out[0]));",
            f"for (int i = 0; i < {n}; ++i) {{",
            # Assembled byte by byte rather than by memcpy or a union: no header to include and no
            # type punning, in a fragment that is spliced into someone else's translation unit.
            "  unsigned long long merlin_word = 0;",
            "  const unsigned char *merlin_raw = (const unsigned char *)&merlin_out[i];",
            "  for (unsigned k = 0; k < sizeof(merlin_out[0]); ++k)",
            "    merlin_word |= (unsigned long long)merlin_raw[k] << (8u * k);",
            '  printf("MERLIN_OUT %d %d %016llx\\n", merlin_step, i, merlin_word);',
            "}",
        ]
    else:
        lines += [
            f'printf("MERLIN_GATE_MODE digest_and_argmax elements=%d steps=%d '
            f'line_budget=%d\\n",',
            f"       {n}, {int(gate.steps)}, {int(line_budget)});",
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
        per_row = n // max(int(gate.rows), 1)
        lines += [
            "/* No literal argmax: the gate checks AGREEMENT with the reference's own ranking. For a",
            "   language model that ranking is PER TOKEN -- one argmax over every logit in the",
            "   output agrees whenever the single largest lands in the same place and says nothing",
            "   about the other rows -- so each declared row is ranked separately. */",
            f"int merlin_rows_disagreeing = 0;",
            f"for (int r = 0; r < {int(gate.rows)}; ++r) {{",
            f"  const int merlin_base = r * {per_row};",
            "  int merlin_row_argmax = 0, merlin_row_ref = 0;",
            f"  for (int i = 1; i < {per_row}; ++i) {{",
            "    if (merlin_out[merlin_base + i] > merlin_out[merlin_base + merlin_row_argmax])",
            "      merlin_row_argmax = i;",
            "    if (merlin_want[merlin_base + i] > merlin_want[merlin_base + merlin_row_ref])",
            "      merlin_row_ref = i;",
            "  }",
            "  if (merlin_row_argmax != merlin_row_ref) ++merlin_rows_disagreeing;",
            '  printf("MERLIN_TOP1 step=%d row=%d got=%d want=%d\\n", merlin_step, r,',
            "         merlin_row_argmax, merlin_row_ref);",
            "}",
            f'printf("MERLIN_GATE_EXPECT rows=%d disagreeing=%d\\n", {int(gate.rows)},',
            "       merlin_rows_disagreeing);",
            "const int merlin_argmax_ok = (merlin_rows_disagreeing == 0);",
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
    step_fn = ("static int merlin_gate_step(int merlin_step_index) {\n"
               + body + "\n" + lines[-1] + "\n}")
    if int(gate.steps) <= 1:
        # A single-step gate: the aggregate IS the one step, and it grades step 0.
        return (step_fn + "\n\n"
                "static int merlin_gate_check(void) { return merlin_gate_step(0); }")
    # A TRAJECTORY. Every step is graded and the failures are ACCUMULATED, because a session's last
    # step passing says nothing about the nine before it -- and a carry defect shows up as a
    # divergence that grows, so grading only the end is where it is largest and grading only the
    # start is where it is invisible.
    return (step_fn + "\n\n"
            "static int merlin_gate_check(void) {\n"
            "  int merlin_gate_failures = 0;\n"
            f"  for (int s = 0; s < {int(gate.steps)}; ++s)\n"
            "    merlin_gate_failures += merlin_gate_step(s);\n"
            f'  printf("MERLIN_GATE_TRAJECTORY steps=%d failed=%d\\n", {int(gate.steps)},\n'
            "         merlin_gate_failures);\n"
            "  return merlin_gate_failures != 0 ? 1 : 0;\n"
            "}")


def render_bundle_harness(plan: PackPlan, gate: CorrectnessGate, *, entry_symbol: str,
                          output_tensor: str, output_ctype: str = "float",
                          counter_bracket: object = None,
                          reset_after_warm: str | None = None,
                          const_blob_base: int | None = None,
                          near_additional_bytes: int = 0,
                          console_line_budget: int = CONSOLE_LINE_BUDGET) -> dict[str, Any]:
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

    ``const_blob_base`` moves the const blob to a FIXED ABSOLUTE address, addressed by a
    compile-time literal. Required when the image would otherwise exceed the PC-relative reach:
    tiny_llama's plan projects 2.237 GiB, and linking its 1.209 GiB blob beside the code pushes
    ordinary symbols out of the window -- a failure that is silent, because the program links and
    reads the wrong bytes. With it, only the near region (the mutable arena, the compiler's static
    arena, code) has to be reachable, and ``near_additional_bytes`` is what the plan cannot see of
    that. Both are checked here rather than discovered after a link.
    """
    if not entry_symbol or not entry_symbol.isidentifier():
        raise BundleHarnessError("the kernel entry symbol must be one plain C identifier")
    # THE ABI's OWN ORDER, never `const + mutable`. Those two coincide only while every read
    # argument precedes every write one; a recurrent session moves a carried input into the mutable
    # blob, and the concatenation then passes every pointer after the first carry to the wrong
    # parameter. Measured on SmolVLA: 1,163 arguments, and the concatenation disagrees.
    args = list(plan.arguments) if plan.abi_order else [*plan.const, *plan.mutable]
    if not args:
        raise BundleHarnessError("the pack plan lays out no arguments, so there is nothing to call")

    if near_additional_bytes < 0:
        raise BundleHarnessError("near_additional_bytes cannot be negative")
    far = const_blob_base is not None
    # WHAT HAS TO BE REACHABLE. With a far blob, only the near region does; without one, the whole
    # image does. Checked here so an unreachable layout is refused rather than linked.
    near_bytes = int(plan.mutable_bytes) + int(near_additional_bytes)
    reachable_bytes = near_bytes if far else near_bytes + int(plan.const_bytes)
    if reachable_bytes >= PC_RELATIVE_REACH_BYTES:
        detail = ("even with the const blob addressed absolutely, the near region "
                  if far else
                  "the const blob is linked beside the code, so the whole image ")
        raise BundleHarnessError(
            f"{detail}spans {reachable_bytes} bytes, at or past the {PC_RELATIVE_REACH_BYTES}-byte "
            f"PC-relative reach. A reference that cannot be reached does not fail to link: it "
            f"reads the wrong bytes. "
            + ("Reduce the near region -- the mutable arena and the compiler's static arena are "
               "what it holds." if far else
               f"Pass const_blob_base to address the {plan.const_bytes}-byte const blob from a "
               f"compile-time literal instead."))
    if far and int(const_blob_base) <= 0:
        raise BundleHarnessError("const_blob_base must be a positive absolute address")

    graded = next((t for t in plan.mutable
                   if t.tensor == output_tensor and t.role == "argument"), None)
    if graded is None:
        available = [t.tensor for t in plan.mutable]
        raise BundleHarnessError(
            f"the graded output {output_tensor!r} is not a write argument in this plan "
            f"(have {available[:8]}); reading the result from a guessed offset is correct "
            f"arithmetic on the wrong bytes")
    graded_elements = 1
    for extent in graded.shape:
        graded_elements *= int(extent)
    if graded_elements != gate.output_elements:
        raise BundleHarnessError(
            f"the gate grades {gate.output_elements} element(s) per step but "
            f"{output_tensor!r} holds {graded_elements}; a gate over a different count than the "
            f"tensor has is checking either padding or someone else's tensor")

    steps = int(gate.steps)
    record = ({"offset": graded.offset, "elements": graded_elements, "ctype": output_ctype}
              if steps > 1 else None)
    # THE CONSOLE BUDGET IS A TOTAL. `gate.prints_values` asks about one step's elements, which is
    # the right question for a one-shot program and the wrong one for a session.
    if console_line_budget < 0:
        raise BundleHarnessError("console_line_budget cannot be negative")
    total_value_lines = steps * graded_elements
    dump_values = gate.prints_values and total_value_lines <= console_line_budget
    gate_fn = _gate_check(gate, output_offset=graded.offset, output_ctype=output_ctype,
                          step_expression="merlin_step_index", from_trajectory=steps > 1,
                          dump_values=dump_values, line_budget=console_line_budget)
    const_declaration = (
        [f"/* The const blob is NOT a symbol here: it lives at the fixed absolute address",
         f"   {CONST_BASE_MACRO}, supplied as a compile-time literal so every reference to it",
         f"   compiles to `li` and emits no relocation. A relocation is what a PC-relative code",
         f"   model cannot satisfy across a {plan.const_bytes}-byte blob. */",
         f"#ifndef {CONST_BASE_MACRO}",
         f'#error "{CONST_BASE_MACRO} must be defined: the const blob is addressed absolutely"',
         "#endif"]
        if far else
        [f"extern const unsigned char {CONST_SYMBOL}[];"])
    declarations = "\n".join([
        *const_declaration,
        f"extern unsigned char {MUTABLE_SYMBOL}[];",
        f"/* {steps} step(s) x {int(gate.output_elements)} element(s). */",
        f"extern const {output_ctype} merlin_reference[];",
        *((f"/* Every step's graded output, retained so the gate runs after the counted window. */",
           f"static {output_ctype} {TRAJECTORY_SYMBOL}[{steps * graded_elements}];")
          if steps > 1 else ()),
        f"/* {len(args)} pointer arguments, in the kernel ABI's own declared order. */",
        f"extern void {entry_symbol}(" + ", ".join(["void *"] * len(args)) + ");",
        "",
        gate_fn,
    ])

    pointers = ",\n    ".join(pointer_expression(t.storage, t.offset, const_is_far=far)
                               for t in args)
    call = f"{entry_symbol}(\n    {pointers});"
    reseed = render_reseed(plan, const_is_far=far)
    session = plan.carried or steps > 1
    body = (render_session_loop(plan, steps=steps, call=call, record=record) if session else call)
    recorded_bytes = (steps * graded_elements * _CTYPE_BYTES.get(output_ctype, 0)
                      if record is not None else 0)

    for fragment in (declarations, body, reseed):
        assert_console_portable(fragment)

    return {
        "declarations": declarations,
        # ONE body for both the warm and the measured invocation -- they must be the same program.
        "call": body,
        "reseed": reseed,
        "steps": steps,
        "carried": [dict(row) for row in plan.carried],
        # Host work this shape adds INSIDE the measured window, stated rather than absorbed: the
        # per-step retain copy, plus the carries the session genuinely requires.
        "in_window_host_bytes": {
            "trajectory_retain": recorded_bytes,
            "state_carry": steps * sum(int(row["bytes"]) for row in plan.carried)},
        # The metric is published only on a clean gate. The check runs after the closing cycle
        # read, so its own cost is outside the measured window.
        "validate": "merlin_gate_check()",
        "gate": gate_fn,
        "n_arguments": len(args),
        "graded_output": {"tensor": graded.tensor, "offset": graded.offset,
                          "elements": graded_elements, "steps": steps},
        "const_bytes": plan.const_bytes,
        "mutable_bytes": plan.mutable_bytes,
        "const_blob_base": (int(const_blob_base) if far else None),
        "const_addressing": ("absolute_literal" if far else "linker_symbol"),
        "reachable_bytes": reachable_bytes,
        "pc_relative_reach_bytes": PC_RELATIVE_REACH_BYTES,
        "dumps_values": dump_values,
        "total_value_lines": total_value_lines if dump_values else 0,
        "console_line_budget": console_line_budget,
        "gate_declaration": gate.to_dict(),
    }


# ---------------------------------------------------------------------------------------------
# Freestanding support: symbols the target's baremetal environment does not provide
# ---------------------------------------------------------------------------------------------
#
# ResNet-50's kernel references nothing outside `memcpy`/`memset`, so this never came up. SmolVLA's
# flow-matching time embedding calls `sin`, `cos` and `pow`, and newlib's `pow` reaches an errno
# write, so the link fails on `__errno` -- a symbol the curated baremetal environment has no
# definition for. Vendored support trees are not ours to edit, and adding a stub straight into a
# link line is how an unrelated missing symbol later gets satisfied by accident.
#
# So a shim is only ever emitted for a symbol listed here WITH the argument for why the definition
# is honest, and an unresolved symbol not on the list is REFUSED by name. The refusal is the
# valuable half: a program that turns out to need real functionality must not link against a stub
# that returns zero.

#: Symbols this repo can give a freestanding single-threaded program an honest definition for.
#: ``why`` is not a comment -- :func:`render_freestanding_support` emits it into the generated C, so
#: a reader of the harness sees the argument alongside the definition.
FREESTANDING_SHIMS: Mapping[str, Mapping[str, str]] = {
    "__errno": {
        "definition": ("static int merlin_errno_storage;\n"
                       "int *__errno(void) { return &merlin_errno_storage; }"),
        "why": ("newlib's libm writes errno on a domain or range error, and generated kernel code "
                "never reads it, so STORAGE is the whole requirement and providing it changes no "
                "computed value. If generated code ever read errno this shim would be wrong and "
                "plausible, which is exactly why it is listed with its justification instead of "
                "being added as a link fix"),
    },
}


def unresolved_symbols(linker_output: str) -> tuple[str, ...]:
    """Symbol names a linker reported as undefined, parsed structurally from its own message.

    Read from what the linker actually said rather than predicted from the object, because the
    question is not "what does this object reference" (``sin``, ``pow`` and ``memcpy`` are all
    referenced and all resolve) but "what did the environment fail to supply".
    """
    marker = "undefined reference to "
    found: list[str] = []
    for line in linker_output.splitlines():
        _, sep, tail = line.partition(marker)
        if not sep:
            continue
        tail = tail.strip()
        if not tail:
            continue
        opener = tail[0]
        closers = {"`": "'", "'": "'", '"': '"'}
        if opener not in closers:
            continue
        name, sep, _ = tail[1:].partition(closers[opener])
        if sep and name and name not in found:
            found.append(name)
    return tuple(found)


def render_freestanding_support(symbols: Sequence[str]) -> str:
    """C definitions for ``symbols``, or refuse and name the ones with no honest definition.

    ``symbols`` are the names a link actually failed on. Every one must be listed in
    :data:`FREESTANDING_SHIMS`; anything else raises, because stubbing an unknown symbol produces a
    program that links and computes something other than what it declares.
    """
    unknown = [name for name in symbols if name not in FREESTANDING_SHIMS]
    if unknown:
        raise BundleHarnessError(
            f"the link is unresolved on {sorted(unknown)}, for which this repo has no honest "
            f"freestanding definition (it can supply {sorted(FREESTANDING_SHIMS)}). A stub would "
            f"let the program link and compute something other than what it declares; supply the "
            f"real symbol through the target's support sources, or add it here WITH the argument "
            f"for why a stub is faithful")
    if not symbols:
        return "/* the environment resolved every referenced symbol: no shim needed */"
    blocks = ["/* Freestanding support. Each definition carries the argument for why it is",
              "   faithful; see merlin.targetgen.bundle_harness.FREESTANDING_SHIMS. */"]
    for name in symbols:
        shim = FREESTANDING_SHIMS[name]
        blocks.append(f"/* {name}: {shim['why']} */")
        blocks.append(str(shim["definition"]))
    return "\n".join(blocks)


# ---------------------------------------------------------------------------------------------
# Placing a far const blob, so the C literal and the linker cannot disagree
# ---------------------------------------------------------------------------------------------
#
# A far blob has two halves that must agree: the address the harness compiles into an `li`, and the
# address the linker actually puts the bytes at. If they disagree the program still links and still
# runs -- it reads whatever is at the literal. So both come from ONE call here.
#
# The placement is a linker OPTION, not a script edit. `INSERT AFTER` cannot be used: a script
# containing INSERT augments ld's DEFAULT script, so combined with the target's own `-T` script the
# insert point is not found ("`.text` not found for insert"). `--section-start` needs no script at
# all, which leaves the target's curated linker script authoritative -- and that script is vendored,
# so not editing it is the point rather than a convenience.

#: The section a far const blob is emitted into. Named here once so the assembly that defines it and
#: the linker flag that places it cannot drift apart.
FAR_BLOB_SECTION = ".merlin_const_blob"


def far_blob_link_flags(const_blob_base: int, *, section: str = FAR_BLOB_SECTION) -> tuple[str, ...]:
    """Linker flags placing ``section`` at ``const_blob_base``, as its own load segment."""
    if not isinstance(const_blob_base, int) or isinstance(const_blob_base, bool):
        raise BundleHarnessError("const_blob_base must be an integer address")
    if const_blob_base <= 0:
        raise BundleHarnessError("const_blob_base must be a positive absolute address")
    if not section.startswith("."):
        raise BundleHarnessError(f"section {section!r} must be an ELF section name")
    return (f"-Wl,--section-start={section}={const_blob_base:#x}",)


def far_blob_compile_flags(const_blob_base: int) -> tuple[str, ...]:
    """The compile-time literal the harness reaches the blob by. Same number, one source."""
    if not isinstance(const_blob_base, int) or isinstance(const_blob_base, bool):
        raise BundleHarnessError("const_blob_base must be an integer address")
    if const_blob_base <= 0:
        raise BundleHarnessError("const_blob_base must be a positive absolute address")
    return (f"-D{CONST_BASE_MACRO}={const_blob_base:#x}UL",)


def render_far_blob_assembly(*, blob_path: str, section: str = FAR_BLOB_SECTION) -> str:
    """Assembly placing ``blob_path``'s bytes in ``section``, with no symbol anyone must reach.

    A symbol is emitted for a reader's benefit and deliberately not used by the harness: taking its
    address would be a relocation, which is the thing the absolute literal exists to avoid.
    """
    if not blob_path:
        raise BundleHarnessError("the blob path is required")
    return "\n".join([
        f'/* {section} is placed at an absolute address by far_blob_link_flags(); the harness',
        f'   reaches it through the {CONST_BASE_MACRO} literal and never through this symbol. */',
        f'    .section {section}, "a"',
        "    .balign 64",
        "    .global merlin_far_const_blob_start",
        "merlin_far_const_blob_start:",
        f'    .incbin "{blob_path}"',
        "    .global merlin_far_const_blob_end",
        "merlin_far_const_blob_end:",
        "",
    ])
