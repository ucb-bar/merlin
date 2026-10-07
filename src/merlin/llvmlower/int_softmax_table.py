"""``int_softmax_table``: the integer softmax's per-element work, restructured exactly (default off).

A capture made with integer nonlinears (``targetgen/_integer_nonlinear.py``) spells softmax as a row
maximum, a fixed exponent grid index ``q = clamp(round((x - max) / S), lo, hi)``, an integer exp and a
64-bit floor division per element, an i64 row sum, ``P = p / T``, and then the next contraction's
per-row int8 quantization of ``P`` -- its row minimum and maximum, then a divide, round and clamp per
element. Measured on one SmolVLA SigLIP layer, the integer exp and division alone were 46 spike
instructions per element of a 12x1024x1024 score tensor.

This rewrite runs over the CAPTURED IR (the capture is unchanged) before the lowering pipeline, in the
compiler's Python (:data:`RUNNER_PRELUDE`, the source of ``_int_softmax_table_rt.py``). Each piece is
the same value as what it replaces:

* the numerator becomes a read of a table of every grid index, evaluated at compile time with the
  IR's own integer semantics;
* the clamp's upper bound (which never binds, ``x - max <= 0``) is dropped and its lower bound
  becomes a compare-and-select, so NaN -- a row with no finite maximum, poison in the original --
  also lands inside the table;
* the row sum accumulates in i32 when it provably cannot overflow;
* ``P``'s quantization is computed once per row on its candidate values ``k / T`` and each element
  reads its int8 at ``p`` (the row's largest ``P`` is ``max / T``, and the row minimum only enters
  clamped with 0 -- both checked on the IR);
* a constant step on the scores applied behind a reshape (the attention scale) moves before it, so it
  fuses into the op that produced them.

The match is STRUCTURAL -- the ops above, never a model, shape or target name -- and anything that
differs leaves the IR as it was, with the reason in the report. A test executes the original and the
rewritten module on the same inputs and requires bit-identical outputs.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

FEATURE = "int_softmax_table"
REPORT_PREFIX = "OK int_softmax_table "
#: The runner argv slot that gates the rewrite (after the data layout, so no existing slot moves).
ARGV_INDEX = 20
_RT_SOURCE = Path(__file__).with_name("_int_softmax_table_rt.py")


def runtime_source() -> str:
    """The runner half, verbatim (it is only ever executed by the compiler's Python)."""
    return _RT_SOURCE.read_text(encoding="utf-8")


#: Spliced into every lowering runner. ``_INT_SOFTMAX_TABLE`` gates the call each runner makes right
#: after it parses the module, before any other pre-pipeline rewrite.
RUNNER_PRELUDE = (
    "\n"
    + runtime_source()
    + f"\n\n_INT_SOFTMAX_TABLE = len(sys.argv) > {ARGV_INDEX} and sys.argv[{ARGV_INDEX}] == '1'\n\n\n"
    "def _ist_run_and_report(ctx, module):\n"
    "    import json as _ist_json\n"
    "    print(_IST_TOKEN + _ist_json.dumps(_int_softmax_table(ctx, module), sort_keys=True), flush=True)\n"
)

#: What each runner variant executes after parsing the module.
RUNNER_CALL = "if _INT_SOFTMAX_TABLE:\n    _ist_run_and_report(ctx, module)\n"


def _feature():
    from .impr_features import ImprFeature

    return ImprFeature(
        name=FEATURE,
        action_class="PASS",
        description=(
            "Restructure every integer softmax in the captured IR, exactly: the per-element integer "
            "exp and floor division become a table read (evaluated at compile time with the IR's own "
            "integer semantics), the never-binding upper clamp is dropped and the lower one becomes a "
            "compare-and-select, the row sum accumulates in i32 when it cannot overflow, the next "
            "contraction's per-row int8 quantization of the probabilities runs once per row on their "
            "candidate values, and the attention scale moves before the reshape that hid it from "
            "fusion. Structure-matched, never by name; default off; runner-gated."
        ),
    )


def ensure_registered() -> str:
    from .impr_features import known, register

    if FEATURE not in known():
        register(_feature())
    return FEATURE


def require_report(stdout: str, work: str | Path) -> dict:
    """The runner's report for a requested rewrite, kept beside the lowered IR.

    A requested rewrite that never reported did not run (a runner variant that skipped it would emit
    a valid baseline object), so its absence is an error. Matching nothing is not: a module without an
    integer softmax is unchanged, and the report says why each softmax-shaped region was left alone.
    """
    lines = [line[len(REPORT_PREFIX) :] for line in stdout.splitlines() if line.startswith(REPORT_PREFIX)]
    if len(lines) != 1:
        raise ValueError(f"{FEATURE} was requested but the lowering runner reported {len(lines)} times, not once")
    report = json.loads(lines[0])
    (Path(work) / f"{FEATURE}_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def apply_for_test(mlir_text: str, *, timeout: int = 600) -> tuple[str, dict]:
    """Apply the shipped rewrite to ``mlir_text`` in the compiler's Python; ``(rewritten text, report)``."""
    from .toolchain import m2m_python

    work = Path(tempfile.mkdtemp(prefix="merlin_int_softmax_table_"))
    src, dst, script = work / "in.mlir", work / "out.mlir", work / "run.py"
    src.write_text(mlir_text, encoding="utf-8")
    script.write_text(
        "import json, sys\nfrom torch_mlir import ir\n" + runtime_source() + "\nctx = ir.Context()\n"
        "with open(sys.argv[1]) as f:\n    module = ir.Module.parse(f.read(), ctx)\n"
        "report = _int_softmax_table(ctx, module)\n"
        "with open(sys.argv[2], 'w') as f:\n    f.write(str(module.operation))\n"
        "print(_IST_TOKEN + json.dumps(report, sort_keys=True))\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [str(m2m_python()), str(script), str(src), str(dst)], capture_output=True, text=True, timeout=timeout
    )
    if proc.returncode != 0:
        raise RuntimeError(f"{FEATURE} rewrite failed:\n{proc.stdout}\n{proc.stderr}")
    return dst.read_text(encoding="utf-8"), require_report(proc.stdout, work)


# Direct imports (tests, controlled build scripts) see a resolvable feature name; the lowering also
# registers it explicitly before normalizing a feature set.
ensure_registered()
