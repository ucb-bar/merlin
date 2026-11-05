#!/usr/bin/env python3
"""gen_spec_int8_form.py MERLIN_ROOT [SLUG=div_Scalar] : the spec_int8 form of a DEVICE57 overload that has no integer capture.

Operator decision (Loren, 2026-11-04, "Grade under spec_int8"): aten.div.Scalar is graded the way the software spec defines
it, like Merlin's own corpus_193 row (ref_policy spec_int8, device, pass), instead of bit-exact IEEE f32.

The PROGRAM handed to the author is unchanged: the canonical f32 capture canon193/064 (`arith.divf x, 2.0` on
tensor<2x3xf32>).  Only the REFERENCE changes: it is the spec's declared quantized model evaluated exactly, and the
harness compares it EXACTLY (drive193.sh policy `exact`: numpy array_equal on the f32 output words; rtol = atol = 0),
the same comparison every other DEVICE57 int8 form gets and the same rtol=0/atol=0 Merlin's core_aten_eval applies to a
spec_int8 numeric_reference.  run_case.py's tolerance policy `spec_int8` (2*max|g|/127 + 5%) is NOT used.

Declared model (MERLIN_ROOT/examples/gemmini/target/software-spec.yaml, numerical_semantics; parameters from Merlin's
merlin.llvmlower.quantized_native.parameters, which raises on any other contract):
  calibration_domain [-1, 1], scale_policy enclosing_power_of_two -> s = 2^e, e = ceil(log2(max|domain| / 127))
  entry   (quantize_entries):   q  = RNE(clamp(x, -128 s, 127 s) / s)          NaN -> 0
  body    (exact rational):     y  = q * s / d                                   d = the capture's f32 literal divisor
  output  (output_boundary):    q' = clamp(RNE(y / s), -128, 127), result = q' * s   (one rounding, ties to even)
  = `quantized_native_fallback`: arithmetic exact_rational_native_family, output declared_byte_quantize_then_single_dtype_dequantize.
VERIFICATION INSTRUMENT: never an author input."""
import json, math, re, shutil, sys
from fractions import Fraction
from pathlib import Path

H = Path("/scratch2/loren/backend-verify-author")
ROOT = Path(sys.argv[1]); SLUG = sys.argv[2] if len(sys.argv) > 2 else "div_Scalar"
CASES = {"div_Scalar": ("aten.div.Scalar", "064")}
overload, n = CASES[SLUG]
sys.path.insert(0, str(ROOT / "src"))
import yaml  # noqa: E402
from merlin.llvmlower.quantized_native import parameters  # noqa: E402

spec = yaml.safe_load(open(ROOT / "examples/gemmini/target/software-spec.yaml"))
P = parameters(spec)
s = Fraction(P["scale_numerator"], P["scale_denominator"])
lo, hi = P["quant_min"], P["quant_max"]

src = H / "canon193" / n
text = (src / "capsule.linalg.mlir").read_text()
# Structural guard: one parallel linalg.generic whose body is exactly one arith.divf of the input by a splatted f32 literal.
body = re.findall(r'"(arith|math)\.([a-z_]+)"', text)
assert [b for b in body if b[1] not in ("constant",)] == [("arith", "divf")], body
(divisor,) = [float(v) for v in re.findall(r'arith\.constant"\(\) <\{value = ([-0-9.e+]+) : f32\}>', text)]
d = Fraction(divisor)
assert d != 0


def rne(x: Fraction) -> int:
    f = math.floor(x); r = x - f
    return f + (1 if r > Fraction(1, 2) or (r == Fraction(1, 2) and f % 2) else 0)


def model(x: float) -> float:
    if math.isnan(x): x = 0.0
    xf = min(max(Fraction(x), lo * s), hi * s)
    q = rne(xf / s)
    y = Fraction(q) * s / d
    q2 = min(max(rne(y / s), lo), hi)
    out = Fraction(q2) * s
    assert float(out) == out  # a multiple of 2^e inside the byte range: exact in f32
    return float(out)


inputs = json.load(open(src / "inputs.json"))
golden = [[model(v) for v in row] for row in inputs[0]]
req = json.load(open(src / "request.json"))
req["expected"] = dict(req["expected"], values=golden)
req["comparison"] = "equal"; req["comparison_parameters"] = {}
req["numeric_reference"] = dict(
    policy="spec_int8", engine="tools/gen_spec_int8_form.py (exact rational evaluation of the declared quantized model)",
    contract={k: P[k] for k in ("exponent", "quant_min", "quant_max", "zero_point", "nonfinite")},
    divisor=str(d), software_spec=str(ROOT / "examples/gemmini/target/software-spec.yaml"),
    fp32_eager_expected=json.load(open(src / "golden.json")),
    comparison="exact: array_equal on the f32 output words (rtol = atol = 0)")
out = H / "forms93" / SLUG; out.mkdir(parents=True, exist_ok=True)
shutil.copy(src / "capsule.linalg.mlir", out / "capsule.linalg.mlir")
shutil.copy(src / "inputs.json", out / "inputs.json")
json.dump(golden, open(out / "golden.json", "w"), indent=1)
json.dump(req, open(out / "request.json", "w"), indent=1)
# No op-capsule registry capsule exists for this form: the trace check keeps expected = {} exactly as for the f32 capture.
json.dump(dict(source="spec_int8 numeric reference (no registry capsule; expected = {} as for the f32 capture)", capsule={}),
          open(out / "capsule_meta.json", "w"), indent=1)
print(SLUG, "exponent", P["exponent"], "divisor", d, "golden", golden)
