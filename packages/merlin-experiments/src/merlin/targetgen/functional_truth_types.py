"""User-reviewed functional truth for Phase 0 (functional, all native ops).

Vendored semantic rule functions from phase0-release/truth/truth_rules.py.
No placement, source input values, run artifacts or operator coverage roster
participates. See docs/compiler/phase0_functional_truth.md for provenance.
"""

import math
import struct

# dtype classes
FLOAT = {"float16", "bfloat16", "float32", "float64"}
COMPLEX = {"complex32", "complex64", "complex128"}
INT = {"int8", "int16", "int32", "int64", "uint8"}
BOOL = {"bool"}
INT_RANGE = {
    "int8": (-128, 127),
    "uint8": (0, 255),
    "int16": (-(2**15), 2**15 - 1),
    "int32": (-(2**31), 2**31 - 1),
    "int64": (-(2**63), 2**63 - 1),
    "bool": (0, 1),
}
INT_BITS = {"int8": 8, "uint8": 8, "int16": 16, "int32": 32, "int64": 64}


def dclass(dt):
    if dt in FLOAT:
        return "float"
    if dt in COMPLEX:
        return "complex"
    if dt in INT:
        return "int"
    if dt in BOOL:
        return "bool"
    return "other"


def integral(dt):
    """Integer-valued domain (int or bool)."""
    return dclass(dt) in ("int", "bool")


# ----------------------------------------------------------------------------------------------
# Truth categories, family ids and hack / not-native reason classes
MUST, TRIV, HACK, CPLX, NN = (
    "must_accelerate",
    "trivially_covered",
    "exception_hack",
    "exception_complex_composition",
    "not_native",
)
TRIVM = "trivially_covered_metadata"
FAM = {
    "N1": "contraction (matmul/conv/linear map on the systolic array)",
    "N2": "additive reduction / static-window sum (accumulate, ones-vector matmul, 1/N readout scale)",
    "N3": "affine map with compile-time scalars (operand sum, scale, shift, readout ReLU)",
    "N3c": (
        "complex pointwise runtime x literal: (re, im) -> constant 2x2 real lin"
        "ear map (+ constant shift) per element, following eager's full complex"
        " formula"
    ),
    "N4": "pure movement / conversion (static byte selection, view, broadcast, pad, widening)",
    "N5": "uniform literal fill",
    "N6": (
        "identity specialization (op equals identity / uniform fill on the whol"
        "e dtype domain or for the given arguments/shapes)"
    ),
    "T0": "empty: numel 0 on every output",
    "T1": "metadata-only, no work (allocation / size query / metadata resize)",
    "N2e": (
        "integer window mean: exact device window sum + exact host readout epil"
        "ogue (eager C integer division, truncation), same pattern as requant"
    ),
    "N1t": (
        "prefix sum as a contraction with a constant lower-triangular 0/1 matri"
        "x (same linear-map family as the ones-vector sum)"
    ),
    "X_ieee_special": "not native: IEEE special-value outputs outside the quantized contract (division by a literal 0)",
    "H1": "hack: Boolean OR (add + compare), elementwise or as a reduction",
    "H2": "hack: elementwise product of runtime operands (runtime Hadamard), incl. Boolean AND and product reductions",
    "H3": "hack: nonuniform literal result table",
    "H4": "hack: saturation comparator",
    "H5": "hack: readout-offset rounding (integer rounding of a native linear result)",
    "C1": "complex x complex runtime contraction (multi-matmul composition)",
    "X_transcendental": "not native: transcendental / nonlinear function",
    "X_comparison": "not native: comparison producing a data-dependent mask (sign, compare, max/min select)",
    "X_rounding": (
        "not native: rounding of non-integral values (floor/ceil/round/trunc of floats, integer floor division)"
    ),
    "X_runtime_division": "not native: division / modulo by a runtime tensor or by a non-unit constant",
    "X_bitwise": "not native: bitwise function that is not a byte selection (xor, not, masks)",
    "X_bool_table": "not native: bool-domain result is a nonuniform 2-entry literal table",
    "X_select_sort": "not native: data-dependent selection (max/min reductions, arg-indices, sort, topk)",
    "X_index": "not native: data-dependent addressing (gather/scatter/embedding/index by runtime indices)",
    "X_random": "not native: random generation",
    "X_composition": (
        "not native: composition of several units / runtime products and reductions (norms, softmax, var, distances)"
    ),
    "X_interp": "not native: nonuniform-weight interpolation / data-dependent sampling",
    "X_fft": "not native: DFT (constant complex matrix, inexact / multi-op)",
    "X_shape": "not native: data-dependent output shape",
    "X_host_scalar": "not native: result is a host Scalar read back from tensor data",
    "X_select_runtime": "not native: elementwise select by a runtime condition / runtime bounds",
    "X_conversion": "not native: conversion that rounds or compares (float->int truncation, x->bool)",
}


class V:
    __slots__ = ("truth", "family", "reason", "cls", "eq", "amb")

    def __init__(self, truth, family, reason, cls=None, eq=None, amb=None):
        self.truth, self.family, self.reason, self.cls, self.eq, self.amb = truth, family, reason, cls, eq, amb


# ----------------------------------------------------------------------------------------------
# Value normalisation (structure only; tensor VALUES are dropped for the derivation)
def norm(o):
    if isinstance(o, dict):
        k = o.get("kind")
        if k == "tensor":
            return Tensor(o)
        if k in ("tuple", "list"):
            return [norm(x) for x in o.get("items", [])]
        if k in ("dtype", "memory_format", "layout", "device"):
            return o.get("value")
        if k == "float":
            return float(o["value"])
        if k == "complex":
            re_, im = o.get("real"), o.get("imag")
            return complex(norm(re_) if isinstance(re_, dict) else re_, norm(im) if isinstance(im, dict) else im)
        return {kk: norm(v) for kk, v in o.items()}
    if isinstance(o, list):
        return [norm(x) for x in o]
    return o


class Tensor:
    __slots__ = ("dtype", "shape", "stride", "offset", "raw")

    def __init__(self, d):
        self.dtype, self.shape = d.get("dtype"), list(d.get("shape") or [])
        self.stride, self.offset = d.get("stride"), d.get("storage_offset")
        self.raw = d

    @property
    def numel(self):
        n = 1
        for s in self.shape:
            n *= s
        return n

    def __repr__(self):
        return f"{self.dtype}{self.shape}"


def tensors(o, acc=None):
    acc = [] if acc is None else acc
    if isinstance(o, Tensor):
        acc.append(o)
    elif isinstance(o, list):
        for x in o:
            tensors(x, acc)
    elif isinstance(o, dict):
        for x in o.values():
            tensors(x, acc)
    return acc


# ----------------------------------------------------------------------------------------------
# Schema binding
def split_top(s):
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


def parse_default(s):
    s = s.strip()
    if s == "None":
        return None
    if s in ("True", "False"):
        return s == "True"
    if s.startswith("["):
        inner = s[1:-1].strip()
        return [parse_default(x) for x in split_top(inner)] if inner else []
    if s.startswith(("'", '"')):
        return s[1:-1]
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        return s


def parse_schema(schema):
    start = schema.index("(")
    depth = 0
    for i in range(start, len(schema)):
        depth += {"(": 1, ")": -1}.get(schema[i], 0)
        if depth == 0:
            break
    params, ret = schema[start + 1 : i], schema[i + 1 :].partition("->")[2].strip()
    out = []
    kwonly = False
    for p in split_top(params):
        if p == "*":
            kwonly = True
            continue
        typ, _, rest = p.partition(" ")
        name, eq, dflt = rest.partition("=")
        out.append(
            {
                "type": typ,
                "name": name.strip(),
                "has_default": bool(eq),
                "default": parse_default(dflt) if eq else None,
                "kwonly": kwonly,
            }
        )
    return out, ret


class Case:
    def __init__(self, raw, source):
        self.source = source
        self.ov = raw["overload"]
        self.case_id = raw.get("case_id") or ("corpus193::" + raw["overload"])
        self.schema = raw["schema"]
        params, _ = parse_schema(self.schema)
        args = norm(raw["arguments"].get("args", []))
        kwargs = norm(raw["arguments"].get("kwargs", {})) or {}
        self.a = {}
        pos = [p for p in params]
        for i, v in enumerate(args):
            self.a[pos[i]["name"]] = v
        for k, v in kwargs.items():
            self.a[k] = v
        for p in params:
            if p["name"] not in self.a:
                self.a[p["name"]] = p["default"]
        self.ins = tensors(args) + tensors(kwargs)
        exp = norm(raw["expected"])
        self.outs = tensors(exp)
        self.out_scalar = not isinstance(exp, (Tensor, list))
        pa = raw.get("partition_assignment") or {}
        self.dt = pa.get("dtype") or (self.ins[0].dtype if self.ins else (self.outs[0].dtype if self.outs else None))
        self.comparison = raw.get("comparison")

    # helpers
    def x(self):
        return self.ins[0]

    def out(self, i=0):
        return self.outs[i]

    def reduce_size(self, inp=None, out=None):
        """Number of input elements combined into each output element (input numel / output numel)."""
        inp = inp or self.ins[0]
        out = out or self.outs[0]
        if out.numel == 0:
            return None
        return inp.numel // out.numel if inp.numel else 0

    def dim_size(self, t, d):
        if not t.shape:
            return 1
        d = d if d >= 0 else d + len(t.shape)
        return t.shape[d]


# ----------------------------------------------------------------------------------------------
# Numeric helpers for bool-domain evaluation and IEEE rounding of the result dtype
def round_to(dt, v):
    if isinstance(v, bool):
        v = int(v)
    if dt in ("float32",):
        try:
            return struct.unpack("f", struct.pack("f", v))[0]
        except OverflowError:
            return math.copysign(math.inf, v)
    if dt == "float16":
        try:
            return struct.unpack("e", struct.pack("e", v))[0]
        except OverflowError:
            return math.copysign(math.inf, v)
    if dt == "bfloat16":
        f = round_to("float32", v)
        if math.isnan(f) or math.isinf(f):
            return f
        b = struct.unpack("I", struct.pack("f", f))[0]
        b = (b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000
        return struct.unpack("f", struct.pack("I", b))[0]
    if dt == "bool":
        return 1 if v else 0
    return v


def same(a, b):
    """Equality on result values: NaN == NaN, +0 == -0 (IEEE equality up to the sign of zero)."""
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def table2(f, out_dt):
    """Evaluate a bool-domain function: returns 'identity', 'uniform' or 'table'."""
    v0, v1 = round_to(out_dt, f(0)), round_to(out_dt, f(1))
    if same(v0, v1):
        return "uniform", (v0, v1)
    if same(v0, 0) and same(v1, 1):
        return "identity", (v0, v1)
    return "table", (v0, v1)


def safe(fn):
    def g(x):
        try:
            return fn(x)
        except ZeroDivisionError:
            return math.inf
        except (ValueError, OverflowError):
            return math.nan

    return g


UNARY_REF = {
    "acos": math.acos,
    "asin": math.asin,
    "atan": math.atan,
    "cos": math.cos,
    "sin": math.sin,
    "tan": math.tan,
    "cosh": math.cosh,
    "sinh": math.sinh,
    "tanh": math.tanh,
    "erf": math.erf,
    "exp": math.exp,
    "expm1": math.expm1,
    "asinh": math.asinh,
    "sqrt": math.sqrt,
    "acosh": lambda x: math.acosh(x) if x >= 1 else math.nan,
    "atanh": lambda x: math.inf if x == 1 else math.atanh(x),
    "log": lambda x: -math.inf if x == 0 else math.log(x),
    "log2": lambda x: -math.inf if x == 0 else math.log2(x),
    "log10": lambda x: -math.inf if x == 0 else math.log10(x),
    "log1p": math.log1p,
    "reciprocal": lambda x: math.inf if x == 0 else 1 / x,
    "rsqrt": lambda x: math.inf if x == 0 else 1 / math.sqrt(x),
    "sigmoid": lambda x: 1 / (1 + math.exp(-x)),
}


def is_pow2_recip_int(c):
    """True if 1/c is an integer power of two in magnitude (c = +-2^-k, k >= 0): x/c is then an
    exact integer for every integer x, so floor/trunc are no-ops."""
    if not isinstance(c, (int, float)) or c == 0 or math.isinf(c) or math.isnan(c):
        return False
    m, e = math.frexp(abs(c))
    return m == 0.5 and e <= 1  # |c| = 2^(e-1) with e-1 <= 0


def scalar_val(v):
    if isinstance(v, bool):
        return int(v)
    return v


# ----------------------------------------------------------------------------------------------
# Conversions
def conversion(src, dst):
    """Classify a dtype conversion. Returns (ok, note)."""
    if src == dst:
        return True, "same dtype (copy)"
    s, d = dclass(src), dclass(dst)
    if s == "bool":
        return True, "bool widening (0/1)"
    if d == "bool":
        return False, "x != 0 comparator (H4)"
    if s == "int" and d == "int":
        return True, "integer widening (accumulator readout) / narrowing (low-byte selection, modular)"
    if s == "int" and d in ("float", "complex"):
        return True, "integer to float: exact host requant (RNE)"
    if s == "float" and d in ("float", "complex"):
        return True, "float to float: quantized contract identity + host requant"
    if s == "complex" and d == "complex":
        return True, "complex precision change: per-component float conversion"
    if s == "complex" and d == "float":
        return True, "complex to real: real-part byte selection"
    if s in ("float", "complex") and d == "int":
        return False, "float to int truncation toward zero (H5 rounding)"
    return False, f"unclassified conversion {src}->{dst}"


# ----------------------------------------------------------------------------------------------
# Rule helpers producing verdicts
def quant_note(c):
    dts = {t.dtype for t in c.ins + c.outs}
    notes = []
    if dts & FLOAT:
        notes.append("float dtypes via the int8 quantized contract (host quantize -> accelerator -> exact host requant)")
    if dts & COMPLEX:
        notes.append("complex: per-component")
    if dts & INT:
        notes.append("integers: exact limb arithmetic with modular wrap")
    return "; ".join(notes)


N3C_EQ = {
    "mul": (
        "mul(z, s) per element: (re, im) -> [[sr, -si], [si, sr]] (re, im) with"
        " eager's full complex formula (re*sr - im*si, re*si + im*sr), so a rea"
        "l s still carries the 0*im / 0*re terms (zero sign and NaN/inf couplin"
        "g follow eager)"
    ),
    "div": (
        "div(z, s) per element: eager complex division by the literal s; for s "
        "= c+0i the map is (re, im) -> eager's formula with the 0 imaginary par"
        "t kept (NaN/inf coupling follows eager; s = 0 gives the eager non-fini"
        "te pattern)"
    ),
    "add": (
        "add/sub(z, w | s, alpha) per element: (re, im) -> (re +- alpha*Re w, i"
        "m +- alpha*Im w) with alpha*w evaluated by eager's complex product (al"
        "pha+0i)*w"
    ),
    "neg": "neg(z) per element: (re, im) -> (-re, -im)",
    "other": (
        "complex runtime x literal pointwise map: constant 2x2 real linear map "
        "plus constant shift per element, eager formula"
    ),
}


def must(fam, reason, c, cls=None, eq=None, amb=None):
    q = quant_note(c)
    if fam == "N3" and any(dclass(t.dtype) == "complex" for t in c.ins):
        fam = "N3c"
        n = c.ov.split(".")[1]
        key = "add" if n in ("add", "sub") else (n if n in N3C_EQ else "other")
        cls = cls or f"{c.ov.replace('aten.', '')} complex runtime x literal = 2x2 linear map"
        eq = eq or N3C_EQ[key]
        reason = reason + "; complex runtime x literal as a constant 2x2 real linear map per element (eager formula)"
    return V(MUST, fam, reason + ("; " + q if q else ""), cls, eq, amb)


def hack(h, reason, cls=None, eq=None):
    return V(HACK, h, reason, cls, eq)


def nn(x, reason, cls=None, eq=None, amb=None):
    return V(NN, x, reason, cls, eq, amb)


def conversions_ok(c, srcs, dst):
    bad = []
    for t in srcs:
        ok, note = conversion(t.dtype, dst)
        if not ok:
            bad.append((t.dtype, dst, note))
    return bad


# ---------------------------------------------------------------- native-51 handlers
