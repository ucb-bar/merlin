#!/usr/bin/env python3
"""Unified capsule-corpus generator — ONE generator for every target.

Replaces the two forked generators (this file's gemmini/integer predecessor + ``atlas/generate_atlas_corpus.py``
atlas/float). For each target it loads the declarative ``profiles/<target>.yaml`` (the target-agnostic test
DEFINITION — op + shapes-in-tiles + epilogue, plus the numeric datapath), derives the per-target binding from
the target's descriptor via :mod:`merlin.targetgen.corpus_spec` (dtypes, tile dim, instruction classes, oracle
tiers — nothing hand-set per target), builds each capsule, computes its golden with the regime's engine
(integer = the :mod:`capsule_golden` recompute; float = the external ``specir`` fp8/bf16 refmodel), and writes
the 5-file capsule dir into the target's own corpus root (``Path(te.capsule_corpus).parent`` — gemmini at the
contract root, atlas under ``atlas/``). Only capsules named in a profile are (over)written; hand-authored
capsules (e.g. gemmini's movement/conv) are left untouched.

Run:  PYTHONPATH=/scratch2/agustin/mvp-lhwir/spec .venv/bin/python \
          merlin/contract/capsules/generate_corpus.py            # all targets with a profile
      ... merlin/contract/capsules/generate_corpus.py --target gemmini
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import os
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "merlin" / "python"))

from merlin.targetgen import capsule_golden as CG            # noqa: E402
from merlin.targetgen import corpus_spec as CS               # noqa: E402
from merlin.targetgen.target_experiment import load_target_experiment  # noqa: E402

HERE = Path(__file__).resolve().parent
PROFILES = HERE / "profiles"


# ------------------------------------------------------------------------------------------------
# float golden engine (generation-time only; needs the external specir refmodel)
# ------------------------------------------------------------------------------------------------
def _specir():
    root = os.environ.get("SPECIR_ROOT", "/scratch2/agustin/mvp-lhwir/spec")
    if root not in sys.path:
        sys.path.insert(0, root)
    from specir.oracle import dtypes as D
    from specir.oracle.refmodel import fp_reduce
    return D, fp_reduce


# specir fp8 format handle per canonical operand dtype token (fail closed if the refmodel lacks it).
_SPECIR_FP8_ATTR = {"fp8_e4m3": "FP8_E4M3", "fp8_e5m2": "FP8_E5M2"}


def _specir_fp8(D, fmt_token: str):
    attr = _SPECIR_FP8_ATTR.get(fmt_token)
    if attr is None or not hasattr(D, attr):
        raise ValueError(f"specir refmodel has no fp8 format for operand dtype {fmt_token!r} "
                         f"(known: {sorted(_SPECIR_FP8_ATTR)})")
    return getattr(D, attr)


def _det_fp8(D, name, shape, salt, fmt_token, d_fp8):
    """Structured, format-DERIVED operand bytes: distinct rows AND columns + asymmetric (so a wrong row
    stride / base offset / transposed load changes the output), spanning the fp8 format's representable
    range. Replaces the old 11-magnitude flat-hash fill (~6 distinct values, ~11/32 distinct rows) that hid
    those bug classes. See merlin.targetgen.corpus_operands."""
    from merlin.targetgen import corpus_operands as CO
    salt_int = sum((i + 1) * ord(c) for i, c in enumerate(f"{salt}|{name}")) or 1
    vals = CO.operand_values(tuple(shape), fmt_token, salt_int)
    raw = [D.encode_float(v, d_fp8) for v in vals]
    # Self-enforcing rigor: fail generation loudly if the ENCODED bytes are not distinct-per-row/col +
    # asymmetric (e.g. a future palette/fill change, or an encode that collapsed distinct values). A weak
    # operand silently hides addressing/stride/transpose bugs — never let a regeneration ship one.
    if len(shape) == 2:
        problems = CO.rigor_findings([float(b) for b in raw], tuple(shape))
        if problems:
            raise AssertionError(f"non-rigorous operand {name}{tuple(shape)}: {problems}")
    return raw, vals


def _float_golden(entry, binding):
    """A capsule's fp8->bf16 golden + input provenance from the specir refmodel (independent of the RTL)."""
    D, fp_reduce = _specir()
    fmt_token = binding.operand_dtype                    # e.g. "fp8_e4m3" — DERIVED, not assumed
    FP8, BF16 = _specir_fp8(D, fmt_token), D.BF16
    salt, dim = entry["name"], binding.tile_dim
    prov, outputs = {}, {}

    def reg(name, shape):
        raw, vals = _det_fp8(D, name, shape, salt, fmt_token, FP8)
        prov[name] = {"shape": list(shape), "fp8_raw_hex": [f"0x{r:02x}" for r in raw], "decoded": vals}
        return raw

    def rnd(x):
        return D.round_to_format(x, BF16, "rne")

    def mm(a_raw, ashape, w_raw, wshape):
        m, k = ashape
        _, n = wshape
        out = [[0] * n for _ in range(m)]
        for i in range(m):
            for j in range(n):
                prods = [rnd(D.decode_float_exact(a_raw[i * k + p], FP8)
                             * D.decode_float_exact(w_raw[p * n + j], FP8)) for p in range(k)]
                out[i][j] = fp_reduce(prods, BF16, order="index_sequential", cadence="per_step", rm="rne")
        return out

    def floats(y):
        return [[D.decode_float(v, BF16) for v in row] for row in y]

    op = entry.get("op", "matmul")
    if op in ("matmul", "linear"):
        M = entry.get("M", entry.get("M_tiles", 1) * dim)
        K = entry.get("K", entry.get("K_tiles", 1) * dim)
        N = entry.get("N", entry.get("N_tiles", 1) * dim)
        a = reg(entry.get("lhs", "A0"), (M, K))
        w = reg(entry.get("weight", "W"), (K, N))
        y = mm(a, (M, K), w, (K, N))
        epi = entry.get("epilogue", [])
        if "acc_scale" in epi:
            s = Fraction(entry["acc_scale"]).limit_denominator(1 << 20)
            y = [[rnd(D.decode_float_exact(v, BF16) * s) for v in row] for row in y]
        if "relu" in epi:
            y = [[v if D.decode_float(v, BF16) > 0 else 0 for v in row] for row in y]
        outputs[entry.get("out", "Y0")] = floats(y)
    elif op == "movement":
        M = entry.get("M", entry.get("M_tiles", 1) * dim)
        N = entry.get("N", entry.get("N_tiles", 1) * dim)
        x = reg(entry.get("src", "X"), (M, N))
        outputs[entry.get("out", "Y0")] = floats([[rnd(D.decode_float_exact(x[i * N + j], FP8))
                                                   for j in range(N)] for i in range(M)])
    elif op == "resident_reuse":
        K = entry.get("K_tiles", 1) * dim
        N = entry.get("N_tiles", 1) * dim
        w = reg(entry["weight"], (K, N))
        for m in entry["matmuls"]:
            M = m.get("M_tiles", 1) * dim
            a = reg(m["lhs"], (M, K))
            outputs[m["out"]] = floats(mm(a, (M, K), w, (K, N)))
    elif op == "attention_qk":
        M = entry.get("M_tiles", 1) * dim
        Kd = entry.get("K_tiles", 1) * dim
        q = reg(entry.get("q", "Q"), (M, Kd))
        k = reg(entry.get("k", "K"), (M, Kd))
        kt = [0] * (M * Kd)
        for i in range(M):
            for j in range(Kd):
                kt[j * M + i] = k[i * Kd + j]
        outputs[entry.get("out", "Y0")] = floats(mm(q, (M, Kd), kt, (Kd, M)))
    else:
        raise ValueError(f"no float golden for op {op!r}")
    return outputs, prov


# ------------------------------------------------------------------------------------------------
# MX (microscaling block-scaled FP) golden engine — HARDWARE semantics via mlc's mx_ref, NOT specir
# (specir is the atlas fp8 refmodel; MX is a different datapath: 16-deep systolic per-column accumulate
# schedule + one E8M0 scale per 32-element K group). mx_ref is transcribed bit-exactly from the target's
# own reference (radiance-kernels lib/golden/{mx_fp_math.h,mx_golden.cpp}, mirroring the RTL).
# ------------------------------------------------------------------------------------------------
def _mx_ref():
    """Import mlc's ``validate/mx_ref.py`` BY FILE PATH (like the specir import) so we do NOT trigger
    ``mlc/validate/__init__.py`` (which carries concurrent work and heavy imports)."""
    root = os.environ.get("MERLIN_MLC_DIR", "/scratch2/agustin/mvp-lhwir/modeling")
    path = Path(root) / "mlc" / "validate" / "mx_ref.py"
    if not path.exists():
        raise FileNotFoundError(f"mx_ref not found at {path} (set MERLIN_MLC_DIR to the mlc modeling root)")
    spec = importlib.util.spec_from_file_location("merlin_mx_ref", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _salt(name: str, tensor: str) -> int:
    return sum((i + 1) * ord(c) for i, c in enumerate(f"{name}|{tensor}")) or 1


def _mx_value_codes(mx, fmt_token: str):
    """value(float) -> device code, DERIVED by decoding every code with mx_ref's own decoder (no baked
    table). fp8: 8-bit e4m3 code; fp4: 4-bit e2m1 nibble; fp6: 6-bit e3m2 code."""
    if fmt_token == "fp8_e4m3":
        rng, dec = range(256), mx.fp8_e4m3_decode
    elif fmt_token == "fp4_e2m1":
        rng, dec = range(16), mx.fp4_e2m1_decode
    elif fmt_token == "fp6_e3m2":
        rng, dec = range(64), mx.fp6_e3m2_decode
    else:
        raise ValueError(f"no MX code table for {fmt_token!r}")
    table: dict[float, int] = {}
    for c in rng:
        v = dec(c)
        if v == v and abs(v) != float("inf"):        # finite; keep the FIRST (lowest) code per value
            table.setdefault(float(v), c)
    return table


def _mx_golden(entry, binding):
    """MX matmul golden (bf16 output) + provenance, computed by mx_ref in hardware semantics. Operands are
    format-derived + rigor-gated; the E8M0 block-scale streams are rigor-gated too (a mis-indexed per-lane
    scale must change the output)."""
    from merlin.runtime.fp8_formats import canonical_float, e8m0_decode
    from merlin.targetgen import corpus_operands as CO
    mx = _mx_ref()
    tok = canonical_float(binding.operand_dtype)          # fp8_e4m3 / fp6_e3m2 / fp4_e2m1
    op = entry.get("op", "matmul")
    if op not in ("matmul", "linear"):
        raise ValueError(f"MX regime supports matmul/linear only (got op {op!r} in {entry['name']!r})")
    dim = binding.tile_dim
    M = entry.get("M", entry.get("M_tiles", 1) * dim)
    K = entry.get("K", entry.get("K_tiles", 1) * dim)
    N = entry.get("N", entry.get("N_tiles", 1) * dim)
    if tok == "fp8_e4m3":
        fmt, max_alpha, G = mx.FMT_FP8, None, 0
    elif tok == "fp4_e2m1":
        fmt, max_alpha, G = mx.FMT_FP4, None, 0
    elif tok == "fp6_e3m2":
        fmt, max_alpha, G = mx.FMT_FP6, 16, 5             # single 16-entry LUT (fp6 is LUT-indexed)
    else:
        raise ValueError(f"unsupported MX operand dtype {tok!r}")
    codes = _mx_value_codes(mx, tok)
    lhs, weight, out = entry.get("lhs", "A0"), entry.get("weight", "W"), entry.get("out", "Y0")

    def synth(name, shape):
        # MX operands are kept small (|v| <= 4): a wide E8M0 block scale over a long-K bf16 accumulate would
        # otherwise saturate to inf (a golden any broken kernel matches). See rand_fp8 in lib/golden.
        vals = CO.operand_values(shape, tok, _salt(entry["name"], name), max_alphabet=max_alpha, mag_cap=4.0)
        problems = CO.rigor_findings(vals, shape)
        if problems:
            raise AssertionError(f"non-rigorous MX operand {name}{shape}: {problems}")
        return np.array(vals, dtype=np.float64).reshape(shape)

    A = synth(lhs, (M, K))
    W = synth(weight, (K, N))

    def enc(v):
        c = codes.get(float(np.float32(v)))
        if c is None:                                     # exactly-representable palette -> exact hit expected
            raise AssertionError(f"MX value {v!r} not exactly representable in {tok}")
        return c

    A_codes = np.vectorize(enc)(A).astype(np.uint8)
    B_codes = np.vectorize(enc)(W).astype(np.uint8)
    GK = K // mx.GROUP
    SA = np.array(CO.e8m0_scale_codes((GK, M), _salt(entry["name"], "SA")), dtype=np.uint8)
    SB = np.array(CO.e8m0_scale_codes((GK, N), _salt(entry["name"], "SB")), dtype=np.uint8)
    for nm, sc in (("SA", SA), ("SB", SB)):
        prob = CO.scale_rigor_findings(sc.tolist())
        if prob:
            raise AssertionError(f"non-rigorous E8M0 scale stream {nm}{sc.shape}: {prob}")

    lutA = lutB = None
    if fmt == mx.FMT_FP8:
        Ab, Bb = A_codes, B_codes
    else:
        if fmt == mx.FMT_FP6:                             # nibbles index a shared 16-entry LUT of e3m2 codes
            lut = np.array(sorted({int(c) for c in A_codes.reshape(-1)} |
                                  {int(c) for c in B_codes.reshape(-1)}), dtype=np.uint8)
            assert lut.size <= 16, f"fp6 LUT overflow ({lut.size} > 16)"
            lut = np.pad(lut, (0, 16 - lut.size))[:16]
            idx = {int(v): i for i, v in enumerate(lut)}
            A_nib = np.vectorize(lambda c: idx[int(c)])(A_codes).astype(np.uint8)
            B_nib = np.vectorize(lambda c: idx[int(c)])(B_codes).astype(np.uint8)
            # mx_ref indexes the LUT as ``L[(row_or_col >> G) * 16 + nib]`` — ONE 16-entry block per
            # ``1<<G`` rows (A) / cols (B). Supply exactly that many blocks (a single global palette shared
            # by all groups is replicated: every block is identical, so ``(g)*16 + nib`` always resolves to
            # lut[nib]). Prior code shipped a lone block, so any fp6 capsule with M or N > 1<<G (e.g. N=64)
            # indexed past it and crashed.
            grp = 1 << G
            nblk_A = (A_codes.shape[0] + grp - 1) // grp      # blocks along A rows (M)
            nblk_B = (B_codes.shape[1] + grp - 1) // grp      # blocks along B cols (N)
            lutA = np.tile(lut.reshape(1, 16), (nblk_A, 1))
            lutB = np.tile(lut.reshape(1, 16), (nblk_B, 1))
        else:
            A_nib, B_nib = A_codes, B_codes               # fp4 nibble == code
        Ab = ((A_nib[1::2, :] << 4) | (A_nib[0::2, :] & 0xF)).astype(np.uint8)     # pack along M
        Bb = ((B_nib[:, 1::2] << 4) | (B_nib[:, 0::2] & 0xF)).astype(np.uint8)     # pack along N

    C = mx.mx_matmul(Ab, Bb, SA, SB, M, N, K, fmt=fmt, lutA=lutA, lutB=lutB, G=G)
    y = [[float(mx.bf16_to_f32(int(C[i, j]))) for j in range(N)] for i in range(M)]
    prov = {
        lhs: {"shape": [M, K], "decoded": A.reshape(-1).tolist()},
        weight: {"shape": [K, N], "decoded": W.reshape(-1).tolist()},
        "SA_e8m0_codes": SA.tolist(), "SB_e8m0_codes": SB.tolist(),
        "scale_example": {"SA[0][0]": int(SA[0, 0]), "as_scale": e8m0_decode(int(SA[0, 0]))},
    }
    return {out: y}, prov


def _simt_golden(entry, binding):
    """SIMT (CVFPU) golden in ordinary IEEE float — fp32 accumulate, format-rounded operands. Covers the
    matmul / attention / rmsnorm shapes; independent of any accelerator model (the SIMT cores do plain IEEE
    math). Operands are format-derived + rigor-gated."""
    from merlin.runtime.fp8_formats import canonical_float
    from merlin.targetgen import corpus_operands as CO
    tok = canonical_float(binding.operand_dtype)          # fp16 / bf16 / f32
    dim = binding.tile_dim
    op = entry.get("op", "matmul")

    def q(arr):
        a = np.asarray(arr, dtype=np.float64)
        if tok == "fp16":
            return a.astype(np.float16).astype(np.float64)
        if tok == "bf16":                                 # operands are exact bf16 already; identity round
            u = a.astype(np.float32).view(np.uint32)
            return ((u >> 16) << 16).view(np.float32).astype(np.float64)
        return a.astype(np.float32).astype(np.float64)

    def synth(name, shape):
        vals = CO.operand_values(shape, tok, _salt(entry["name"], name))
        problems = CO.rigor_findings(vals, shape)
        if problems:
            raise AssertionError(f"non-rigorous SIMT operand {name}{shape}: {problems}")
        return q(np.array(vals, dtype=np.float64).reshape(shape))

    def rnd_out(y):
        return [[float(np.float32(v)) for v in row] for row in np.asarray(y)]

    prov, outputs = {}, {}
    if op in ("matmul", "linear"):
        M = entry.get("M", entry.get("M_tiles", 1) * dim)
        K = entry.get("K", entry.get("K_tiles", 1) * dim)
        N = entry.get("N", entry.get("N_tiles", 1) * dim)
        A = synth(entry.get("lhs", "A0"), (M, K))
        W = synth(entry.get("weight", "W"), (K, N))
        y = (A.astype(np.float32) @ W.astype(np.float32)).astype(np.float64)
        epi = entry.get("epilogue", [])
        if "acc_scale" in epi:
            y = y * float(entry["acc_scale"])
        if "relu" in epi:
            y = np.maximum(y, 0.0)
        prov[entry.get("lhs", "A0")] = {"shape": [M, K], "decoded": A.reshape(-1).tolist()}
        prov[entry.get("weight", "W")] = {"shape": [K, N], "decoded": W.reshape(-1).tolist()}
        outputs[entry.get("out", "Y0")] = rnd_out(y)
    elif op == "attention_qk":
        M = entry.get("M_tiles", 1) * dim
        Kd = entry.get("K_tiles", 1) * dim
        Q = synth(entry.get("q", "Q"), (M, Kd))
        Kk = synth(entry.get("k", "K"), (M, Kd))
        y = (Q.astype(np.float32) @ Kk.astype(np.float32).T).astype(np.float64)
        prov[entry.get("q", "Q")] = {"shape": [M, Kd], "decoded": Q.reshape(-1).tolist()}
        prov[entry.get("k", "K")] = {"shape": [M, Kd], "decoded": Kk.reshape(-1).tolist()}
        outputs[entry.get("out", "Y0")] = rnd_out(y)
    elif op == "rmsnorm":
        M = entry.get("M", entry.get("M_tiles", 1) * dim)
        K = entry.get("K", entry.get("K_tiles", 1) * dim)
        eps = float(entry.get("eps", 1.0 / 65536.0))
        X = synth(entry.get("src", "X"), (M, K))
        gamma = synth(entry.get("gamma", "G"), (1, K))[0]
        y = np.empty((M, K), dtype=np.float64)
        for m in range(M):
            row = X[m].astype(np.float32)
            ss = np.float32(0.0)
            for k in range(K):
                ss = np.float32(ss + np.float32(row[k] * row[k]))
            mean = np.float32(ss / np.float32(K))
            rms = np.float32(1.0) / np.float32(np.sqrt(np.float32(mean + np.float32(eps))))
            for k in range(K):
                y[m, k] = float(np.float32(np.float32(row[k] * rms) * np.float32(gamma[k])))
        prov[entry.get("src", "X")] = {"shape": [M, K], "decoded": X.reshape(-1).tolist()}
        prov[entry.get("gamma", "G")] = {"shape": [1, K], "decoded": gamma.tolist()}
        outputs[entry.get("out", "Y0")] = rnd_out(y)
    else:
        raise ValueError(f"no SIMT golden for op {op!r}")
    return outputs, prov


def _entry_regime(entry, binding):
    """Route an entry to its numeric regime + return a per-entry binding (operand/accum overridden). ``int``
    (gemmini), ``specir`` (atlas fp8), ``mx`` (microscaling block-scaled FP), ``simt`` (IEEE fp16/bf16/f32).
    Routed purely by the entry's operand dtype token — no target name."""
    from merlin.runtime.fp8_formats import canonical_float
    tok = entry.get("operand_dtype") or binding.operand_dtype
    if tok in ("mxfp4", "mxfp6", "mxfp8"):
        regime, acc = "mx", "bf16"
    else:
        try:
            canon = canonical_float(tok)
        except KeyError:
            canon = None
        if canon in ("fp8_e4m3", "fp8_e5m2"):
            regime, acc = "specir", binding.accum_dtype
        elif canon in ("fp16", "bf16", "f32"):
            regime, acc = "simt", "f32"
        else:
            regime, acc = "int", binding.accum_dtype
    eb = dataclasses.replace(
        binding, operand_dtype=tok, accum_dtype=acc, integer=(regime == "int"),
        compare=("exact_int" if regime == "int" else "tolerance_float"))
    return regime, eb


# ------------------------------------------------------------------------------------------------
def _write_capsule(entry, binding, out_root):
    regime, eb = _entry_regime(entry, binding)
    # Whole-model capsule: a small representative network lowered end-to-end via model2MLIR, graded vs its
    # host torch-eager output, GATED so it runs only after the op suite proves itself. Additive: skipped
    # (loudly) when the m2m venv is absent.
    if entry.get("kind") == "model" or entry.get("op") == "model":
        from merlin.targetgen import capsule_source as CSRC
        src = CSRC.PytorchRefSource()
        if not src.available():
            print(f"  [skip] {entry['name']}: model capsule needs the m2m venv (set MERLIN_M2M_PYTHON)")
            return None
        return CSRC.write_model_capsule(entry, eb, out_root, source=src)
    # PREFERRED source: a capsule defined in PyTorch (frontend-faithful), lowered to linalg via model2MLIR
    # with a host torch-eager golden. Opt in per entry (``source: pytorch``). Restricted to the float
    # regime: a host-eager float reference is graded with tolerance, matching the merlin_iface float
    # interface; int/MX datapaths keep the direct-MLIR engines below (the endorsed fallback for the
    # dtypes torch/torchAO does not faithfully model, e.g. int8xint8 systolic or block-scaled MX).
    if entry.get("source") == "pytorch" or entry.get("pytorch_ref"):
        if regime != "simt":
            raise ValueError(f"pytorch source for capsule {entry['name']!r} needs a float dtype "
                             f"(got regime {regime!r} for {eb.operand_dtype!r}); author int/MX capsules "
                             f"via the direct-MLIR engine")
        from merlin.targetgen import capsule_source as CSRC
        src = CSRC.PytorchRefSource()
        if not src.available():
            # A pytorch capsule needs the m2m venv (torch) at generation time. It is additive: skip it
            # (loudly) rather than sink the whole target, so a checkout without the venv still regenerates
            # the direct-MLIR corpus. A capture that STARTS but fails (opaque/crash) still raises.
            print(f"  [skip] {entry['name']}: pytorch source needs the m2m venv (set MERLIN_M2M_PYTHON)")
            return None
        return CSRC.write_pytorch_capsule(entry, eb, out_root, source=src)
    # Spec source: a capsule whose PROGRAM + bit-exact golden come from the specir verification spec itself
    # (``spec_ref: '<gen>:op.<name>'``). Additive: a gen without a specir program emitter (or no specir) is
    # skipped loudly rather than sinking the target.
    if entry.get("source") == "spec" or entry.get("spec_ref"):
        from merlin.targetgen import capsule_source as CSRC
        src = CSRC.SpecRefSource()
        if not src.available():
            print(f"  [skip] {entry['name']}: spec source needs specir (set SPECIR_ROOT)")
            return None
        try:
            return CSRC.write_spec_capsule(entry, eb, out_root, source=src)
        except CSRC.SpecProgramUnavailable as e:
            print(f"  [skip] {entry['name']}: {e}")
            return None
    cap, mlir = CS.build(entry, eb)
    d = Path(out_root) / entry["cat"] / entry["name"]
    d.mkdir(parents=True, exist_ok=True)
    (d / "capsule.yaml").write_text(yaml.safe_dump(cap, sort_keys=False), encoding="utf-8")
    (d / "capsule.interface.mlir").write_text(mlir, encoding="utf-8")
    (d / "expected_instruction_coverage.yaml").write_text(
        yaml.safe_dump(cap["expected"], sort_keys=False), encoding="utf-8")
    if regime == "int":
        (d / "golden.yaml").write_text(yaml.safe_dump(
            {"golden_source": "merlin_tensor_int", "outputs": CG.golden({**cap, "__dir__": ""})},
            sort_keys=False), encoding="utf-8")
    elif regime == "specir":
        outputs, prov = _float_golden(entry, eb)
        (d / "golden.yaml").write_text(yaml.safe_dump({
            "golden_source": "specir_refmodel_fp8_bf16",
            "oracle_provenance": {
                "engine": "specir.oracle.dtypes + specir.oracle.refmodel.fp_reduce",
                "datapath": "acc <- round_bf16(acc + round_bf16(a*w)); k index_sequential; per_step; rne",
                "operand_dtype": eb.cap_dtype(eb.operand_dtype),
                "accum_dtype": eb.cap_dtype(eb.accum_dtype), "output_dtype": "bf16",
                "note": "INDEPENDENT of the target RTL (not self-oracle); specir refmodel is the reference.",
                "grade_policy": {"compare": eb.compare, "atol": eb.atol, "rtol": eb.rtol},
                "inputs": prov},
            "outputs": outputs}, sort_keys=False), encoding="utf-8")
    elif regime == "mx":
        outputs, prov = _mx_golden(entry, eb)
        (d / "golden.yaml").write_text(yaml.safe_dump({
            "golden_source": "mlc_mx_ref_hardware_semantics",
            "oracle_provenance": {
                "engine": "mlc.validate.mx_ref.mx_matmul (transcribed from radiance-kernels "
                          "lib/golden/{mx_fp_math.h,mx_golden.cpp}; mirrors the RTL, bit-exact vs spike)",
                "datapath": "16-deep systolic per-column acc schedule (ACC_E/ACC_M); one E8M0 scale per "
                            "32-elt K group; bf16 accumulate",
                "operand_dtype": eb.cap_dtype(eb.operand_dtype), "block_scale": "e8m0", "output_dtype": "bf16",
                "note": "NOT specir (specir is atlas fp8); MX is a distinct block-scaled datapath.",
                "grade_policy": {"compare": eb.compare, "atol": eb.atol, "rtol": eb.rtol},
                "inputs": prov},
            "outputs": outputs}, sort_keys=False), encoding="utf-8")
    else:                                                     # simt (IEEE fp16/bf16/f32)
        outputs, prov = _simt_golden(entry, eb)
        (d / "golden.yaml").write_text(yaml.safe_dump({
            "golden_source": "ieee_simt_f32_accumulate",
            "oracle_provenance": {
                "engine": "numpy IEEE float (CVFPU fp32 accumulate; format-rounded operands)",
                "operand_dtype": eb.cap_dtype(eb.operand_dtype), "accum_dtype": "f32", "output_dtype": "f32",
                "note": "SIMT cores do ordinary IEEE math; reference is independent of any accelerator model.",
                "grade_policy": {"compare": eb.compare, "atol": eb.atol, "rtol": eb.rtol},
                "inputs": prov},
            "outputs": outputs}, sort_keys=False), encoding="utf-8")
    return d


def _descriptor_for(target: str) -> Path:
    from merlin.common.paths import repo_root
    return (repo_root() / "merlin" / "experiments" / "capsule_bench" / "targets" / target
            / "target_experiment.yaml")


def _ensure_contract_on_path(descriptor: Path) -> None:
    """If the descriptor names an out-of-tree ``target_contract`` (e.g. radiance's contract lives under
    the ``radiance`` target package), prepend its package root to ``MERLIN_TARGET_PATH`` so the registry
    resolves the manifest. Read from the descriptor, so it stays target-agnostic."""
    from merlin.common.paths import repo_root
    raw = yaml.safe_load(descriptor.read_text())
    tc = (raw.get("hardware_spec") or {}).get("target_contract")
    if not tc:
        return
    pkg = (repo_root() / tc).resolve().parent.parent      # .../contracts/target_contract.yaml -> package root
    cur = os.environ.get("MERLIN_TARGET_PATH", "")
    if str(pkg) not in cur.split(os.pathsep):
        os.environ["MERLIN_TARGET_PATH"] = os.pathsep.join([str(pkg), cur]) if cur else str(pkg)


def generate_target(target: str) -> list[Path]:
    descriptor = _descriptor_for(target)
    _ensure_contract_on_path(descriptor)
    te = load_target_experiment(descriptor)
    profile = yaml.safe_load((PROFILES / f"{target}.yaml").read_text())
    binding = CS.derive_binding(te, profile.get("datapath", {}))
    out_root = Path(te.capsule_corpus).parent                 # target's corpus root, derived (no move)
    written = [w for w in (_write_capsule(e, binding, out_root) for e in profile["capsules"]) if w]
    return written


def build_comparison_manifest(targets: list[str]) -> dict:
    """Group capsules that exercise the SAME op across targets into comparison sets, so a shared op (e.g.
    rmsnorm/gelu/gemv_batched) can be compared across each target's own precision (MXFP8 on mx vs FP8-E4M3
    on atlas vs fp16 on radiance). Keyed by ``comparison_group`` when the profile declares one, else by op."""
    groups: dict[str, list[dict]] = {}
    for t in targets:
        prof = yaml.safe_load((PROFILES / f"{t}.yaml").read_text())
        for e in prof["capsules"]:
            if e.get("kind") == "model" or e.get("op") == "model":
                continue
            key = e.get("comparison_group") or e.get("op", "unknown")
            groups.setdefault(key, []).append(
                {"target": t, "name": e["name"],
                 "dtype": e.get("operand_dtype", prof.get("datapath", {}).get("operand_dtype", "")),
                 "label": e.get("label", "public")})
    # a comparison set is only interesting when >1 target covers the op
    cross = {k: v for k, v in sorted(groups.items()) if len({m["target"] for m in v}) > 1}
    return {"comparison_sets": cross,
            "note": "each set is one op exercised across multiple targets in each target's own precision; "
                    "same inner op name across targets makes target-vs-target numerics directly comparable"}


def write_comparison_manifest(targets: list[str]) -> Path:
    from merlin.common.paths import artifacts_dir
    manifest = build_comparison_manifest(targets)
    out = Path(artifacts_dir()) / "compare" / "capsule_comparison_manifest.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(manifest, sort_keys=True), encoding="utf-8")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Unified descriptor-driven capsule-corpus generator.")
    ap.add_argument("--target", default=None, help="one target (default: every target with a profile)")
    ap.add_argument("--comparison-manifest", action="store_true",
                    help="also emit the cross-target op-comparison manifest under out/artifacts/compare/")
    a = ap.parse_args(argv)
    targets = [a.target] if a.target else sorted(p.stem for p in PROFILES.glob("*.yaml"))
    for t in targets:
        written = generate_target(t)
        print(f"{t}: wrote {len(written)} capsules -> {written[0].parent.parent if written else '(none)'}")
    if a.comparison_manifest or not a.target:
        allt = sorted(p.stem for p in PROFILES.glob("*.yaml"))
        m = write_comparison_manifest(allt)
        print(f"comparison manifest: {m} ({len(build_comparison_manifest(allt)['comparison_sets'])} sets)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
