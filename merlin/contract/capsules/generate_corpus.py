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
import os
import sys
from fractions import Fraction
from pathlib import Path

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
    return [D.encode_float(v, d_fp8) for v in vals], vals


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
def _write_capsule(entry, binding, out_root):
    cap, mlir = CS.build(entry, binding)
    d = Path(out_root) / entry["cat"] / entry["name"]
    d.mkdir(parents=True, exist_ok=True)
    (d / "capsule.yaml").write_text(yaml.safe_dump(cap, sort_keys=False), encoding="utf-8")
    (d / "capsule.interface.mlir").write_text(mlir, encoding="utf-8")
    (d / "expected_instruction_coverage.yaml").write_text(
        yaml.safe_dump(cap["expected"], sort_keys=False), encoding="utf-8")
    if binding.integer:
        (d / "golden.yaml").write_text(yaml.safe_dump(
            {"golden_source": "merlin_tensor_int", "outputs": CG.golden({**cap, "__dir__": ""})},
            sort_keys=False), encoding="utf-8")
    else:
        outputs, prov = _float_golden(entry, binding)
        (d / "golden.yaml").write_text(yaml.safe_dump({
            "golden_source": "specir_refmodel_fp8_bf16",
            "oracle_provenance": {
                "engine": "specir.oracle.dtypes + specir.oracle.refmodel.fp_reduce",
                "datapath": "acc <- round_bf16(acc + round_bf16(a*w)); k index_sequential; per_step; rne",
                "operand_dtype": binding.cap_dtype(binding.operand_dtype),
                "accum_dtype": binding.cap_dtype(binding.accum_dtype), "output_dtype": "bf16",
                "note": "INDEPENDENT of the target RTL (not self-oracle); specir refmodel is the reference.",
                "grade_policy": {"compare": binding.compare, "atol": binding.atol, "rtol": binding.rtol},
                "inputs": prov},
            "outputs": outputs}, sort_keys=False), encoding="utf-8")
    return d


def _descriptor_for(target: str) -> Path:
    from merlin.common.paths import repo_root
    return (repo_root() / "merlin" / "experiments" / "capsule_bench" / "targets" / target
            / "target_experiment.yaml")


def generate_target(target: str) -> list[Path]:
    te = load_target_experiment(_descriptor_for(target))
    profile = yaml.safe_load((PROFILES / f"{target}.yaml").read_text())
    binding = CS.derive_binding(te, profile.get("datapath", {}))
    out_root = Path(te.capsule_corpus).parent                 # target's corpus root, derived (no move)
    written = [_write_capsule(e, binding, out_root) for e in profile["capsules"]]
    return written


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Unified descriptor-driven capsule-corpus generator.")
    ap.add_argument("--target", default=None, help="one target (default: every target with a profile)")
    a = ap.parse_args(argv)
    targets = [a.target] if a.target else sorted(p.stem for p in PROFILES.glob("*.yaml"))
    for t in targets:
        written = generate_target(t)
        print(f"{t}: wrote {len(written)} capsules -> {written[0].parent.parent if written else '(none)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
