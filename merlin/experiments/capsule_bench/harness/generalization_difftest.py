#!/usr/bin/env python3
"""Generalization differential-test — the Phase-D measurement loop that turns the capability-derived probe
generators (``capability_probes`` / ``semantic_fuzzer``) into an actual Acceleratable-Region-Recall NUMBER
over UNSEEN workloads (not the hand corpus).

For each probe drawn from the target's declared capability closure it MATERIALIZES a self-contained capsule
(interface MLIR via ``merlin_iface`` + a numpy CPU-reference ``golden.yaml``), grades it through the REAL
reference-backend + oracle path (:func:`capsule_runner.run_capsule`), and reports per-family / per-axis recall.
So "matmul supported" stops meaning "one 16x16 GEMM passed" and starts meaning "the compiler lowers the
declared closure across unseen shapes/dtypes". Target-general: everything is derived from the active
descriptor (``MERLIN_TARGET_EXPERIMENT`` → ``C.TARGET``), no target literal.

Scope (honest, logged-not-dropped): the families reachable through the reference emitter TODAY via a clean
single-op grammar — contraction, normalization, softmax, attention (scores). Non-MX dtypes only (fp32/fp16/
bf16/int8): the MX-dtype probes carry corpus-seeded operands with no from-random CPU reference, and the
4096-wide skinny shapes are impractical to simulate. Reduction/movement have no single-op emitter yet;
synchronization is a runtime no-op family — all excluded and recorded in ``skipped``.

Usage: MERLIN_TARGET_EXPERIMENT=<descriptor> generalization_difftest.py [--max-dim 64] [--families a,b]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402 — active target (descriptor-driven), bootstraps merlin/python
from merlin.targetgen import capability_probes as CP  # noqa: E402
from merlin.targetgen import coverage_report as CR  # noqa: E402
from merlin.targetgen import eligibility as EL  # noqa: E402
from merlin.targetgen.capsule_common import load_capsule  # noqa: E402
from merlin.targetgen.capsule_runner import qa_loop_adapters, run_capsule  # noqa: E402

TARGET = C.TARGET
PKG_ROOT = C.REPO / "out/artifacts/targets" / TARGET


def default_package() -> Path | None:
    """The target's reference backend, DERIVED — never a per-target directory name.

    ``reference_v0`` is what one target happens to call it; another calls it ``agent_spec_v1_mlir_oot``,
    so naming it here quietly restricted this measurement to a single target. Pick the package whose
    manifest marks it ``integrity_exempt`` (the reference backend is the one package allowed to import
    Merlin internals), else any package carrying a manifest. None when the target ships none.
    """
    exempt, any_pkg = [], []
    for m in sorted(PKG_ROOT.glob("*/manifest.yaml")):
        any_pkg.append(m.parent)
        try:
            if (yaml.safe_load(m.read_text(encoding="utf-8")) or {}).get("integrity_exempt"):
                exempt.append(m.parent)
        except Exception:  # noqa: BLE001 -- an unreadable manifest just isn't a candidate
            continue
    pool = exempt or any_pkg
    return pool[0] if pool else None
CONTRACT = C.REPO / "merlin" / "contract"
OUTDIR = C.RUNS / "genmatrix" / "caps"
RUNS = C.RUNS / "genmatrix"
GRADEABLE_DTYPE = {"fp32": "f32", "fp16": "f16", "bf16": "bf16", "int8": "i8"}
NORM_C = 16                    # feature width for row-op families (probes vary the ROW count)
MAX_DIM_DEFAULT = 64           # skip 4096-wide skinny probes (impractical to simulate)


def datapath_policy() -> dict:
    """The target's NUMERIC datapath, from its own corpus profile — not assumed to be float.

    Probes were materialized with `output_dtype: f32` and `compare: tolerance_float`, which is right for
    a SIMT float core and wrong for an integer systolic one: gemmini computes i8 x i8 -> i32 EXACTLY, and
    its shipped capsules declare `compare: exact_int`. Grading an integer datapath against a float golden
    mismatches every probe, including the canonical tile the same backend passes on a shipped capsule --
    a recall of 0 that measures the materializer, not the compiler.
    """
    prof = C.REPO / "merlin/contract/capsules/profiles" / f"{TARGET}.yaml"
    dp = {}
    if prof.is_file():
        dp = (yaml.safe_load(prof.read_text()) or {}).get("datapath") or {}
    compare = dp.get("compare", "tolerance_float")
    exact = compare in ("exact_int", "exact")
    return {"compare": compare, "exact": exact,
            "operand_dtype": "i8" if exact else None,      # float path keeps the probe's own dtype
            "acc_dtype": "i32" if exact else "f32"}


def _round_dtype(x: np.ndarray, dt: str) -> np.ndarray:
    if dt == "fp32":
        return x.astype(np.float32)
    if dt == "fp16":
        return x.astype(np.float16).astype(np.float32)
    if dt == "bf16":
        u = x.astype(np.float32).view(np.uint32)
        return ((u + 0x8000) & 0xFFFF0000).view(np.float32)
    if dt == "int8":
        return np.clip(np.round(x * 8), -128, 127).astype(np.int8).astype(np.float32)
    raise ValueError(dt)


def _iface_header() -> str:
    return (f'module attributes {{merlin_iface.version = "0.1", merlin_iface.target = "{TARGET}", '
            f'merlin_iface.abi_version = "0.1"}} {{\n')


def _write_capsule(probe, iface, *, inputs, out, op, op_attrs, ct) -> Path:
    pol = datapath_policy()
    cdir = OUTDIR / probe.name.replace(".", "_")
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "capsule.interface.mlir").write_text(iface)
    (cdir / "capsule.yaml").write_text(yaml.safe_dump({
        "name": probe.name.replace(".", "_"), "kind": "isa", "label": "public",
        "source_role": "handauthored_compiler_test",
        "source_reference": f"generalization probe {probe.name} axis={probe.axis}",
        "interface_mlir": "capsule.interface.mlir",
        "inputs": [{"name": nm, "role": role, "shape": sh, "dtype": ct} for nm, role, sh, _ in inputs],
        "operation": {"op": op, "attributes": op_attrs},
        "numeric_policy": ({"compare": pol["compare"], "dtype": pol["acc_dtype"]} if pol["exact"] else
                           {"compare": "tolerance_float", "dtype": "f32", "atol": 0.03125, "rtol": 0.015625}),
        "expected": {"instruction_classes": [], "modes": {}},
        "required_oracle_tiers": ["L0", "L1", "L2"], "vcs": "optional",
    }, sort_keys=False))
    if pol["exact"]:
        # Integer datapath: declare `merlin_tensor_int` and ship NO decoded operands, so the grader
        # materializes the leaves and recomputes the golden on its own exact-integer engine — the same
        # path every shipped integer capsule takes. A numpy float reference would introduce a second
        # arithmetic definition for a datapath that has exactly one.
        (cdir / "golden.yaml").write_text(yaml.safe_dump({
            "golden_source": "merlin_tensor_int",
            "oracle_provenance": {"engine": "merlin Tensor exact-integer recompute (grader-side)",
                                  "operand_dtype": ct, "accum_dtype": pol["acc_dtype"],
                                  "output_dtype": pol["acc_dtype"],
                                  "grade_policy": {"compare": pol["compare"]}},
        }, sort_keys=False))
    else:
        (cdir / "golden.yaml").write_text(yaml.safe_dump({
            "golden_source": "ieee_simt_f32_accumulate",
            "oracle_provenance": {"engine": f"numpy IEEE float {op} (independent CPU reference)",
                                  "operand_dtype": ct, "accum_dtype": "f32", "output_dtype": "f32",
                                  "grade_policy": {"compare": "tolerance_float", "atol": 0.03125,
                                                   "rtol": 0.015625},
                                  "inputs": {nm: {"shape": sh, "decoded": arr.reshape(-1).tolist()}
                                             for nm, _, sh, arr in inputs}},
            "outputs": {"Y0": out.reshape(-1).tolist()},
        }, sort_keys=False))
    return cdir


def materialize_contraction(probe, seed):
    d = probe.descriptor
    pol = datapath_policy()
    # An integer datapath has one operand dtype and one accumulator width, both from the profile; a float
    # one keeps the probe's declared dtype. Emitting f32 at an i8 x i8 -> i32 core is what made every
    # probe mismatch, canonical tile included.
    ct = pol["operand_dtype"] or GRADEABLE_DTYPE.get(d.in_dtype)
    if ct is None:
        return None
    acc = pol["acc_dtype"]
    m, k, n = int(d.m), int(d.k), int(d.n)
    rng = np.random.default_rng(seed)
    if pol["exact"]:
        A = W = None                       # grader materializes the leaves and recomputes the golden
        Y = None
    else:
        A = _round_dtype(rng.standard_normal((m, k)), d.in_dtype)
        W = _round_dtype(rng.standard_normal((k, n)), d.in_dtype)
        Y = (A.astype(np.float32) @ W.astype(np.float32)).astype(np.float32)
    iface = (_iface_header()
             + f'  %W = merlin_iface.tensor {{name = "W", role = "weight"}} : tensor<{k}x{n}x{ct}>\n'
             + f'  %A0 = merlin_iface.tensor {{name = "A0", role = "input"}} : tensor<{m}x{k}x{ct}>\n'
             + f'  %W_res = merlin_iface.resident_pack %W {{layout = "packed_rhs"}} : (tensor<{k}x{n}x{ct}>) -> !merlin_iface.resident\n'
             + f'  %acc0 = merlin_iface.matmul %A0, %W_res : (tensor<{m}x{k}x{ct}>, !merlin_iface.resident) -> !merlin_iface.acc<{acc}>\n'
             + f'  %Y0 = merlin_iface.commit %acc0 {{name = "Y0", epilogue = [], output_dtype = "{acc}"}} : (!merlin_iface.acc<{acc}>) -> tensor<{m}x{n}x{acc}>\n'
             + '  merlin_iface.evict %W_res : (!merlin_iface.resident) -> ()\n}\n')
    return _write_capsule(probe, iface, inputs=[("W", "weight", [k, n], W), ("A0", "input", [m, k], A)],
                          out=Y, op="matmul", ct=ct,
                          op_attrs={"lhs": "A0", "weight": "W", "out": "Y0", "epilogue": [],
                                    "output_dtype": acc})


def materialize_normalization(probe, seed):
    d = probe.descriptor
    ct = GRADEABLE_DTYPE.get(d.in_dtype)
    if ct is None:
        return None
    m, c = int(d.m), NORM_C
    rng = np.random.default_rng(seed)
    X = _round_dtype(rng.standard_normal((m, c)), d.in_dtype)
    G = _round_dtype(rng.standard_normal((1, c)), d.in_dtype)
    Xf, Gf = X.astype(np.float32), G.astype(np.float32)
    Y = (Xf / np.sqrt(np.mean(Xf * Xf, axis=1, keepdims=True) + 1e-5)) * Gf
    iface = (_iface_header()
             + f'  %X = merlin_iface.tensor {{name = "X", role = "input"}} : tensor<{m}x{c}x{ct}>\n'
             + f'  %G = merlin_iface.tensor {{name = "G", role = "weight"}} : tensor<1x{c}x{ct}>\n'
             + f'  %Y0 = merlin_iface.rmsnorm %X, %G {{name = "Y0", eps = 1.000000000e-05 : f64, output_dtype = "f32"}} : (tensor<{m}x{c}x{ct}>, tensor<1x{c}x{ct}>) -> tensor<{m}x{c}xf32>\n}}\n')
    return _write_capsule(probe, iface, inputs=[("X", "input", [m, c], X), ("G", "weight", [1, c], G)],
                          out=Y, op="rmsnorm", ct=ct,
                          op_attrs={"src": "X", "gamma": "G", "out": "Y0", "eps": 1e-5, "output_dtype": "f32"})


def materialize_softmax(probe, seed):
    d = probe.descriptor
    ct = GRADEABLE_DTYPE.get(d.in_dtype)
    if ct is None:
        return None
    m, c = int(d.m), NORM_C
    rng = np.random.default_rng(seed)
    X = _round_dtype(rng.standard_normal((m, c)), d.in_dtype)
    Xf = X.astype(np.float32)
    e = np.exp(Xf - Xf.max(axis=1, keepdims=True))
    Y = e / e.sum(axis=1, keepdims=True)
    iface = (_iface_header()
             + f'  %X = merlin_iface.tensor {{name = "X", role = "input"}} : tensor<{m}x{c}x{ct}>\n'
             + f'  %Y0 = merlin_iface.softmax %X {{name = "Y0", output_dtype = "f32"}} : (tensor<{m}x{c}x{ct}>) -> tensor<{m}x{c}xf32>\n}}\n')
    return _write_capsule(probe, iface, inputs=[("X", "input", [m, c], X)], out=Y, op="softmax", ct=ct,
                          op_attrs={"src": "X", "out": "Y0", "output_dtype": "f32"})


def materialize_attention(probe, seed):
    d = probe.descriptor
    ct = GRADEABLE_DTYPE.get(d.in_dtype)
    if ct is None:
        return None
    m, k, n = int(d.m), int(d.k), int(d.n)
    rng = np.random.default_rng(seed)
    Q = _round_dtype(rng.standard_normal((m, k)), d.in_dtype)
    K = _round_dtype(rng.standard_normal((n, k)), d.in_dtype)
    S = (Q.astype(np.float32) @ K.astype(np.float32).T).astype(np.float32)
    iface = (_iface_header()
             + f'  %Q = merlin_iface.tensor {{name = "Q", role = "input"}} : tensor<{m}x{k}x{ct}>\n'
             + f'  %K = merlin_iface.tensor {{name = "K", role = "input"}} : tensor<{n}x{k}x{ct}>\n'
             + f'  %Y0 = merlin_iface.attention_qk %Q, %K {{name = "Y0", output_dtype = "f32"}} : (tensor<{m}x{k}x{ct}>, tensor<{n}x{k}x{ct}>) -> tensor<{m}x{n}xf32>\n}}\n')
    return _write_capsule(probe, iface, inputs=[("Q", "input", [m, k], Q), ("K", "input", [n, k], K)],
                          out=S, op="attention_qk", ct=ct,
                          op_attrs={"q": "Q", "k": "K", "out": "Y0", "output_dtype": "f32"})


FAMILY_MAT = {"contraction": materialize_contraction, "normalization": materialize_normalization,
              "softmax": materialize_softmax, "attention": materialize_attention}


def grade(cdir: Path, adapters, pkg: Path) -> str:
    cap = load_capsule(str(cdir), contract=str(CONTRACT))
    res = run_capsule(cap, pkg, runs_root=RUNS, run_id=cap["name"], contract=str(CONTRACT),
                      target=TARGET, timeout=600, oracle_adapters=adapters)
    l2 = (res.get("tiers", {}) or {}).get("L2", {}).get("status")
    return "pass" if res.get("status") == "pass" or l2 == "pass" else "fail"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-dim", type=int, default=MAX_DIM_DEFAULT)
    ap.add_argument("--families", default=",".join(FAMILY_MAT))
    ap.add_argument("--package", default=None,
                    help="backend package to measure (default: the target's derived reference backend). "
                         "Point it at a run's frozen submission/ to measure THAT compiler's recall.")
    a = ap.parse_args(argv)
    pkg = Path(a.package) if a.package else default_package()
    if pkg is None or not (pkg / "manifest.yaml").is_file():
        print(f"no backend package to measure under {PKG_ROOT} (pass --package)", file=sys.stderr)
        return 2
    fams = set(a.families.split(","))
    adapters = qa_loop_adapters(TARGET)
    cap_map = EL.capability_map_for_target(TARGET)
    probes = [p for p in CP.synthesize(cap_map)
              if p.descriptor.resolved_family() in (fams & set(FAMILY_MAT))]
    rows, skipped = [], []
    for i, p in enumerate(probes):
        d = p.descriptor
        fam = d.resolved_family()
        if d.in_dtype not in GRADEABLE_DTYPE:
            skipped.append((p.name, "mx-dtype (seeded operands, no from-float CPU ref)"))
            continue
        dims = [int(v) for v in (d.m, d.k, d.n) if v is not None]
        if dims and max(dims) > a.max_dim:
            skipped.append((p.name, f"shape>{a.max_dim} (impractical to simulate)"))
            continue
        if int(getattr(d, "batch", 1) or 1) > 1 or getattr(d, "layout", None):
            skipped.append((p.name, "batch/layout axis (materializer is plain 2-D; not yet handled)"))
            continue
        cdir = FAMILY_MAT[fam](p, seed=1000 + i)
        try:
            verdict = grade(cdir, adapters, pkg)
        except Exception as e:  # noqa: BLE001
            verdict = f"error:{type(e).__name__}:{str(e)[:80]}"
        rows.append({"probe": p.name, "family": fam, "axis": p.axis, "verdict": verdict})
        print(json.dumps(rows[-1]), flush=True)

    def _recall(key):
        agg = {}
        for r in rows:
            b = agg.setdefault(r[key], [0, 0])
            b[1] += 1
            b[0] += int(r["verdict"] == "pass")
        return {k: f"{p}/{t}" for k, (p, t) in sorted(agg.items())}

    n_pass = sum(1 for r in rows if r["verdict"] == "pass")
    summary = {"target": TARGET, "package": str(pkg), "graded": len(rows), "overall": f"{n_pass}/{len(rows)}",
               "acceleratable_region_recall_sampled": (n_pass / len(rows)) if rows else None,
               "per_family_recall": _recall("family"), "per_axis_recall": _recall("axis"),
               "skipped": skipped}
    out = C.REPORTS / "generalization_arr.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print("SUMMARY", json.dumps(summary, indent=2))
    print("persisted →", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
