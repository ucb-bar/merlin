#!/usr/bin/env python3
"""Split an int8 (W8A8) residual into **policy** and **arithmetic**, one bundle per process.

A W8A8 grade against an INDEPENDENT reference (``golden_w8a8.independent.npy``, torchao
``int8_dyn_act_int8_weight`` in torch eager) is only an arithmetic verdict if both sides quantize
the SAME operations. They do not. torchao's scheme rewrites ``nn.Linear`` and nothing else, while
``quant_passes.apply_quant`` rewrites every recognized contraction — the attention ``bmm``s and,
on a spectral model, the DFT matmuls included — plus softmax / GELU / SiLU / rsqrt. The measured
residual is therefore *error + a more aggressive quantization policy, summed*, and reporting it as
error alone is a category mistake in the direction that makes us look wrong.

This tool measures the two separately by REPLAYING the same model through the same host runtime
with a progressively wider quant reach (``run_model(quant_passes=..., quant_select=...)``):

  * ``weight_only``          — ``int8_compute=False``: no activation quantization at all.
  * ``contraction_linear``   — the contraction pass restricted to Linear-descended contractions,
    i.e. exactly the operations torchao quantized. This is the apples-to-apples arm; what remains
    of the residual here is ARITHMETIC.
  * ``contraction``          — every contraction (adds attention bmm / spectral DFT / im2col conv
    matmul). The step from the previous arm is POLICY.
  * ``contraction_conv``     — plus the true conv pass.
  * ``all``                  — plus softmax / GELU / SiLU / rsqrt: the shipped datapath.

"Linear-descended" is not a fact this repo may assume: it is read off the capture's own provenance
(``prov.aten``), and the set of aten ops that means "an ``nn.Linear`` matmul" is a CLI parameter
(``--linear-aten``), not a literal in library code. The count is cross-checked against the
reference's own ``n_quantized`` from ``golden_w8a8.independent.npy.provenance.json``: if the number
of contractions we admit does not equal the number of weights torchao quantized, the arm is NOT
apples-to-apples and the run says so instead of quietly reporting a number.

Grading is ``zephyr_model._gate`` with both references under their correct tier keys
(``fp32`` = ``golden.npy``, ``w8a8`` = the independent reference), so ``tiers`` / ``tier_ok`` /
``ok`` mean what the gate says they mean. No threshold is touched.

ONE BUNDLE PER PROCESS is deliberate; see ``--help`` on ``bundle``.

Usage:
    w8a8_policy_ablation.py <bundle> --out <path.json>
    w8a8_policy_ablation.py <bundle> --classify-only
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from merlin.common.artifacts import recaptures_dir
from merlin.llvmlower.quant_passes import known
from merlin.runtime.backends.zephyr_model import _gate

#: aten ops that a ``torch.nn.Linear`` decomposes to under torch.export. Passed as a CLI default,
#: never consulted from library code — a different frontend spells this differently and the tool
#: must not pretend otherwise.
DEFAULT_LINEAR_ATEN = "aten.mm.default,aten.addmm.default,aten.linear.default"

INDEP_NAME = "golden_w8a8.independent.npy"


def _attr_str(op, key: str) -> str | None:
    a = op.attributes.get(key)
    if a is None:
        return None
    d = getattr(a, "data", None)
    return d if isinstance(d, str) else None


def _prepared_module(bundle: Path):
    """Parse + apply the same pre-quant normalization ``run_model`` does, so the ops offered to the
    quant passes here are the ops offered there."""
    from merlin.frontends.linalg_mlir import parse_mlir_file
    from merlin.llvmlower.passes_xdsl import collapse_overrank_matmul
    from merlin.llvmlower.torchao_affine import lower_torchao_affine_quant
    from merlin.runtime.dispatch_runtime import _propagate_quant_inner
    module = parse_mlir_file(bundle / "model.mlir")
    lower_torchao_affine_quant(module)
    collapse_overrank_matmul(module)
    _propagate_quant_inner(module)
    return module


def classify(bundle: Path) -> dict:
    """What each quant pass would rewrite, bucketed by the capture's own ``prov.aten``.

    Every pass is offered the PRISTINE module and the recording predicate refuses everything, so
    nothing is rewritten and the six counts are independent of each other's edits (which is what
    makes them comparable to the reference's per-construct weight count).
    """
    from merlin.llvmlower.quant_passes import registry
    module = _prepared_module(bundle)
    out: dict[str, dict[str, int]] = {}
    for name, qp in registry().items():
        seen: dict[str, int] = {}

        def rec(op, _seen=seen):
            key = _attr_str(op, "prov.aten") or "<untagged>"
            _seen[key] = _seen.get(key, 0) + 1
            return False                      # record only; never rewrite

        qp.fn(module, select=rec)
        out[name] = dict(sorted(seen.items(), key=lambda kv: -kv[1]))
    return out


def _references(bundle: Path) -> dict[str, np.ndarray]:
    refs: dict[str, np.ndarray] = {}
    g = bundle / "golden.npy"
    if g.is_file():
        refs["fp32"] = np.load(g).astype(np.float32).ravel()
    i = bundle / INDEP_NAME
    if i.is_file():
        refs["w8a8"] = np.load(i).astype(np.float32).ravel()
    return refs


def _flat(res: dict) -> np.ndarray:
    from merlin.runtime.dispatch_runtime import bf16_to_f32
    raw = res["output"]
    a = np.asarray(raw)
    return (bf16_to_f32(a) if a.dtype == np.uint16 else a.astype(np.float32)).ravel()


def aten_arm(name: str, atens: set[str]) -> tuple[str, dict]:
    """A contraction-only arm restricted to contractions carrying one of ``atens``.

    This is how a *within-contraction* attribution is made: the step from "Linear only" to "Linear
    + attention bmm" and the step to "+ spectral DFT" are different policy decisions with different
    risk, and a single "every contraction" arm cannot tell which one moved the number.
    """
    def sel(op, _a=atens) -> bool:
        return (_attr_str(op, "prov.aten") or "") in _a
    return (name, {"int8_compute": True, "quant_passes": ["contraction_int8"], "quant_select": sel})


def residual_structure(run: np.ndarray, indep: np.ndarray, fp32: np.ndarray) -> dict:
    """Is the residual against the independent reference DIFFUSE ROUNDING or a LOCALIZED DEFECT?

    Two implementations of the same quantized program disagree by accumulation order and tie
    rounding: the disagreement is spread over most outputs, is SMALLER than what quantizing at all
    already costs, and is uncorrelated with that cost. A broken op does the opposite — it
    concentrates in the elements that flow through it, and it tracks the quantization error because
    it IS a quantization step gone wrong. Reported as three numbers so the claim is checkable
    rather than asserted:

      * ``ratio_mean`` / ``ratio_max`` — our deviation over the reference's OWN deviation from
        fp32 (the floor that correct W8A8 already costs on this output). < 1 means we are closer
        to the reference than the reference is to full precision.
      * ``corr_with_floor`` — correlation of the two per-element deviation magnitudes. ~0 means
        independent noise; ~1 means we are amplifying the same error the quantizer makes.
      * ``diffuseness`` — fraction of elements deviating by more than 10% of the max deviation.
        High = spread everywhere (rounding); low = a handful of elements (a localized blow-up).
    """
    k = min(len(run), len(indep), len(fp32))
    d_run = np.abs(run[:k] - indep[:k]).astype(np.float64)
    d_floor = np.abs(indep[:k] - fp32[:k]).astype(np.float64)
    if d_floor.max() <= 0:
        return {"measurable": False}
    return {
        "measurable": True,
        "ratio_mean": float(d_run.mean() / max(1e-30, d_floor.mean())),
        "ratio_max": float(d_run.max() / d_floor.max()),
        "corr_with_floor": float(np.corrcoef(d_run, d_floor)[0, 1]) if k > 1 else 0.0,
        "diffuseness": float((d_run > 0.1 * d_run.max()).mean()) if d_run.max() > 0 else 0.0,
        "floor_mean_abs": float(d_floor.mean()),
        "run_mean_abs": float(d_run.mean()),
    }


def pass_arm(name: str, passes: list[str]) -> tuple[str, dict]:
    """An arm running an arbitrary pass subset over the full op set — how the step from "every
    contraction" to the shipped datapath divides across softmax / GELU / SiLU / rsqrt."""
    return (name, {"int8_compute": True, "quant_passes": passes})


def arms(linear_aten: set[str], extra: "list[tuple[str, set[str]]] | None" = None,
         pass_extra: "list[tuple[str, list[str]]] | None" = None
         ) -> list[tuple[str, dict]]:
    """(name, run_model kwargs) in widening-reach order."""
    def is_linear(op) -> bool:
        return (_attr_str(op, "prov.aten") or "") in linear_aten
    base = [
        ("weight_only", {"int8_compute": False}),
        ("contraction_linear", {"int8_compute": True, "quant_passes": ["contraction_int8"],
                                "quant_select": is_linear}),
        ("contraction", {"int8_compute": True, "quant_passes": ["contraction_int8"]}),
        ("contraction_conv", {"int8_compute": True,
                              "quant_passes": ["contraction_int8", "conv_int8"]}),
        ("all", {"int8_compute": True, "quant_passes": list(known())}),
    ]
    # extra arms slot in after the apples-to-apples one: they widen the contraction reach a
    # named group at a time, before the "every contraction" arm.
    return (base[:2] + [aten_arm(n, a) for n, a in (extra or [])] + base[2:]
            + [pass_arm(n, ps) for n, ps in (pass_extra or [])])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bundle", help="recapture bundle name (ONE per process: two model libraries "
                                   "both export _mlir_ciface_forward and the second dlopen wins)")
    ap.add_argument("--out", help="write the ablation JSON here")
    ap.add_argument("--classify-only", action="store_true")
    ap.add_argument("--linear-aten", default=DEFAULT_LINEAR_ATEN,
                    help="comma-separated prov.aten values that mean 'an nn.Linear matmul'")
    ap.add_argument("--arms", default="", help="comma-separated subset of the arm names")
    ap.add_argument("--aten-arm", action="append", default=[], metavar="NAME=ATEN,ATEN",
                    help="extra contraction-only arm over exactly these prov.aten values "
                         "(repeatable); attributes the residual to one op group at a time")
    ap.add_argument("--pass-arm", action="append", default=[], metavar="NAME=PASS,PASS",
                    help="extra arm over exactly this quant-pass subset (repeatable); attributes "
                         "the step from 'every contraction' to the shipped datapath")
    ap.add_argument("--workdir", default=None, help="scratch dir for compiled kernels")
    ap.add_argument("--dump-outputs", default=None, metavar="DIR",
                    help="save each arm's raw output as <DIR>/<arm>.npy, so the residual's "
                         "STRUCTURE (a few quantization-boundary flips vs a systematic bias) can "
                         "be examined without re-running the model")
    args = ap.parse_args()

    bundle = recaptures_dir() / args.bundle
    if not (bundle / "model.mlir").is_file():
        print(f"no such bundle: {bundle}", file=sys.stderr)
        return 2

    cls = classify(bundle)
    n_linear = sum(v for k, v in cls["contraction_int8"].items()
                   if k in set(args.linear_aten.split(",")))
    n_contraction = sum(cls["contraction_int8"].values())

    # Cross-check the apples-to-apples arm against the reference's OWN accounting.
    prov_path = bundle / (INDEP_NAME + ".provenance.json")
    prov = json.loads(prov_path.read_text()) if prov_path.is_file() else {}
    n_ref_quantized = prov.get("weights", {}).get("n_quantized")
    parity = (None if n_ref_quantized is None else bool(n_linear == n_ref_quantized))

    report: dict = {
        "bundle": args.bundle,
        "reference": INDEP_NAME,
        "reference_n_quantized_weights": n_ref_quantized,
        "linear_aten": sorted(set(args.linear_aten.split(","))),
        "n_contraction_sites": n_contraction,
        "n_linear_contraction_sites": n_linear,
        "apples_to_apples": parity,
        "classification": cls,
        "runs": {},
    }
    if args.classify_only:
        print(json.dumps(report, indent=2))
        return 0

    from merlin.runtime.dispatch_runtime import run_model
    refs = _references(bundle)
    if "w8a8" not in refs:
        print(f"{args.bundle}: no {INDEP_NAME} — nothing independent to grade against",
              file=sys.stderr)
        return 2

    want = set(a for a in args.arms.split(",") if a) or None
    root = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix="w8a8abl_"))
    root.mkdir(parents=True, exist_ok=True)
    extra = []
    for spec in args.aten_arm:
        nm, _, vals = spec.partition("=")
        extra.append((nm, set(v for v in vals.split(",") if v)))
    pass_extra = []
    for spec in args.pass_arm:
        nm, _, vals = spec.partition("=")
        pass_extra.append((nm, [v for v in vals.split(",") if v]))
    for name, kw in arms(set(args.linear_aten.split(",")), extra, pass_extra):
        if want is not None and name not in want:
            continue
        wd = root / name
        wd.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        res = run_model(str(bundle), wd, **kw)
        flat = _flat(res)
        if args.dump_outputs:
            d = Path(args.dump_outputs); d.mkdir(parents=True, exist_ok=True)
            np.save(d / f"{name}.npy", flat)
        g = _gate(flat, refs)
        entry = {k: v for k, v in g.items() if k not in ("golden",)}
        if "fp32" in refs:
            entry["residual_structure"] = residual_structure(flat, refs["w8a8"], refs["fp32"])
        entry["seconds"] = round(time.time() - t0, 1)
        entry["n_kernels"] = res.get("n_kernels")
        report["runs"][name] = entry
        if args.out:                       # write after EVERY arm: a long multi-arm run that
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)   # dies mid-way must not
            Path(args.out).write_text(json.dumps(report, indent=2))    # lose the arms it finished
        print(f"{args.bundle:28s} {name:20s} "
              f"w8a8_cos={g.get('w8a8_cos'):.6f} rel={g.get('w8a8_rel'):.5f} "
              f"max_rel={g.get('w8a8_max_rel'):.4g} "
              f"fp32_cos={g.get('fp32_cos'):.6f} tier_ok={g.get('tier_ok')} ok={g.get('ok')}",
              flush=True)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
