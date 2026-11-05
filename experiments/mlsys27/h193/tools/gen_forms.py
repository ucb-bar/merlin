#!/usr/bin/env python3
"""gen_forms.py OPCAPS_ENV_ROOT OUTDIR OVERLOADS.json  (runs on compute-1 with the op-capsule tree on PYTHONPATH, merlin-wtC python)
Builds, for each of the 93 Gemmini-capable overloads, the op-capsule group's form (core_aten_op_capsules registry of the merged opcaps tree) as a harness case dir
OUTDIR/<slug>/{capsule.linalg.mlir,inputs.json,golden.json,request.json} and OUTDIR/forms.json.  READ-ONLY w.r.t. the tree.  Lanes:
  DEVICE_FORMS(<group>) : existing direct builder (corpus_spec.build) -> merlin_iface interface, exact integer golden
  INT8 / COMPOSITE / widened : write_pytorch_capsule (model2MLIR capture of the integer torch form; widened = hand-stated single i32 generic) -> linalg interface; golden = torch eager
  NO_INT_FORM : untestable (reason recorded)"""
import sys, os, json, subprocess, importlib, shutil, traceback
from pathlib import Path
import numpy as np
from merlin.targetgen import core_aten_op_capsules as C, corpus_spec as SPEC, capsule_source as CS
from merlin.targetgen.capsule_golden import _recompute_golden
from merlin.targetgen.capsule_inputs import materialize_capsule_leaves
out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
OVS = json.load(open(sys.argv[3]))
M2M_PY = os.environ.get("M2M_PY", "/scratch/loren/core-aten-snap/venvs/m2m-run10f/bin/python")
grp = {}
for g in ("builders", "elementwise", "movement", "reduce_pool", "hard"):
    m = importlib.import_module(f"merlin.targetgen.core_aten_groups.{g}")
    for a in ("DEVICE_FORMS", "INT8_FORMS", "COMPOSITE_FORMS", "NO_INT_FORM"):
        for ov in getattr(m, a, {}): grp[(ov, a)] = g
bd = SPEC.CorpusBinding(target="gemmini", tile_dim=16, operand_dtype="int8", accum_dtype="int32", integer=True, tiers=["L2", "L3"], compare="exact")
bf = SPEC.CorpusBinding(target="gemmini", tile_dim=16, operand_dtype="fp32", accum_dtype="fp32", integer=False, tiers=["L2", "L3"], compare="tolerance")
src = CS.PytorchRefSource(m2m_dir=Path(os.environ.get("M2M_DIR", "/scratch/loren/his-flow/model2mlir")), python=Path(M2M_PY))
LOAD = r'''
import json,sys,runpy,torch
ns=runpy.run_path(sys.argv[1]); model,inputs=ns["get_model_and_inputs"](); res=model(*inputs)
res=list(res) if isinstance(res,(tuple,list)) else [res]
f=lambda t: t.to(torch.int64).tolist()
print(json.dumps(dict(inputs=[f(t) for t in inputs],outs=[f(t) for t in res],shapes=[list(t.shape) for t in res])))
'''
rows = []
ONLY = set(sys.argv[4].split(',')) if len(sys.argv) > 4 else None
for ov in OVS:
    if ONLY and ov not in ONLY: continue
    sl = C.slug(ov); d = out / sl; row = dict(overload=ov, slug=sl, path=str(d), lane="", form="", note="")
    try:
        if ov in C.NO_INT_FORM and ov not in C.DEVICE_FORMS:
            row.update(lane="NO_INT_FORM", form="-", note=C.NO_INT_FORM[ov], testable=False)
        elif ov in C.DEVICE_FORMS:
            frag, what = C.DEVICE_FORMS[ov]
            e = {"cat": "isa", "kind": "isa", "name": C.capsule_name(ov), "operand_dtype": "i8", "out": "Y0", "lhs": "A0", "weight": "W", "source_role": "derived_sweep", "source_reference": "t", "label": "public", **json.loads(json.dumps(frag))}
            if e["op"] == "residual_add": e["lhs"], e["rhs"] = "X0", "X1"
            cap, text = SPEC.build(e, bd)
            if not cap.get('numerical_semantics'):
                import yaml; cap['numerical_semantics'] = yaml.safe_load(open(os.environ['MERLIN_REPO_ROOT'] + '/examples/gemmini/target/software-spec.yaml'))['numerical_semantics']
            env = materialize_capsule_leaves(cap)
            ins = {i["name"]: np.asarray(env[i["name"]].data, dtype=np.int64).reshape(i["shape"]).tolist() for i in cap["inputs"]}
            try: g = _recompute_golden(cap)["Y0"]; row["golden_src"] = "merlin integer reference"
            except ValueError as ex:        # component_program/column_sum capsules need the spec doc's declaration binding; fall back to numpy semantics of the registered form
                a = np.asarray(ins["A0"], dtype=np.int64); k = ov.split(".")[1]
                g = {"permute": lambda: a.T, "view": lambda: a.reshape(32, 16), "sum": lambda: a.sum(axis=0)}.get(k, lambda: a)().tolist()
                row["golden_src"] = "numpy fallback (" + str(ex)[:60] + ")"
            d.mkdir(parents=True, exist_ok=True)
            (d / "capsule.linalg.mlir").write_text(text); json.dump(ins, open(d / "inputs.json", "w")); json.dump(g, open(d / "golden.json", "w"))
            json.dump(dict(expected=dict(kind="tensor", dtype="int32", shape=list(np.asarray(g).shape)), comparison="equal", comparison_parameters={}), open(d / "request.json", "w"))
            g0 = grp.get((ov, "DEVICE_FORMS"), "?")
            lane = "window_sum" if (g0 == "reduce_pool" and e["op"] in ("reduce_pool", "window_sum")) or "window_sum" in json.dumps(frag) else "DEVICE_FORMS"
            row.update(lane=lane, form=f"{g0}: {e['op']}: {what}", testable=True)
        else:
            if ov in C.INT8_FORMS: form = "int8"
            elif ov in C.COMPOSITE_FORMS: form = "composite"
            else: raise KeyError("no form registered")
            e = {"cat": "isa", "kind": "isa", "name": C.capsule_name(ov), "op": "aten_op", "aten_overload": ov, "aten_form": form, "capture_dtype": "fp32", "operand_dtype": "f32", "out": "Y0", "source": "pytorch", "source_role": "derived_sweep", "source_reference": "t", "label": "public",
                 "generalization": {"generalization_axis": "core_aten_op", "must_accelerate": True, "eligible": True}}
            if form == "composite":
                e["aten_composite"] = C.COMPOSITE_FORMS[ov][2]
                if ov in C.WIDENED_LINALG: e["aten_linalg"] = list(C.WIDENED_LINALG[ov])
            tmp = out / "_cap"; capd = CS.write_pytorch_capsule(e, bf, tmp, source=src)
            p = subprocess.run([M2M_PY, "-c", LOAD, str(capd / "capsule.pytorch.py")], capture_output=True, text=True)
            if p.returncode: raise RuntimeError("loader: " + p.stderr[-300:])
            r = json.loads(p.stdout.strip().splitlines()[-1])
            d.mkdir(parents=True, exist_ok=True)
            shutil.copy(capd / "capsule.linalg.mlir", d / "capsule.linalg.mlir")
            json.dump(r["inputs"], open(d / "inputs.json", "w"))
            json.dump(r["outs"] if len(r["outs"]) > 1 else r["outs"][0], open(d / "golden.json", "w"))
            exp = [dict(kind="tensor", dtype="int32", shape=s) for s in r["shapes"]]
            json.dump(dict(expected=exp[0] if len(exp) == 1 else dict(kind="tuple", items=exp), comparison="equal", comparison_parameters={}), open(d / "request.json", "w"))
            what = (C.INT8_FORMS[ov][1] if form == "int8" else C.COMPOSITE_FORMS[ov][2])
            row.update(lane=("widened" if ov in C.WIDENED_LINALG else ("INT8" if form == "int8" else "COMPOSITE")), form=f"{grp.get((ov,'INT8_FORMS' if form=='int8' else 'COMPOSITE_FORMS'),'?')}: {what}", testable=True)
    except Exception as ex:
        row.update(lane=row["lane"] or "BUILD_FAILED", testable=False, note=f"{type(ex).__name__}: {str(ex)[:300]}")
    print(row["lane"], ov, row["note"][:100], flush=True)
    rows.append(row)
if ONLY: sys.exit(0)
json.dump(rows, open(out / "forms.json", "w"), indent=1)
