#!/usr/bin/env python3
"""gen_expected.py FORMS_DIR  (compute-1, opcaps env python = merlin-wtC with /scratch/loren/opcaps/merlin/src on PYTHONPATH; READ-ONLY w.r.t. the tree)
Writes FORMS_DIR/<slug>/capsule_meta.json = the op-capsule group's capsule dict (its `expected` structural declaration: instruction_classes/modes/...), re-derived exactly as gen_forms.py did:
  DEVICE_FORMS / window_sum lanes : corpus_spec.build(...) capsule dict;  INT8/COMPOSITE/widened : FORMS_DIR/_cap/isa/<capsule_name>/capsule.yaml (the registry's written capsule)."""
import sys, json, yaml
from pathlib import Path
from merlin.targetgen import core_aten_op_capsules as C, corpus_spec as SPEC
F = Path(sys.argv[1]); rows = json.load(open(F / "forms.json"))
bd = SPEC.CorpusBinding(target="gemmini", tile_dim=16, operand_dtype="int8", accum_dtype="int32", integer=True, tiers=["L2", "L3"], compare="exact")
n = 0
for r in rows:
    if not r.get("testable"): continue
    ov, d = r["overload"], F / r["slug"]
    if r["lane"] in ("DEVICE_FORMS", "window_sum"):
        frag, _ = C.DEVICE_FORMS[ov]
        e = {"cat": "isa", "kind": "isa", "name": C.capsule_name(ov), "operand_dtype": "i8", "out": "Y0", "lhs": "A0", "weight": "W", "source_role": "derived_sweep", "source_reference": "t", "label": "public", **json.loads(json.dumps(frag))}
        if e["op"] == "residual_add": e["lhs"], e["rhs"] = "X0", "X1"
        cap, _t = SPEC.build(e, bd); src = "corpus_spec.build"
    else:
        cap = yaml.safe_load(open(F / "_cap" / "isa" / C.capsule_name(ov) / "capsule.yaml")); src = "registry capsule.yaml"
    json.dump(dict(source=src, capsule=json.loads(json.dumps(cap, default=str))), open(d / "capsule_meta.json", "w"), indent=1); n += 1
    print(r["lane"], ov, "expected=", json.dumps(cap.get("expected"), default=str)[:160])
print("wrote", n)
