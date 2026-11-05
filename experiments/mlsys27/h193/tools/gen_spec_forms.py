#!/usr/bin/env python3
"""gen_spec_forms.py MERLIN_ROOT : spec-declared forms for the ten DEVICE57 ops that the 193 harness used to grade on a
capture spelling (INT8 / COMPOSITE / NO_INT_FORM lanes).

Operator decision (Loren, 2026-11-04): grade these ten on their SPEC-DECLARED forms, the way the other 47 ops
(DEVICE_FORMS / window_sum / widened lanes) are graded.  Each case is built exactly like gen_forms.py's DEVICE_FORMS
lane: Merlin's own corpus builder (corpus_spec.build) on an entry whose fields come from the spec form table
(corpus_form_table.FORM_TABLE[(family, form)].make at the spec's aligned cell: M = tile, K = 2*tile, N = tile, tile 16),
inputs = the builder's canonical leaves (capsule_inputs.materialize_capsule_leaves), reference = Merlin's exact integer
golden (capsule_golden._recompute_golden), compared EXACTLY (drive193.sh policy `exact`).  Component programs get the
spec's numerical_semantics and the Phase 0 writer's integer bound (writer._integer_reference_bound) exactly as Phase 0
writes them.  capsule_meta.json = the built capsule (its `expected` drives the trace check, as gen_expected.py does for
DEVICE_FORMS).

div.Scalar is NOT built here: its spec form is its spec_int8 form (tools/gen_spec_int8_form.py).
Run with the frozen Merlin branch (mlsys27/test-generation-coverage) on PYTHONPATH (src + packages/*/src).
VERIFICATION INSTRUMENT: never an author input."""
import os, json, sys
from pathlib import Path

import numpy as np
import yaml

H = Path(os.environ.get("H193_H", os.environ.get("H", "/scratch2/loren/backend-verify-author")))
ROOT = Path(sys.argv[1])
from merlin.targetgen import corpus_spec as SPEC  # noqa: E402
from merlin.targetgen.capsule_golden import _recompute_golden  # noqa: E402
from merlin.targetgen.capsule_inputs import materialize_capsule_leaves  # noqa: E402
from merlin.targetgen.corpus_form_table import FORM_TABLE, entry_fields  # noqa: E402

TILE = 16
ALIGNED = {"M": TILE, "K": 2 * TILE, "N": TILE}  # corpus_shapes.extents_for("aligned") at tile 16
# overload -> (spec row, family, form)
OPS = {
    "aten.cumsum.default": ("reduction_prefix_sum", "reduction", "cumsum"),
    "aten.diagonal.default": ("movement", "movement", "diagonal"),
    "aten.select.int": ("movement", "movement", "select"),
    "aten.split_with_sizes.default": ("movement", "movement", "split_sizes"),
    "aten.convolution_backward.default": ("contraction (convolution_backward forms)", "contraction", "convolution_backward_spatial"),
    "aten.upsample_nearest2d.vec": ("movement", "movement", "upsample_nearest"),
    "aten.full.default": ("movement", "movement", "fill"),
    "aten.full_like.default": ("movement", "movement", "fill"),
    "aten.scalar_tensor.default": ("movement", "movement", "fill_scalar"),
}
spec = yaml.safe_load(open(ROOT / "examples/gemmini/target/software-spec.yaml"))
bd = SPEC.CorpusBinding(target="gemmini", tile_dim=TILE, operand_dtype="int8", accum_dtype="int32", integer=True,
                        tiers=["L2", "L3"], compare="exact")
rows = []
for ov, (row, family, form) in OPS.items():
    slug = "spec_" + ov.removeprefix("aten.").replace(".", "_")
    fe = FORM_TABLE[(family, form)]
    fields = entry_fields(family, form, ALIGNED["M"], ALIGNED["N"])
    e = {"cat": "isa", "kind": "isa", "name": "SY_atenop_" + slug, "op": fe.op, "operand_dtype": "i8", "out": "Y0",
         "lhs": "A0", "weight": "W", "source_role": "derived_sweep", "source_reference": "t", "label": "public",
         **ALIGNED, **json.loads(json.dumps(fields))}
    if fe.op in ("prefix_sum", "conv_backward"):  # component programs carry the selected numeric policy (Phase 0 writer)
        e["numerical_semantics"] = spec["numerical_semantics"]
    cap, text = SPEC.build(e, bd)
    if cap["operation"]["op"] == "component_program":
        from merlin_experiments.phase0.writer import _integer_reference_bound
        cap["numerical_semantics"] = json.loads(json.dumps(e.get("numerical_semantics") or {}))
        cap["integer_partial_sum_bound"] = _integer_reference_bound(e, cap)
    if not cap.get("numerical_semantics"):
        cap["numerical_semantics"] = spec["numerical_semantics"]
    env = materialize_capsule_leaves(cap)
    ins = {i["name"]: np.asarray(env[i["name"]].data, dtype=np.int64).reshape(i["shape"]).tolist() for i in cap["inputs"]}
    gold = _recompute_golden(cap)
    names = list(gold)
    vals = [np.asarray(getattr(gold[k], "data", gold[k]), dtype=np.int64).reshape(np.asarray(getattr(gold[k], "shape", None) or np.asarray(gold[k]).shape)).tolist()
            if hasattr(gold[k], "data") else np.asarray(gold[k]).tolist() for k in names]
    shapes = [list(np.asarray(v).shape) for v in vals]
    d = H / "forms93" / slug; d.mkdir(parents=True, exist_ok=True)
    (d / "capsule.linalg.mlir").write_text(text)
    json.dump(ins, open(d / "inputs.json", "w"))
    if len(vals) == 1:
        golden, expected = vals[0], dict(kind="tensor", dtype="int32", shape=shapes[0])
    else:
        golden, expected = vals, dict(kind="tuple", items=[dict(kind="tensor", dtype="int32", shape=s, name=n) for n, s in zip(names, shapes)])
    json.dump(golden, open(d / "golden.json", "w"))
    json.dump(dict(expected=expected, comparison="equal", comparison_parameters={}), open(d / "request.json", "w"))
    json.dump(dict(source="corpus_spec.build (spec-declared form; tools/gen_spec_forms.py)", spec_row=row, family=family,
                   form=form, capsule=json.loads(json.dumps(cap, default=str))), open(d / "capsule_meta.json", "w"), indent=1)
    note = f"spec form: row {row}, {family}/{form} via corpus_form_table at the aligned cell (16x16); builder op {fe.op}"
    rows.append(dict(overload=ov, slug=slug, spec_row=row, family=family, form=form, builder_op=fe.op, note=note,
                     inputs={k: list(np.asarray(v).shape) for k, v in ins.items()}, outputs=dict(zip(names, shapes)),
                     expected=cap.get("expected")))
    print(ov, slug, fe.op, rows[-1]["inputs"], rows[-1]["outputs"])
json.dump(rows, open(H / "forms93" / "spec_forms.json", "w"), indent=1)
