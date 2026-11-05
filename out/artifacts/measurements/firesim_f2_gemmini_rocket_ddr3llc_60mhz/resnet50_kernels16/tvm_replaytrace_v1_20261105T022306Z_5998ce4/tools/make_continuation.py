#!/usr/bin/env python3
"""Build the continuation study: stock graph alternatives (fusion 0-3 x CSE) for each initial owner.

For every initial row the alternative plans are re-extracted from the frozen program (fresh compile of the
changed graph). A task whose structural workload key is already owned by the initial study reuses that
record and is NOT tuned again. Remaining distinct new tasks are tuned under the owner's reserved
allowance: the reserved trials (25 - initial) are split across that owner's distinct alternative plans.

usage: make_continuation.py <initial-study.json> <output-dir>
"""
import argparse, json, os, sys
from pathlib import Path

import tvm
from program_io import import_program
from tune_models import extract_inventory
from tvm.relax.backend.contrib.gemmini_search import WORKLOAD_ATTR
from tvm.relax.backend.contrib.gemmini_target import GemminiCapabilities

p = argparse.ArgumentParser()
p.add_argument("study", type=Path)
p.add_argument("output", type=Path)
a = p.parse_args()
study = json.loads(a.study.read_text())
root = a.study.resolve().parent
out = a.output.resolve()
out.mkdir(parents=True, exist_ok=False)
owned_keys = {k for r in study["campaigns"] for k in r["workload_keys"]}
default = (2, False)
plans = [(f, c) for c in (False, True) for f in (2, 3, 1, 0) if (f, c) != default]
rows, report = [], []
for row in study["campaigns"]:
    program = (root / row["program"]).resolve()
    fixture = (root / row["fixture"]).resolve()
    reserve = 25 - row["trials"]
    cands = []
    seen_signatures = set()
    for fusion, cse in plans:
        prepared, tasks, inventory = extract_inventory(
            import_program(program), GemminiCapabilities(), fusion_level=fusion, prepack_weights=False,
            bias_preload=False, eliminate_common_subexpr=cse)
        kinds = {r["task"]: r["kind"] for r in inventory}
        keys = []
        for t in tasks:
            if kinds[t.task_name] == "contraction":
                keys.append(str(t.dispatched[0]["main"].attrs[WORKLOAD_ATTR]))
        new = sorted(set(keys) - owned_keys)
        sig = tuple(sorted(keys))
        report.append(dict(id=row["id"], fusion=fusion, cse=cse, contractions=len(keys), new_keys=new,
                           same_contraction_set_as_earlier_plan=sig in seen_signatures))
        seen_signatures.add(sig)
        if new:
            cands.append((fusion, cse, new))
    cands = cands[:reserve] if reserve else []
    for i, (fusion, cse, new) in enumerate(cands):
        n = reserve // len(cands) + (1 if i < reserve % len(cands) else 0)
        if n < 1:
            continue
        for k in new:
            owned_keys.add(k)  # a later plan must not re-own a structurally identical task
        rows.append(dict(row, id="%s@fuse%d%s" % (row["id"], fusion, "-cse" if cse else ""),
                         program=os.path.relpath(program, out), fixture=os.path.relpath(fixture, out),
                         phase="continuation", trials=n, workload_keys=new, graph_fusion_level=fusion, graph_cse=cse))
    print(row["id"], "distinct alt plans with new tasks:", len(cands), flush=True)
cont = dict(study, campaigns=rows, coverage=[], continuation_of=str(a.study.resolve()))
(out / "study.json").write_text(json.dumps(cont, indent=2) + "\n")
(out / "plan-report.json").write_text(json.dumps(report, indent=1) + "\n")
print(json.dumps(dict(study=str(out / "study.json"), continuation_rows=len(rows), trials=sum(r["trials"] for r in rows))))
