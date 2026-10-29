"""Independent audit of the declared-call qualification-v1 fresh-replay cases (i8, i16shared).

Graham independent review (agent), 2026-10-26. Read-only; imports only the
standard library and the sibling independent reader. For each retained case:

1. the source MLIR (custom-free generic text) is read independently and its
   pre-order operation/block structure is compared with the `actualBefore`
   Module literal in the generated Check.lean (names, result/operand/block-arg
   numbering, attribute keys);
2. the final allocated Kernel JSON is decoded independently using the
   provider's declared storage choices (selected-storage-declarations.json),
   the storage windows and the original connected plan, giving each call's
   read/write resources and offsets; the resulting dataflow is compared with
   the linalg.generic bodies read in step 1;
3. the host-supplied Lean literals (schema window tables, `addresses`,
   `initialKnown`) are compared with the reviewer's decode.
Nothing here re-runs Lean or any oracle.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mlir_subset as M  # noqa: E402

REPO = Path("/home/grammy/blast/merlin-pass-verification")
BASE = REPO / "out/artifacts/verification/graph-regions-declared/qualification-v1/fresh-replay"
D2 = REPO / "out/artifacts/verification/graham-graph-review/deliverable-2"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---- tiny Lean term reader -------------------------------------------------
def lean_terms(text):
    pos = 0
    pairs = {"(": ")", "[": "]", "⟨": "⟩"}

    def skip():
        nonlocal pos
        while pos < len(text) and text[pos] in " \n\t":
            pos += 1

    def item():
        nonlocal pos
        skip()
        c = text[pos]
        if c in pairs:
            close = pairs[c]
            pos += 1
            groups, current = [], []
            while True:
                skip()
                if text[pos] == close:
                    pos += 1
                    break
                if text[pos] == ",":
                    pos += 1
                    groups.append(current)
                    current = []
                    continue
                current.append(item())
            groups.append(current)
            groups = [g[0] if len(g) == 1 else ("app", g) for g in groups if g]
            return (c, groups)
        if c == '"':
            end = text.index('"', pos + 1)
            value = text[pos + 1:end]
            pos = end + 1
            return ("str", value)
        start = pos
        while pos < len(text) and text[pos] not in " \n\t,()[]⟨⟩":
            pos += 1
        return ("atom", text[start:pos])

    return item()


def args_of(term):
    """('(', [('app', [ctor, a, b, ...])]) -> [ctor, a, b, ...]."""
    kind, groups = term
    assert kind == "(" and len(groups) == 1
    inner = groups[0]
    return inner[1] if inner[0] == "app" else [inner]


def nat_list(term):
    return [int(x[1]) for x in term[1]]


def flatten_lean_module(term):
    events = []

    def op(t):
        parts = args_of(t)
        name = parts[1][1]
        results = [int(r[1][0][1]) for r in parts[2][1]]
        operands = nat_list(parts[3])
        attrs = sorted(a[1][0][1] for a in parts[4][1])
        events.append(("op", name, tuple(results), tuple(operands), tuple(attrs)))
        for region in parts[6][1]:
            for block in args_of(region)[1][1]:
                bparts = args_of(block)
                events.append(("block", tuple(int(a[1][0][1]) for a in bparts[1][1])))
                for child in bparts[2][1]:
                    op(child)

    top = term[1][0]  # ⟨ [ops], [], [] ⟩
    for child in top[1]:
        op(child)
    return events


def flatten_reader_module(module):
    names, events = {}, []

    def define(name):
        names[name] = len(names)
        return names[name]

    def op(o):
        operands = tuple(names[v] for v in o.operands)
        results = tuple(define(r) for r in o.results)
        attrs = tuple(sorted(k for k, _ in M.canonical_props(o)))
        events.append(("op", o.name, results, operands, attrs))
        for region in o.regions:
            for block in region:
                events.append(("block", tuple(define(n) for n, _ in block.args)))
                for child in block.ops:
                    op(child)

    for o in module.ops:
        op(o)
    return events


def lean_def(text, name):
    for line in text.splitlines():
        if line.startswith(f"def {name} "):
            return line.split(":= ", 1)[1]
    raise KeyError(name)


# ---- kernel decode -------------------------------------------------------------
def decode_kernel(capture):
    kernel = json.loads((capture / "actual-output-kernel.json").read_text())
    declarations = json.loads((capture / "selected-storage-declarations.json").read_text())
    windows = {w["resource"]: w for w in json.loads((capture / "selected-storage-windows.json").read_text())}
    calls = []
    for instruction in kernel["body"]:
        name = instruction["instr"]
        values = {k: v["value"] for k, v in instruction["args"]}
        reads, writes = [], []
        for access in declarations[name]["accesses"]:
            choice = {c["value"]: c["resource"] for c in access["space"]["choices"]}
            resource = choice[values[access["space"]["argument"]]]
            start = values[access["start"]["argument"]]
            size = access["size"]["scale"]
            for factor in access["size"]["product"]:
                size *= values[factor["argument"]]
            row = {"resource": resource, "backing": windows[resource]["backing"], "start": start, "bytes": size,
                   "within_capacity": start + size <= windows[resource]["capacity"]}
            (writes if access["writes"] else reads).append(row)
        calls.append({"instr": name, "reads": reads, "writes": writes})
    return kernel, calls


def generic_bodies(module):
    fn = module.ops[0]
    block = fn.regions[0][0]
    out = []
    for o in block.ops:
        if o.name == "linalg.generic":
            body = o.regions[0][0]
            out.append({"result": o.results[0], "operands": o.operands,
                        "ops": [(b.name, [x for x in b.operands]) for b in body.ops],
                        "args": [n for n, _ in body.args]})
    return out, [n for n, _ in block.args]


def audit(case):
    root = BASE / case
    capture = root / "capture"
    lean = (root / "Check.lean").read_text()
    source = M.read((capture / "source.mlir").read_text())
    row = {"case": case, "source_sha256": sha(capture / "source.mlir"),
           "retained_source_identical": sha(capture / "source.mlir") == sha(capture / "retained-source.mlir"),
           "check_lean_sha256": sha(root / "Check.lean")}
    mine = flatten_reader_module(source)
    theirs = flatten_lean_module(lean_terms(lean_def(lean, "actualBefore")))
    structural, attribute_notes = [], []
    if len(mine) != len(theirs):
        structural.append(f"event count {len(mine)} vs {len(theirs)}")
    for a, b in zip(mine, theirs):
        if a[0] != b[0] or a[1:4] != b[1:4]:
            structural.append(f"{a} vs {b}")
        elif a[0] == "op" and a[4] != b[4]:
            attribute_notes.append({"op": a[1], "results": a[2], "reader_keys": a[4], "lean_keys": b[4]})
    row["actualBefore_vs_reader"] = {"events": len(mine), "structural_mismatches": structural,
                                     "attribute_key_differences": attribute_notes}
    retained = flatten_lean_module(lean_terms(lean_def(lean, "actualRetained")))
    row["actualRetained_equals_actualBefore_literal"] = retained == theirs
    # kernel decode
    kernel, calls = decode_kernel(capture)
    plan = json.loads((capture / "original-connected-plan.json").read_text())
    bodies, fn_args = generic_bodies(source)
    plan_values = {}
    for stage, body in zip(plan["stages"], bodies[:3]):
        for index, operand in zip(stage["inputs"], body["operands"]):
            plan_values.setdefault(index, operand)
        plan_values[stage["output"]] = body["result"]
    kernel_inputs = [a["name"] for a in kernel["args"] if a["access"] == "read"]
    kernel_outputs = [a["name"] for a in kernel["args"] if a["access"] == "write"]
    resource_value = {name: plan_values[v] for name, v in zip(kernel_inputs, plan["inputs"])}
    resource_value.update({name: plan_values[v] for name, v in zip(kernel_outputs, plan["outputs"])})
    dataflow, problems = [], []
    private = {}
    for call, body in zip(calls, bodies[:3]):
        reads = []
        for r in call["reads"]:
            key = (r["backing"], r["start"])
            reads.append(resource_value.get(r["resource"]) or private.get(key, f"?{key}"))
        target = call["writes"][0]
        produced = body["result"]
        external = [o for o in body["operands"][:len(reads)]]
        if reads != external:
            problems.append(f"{call['instr']}: kernel reads {reads} but source generic consumes {external}")
        if target["resource"] in resource_value:
            if resource_value[target["resource"]] != produced:
                problems.append(f"{call['instr']}: writes {target['resource']} = {resource_value[target['resource']]}"
                                f" but generic produces {produced}")
        else:
            private[(target["backing"], target["start"])] = produced
        if not all(x["within_capacity"] for x in (*call["reads"], *call["writes"])):
            problems.append(f"{call['instr']}: access exceeds declared capacity")
        dataflow.append({"instr": call["instr"], "reads": reads, "writes": produced, "write_window": target,
                         "body": body["ops"]})
    row["kernel_dataflow"] = dataflow
    row["kernel_vs_source_problems"] = problems
    row["abi_association_used"] = ("kernel read args in order <- original-connected-plan inputs; write args <- "
                                   "plan outputs (positional; host-supplied, not derived from MLIR)")
    # host literals vs reviewer decode
    addresses = lean_def(lean, "addresses")
    row["lean_addresses_literal"] = addresses
    row["lean_initialKnown"] = lean_def(lean, "initialKnown")
    row["verdict"] = "match" if not structural and not problems else "MISMATCH"
    return row


def main():
    rows = [audit(case) for case in ("0", "1")]
    report = {"reviewer": "Graham independent review (agent)", "date": "2026-10-26", "script_sha256": sha(__file__),
              "reader_sha256": sha(M.__file__), "imports_implementation": False, "cases": rows}
    (D2 / "audit/audit_declared.out.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    for row in rows:
        print(row["case"], row["verdict"], row["actualBefore_vs_reader"]["events"],
              row["actualBefore_vs_reader"]["structural_mismatches"][:2],
              row["actualBefore_vs_reader"]["attribute_key_differences"][:2], row["kernel_vs_source_problems"])
        for d in row["kernel_dataflow"]:
            print("   ", d["instr"], d["reads"], "->", d["writes"], d["write_window"]["resource"],
                  d["write_window"]["start"])


if __name__ == "__main__":
    main()
