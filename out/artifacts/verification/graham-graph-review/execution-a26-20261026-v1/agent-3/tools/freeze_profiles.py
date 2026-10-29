"""Freeze source-only reproductions; never import or execute producer code."""

import hashlib
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from plan_core import closure, header, module_from_path, topo

ROOT = pathlib.Path.cwd().resolve()
REPORT = pathlib.Path(__file__).resolve().parent.parent
BUILD = ROOT / "out/build/verification-a26-20261026-v1/profiles"
STAGED = BUILD / "capsule-staged-v1/profiles"
INLINE = ROOT / "out/artifacts/verification/graph-pipeline-gate/development/unroll-inline-v1"


def pin(data):
    return {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def read_bound(path, reference):
    if not path.resolve().is_relative_to(ROOT) or path.is_symlink():
        raise ValueError("Source escaped checkout")
    raw = path.read_bytes()
    if pin(raw) != {"sha256": reference["sha256"], "bytes": reference.get("bytes", reference.get("size"))}:
        raise ValueError("Source pin changed: " + str(path))
    return raw


def write_new(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        f.write(raw)


def relative(path):
    return str(path.relative_to(ROOT))


profiles = []
for name in ("Q-schedule-successor-v1", "Q-unroll-tail-successor-v1",
             "W-caller-context-successor-v1", "W-entries-successor-v1"):
    folder = STAGED / name
    order = json.loads((folder / "BUILD_ORDER.json").read_bytes())
    sources = []
    for module in order["dependency_order"]:
        p = folder / "project" / (module.replace(".", "/") + ".lean")
        raw = p.read_bytes()
        sources.append({"module": module, "path": relative(p), **pin(raw), "header": header(raw)})
    cases = []
    for case_path in sorted((folder / "queries").iterdir()):
        metadata = json.loads((case_path / "roots.json").read_bytes())
        query = case_path / (metadata["module"].replace(".", "/") + ".lean")
        cases.append({"id": case_path.name, "module": metadata["module"],
                      "path": relative(query), **pin(query.read_bytes()), "roots": metadata,
                      "witness": None})
    profiles.append({"id": name, "project_root": relative(folder / "project"),
                     "dependency_order": order["dependency_order"], "DAG": order["DAG"],
                     "external_module_names_only": order["external_names_only"],
                     "sources": sources, "cases": cases})

source_inventory = json.loads((INLINE / "SOURCE-INVENTORY.json").read_bytes())
query_inventory = json.loads((INLINE / "QUERY-INVENTORY.json").read_bytes())
manifest = json.loads((INLINE / "MANIFEST.json").read_bytes())
manifest_by_path = {r["path"]: r for r in manifest["rows"]}
all_sources = {}
graph = {}
for group in ("owned", "dependencies"):
    for row in source_inventory[group]:
        module = module_from_path(row["path"])
        if module is None:
            continue
        raw = read_bound(INLINE / row["snapshot"], row)
        current = {"module": module, "original_snapshot": relative(INLINE / row["snapshot"]),
                   **pin(raw), "header": header(raw), "raw": raw}
        if module in all_sources and all_sources[module]["sha256"] != current["sha256"]:
            raise ValueError("Conflicting inline module identities")
        all_sources[module] = current
        graph[module] = [m for m in current["header"]["imports"] if m.startswith(("Merlin.", "MerlinVerify."))]

inline_root = BUILD / "inline-staged-v1"
inline_root.mkdir(exist_ok=False)
case_records = []
query_imports = set()
for case in query_inventory["cases"]:
    original_case = INLINE / query_inventory["authoritative_examples"] / case["case"]
    metadata = json.loads((original_case / "REQUEST.json").read_bytes())
    prepared = json.loads((original_case / "PREPARED.json").read_bytes())
    query = original_case / "Query.lean"
    raw = read_bound(query, {"sha256": case["query_sha256"], "bytes": query.stat().st_size})
    if metadata["query"].encode() != raw or prepared["query"].encode() != raw:
        raise ValueError("Inline query/request/preparation association differs")
    if metadata["roots"] != case["roots"] or metadata["final_root"] != query_inventory["final_root"]:
        raise ValueError("Inline exact root association differs")
    if metadata["support_roots"] + [metadata["final_root"]] != metadata["roots"]:
        raise ValueError("Inline root roster order differs")
    query_imports.update(m for m in header(raw)["imports"] if m.startswith(("Merlin.", "MerlinVerify.")))
    dest = inline_root / "queries" / case["case"] / "Query.lean"
    write_new(dest, raw)
    witness = original_case / "Witness.lean"
    witness_rel = witness.relative_to(INLINE).as_posix()
    witness_raw = read_bound(witness, manifest_by_path[witness_rel])
    query_imports.update(m for m in header(witness_raw)["imports"] if m.startswith(("Merlin.", "MerlinVerify.")))
    witness_dest = dest.with_name("Witness.lean")
    write_new(witness_dest, witness_raw)
    roots = {"final_root": metadata["final_root"], "support_roots": metadata["support_roots"],
             "domain": metadata["domain"], "relation": "all_defined_entries_equal_returns_errors_terminal_world",
             "assumptions": metadata["premises"], "query_pin": pin(raw),
             "compile": "UNEXECUTED", "audit": "UNEXECUTED", "qualification": "UNEXECUTED"}
    write_new(dest.with_name("roots.json"), (json.dumps(roots, indent=2) + "\n").encode())
    case_records.append({"id": case["case"], "module": "Query", "path": relative(dest), **pin(raw),
                         "roots": roots,
                         "witness": {"module": "Witness", "path": relative(witness_dest), **pin(witness_raw),
                                     "roots": case["witness_roots"]},
                         "original_case": relative(original_case),
                         "source_sha256": case["source_sha256"], "target_sha256": case["target_sha256"]})

needed = closure(query_imports, graph)
graph = {module: graph[module] for module in needed}
order = topo(graph)
source_records = []
for module in order:
    row = all_sources[module]
    dest = inline_root / "project" / (module.replace(".", "/") + ".lean")
    write_new(dest, row["raw"])
    source_records.append({k: v for k, v in row.items() if k != "raw"} | {"path": relative(dest)})
external_names = {m: sorted(set(s["header"]["imports"]) - needed | ({"Init"} if s["header"]["implicit_Init"] else set()))
                  for m in order for s in [all_sources[m]]}
profile = {"id": "Q-inline-complete-carry-v1", "project_root": relative(inline_root / "project"),
           "dependency_order": order, "DAG": graph, "external_module_names_only": external_names,
           "sources": source_records, "cases": case_records,
           "final_root": query_inventory["final_root"],
           "status": "SOURCE_PROFILE_FROZEN_NOT_COMPILED"}
write_new(inline_root / "BUILD_ORDER.json", (json.dumps({k: profile[k] for k in (
    "id", "dependency_order", "DAG", "external_module_names_only")}, indent=2) + "\n").encode())
profiles.insert(1, profile)
report = {"schema": "merlin.a26.frozen-reproduction-profiles.v1",
          "baseline": "c62f81a8a11347ff53b60101fa75280e69d002a6", "profiles": profiles,
          "native_context": "AWAITING_FRESH_OFFICIAL_SDK", "fresh_compilation": False,
          "qualification": False, "human_approval": False, "public_gate_activation": False}
write_new(REPORT / "FROZEN_PROFILES.json", (json.dumps(report, indent=2) + "\n").encode())
print(json.dumps({"profiles": [{"id": p["id"], "modules": len(p["sources"]),
                               "queries": len(p["cases"]), "witnesses": sum(c["witness"] is not None for c in p["cases"])}
                              for p in profiles], "source_only": True}, indent=2))
