#!/usr/bin/env python3
"""Summarize one public model (or model-slice / layer) case: declaration, quantization, op placement, work.

  python model_summary.py model/CASE [--ws .] [--json]

Reads only the files staged for the case. From ``capsule.yaml``: inputs, the operation attributes
(model, dtypes, quantization scheme, argument order), numeric policy and required tiers, the
integerization receipt (how many qparams, their granularities, accumulator dtypes), and the
``software_screen`` (per-operation host/accelerator placement with counts). From
``frontend-trace.json``: framework op counts per graph stage. From
``model_qualification.group_capsules``: the operator cases that verify each device group in
isolation, each with its operation, shapes, reduction extent and output size from its own
``capsule.yaml`` and ``capsule.interface.mlir``, giving a multiply-accumulate estimate
(output elements x reduction extent x group count) so the dominant work is visible. Files the case
names but the workspace does not stage (weights, answers) are listed as withheld; nothing is
inferred about them. Needs PyYAML (present in the author's interpreter). Read-only.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

#: Fields never echoed: digests of withheld answers are not inputs to a compiler.
WITHHELD_KEYS = frozenset({"golden_sha256"})


def _yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit(f"model_summary needs PyYAML to read {path}: {exc}") from exc
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def tensor_dims(type_text: str) -> list[int] | None:
    """``tensor<9x24xi32>`` -> [9, 24]; None for a dynamic or unreadable type."""
    inner = type_text.partition("tensor<")[2].partition(">")[0]
    if not inner:
        return None
    parts = inner.split("x")
    dims = []
    for part in parts[:-1]:
        if not part.isdecimal():
            return None
        dims.append(int(part))
    return dims


def output_dims(interface_text: str, out_name: str) -> list[int] | None:
    """The result type of the interface op that names ``out_name``: the last tensor type after ``->``."""
    for line in interface_text.splitlines():
        if f'name = "{out_name}"' in line and "->" in line:
            return tensor_dims(line.rpartition("->")[2])
    return None


def capsule_work(case_dir: Path) -> dict:
    """One operator case: op, shapes, reduction extent, output elements and MAC estimate."""
    data = _yaml(case_dir / "capsule.yaml")
    operation = data.get("operation") or {}
    attributes = operation.get("attributes") or {}
    reduction = (data.get("integer_partial_sum_bound") or {}).get("reduction_extent")
    interface = case_dir / str(data.get("interface_mlir") or "capsule.interface.mlir")
    out_name = str(attributes.get("out") or "")
    dims = output_dims(interface.read_text(encoding="utf-8"), out_name) if interface.is_file() and out_name else None
    elements = 1
    for d in dims or ():
        elements *= d
    return {
        "op": operation.get("op"),
        "inputs": {t.get("name"): [t.get("shape"), t.get("dtype"), t.get("role")] for t in data.get("inputs") or ()},
        "output": [out_name, dims, attributes.get("output_dtype")],
        "epilogue": attributes.get("epilogue"),
        "reduction_extent": reduction,
        "output_elements": elements if dims is not None else None,
        "macs": elements * int(reduction) if dims is not None and isinstance(reduction, int) else None,
        "semantic": (data.get("semantic") or {}).get("semantic_family"),
    }


def summarize(case_dir: Path, ws: Path) -> dict:
    data = _yaml(case_dir / "capsule.yaml")
    operation = data.get("operation") or {}
    attributes = operation.get("attributes") or {}
    staged = sorted(p.name for p in case_dir.iterdir())
    named = [attributes.get(k) for k in ("weights", "weights_manifest") if attributes.get(k)]
    out: dict = {
        "name": data.get("name"), "kind": data.get("kind"), "label": data.get("label"),
        "staged_files": staged, "withheld": sorted(str(n) for n in named if str(n) not in staged),
        "inputs": data.get("inputs"),
        "attributes": {k: v for k, v in attributes.items() if not isinstance(v, (dict, list)) or k == "arg_order"},
        "numeric_policy": data.get("numeric_policy"), "required_oracle_tiers": data.get("required_oracle_tiers"),
        "gate": data.get("gate"),
    }
    receipt = attributes.get("integerization_receipt") or {}
    if receipt:
        qparams = receipt.get("qparams") or []
        out["quantization"] = {
            "scheme": attributes.get("quant_scheme"), "receipt_schema": receipt.get("schema"),
            "qparams": len(qparams),
            "activation_granularity": dict(Counter(str((q.get("activation") or {}).get("granularity")) for q in qparams)),
            "weight_granularity": dict(Counter(str((q.get("weight") or {}).get("granularity")) for q in qparams)),
            "accumulator_dtypes": dict(Counter(str(q.get("accumulator_dtype")) for q in qparams)),
            "precision_decisions": len(receipt.get("precision_decisions") or ()),
            "remaining_dequant_count": receipt.get("remaining_dequant_count"),
        }
    screen = data.get("software_screen") or {}
    if screen:
        merged: Counter = Counter()
        for d in screen.get("decisions") or ():
            merged[(str(d.get("placement")), str(d.get("operation")), str(d.get("classification")))] += int(
                d.get("count") or 1)
        out["placement"] = {"status": screen.get("status"), "totals": screen.get("placements"),
                            "by_operation": [{"placement": p, "operation": o, "classification": c, "count": n}
                                             for (p, o, c), n in sorted(merged.items(), key=lambda kv: -kv[1])]}
    trace_path = case_dir / "frontend-trace.json"
    if trace_path.is_file():
        graphs = (json.loads(trace_path.read_text(encoding="utf-8")).get("graphs") or {})
        out["frontend_ops"] = {stage: dict(Counter(g.get("by_target") or {}).most_common(15))
                               for stage, g in graphs.items() if isinstance(g, dict)}
    groups = (data.get("model_qualification") or {}).get("group_capsules") or []
    work = []
    for row in groups:
        entry = {k: v for k, v in row.items() if k not in WITHHELD_KEYS and k != "arithmetic_qualification"}
        target = ws / str(row.get("capsule") or "")
        if (target / "capsule.yaml").is_file():
            entry.update(capsule_work(target))
            if entry.get("macs") is not None:
                entry["total_macs"] = entry["macs"] * int(row.get("count") or 1)
        else:
            entry["note"] = "operator case not staged in this workspace"
        work.append(entry)
    if work:
        total = sum(w.get("total_macs") or 0 for w in work)
        for w in work:
            if total and w.get("total_macs") is not None:
                w["share"] = round(w["total_macs"] / total, 4)
        out["operator_cases"] = sorted(work, key=lambda w: -(w.get("total_macs") or 0))
        out["coverage"] = {k: (data.get("model_qualification") or {}).get(k)
                           for k in ("status", "device_groups", "covered_groups", "scope")}
    elif data.get("kind") != "model":
        out["work"] = capsule_work(case_dir)
    return out


def _print(report: dict) -> None:
    print(f"{report['name']}  kind={report['kind']}  label={report['label']}")
    print(f"  inputs: {report['inputs']}")
    print(f"  attributes: {report['attributes']}")
    print(f"  numeric_policy: {report['numeric_policy']}  tiers: {report['required_oracle_tiers']}")
    if report["withheld"]:
        print(f"  withheld (named, not staged): {report['withheld']}")
    if "quantization" in report:
        print(f"  quantization: {report['quantization']}")
    if "placement" in report:
        p = report["placement"]
        print(f"  placement {p['status']}: {p['totals']}")
        for r in p["by_operation"]:
            print(f"    {r['placement']:<11} x{r['count']:<3} {r['operation']}  ({r['classification']})")
    for w in report.get("operator_cases") or ():
        print(f"  {str(w.get('share', '')):>7} x{w.get('count')} {w.get('capsule')}: {w.get('op')} "
              f"out={w.get('output')} K={w.get('reduction_extent')} macs={w.get('macs')}")
    if "work" in report:
        print(f"  work: {report['work']}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("case_dir", help="e.g. model/CASE, model_slices/CASE or layers/CASE")
    ap.add_argument("--ws", default=".", help="workspace root that group capsule paths are relative to")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    report = summarize(Path(args.case_dir), Path(args.ws))
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        _print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
