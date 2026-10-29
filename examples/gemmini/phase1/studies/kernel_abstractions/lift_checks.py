"""E0 retained-custody audit and E1 real diagnostic requests for the study.

No compiler execution, measured cycles, task completion or semantic parity is
granted by these reports. Exact source/capture gaps remain visible.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from inputs import pin, write_json


def pinned_records(value):
    found = {}

    def visit(node):
        if isinstance(node, dict):
            if isinstance(node.get("path"), str) and isinstance(node.get("sha256"), str):
                found[(node["path"], node["sha256"])] = {"path": node["path"], "sha256": node["sha256"]}
            for member in node.values():
                visit(member)
        elif isinstance(node, list):
            for member in node:
                visit(member)

    visit(value)
    return list(found.values())


def audit_oracle(study):
    study = Path(study)
    document = json.loads(study.read_text())
    checked = []
    for expected in pinned_records(document):
        try:
            actual = pin(expected["path"])
            checked.append(
                {
                    **expected,
                    "status": "matched" if actual["sha256"] == expected["sha256"] else "changed",
                    "actual_sha256": actual["sha256"],
                }
            )
        except OSError as error:
            checked.append({**expected, "status": "unavailable", "reason": type(error).__name__})
    rows = []
    for row in document["rows"]:
        pairs = row.get("ordered_samples", {})
        ours, expert = pairs.get("ours", []), pairs.get("reference", [])
        exact = row.get("outputs_exact") is True
        measured = row.get("evidence") == "retained_FireSim_completed_call"
        ratio = (
            max(sum(ours) / len(ours) / (sum(expert) / len(expert)), ours[-1] / expert[-1])
            if exact and measured and ours and expert
            else None
        )
        rows.append(
            {
                "id": row["id"],
                "reference": row["reference"],
                "evidence": row["evidence"],
                "shape": row["shape"],
                "numerics": row["output"],
                "matched_outputs": exact,
                "screen_ratio": ratio,
                "within_5pct": ratio <= 1.05 if ratio is not None else None,
                "within_8pct": ratio <= 1.08 if ratio is not None else None,
                "source_closure": row.get("source_closure", "unknown"),
                "latest_archive_equivalence": False,
                "qualifications": row.get("qualifications", []),
            }
        )
    return {
        "schema": "private.kernel_abstractions.e0.v1",
        "study": pin(study),
        "pins": checked,
        "counts": {"unique_pins": len(checked), "matched": sum(r["status"] == "matched" for r in checked)},
        "rows": rows,
        "classification": "retained operator capability evidence, not automatic author convergence",
        "qk_gap": "original source/check closure missing; physical ELF/UART/object retained",
        "pv_lm": {row["id"]: row["evidence"] for row in rows if row["id"] in ("attention_pv", "decode_lm")},
        "counterexamples": [
            "near-shape Exo estimates cannot establish exact measured parity",
            "nativei8 scale1 cube differs from retained Exo scale0.5",
            "prepacked-input Exo rows do not inherit common-input checksums",
            "newest whole-model ELF does not inherit older-kernel numerical/ROI equality",
        ],
    }


def check_lifts(
    workspace,
    *,
    facts,
    selection,
    selection_sha256,
    output,
    c_source=None,
    jack_source=None,
    disassembly=None,
    elf=None,
    disassembly_symbol="fx_conv_27",
):
    from merlin_experiments.phase1.feedback import reference_analysis as analysis
    from merlin_experiments.phase1.feedback import reference_experiments as reference
    from merlin_experiments.phase1.feedback.reference_views import source_view

    workspace, output = Path(workspace), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    requests = []
    if c_source:
        requests.append(("public_exo_source", dict(operation="lift", path=c_source, format="auto", limit=1)))
    if jack_source:
        requests.append(
            ("jack_source", dict(operation="lift", path=jack_source, format="c_source", symbol="mm_o", limit=1))
        )
    if disassembly:
        requests.append(
            (
                "jack_conv_disassembly",
                dict(operation="lift", path=disassembly, format="disassembly", symbol=disassembly_symbol, limit=1),
            )
        )
    if elf:
        requests.append(
            ("jack_conv_elf", dict(operation="lift", path=elf, format="elf", symbol=disassembly_symbol, limit=1))
        )
    result = []
    for name, value in requests:
        request = {"action": analysis.ACTION, "reference_analysis": value}
        with reference.invocation(selection, selection_sha256, tasks=False):
            response = analysis.handle(request, workspace=workspace, facts=Path(facts))
        result.append(
            {
                "id": name,
                "request": write_json(output / (name + ".request.json"), request),
                "response": write_json(output / (name + ".response.json"), response),
                "status": response["status"],
                "reason": response.get("reason"),
                "semantic_equivalence": response.get("semantic_equivalence"),
                "performance": response.get("performance"),
            }
        )
    # The existing source parser independently checks original declared result
    # types. A changed output shape must fail before any candidate metadata join.
    from inputs import GEMMS, source_contract

    from merlin.targetgen.contract.interface_emit import emit_interface_mlir, parse_interface_mlir

    source = workspace / "training" / GEMMS[0] / "capsule.interface.mlir"
    raw = source.read_bytes()
    observed = source_view(raw)
    cb = parse_interface_mlir(raw.decode())
    changed = copy.deepcopy(cb)
    commit = next(c for c in changed["commands"] if c["opcode"] == "COMMIT")
    commit["attributes"].update(epilogue=["acc_scale"], acc_scale=0.5)
    scaled = source_view(emit_interface_mlir(changed).encode())
    changed_dtype = copy.deepcopy(cb)
    next(c for c in changed_dtype["commands"] if c["opcode"] == "COMMIT")["attributes"]["output_dtype"] = "i32"
    widened = source_view(emit_interface_mlir(changed_dtype).encode())
    counterexamples = {
        "source": pin(source),
        "scale_contract_differs": source_contract(changed) != source_contract(cb),
        "output_precision_differs": source_contract(changed_dtype) != source_contract(cb),
        "all_outputs_observed": len(observed["command_buffer"]["tensors"]),
        "scale_source_identity_differs": observed["input"] != scaled["input"],
        "widened_source_identity_differs": observed["input"] != widened["input"],
        "authority": "source contracts only; no recovered expert numeric equivalence",
    }
    return {
        "schema": "private.kernel_abstractions.e1.v1",
        "requests": result,
        "counterexamples": counterexamples,
        "coverage": {"real_requests": len(result), "analyzed": sum(r["status"] == "analyzed" for r in result)},
        "unknowns": [
            "static sites are not dynamic counts or elapsed cycles",
            "C syntax does not establish full arithmetic/fusion/quantization semantics",
            "selected ISA facts do not establish every decoded operand or register value",
            "frontend graph binding is a hypothesis until exact producer/source custody closes",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--facts", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--selection-sha256")
    parser.add_argument("--c-source")
    parser.add_argument("--jack-source")
    parser.add_argument("--disassembly")
    parser.add_argument("--elf")
    parser.add_argument("--disassembly-symbol", default="fx_conv_27")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    e0 = audit_oracle(args.study)
    write_json(args.output / "E0.json", e0)
    if args.workspace:
        e1 = check_lifts(
            args.workspace,
            facts=args.facts,
            selection=args.selection,
            selection_sha256=args.selection_sha256,
            output=args.output / "requests",
            c_source=args.c_source,
            jack_source=args.jack_source,
            disassembly=args.disassembly,
            elf=args.elf,
            disassembly_symbol=args.disassembly_symbol,
        )
        write_json(args.output / "E1.json", e1)
    print(json.dumps({"e0": e0["counts"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
