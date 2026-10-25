"""Fresh replay of bound graph-state development evidence; no saved proof authority."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import lean_graph_state as state
from . import lean_graph_state_configuration as configuration


def replay(report_path, destination, *, reference_root):
    original = json.loads(Path(report_path).read_text())
    if original["status"] != "graph_model_checked":
        raise ValueError("this replay mode requires a measured conditional graph-model claim")
    records = [*original["inputs"], *original["retained_inputs"], original["query"], original["proof_profile"]]
    for row in original["source_closure"]:
        records.extend((row["original"], row["retained"]))
    records.extend(original["transitive_audit"]["imported_objects"])
    runtime = original["host_runtime"]
    records.extend((runtime["python"], *runtime["support_sources"]))
    external = original["producer_external_runtime"]
    records.extend((external["python"], *external["loaded_sources"]))
    if not state.fresh(records):
        raise ValueError("recorded input/code/proof/tool dependency identity is stale")
    before, after, contract = (row["path"] for row in original["retained_inputs"])
    if original["schema"] == "merlin.graph-state.composed-boundary.v1":
        measured = state.check_boundary(
            before,
            after,
            destination,
            reference_root=reference_root,
            contract=contract,
            original_contract_sha256=original["contract_sha256"],
            expected_before_sha256=original["inputs"][0]["sha256"],
        )
    elif original["schema"] == "merlin.graph-state.configuration-boundary.v1":
        measured = configuration.check_boundary(
            before,
            after,
            destination,
            reference_root=reference_root,
            contract=contract,
            original_contract_sha256=original["inputs"][2]["sha256"],
        )
    else:
        raise ValueError("unrecognized conditional graph-state claim schema")
    same = measured["status"] == original["status"]
    if same:
        same = (
            measured["query"]["sha256"] == original["query"]["sha256"]
            and measured["transitive_audit"]["statements"] == original["transitive_audit"]["statements"]
            and measured["transitive_audit"]["raw"]["axioms_used"] == original["transitive_audit"]["raw"]["axioms_used"]
        )
    result = {
        "schema": "merlin.graph-state.fresh-replay.v1",
        "original_report": state.file_ref(report_path),
        "fresh_report": state.file_ref(Path(destination) / "report.json"),
        "same_claim": same,
        "public_acceptance_issued": False,
        "source_contract_approved": False,
        "meaning": "fresh actual projection, complete kernel checker and transitive audit; "
        "conditional model evidence only",
    }
    state._write(Path(destination) / "replay.json", result)
    if not same:
        raise ValueError("fresh replay does not reproduce the recorded conditional claim")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(replay(args.report, args.destination, reference_root=args.reference)))


if __name__ == "__main__":
    main()
