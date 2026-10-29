"""Fresh combined-tree baseline checks and separate-process replay evidence."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("family", choices=["elision", "hoist"])
    parser.add_argument("mode", choices=["check", "replay"])
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    record = here / (args.family + "-" + args.mode)
    record.mkdir(exist_ok=False)
    base = Path("out/artifacts/verification/merlin-passes/loop-scope-20261028")
    reference = Path("/home/jeremy/Projects/merlin-graph-proof-intake")
    if args.family == "elision":
        from merlin.verify import configuration_elision_gate as gate
        from merlin.verify.configuration_elision_intake import prepare
        fixtures = base / "NEXT-CONFIGURATION/F"
        case = fixtures / "corpus-v1/positives/multifield"
        candidate = prepare((case / "stage0.mlir").read_text(), (case / "stage1.mlir").read_text(),
            reference=reference, facts=fixtures / "fixture-input-v1/facts.json",
            software=fixtures / "fixture-input-v1/software.yaml", target="configuration_fixture")
    else:
        from merlin.verify import configuration_hoist_gate as gate
        from merlin.verify.configuration_hoist_intake import prepare
        fixtures = base / "NEXT-HOIST"
        data = json.loads((fixtures / "Q/actual-baseline-v1/ACTUAL.json").read_text())
        candidate = prepare(data["source"], data["final"],
            reference="/home/jeremy/Projects/merlin-dce-cse-integration",
            facts=data["facts"]["path"], software=data["software"]["path"], target=data["isa"]["target"],
            source_manifest=fixtures / "COORDINATOR/PRODUCER-SOURCE-MANIFEST-e0484f488.json")
    source_refs = json.loads(candidate.sources_json)
    helper = Path("src/merlin/common/owned_children.py").resolve()
    assert str(helper) in {row["path"] for row in source_refs}
    (record / "INTENT.json").write_text(json.dumps({"candidate": candidate.describe(),
        "mode": args.mode, "helper_bound": str(helper), "old_objects_supplied": False}, indent=2) + "\n")
    build = Path("out/build/configuration-graph-integration-20261028")
    lean = Path("/home/jeremy/.elan/toolchains/leanprover--lean4---v4.35.0/bin/lean")
    destination = build / (args.family + "-" + args.mode)
    result = (gate.check(candidate, lean=lean, destination=destination) if args.mode == "check"
              else gate.replay(build / (args.family + "-check/REPORT.json"), lean=lean, destination=destination))
    outcome = {"status": result.status, "reason": result.reason, "helper_bound": True,
               "source_count": len(source_refs), "fresh_build": True, "new_host_qualification": False}
    if result.completion is not None:
        outcome["report"] = result.completion.report
    if result.status == "checked" and args.mode == "check":
        seen = []
        def consume(text):
            assert text == candidate.final
            (record / "after.txt").write_text(text)
            seen.append(hashlib.sha256(text.encode()).hexdigest())
        gate.deliver(result.completion, candidate, consume)
        assert len(seen) == 1 and gate.validate_completion(result.completion, candidate)
        outcome.update(exact_consumer=True, final_sha256=seen[0])
    (record / "RESULT.json").write_text(json.dumps(outcome, indent=2) + "\n")
    print(json.dumps(outcome), flush=True)
    return 0 if result.status == "checked" else 1


if __name__ == "__main__":
    raise SystemExit(main())
