"""Fresh integration check/replay; historical reports supply no live authority."""
import argparse
import hashlib
import json
from pathlib import Path

from merlin.verify.buffer_reuse_intake import ALLOCATOR, prepare_chain
from merlin.verify import buffer_reuse_gate as gate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["check", "replay"])
    parser.add_argument("--report")
    args = parser.parse_args()
    artifact = Path(__file__).resolve().parent
    root = Path.cwd()
    record = artifact / ("buffer-" + args.mode)
    record.mkdir(exist_ok=False)
    lean = Path("/home/jeremy/.elan/toolchains/leanprover--lean4---v4.35.0/bin/lean")
    destination = root / "out/build/dce-cse-loop-buffer-integration-20261028" / ("buffer-" + args.mode)
    raw = tuple((artifact / "buffer-inputs" / f"stage{i}.mlir").read_text() for i in range(4))
    candidate = prepare_chain(raw, allocator=ALLOCATOR)
    refs = json.loads(candidate.sources_json)
    assert len(refs) == 192
    assert str(root / "src/merlin/common/owned_children.py") in {row["path"] for row in refs}
    (record / "INTENT.json").write_text(json.dumps({"mode": args.mode, "destination": str(destination),
        "candidate": candidate.describe(), "sources": refs, "old_objects_supplied": False}, indent=2) + "\n")
    result = (gate.check(candidate, lean=lean, destination=destination)
              if args.mode == "check" else gate.replay(args.report, lean=lean, destination=destination))
    report = {"status": result.status, "reason": result.reason, "sources": len(refs),
              "fresh_build": True, "new_host_qualification": False}
    if result.completion is not None:
        report["report"] = result.completion.report
    if result.status == "checked" and args.mode == "check":
        seen = []
        def consume(text):
            assert text == raw[-1]
            (record / "after.mlir").write_text(text)
            seen.append(hashlib.sha256(text.encode()).hexdigest())
        gate.deliver(result.completion, candidate, consume)
        assert len(seen) == 1 and gate.validate_completion(result.completion, candidate)
        report.update(exact_final_consumer=True, final_sha256=seen[0])
    (record / "RESULT.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    return 0 if result.status == "checked" else 1


if __name__ == "__main__":
    raise SystemExit(main())
