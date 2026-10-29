"""Verify source/query/metadata pins and recompute the exact source-only plan.

No archive extraction, provider/controller execution, Lean, SDK or native action.
An integrity match is neither root elaboration nor semantic qualification.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_plan import PlanError, Reader, build, decode, relative


def verify(packet, inputs="INPUTS.json", plan="PLAN.json"):
    reader = Reader(packet)
    spec = decode((reader.root / str(relative(inputs))).read_bytes())
    actual = decode((reader.root / str(relative(plan))).read_bytes())
    expected = build(spec, reader)
    if actual != expected:
        raise PlanError("plan differs from exact source/query/root metadata recomputation")
    return {
        "schema": "merlin.fresh-graph-plan-integrity.v1",
        "status": "SOURCE_PLAN_INTEGRITY_MATCHED",
        "profiles": len(expected["profiles"]),
        "source_data_files_checked": len(reader.cache),
        "unavailable_profiles": [p["id"] for p in expected["profiles"] if p["status"] == "UNAVAILABLE"],
        "unavailable_cases": [
            {"profile": p["id"], "case": c["id"], "reason": c.get("reason")}
            for p in expected["profiles"]
            for c in p.get("cases", [])
            if c["status"] == "UNAVAILABLE"
        ],
        "integrity_only": True,
        "kernel_root_resolution": "UNEXECUTED",
        "native_execution_verified": False,
        "graph_or_model_approval": False,
        "human_review_approved": False,
        "all_future_steps": "UNEXECUTED",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--inputs", default="INPUTS.json")
    parser.add_argument("--plan", default="PLAN.json")
    args = parser.parse_args(argv)
    try:
        result = verify(args.packet, args.inputs, args.plan)
    except (PlanError, OSError, KeyError, TypeError, UnicodeError) as error:
        result = {
            "status": "UNAVAILABLE" if isinstance(error, OSError) else "PLAN_INTEGRITY_FAILED",
            "error": str(error),
            "integrity_only": True,
            "native_execution_verified": False,
            "graph_or_model_approval": False,
            "human_review_approved": False,
        }
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "SOURCE_PLAN_INTEGRITY_MATCHED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
