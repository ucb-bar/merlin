"""Replayable whole-program scalar-square claims with undischarged typing, numerics and approval.

This supplements existing scalar-square receipts without rewriting them or native counterexamples.
The source-preservation theorem and approved multiplication-policy theorem have separate premises.
An executable structural match never discharges either set of premises.
"""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

from merlin.common.paths import build_dir, repo_root
from merlin.verify.lean.check import _file_ref
from merlin.verify.lean.scalar_square import _run, dependencies, freshness, replay

MODULE = "MerlinVerify.Driver.ScalarSquareComposition"
CLAIMS = "MerlinVerify.Rules.ScalarSquareClaims"
THEOREMS = [
    *[
        f"MerlinVerify.Rules.ScalarSquareSSA.{name}"
        for name in ("arguments_bind", "results_bind", "site_values", "site_step")
    ],
    *[
        f"MerlinVerify.Rules.ScalarSquareComposition.{name}"
        for name in ("operations_sound", "regions_sound", "blocks_sound", "primitive_regionless", "primitive_call")
    ],
    "MerlinVerify.Check.ScalarSquareProgram.check_sound",
    *[
        f"{CLAIMS}.{name}"
        for name in (
            "calls_equal",
            "source_equal",
            "strict_source_preservation",
            "source_success",
            "approved_multiplication_policy",
            "unapproved_rejected",
            "policy_success",
            "numerical_nonempty",
        )
    ],
]
MODULES = [
    "MerlinVerify.Rules.ScalarSquareSSA",
    "MerlinVerify.Rules.ScalarSquareComposition",
    "MerlinVerify.Check.ScalarSquareProgram",
    CLAIMS,
    MODULE,
]


def prepare() -> None:
    _run(["lake", "build", MODULE])


def bindings() -> list[dict]:
    refs = dependencies()
    for module in MODULES:
        relative = Path(*module.split("."))
        refs.append(_file_ref((repo_root() / "merlin/lean" / relative).with_suffix(".lean")))
        compiled = build_dir() / "lean/build/lib/lean" / relative
        refs.extend(
            _file_ref(path)
            for suffix in (".olean", ".olean.private", ".olean.server", ".ir")
            if (path := compiled.with_suffix(suffix)).is_file()
        )
    refs.append(_file_ref(repo_root() / "src/merlin/verify/lean/scalar_square_composition.py"))
    return refs


def run(before: Path, after: Path) -> dict:
    driver = repo_root() / "merlin/lean/MerlinVerify/Driver/ScalarSquareComposition.lean"
    return json.loads(_run(["lake", "env", "lean", "--run", str(driver), str(before), str(after)]).splitlines()[-1])


def audit() -> dict:
    _run(["lake", "build", "merlin-lean-audit"])
    result = json.loads(_run(["lake", "exe", "merlin-lean-audit", MODULE]).splitlines()[-1])
    if result["violations"]:
        raise RuntimeError(result)
    result["statements"] = json.loads(
        _run(["lake", "exe", "merlin-lean-audit", "--statements", ",".join(THEOREMS), MODULE]).splitlines()[-1]
    )["statements"]
    return result


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def check(base_path: Path, destination: Path, *, compiler_python: Path) -> dict:
    """Supplement an existing receipt; never mark a conditional theorem as instantiated."""
    prepare()
    base = json.loads(base_path.read_text())
    if stale := freshness(base, compiler_python):
        raise ValueError(f"base receipt is stale: {stale}")
    if not base.get("conditional_match"):
        raise ValueError("base boundary did not match")
    destination.mkdir(parents=True, exist_ok=True)
    work = destination / "attempts" / uuid.uuid4().hex
    work.mkdir(parents=True)
    replayed = replay(base_path, work / "base-replay", compiler_python=compiler_python)
    if not replayed["fresh"] or not replayed["matches_recorded"]:
        raise ValueError(replayed)
    before, after = (Path(base["exports"][label]["path"]) for label in ("before", "after"))
    dependencies = bindings()
    report = run(before, after)
    if not report["program_structure_checked"]:
        raise ValueError(f"program correspondence failed: {report}")
    proof = audit()
    _write(work / "report.json", report)
    _write(work / "proof-audit.json", proof)
    result = {
        "schema": "merlin.lean.scalar-square.program-receipt.v1",
        "verdict": "abstracted",
        "base_receipt": _file_ref(base_path),
        "base_replay": replayed,
        "dependencies": dependencies,
        "report": _file_ref(work / "report.json"),
        "proof_audit": _file_ref(work / "proof-audit.json"),
        "strict_source_claim_instantiated": False,
        "policy_claim_instantiated": False,
        "policy_approved": False,
    }
    changed = stale_bindings(result, compiler_python)
    if changed:
        raise ValueError(f"verification inputs changed: {changed}")
    _write(destination / "receipt.json", result)
    return result


def stale_bindings(receipt: dict, compiler_python: Path) -> list[str]:
    refs = [*receipt["dependencies"], receipt["base_receipt"], receipt["report"], receipt["proof_audit"]]
    changed = [r["path"] for r in refs if not Path(r["path"]).is_file() or _file_ref(Path(r["path"])) != r]
    base = json.loads(Path(receipt["base_receipt"]["path"]).read_text())
    return changed + freshness(base, compiler_python)


def replay_program(path: Path, work: Path, *, compiler_python: Path) -> dict:
    receipt = json.loads(path.read_text())
    if stale := stale_bindings(receipt, compiler_python):
        return {"fresh": False, "stale": stale, "matches_recorded": False}
    work.mkdir(parents=True, exist_ok=False)
    base_path = Path(receipt["base_receipt"]["path"])
    base_replay = replay(base_path, work / "base-replay", compiler_python=compiler_python)
    base = json.loads(base_path.read_text())
    actual = run(*(Path(base["exports"][label]["path"]) for label in ("before", "after")))
    recorded = json.loads(Path(receipt["report"]["path"]).read_text())
    proof_matches = audit() == json.loads(Path(receipt["proof_audit"]["path"]).read_text())
    header_matches = (
        receipt.get("schema") == "merlin.lean.scalar-square.program-receipt.v1"
        and receipt.get("verdict") == "abstracted"
        and receipt.get("strict_source_claim_instantiated") is False
        and receipt.get("policy_claim_instantiated") is False
        and receipt.get("policy_approved") is False
    )
    stale = stale_bindings(receipt, compiler_python)
    result = {
        "fresh": not stale,
        "stale": stale,
        "matches_recorded": actual == recorded
        and base_replay["matches_recorded"]
        and base_replay["fresh"]
        and proof_matches
        and header_matches,
        "proof_audit_matches": proof_matches,
        "claim_metadata_matches": header_matches,
        "program_structure_checked": actual["program_structure_checked"],
        "strict_source_claim_instantiated": False,
        "policy_claim_instantiated": False,
    }
    _write(work / "replay.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "replay"))
    parser.add_argument("input", type=Path)
    parser.add_argument("work", type=Path)
    parser.add_argument("--compiler-python", type=Path, required=True)
    args = parser.parse_args()
    result = (
        check(args.input, args.work, compiler_python=args.compiler_python)
        if args.command == "check"
        else replay_program(args.input, args.work, compiler_python=args.compiler_python)
    )
    print(json.dumps(result, indent=2))
    return 0 if result.get("verdict") == "abstracted" or result.get("fresh") and result.get("matches_recorded") else 2


if __name__ == "__main__":
    raise SystemExit(main())
