"""Replayable operational typing evidence; no definedness, numerical or approval claims."""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

from merlin.common.paths import build_dir, repo_root
from merlin.verify.lean.check import _file_ref
from merlin.verify.lean.scalar_square import _run, freshness, replay

MODULE = "MerlinVerify.Driver.ScalarSquareTyping"
NAMESPACE = "MerlinVerify.Rules.ScalarSquareTyping"
THEOREMS = [
    f"{NAMESPACE}.{name}"
    for name in (
        "genericSem_related",
        "typingRule_sound",
        "rejectedQuant_stuck",
        "opsCheck_preserves",
        "regionCheck_preserves",
        "calls_preserve",
        "literal_result",
        "result_facts",
        "primitive_checked",
        "opsCheck_ready",
        "source_typing",
        "closedCheck_sound",
        "closed_results",
        "strict_source_with_checked_typing",
    )
]


def prepare() -> None:
    _run(["lake", "build", MODULE])


def run(export: Path) -> dict:
    driver = repo_root() / "merlin/lean/MerlinVerify/Driver/ScalarSquareTyping.lean"
    return json.loads(_run(["lake", "env", "lean", "--run", str(driver), str(export.resolve())]).splitlines()[-1])


def bindings() -> list[dict]:
    root = repo_root() / "merlin/lean"
    todo, seen, paths = [MODULE], set(), set()
    while todo:
        module = todo.pop()
        if module in seen:
            continue
        seen.add(module)
        relative = Path(*module.split("."))
        source = (root / relative).with_suffix(".lean")
        paths.add(source)
        for line in source.read_text().splitlines():
            words = line.strip().split()
            if words and words[0] == "import":
                todo.extend(word for word in words[1:] if word.startswith("MerlinVerify."))
        compiled = build_dir() / "lean/build/lib/lean" / relative
        paths.update(
            p
            for suffix in (".olean", ".olean.private", ".olean.server", ".ir")
            if (p := compiled.with_suffix(suffix)).is_file()
        )
    paths.update(root / name for name in ("SPEC_LOCK", "lean-toolchain", "lake-manifest.json", "lakefile.toml"))
    paths.add(Path(_run(["lake", "env", "which", "lean"]).strip()).resolve())
    paths.add(Path(__file__).resolve())
    return [_file_ref(path) for path in sorted(paths)]


def audit() -> dict:
    _run(["lake", "build", "merlin-lean-audit"])
    report = json.loads(_run(["lake", "exe", "merlin-lean-audit", MODULE]).splitlines()[-1])
    if report["violations"]:
        raise RuntimeError(report)
    report["statements"] = json.loads(
        _run(
            [
                "lake",
                "exe",
                "merlin-lean-audit",
                "--statements",
                ",".join(THEOREMS),
                MODULE,
            ]
        ).splitlines()[-1]
    )["statements"]
    return report


def write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def check(base_path: Path, destination: Path, *, compiler_python: Path) -> dict:
    """Retain complete before/after typing, with the original boundary receipt as a binding."""
    prepare()
    base = json.loads(base_path.read_text())
    if stale := freshness(base, compiler_python):
        raise ValueError(f"base receipt is stale: {stale}")
    work = destination / "attempts" / uuid.uuid4().hex
    work.mkdir(parents=True)
    dependencies = bindings()
    reports = {side: run(Path(base["exports"][side]["path"])) for side in ("before", "after")}
    if not all(report["source_typing_discharged"] for report in reports.values()):
        raise ValueError(f"closed capture typing failed: {reports}")
    proof = audit()
    write(work / "report.json", reports)
    write(work / "proof-audit.json", proof)
    receipt = {
        "schema": "merlin.lean.scalar-square.typing-receipt.v1",
        "verdict": "typing_only",
        "base_receipt": _file_ref(base_path),
        "dependencies": dependencies,
        "report": _file_ref(work / "report.json"),
        "proof_audit": _file_ref(work / "proof-audit.json"),
        "source_typing_discharged": True,
        "source_definedness_proved": False,
        "successful_execution_proved": False,
        "poison_freedom_proved": False,
        "numerical_law_discharged": False,
        "policy_approved": False,
        "original_source_equivalence_claimed": False,
    }
    if stale := stale_bindings(receipt, compiler_python):
        raise ValueError(f"verification inputs changed: {stale}")
    write(destination / "receipt.json", receipt)
    return receipt


def stale_bindings(receipt: dict, compiler_python: Path) -> list[str]:
    refs = [*receipt["dependencies"], receipt["base_receipt"], receipt["report"], receipt["proof_audit"]]
    changed = [r["path"] for r in refs if not Path(r["path"]).is_file() or _file_ref(Path(r["path"])) != r]
    base = json.loads(Path(receipt["base_receipt"]["path"]).read_text())
    return changed + freshness(base, compiler_python)


def replay_typing(path: Path, work: Path, *, compiler_python: Path) -> dict:
    receipt = json.loads(path.read_text())
    if stale := stale_bindings(receipt, compiler_python):
        return {"fresh": False, "matches_recorded": False, "stale": stale}
    work.mkdir(parents=True, exist_ok=False)
    base_path = Path(receipt["base_receipt"]["path"])
    base = json.loads(base_path.read_text())
    boundary = replay(base_path, work / "boundary-replay", compiler_python=compiler_python)
    actual = {side: run(Path(base["exports"][side]["path"])) for side in ("before", "after")}
    proof_matches = audit() == json.loads(Path(receipt["proof_audit"]["path"]).read_text())
    metadata_matches = (
        receipt.get("schema") == "merlin.lean.scalar-square.typing-receipt.v1"
        and receipt.get("verdict") == "typing_only"
        and receipt.get("source_typing_discharged") is True
        and all(
            receipt.get(key) is False
            for key in (
                "source_definedness_proved",
                "successful_execution_proved",
                "poison_freedom_proved",
                "numerical_law_discharged",
                "policy_approved",
                "original_source_equivalence_claimed",
            )
        )
    )
    result = {
        "fresh": not stale_bindings(receipt, compiler_python),
        "matches_recorded": actual == json.loads(Path(receipt["report"]["path"]).read_text())
        and proof_matches
        and metadata_matches
        and boundary["fresh"]
        and boundary["matches_recorded"],
        "proof_audit_matches": proof_matches,
        "claim_metadata_matches": metadata_matches,
        "source_typing_discharged": actual["before"]["source_typing_discharged"],
        "source_definedness_proved": False,
        "policy_approved": False,
    }
    write(work / "replay.json", result)
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
        else replay_typing(args.input, args.work, compiler_python=args.compiler_python)
    )
    print(json.dumps(result, indent=2))
    return 0 if result.get("source_typing_discharged") and result.get("matches_recorded", True) else 2


if __name__ == "__main__":
    raise SystemExit(main())
