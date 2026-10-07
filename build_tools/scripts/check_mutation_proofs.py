#!/usr/bin/env python3
"""Gate: a check ships with a recorded mutation that makes it FAIL.

This repo's recurring defect is a reading that cannot discriminate the cases it is used to decide,
whose ambiguous answer is the passing one: coverage over a vocabulary that could not name the missing
op; a capability claim that could not tell "hardware cannot" from "nobody checked"; a comparator that
read two schedules out of one shared buffer; a gate wired into no hook at all. Every instance was
found AFTER it had certified something, and every fix was per-defect.

The property that would have caught them is one question asked BEFORE a check ships: *what change to
the world makes this check fail?* A check with no answer is decoration. So every gate names, in
``gate_mutations.yaml`` beside this file, a RUNNABLE proof -- a test that drives the gate into its
failing direction -- plus the mutation that proof applies and, separately, what that mutation
REMOVES.

THE MUTATION MUST REMOVE EVERYTHING THAT MAKES THE CHECK PASS, NOT JUST THE HEADLINE TOKEN. This is
the failure mode that wastes an evening and then gets mis-read as a broken gate: a capability line
was re-introduced in its defective spelling while the corrected version's RTL citations were left in
the surrounding comment block, the gate correctly found the citations, it passed, and the conclusion
very nearly drawn was that the gate did not work. A mutation that leaves an alternative satisfier in
place proves nothing about the gate. That is why ``removes:`` is a separate, mandatory field: it is
the author stating which satisfiers the mutation destroys, and it is read by a reviewer, not by this
script. A proof that only deletes the obvious token and leaves a second path to PASS is a proof of
nothing, and no automated check can tell you that -- writing the sentence is the mechanism.

What this gate decides, structurally and cheaply (the default mode, ~0.2 s):
  * every gate is either PROVEN (named in the registry) or ratcheted -- never silently neither;
  * a registry entry points at a proof file that exists and a test function that is really in it;
  * the proof NAMES ITS GATE, so a proof cannot be a reimplementation of the gate's logic that
    passes while the gate itself is broken;
  * ``mutation`` and ``removes`` are both present, substantive and different from one another;
  * neither ledger has rotted: an entry for a gate that is gone, or a ratchet entry for a gate that
    is now proven, fails until it is removed.

What ``--prove`` decides, by EXECUTION (slower; a CI job and a pre-merge habit, not a pre-commit
hook): each proof is run, must pass, and must be observed to CALL INTO ITS GATE -- traced at the
frame level, so a proof that imports the gate and then asserts something about a local copy of the
logic is reported as uncoupled. An entry may declare ``execution: subprocess`` when its proof drives
the gate as a child process, which the tracer cannot see; that opt-out is PRINTED on every run rather
than silently honoured.

Known debt lives in ``mutation_proof_ratchet.txt``, one gate path per line. It may only shrink
(``check_ratchets_shrink.py`` holds every ``*_ratchet.txt``).

    python build_tools/scripts/check_mutation_proofs.py           # structural audit
    python build_tools/scripts/check_mutation_proofs.py --prove   # run every recorded proof
    python build_tools/scripts/check_mutation_proofs.py --prove --gate check_wiring.py
    python build_tools/scripts/check_mutation_proofs.py --list    # proven / ratcheted roster
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import NamedTuple

import yaml

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "build_tools" / "scripts" / "gate_mutations.yaml"
LEDGER = ROOT / "build_tools" / "scripts" / "mutation_proof_ratchet.txt"
GATE_DIR = "build_tools/scripts"
GATE_PREFIX = "check_"
TESTS_ROOT = "merlin/tests"

#: A one-word `mutation:` or a `removes:` that restates it proves the field was filled in, not that
#: anyone thought about it. Short enough not to be busywork, long enough to be a sentence.
MIN_PROSE = 40


class Failure(NamedTuple):
    subject: str
    message: str

    def render(self) -> str:
        return f"[FAIL] {self.subject}: {self.message}"


def gate_files() -> list[str]:
    base = ROOT / GATE_DIR
    if not base.is_dir():
        return []
    return sorted(f"{GATE_DIR}/{path.name}" for path in base.glob(f"{GATE_PREFIX}*.py"))


def load_registry(text: str | None = None) -> dict[str, dict]:
    """The recorded proofs, keyed by gate path. Pure over the text so a test can feed its own."""
    if text is None:
        text = REGISTRY.read_text(encoding="utf-8") if REGISTRY.is_file() else ""
    document = yaml.safe_load(text) or {}
    proofs = document.get("proofs") if isinstance(document, dict) else None
    if not isinstance(proofs, dict):
        return {}
    return {f"{GATE_DIR}/{name}": entry for name, entry in proofs.items()}


def load_ledger(text: str | None = None) -> list[str]:
    if text is None:
        text = LEDGER.read_text(encoding="utf-8") if LEDGER.is_file() else ""
    return [line.split("#", 1)[0].strip() for line in text.splitlines() if line.split("#", 1)[0].strip()]


def _test_functions(source: str) -> set[str]:
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return set()
    return {node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}


def _names_the_gate(source: str, gate_stem: str) -> bool:
    """Does this proof reach for the gate by name, structurally?

    A proof that never mentions its gate is testing something else -- most often a copy of the gate's
    decision inlined into the test, which passes forever while the gate itself rots. Read out of the
    ``ast`` rather than the text so a mention in a comment does not count.
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and gate_stem in node.value:
            return True
        if isinstance(node, ast.Import) and any(gate_stem in alias.name for alias in node.names):
            return True
        if isinstance(node, ast.ImportFrom):
            if gate_stem in (node.module or "") or any(gate_stem in alias.name for alias in node.names):
                return True
        if isinstance(node, ast.Name) and node.id == gate_stem:
            return True
    return False


def _split_node(node: str) -> tuple[str, str]:
    path, _, test = node.partition("::")
    return path.strip(), test.strip()


def audit(gates: list[str], registry: dict[str, dict], ratchet: list[str], read: object) -> list[Failure]:
    """Every way the roster can be wrong. ``read(relpath)`` returns a file's text or None.

    Pure in its inputs so the gate's own proof can drive each branch without touching the tree.
    """
    failures: list[Failure] = []
    known = set(gates)
    proven = set(registry)
    ratcheted = set(ratchet)

    for gate in sorted(proven - known):
        failures.append(Failure(gate, "recorded in gate_mutations.yaml but no such gate exists; remove the entry"))
    for gate in sorted(ratcheted - known):
        failures.append(Failure(gate, f"ratcheted in {LEDGER.name} but no such gate exists; remove the entry"))
    for gate in sorted(ratcheted & proven):
        failures.append(
            Failure(
                gate,
                f"is both proven and ratcheted; delete its line from {LEDGER.name} -- a ledger that keeps an entry for something already fixed stops meaning anything",
            )
        )
    for gate in gates:
        if gate not in proven and gate not in ratcheted:
            failures.append(
                Failure(
                    gate,
                    "ships with no recorded mutation that makes it fail. Add a proof test and an entry "
                    "to gate_mutations.yaml naming the mutation and everything it removes; a check "
                    "nobody has ever seen fail is indistinguishable from one that cannot.",
                )
            )

    for gate in sorted(proven & known):
        entry = registry[gate]
        if not isinstance(entry, dict):
            failures.append(Failure(gate, "registry entry is not a mapping of proof/mutation/removes"))
            continue
        node = entry.get("proof")
        mutation = entry.get("mutation")
        removes = entry.get("removes")
        if not isinstance(node, str) or "::" not in node:
            failures.append(Failure(gate, "proof must be '<test file>::<test function>'"))
            continue
        for field, value in (("mutation", mutation), ("removes", removes)):
            if not isinstance(value, str) or len(value.strip()) < MIN_PROSE:
                failures.append(
                    Failure(gate, f"{field}: must be a sentence of at least {MIN_PROSE} characters, not {value!r}")
                )
        if (
            isinstance(mutation, str)
            and isinstance(removes, str)
            and " ".join(mutation.split()) == " ".join(removes.split())
        ):
            failures.append(
                Failure(
                    gate,
                    "removes: restates mutation:. They are different questions -- 'removes' is where "
                    "you say which OTHER satisfiers the mutation destroys, and a mutation that leaves "
                    "one standing proves nothing about the gate.",
                )
            )
        path, test = _split_node(node)
        if not path.startswith(f"{TESTS_ROOT}/"):
            failures.append(Failure(gate, f"proof {path} must live under {TESTS_ROOT}/ so the suite runs it"))
            continue
        source = read(path)
        if source is None:
            failures.append(Failure(gate, f"proof file {path} does not exist"))
            continue
        if test not in _test_functions(source):
            failures.append(Failure(gate, f"proof {path} has no test function named {test!r}"))
            continue
        gate_stem = Path(gate).stem
        if not _names_the_gate(source, gate_stem):
            failures.append(
                Failure(
                    gate,
                    f"proof {path} never names {gate_stem} -- it is exercising something other than "
                    f"this gate, most likely a copy of its logic, which passes forever while the gate rots",
                )
            )
    return failures


def _read(relpath: str) -> str | None:
    path = ROOT / relpath
    return path.read_text(encoding="utf-8") if path.is_file() else None


# ------------------------------------------------------------------------------- execution tier


def run_proof(gate: str, node: str) -> tuple[bool, bool, str]:
    """Run one proof. Returns (passed, called_into_gate, detail).

    The coupling half is traced rather than inferred: ``sys.settrace`` records the file of every
    frame entered while the proof runs, so "the proof imported the gate" and "the proof made the gate
    decide something" are distinguishable -- which is the whole point of the exercise.
    """
    import pytest

    gate_path = str(ROOT / gate)
    entered: set[str] = set()

    def tracer(frame, event, arg):  # noqa: ANN001 - CPython trace protocol
        entered.add(frame.f_code.co_filename)
        return None

    previous = sys.gettrace()
    sys.settrace(tracer)
    try:
        code = pytest.main([node, "-q", "--no-header", "-p", "no:cacheprovider"])
    finally:
        sys.settrace(previous)
    passed = int(code) == 0
    return passed, gate_path in entered, f"pytest exit {int(code)}"


def prove(registry: dict[str, dict], only: str | None) -> tuple[list[Failure], list[str]]:
    failures: list[Failure] = []
    notes: list[str] = []
    for gate in sorted(registry):
        if only and Path(gate).name != only:
            continue
        entry = registry[gate]
        if not isinstance(entry, dict) or not isinstance(entry.get("proof"), str):
            continue
        node = entry["proof"]
        passed, coupled, detail = run_proof(gate, node)
        if not passed:
            failures.append(Failure(gate, f"its recorded proof {node} does not pass ({detail})"))
            continue
        if entry.get("execution") == "subprocess":
            notes.append(f"{gate}: coupling UNTRACED (proof drives the gate as a child process)")
            continue
        if not coupled:
            failures.append(
                Failure(
                    gate,
                    f"its proof {node} passes without ever entering {gate}. It is asserting against "
                    f"something else -- point it at the gate's own decision function.",
                )
            )
        else:
            notes.append(f"{gate}: proof passes and enters the gate")
    return failures, notes


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    gates = gate_files()
    registry = load_registry()
    ratchet = load_ledger()

    if "--list" in arguments:
        for gate in gates:
            state = "proven" if gate in registry else ("ratcheted" if gate in set(ratchet) else "UNRECORDED")
            print(f"{state:11s} {gate}")
        return 0

    failures = audit(gates, registry, ratchet, _read)
    for failure in failures:
        print(failure.render())

    if "--prove" in arguments:
        only = None
        if "--gate" in arguments:
            index = arguments.index("--gate")
            only = arguments[index + 1] if index + 1 < len(arguments) else None
            if not only:
                print("[FAIL] --gate needs a gate file name")
                return 2
        run_failures, notes = prove(registry, only)
        for note in notes:
            print(f"[note] {note}")
        for failure in run_failures:
            print(failure.render())
        failures = failures + run_failures

    if failures:
        print(
            f"\n{len(failures)} problem(s). Every check ships with a mutation that makes it fail, and "
            f"that mutation removes EVERY satisfier, not just the headline token -- a mutation that "
            f"leaves a second way to pass proves nothing."
        )
        return 1
    print(
        f"[  ok] mutation proofs: {len(registry)} gate(s) prove they can fail, "
        f"{len(ratchet)} ratcheted, {len(gates)} total."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
