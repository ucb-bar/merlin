"""The meta-gate must be able to fail, or it is the funniest instance of the bug it is for.

``check_mutation_proofs`` requires every gate to ship a recorded, runnable mutation that makes it
fail. A meta-gate that could not itself fail would assert that property about the whole repo on no
evidence -- a reading whose ambiguous answer is the passing one, at the top of the stack.

Each test below mutates ONE thing in an otherwise valid roster. The mutation removes every satisfier
for the failure it provokes, and the first test is the reason that claim is checkable: a roster in
which nothing is wrong must produce NO failures, so a gate that returned a complaint unconditionally
would fail the suite rather than pass it.
"""

from __future__ import annotations

import importlib.util
import sys

import pytest

from merlin.common.paths import repo_root

SCRIPTS = repo_root() / "build_tools" / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


MP = _load("check_mutation_proofs")

GATES = ["build_tools/scripts/check_alpha.py", "build_tools/scripts/check_beta.py"]
PROOF_PATH = "merlin/tests/infra/test_alpha_gate_can_fail.py"
PROOF_SOURCE = (
    "def test_alpha_reports_a_bad_row():\n    module = load('check_alpha')\n    assert module.audit([{'bad': True}])\n"
)
MUTATION = "Feed audit() one row whose declared status is absent from the verdict table."
REMOVES = "Both satisfiers: the row is not ratcheted and the status maps to no clean bucket."


def _registry(**overrides) -> dict[str, dict]:
    entry = {"proof": f"{PROOF_PATH}::test_alpha_reports_a_bad_row", "mutation": MUTATION, "removes": REMOVES}
    entry.update(overrides)
    return {"build_tools/scripts/check_alpha.py": entry}


def _read(source: str = PROOF_SOURCE):
    return lambda path: source if path == PROOF_PATH else None


def _subjects(failures) -> list[str]:
    return [f.subject for f in failures]


def test_a_roster_with_nothing_wrong_produces_no_failures():
    """THE TEST THAT MAKES THE REST MEAN SOMETHING.

    Without it, every assertion below is satisfied by a gate that complains unconditionally, which is
    a check that cannot pass -- the mirror image of a check that cannot fail, and just as useless.
    """
    assert MP.audit(GATES, _registry(), ["build_tools/scripts/check_beta.py"], _read()) == []


def test_a_gate_with_no_recorded_mutation_is_reported():
    """THE PRIMARY MUTATION: a gate ships, and nobody ever said what makes it fail."""
    assert _subjects(MP.audit(GATES, _registry(), [], _read())) == ["build_tools/scripts/check_beta.py"]


def test_a_proof_file_that_does_not_exist_is_reported():
    """A registry entry is a claim about a runnable test; an absent file is a claim about nothing."""
    failures = MP.audit(GATES, _registry(), ["build_tools/scripts/check_beta.py"], lambda path: None)
    assert "does not exist" in failures[0].message


def test_a_proof_whose_test_function_is_not_there_is_reported():
    """The file survived a rename, the test inside it did not, and the entry still reads as proof."""
    failures = MP.audit(
        GATES,
        _registry(),
        ["build_tools/scripts/check_beta.py"],
        _read("def test_something_else():\n    assert True\n"),
    )
    assert "no test function named" in failures[0].message


def test_a_proof_that_never_names_its_gate_is_reported():
    """THE COUPLING TEETH.

    A proof that reimplements the gate's decision inside the test passes forever while the gate rots.
    It is the same defect as the capability line whose corrected citations were left in the comment
    block: something else in the picture satisfies the check, so the check passes and proves nothing
    about the subject. The mutation here removes the only satisfier -- every structural mention of
    `check_alpha` -- rather than, say, only the import.
    """
    failures = MP.audit(
        GATES,
        _registry(),
        ["build_tools/scripts/check_beta.py"],
        _read("def test_alpha_reports_a_bad_row():\n    assert [] == []\n"),
    )
    assert "never names check_alpha" in failures[0].message


def test_removes_that_merely_restates_mutation_is_reported():
    """`removes:` is a different question, and answering it with the same sentence answers neither."""
    failures = MP.audit(GATES, _registry(removes=MUTATION), ["build_tools/scripts/check_beta.py"], _read())
    assert "restates mutation" in failures[0].message


@pytest.mark.parametrize("field", ["mutation", "removes"])
def test_a_one_word_field_is_reported(field):
    failures = MP.audit(GATES, _registry(**{field: "yes"}), ["build_tools/scripts/check_beta.py"], _read())
    assert failures and failures[0].message.startswith(field)


def test_a_proof_outside_the_test_suite_is_reported():
    """A proof the suite never collects is a proof nobody runs -- the unwired-gate defect, again."""
    failures = MP.audit(
        GATES,
        _registry(proof="scratch/probe.py::test_alpha_reports_a_bad_row"),
        ["build_tools/scripts/check_beta.py"],
        _read(),
    )
    assert "must live under merlin/tests/" in failures[0].message


def test_a_gate_both_proven_and_ratcheted_is_reported():
    failures = MP.audit(GATES, _registry(), GATES, _read())
    assert "both proven and ratcheted" in failures[0].message


def test_a_ledger_entry_for_a_gate_that_is_gone_is_reported():
    """So the ledger cannot rot into an allowlist of names nothing checks any more."""
    failures = MP.audit(
        GATES, _registry(), ["build_tools/scripts/check_beta.py", "build_tools/scripts/check_deleted.py"], _read()
    )
    assert _subjects(failures) == ["build_tools/scripts/check_deleted.py"]


def test_a_registry_entry_for_a_gate_that_is_gone_is_reported():
    registry = _registry()
    registry["build_tools/scripts/check_deleted.py"] = registry["build_tools/scripts/check_alpha.py"]
    failures = MP.audit(GATES, registry, ["build_tools/scripts/check_beta.py"], _read())
    assert _subjects(failures) == ["build_tools/scripts/check_deleted.py"]


# ------------------------------------------------------------------- the execution tier can fail


@pytest.mark.slow
def test_prove_reports_a_proof_that_passes_without_ever_entering_its_gate():
    """`--prove` is the half that runs things, so it needs its own failing direction.

    A coupling check that never fires is a check that cannot fail. Here a perfectly good test is
    recorded as the proof of an unrelated gate: it passes, so the run-it-and-see half is satisfied,
    and only the frame-level trace can tell that the gate it claims to prove was never entered. That
    is the mutation -- and it removes the only satisfier, because the named test touches nothing in
    check_wiring at all, not merely its import.
    """
    registry = {
        "build_tools/scripts/check_wiring.py": {
            "proof": (
                "merlin/tests/infra/test_silent_defaults_gate_can_fail.py"
                "::test_a_ledger_entry_whose_finding_is_gone_fails"
            )
        }
    }
    failures, _notes = MP.prove(registry, None)
    assert failures, "an uncoupled proof was accepted; the coupling check cannot fail"
    assert "without ever entering" in failures[0].message


@pytest.mark.slow
def test_prove_reports_a_proof_that_does_not_pass():
    registry = {
        "build_tools/scripts/check_wiring.py": {
            "proof": "merlin/tests/infra/test_check_wiring.py::test_a_function_that_is_not_there"
        }
    }
    failures, _notes = MP.prove(registry, None)
    assert failures and "does not pass" in failures[0].message


# ------------------------------------------------------------------------------ the live roster


def test_the_live_roster_covers_every_gate_in_the_repo():
    """The real thing, over the real tree: no gate may be neither proven nor ratcheted."""
    assert MP.audit(MP.gate_files(), MP.load_registry(), MP.load_ledger(), MP._read) == []


def test_the_live_scan_sees_the_real_gates():
    """A gate list that resolved to nothing would satisfy every assertion above vacuously."""
    assert len(MP.gate_files()) > 20
    assert MP.load_registry(), "the registry parsed to nothing; every gate would read as ratcheted-or-absent"
