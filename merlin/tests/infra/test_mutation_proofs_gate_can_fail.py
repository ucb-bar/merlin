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

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from zipfile import ZipFile

import pytest
from test_check_wiring import pinned_backend as pinned_backend  # noqa: F401 -- shared local Git fixture

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


# ------------------------------------------------------------- concrete CI-preparation witnesses


def test_a_skipped_boundary_case_changes_the_real_runner_verdict(tmp_path, pinned_backend, monkeypatch):
    """Process transport is synthetic; Git custody and JUnit admission are real."""
    monkeypatch.syspath_prepend(str(SCRIPTS))
    gate = _load("check_backend_boundaries")
    source, _, git = pinned_backend
    main_wheel = tmp_path / "merlin-0-py3-none-any.whl"
    with ZipFile(main_wheel, "w") as archive:
        archive.writestr("merlin-0.dist-info/METADATA", "Name: merlin\nVersion: 0\n")
    manifest = json.loads((repo_root() / "build_tools/upstreams/backend-boundaries.json").read_text())
    consumers = manifest.get("source_consumers", [])
    # Use the reviewed support roster as data, including later added consumers;
    # each selected path gets a tiny tracked fixture body, never original code.
    for row in consumers:
        for relative in row["files"]:
            path = source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# bounded static source transport fixture\n")
    if consumers:
        git("add", "--", ".")
        git("commit", "-q", "--no-verify", "-m", "Add selected static consumer fixture paths")
    rows = manifest["backends"]
    for row in rows:
        row.update(
            revision=git("rev-parse", "HEAD"),
            xdsl="0.68.0",
            provider_module="fixture",
            tests=["python/fixture/consumer.py"],
            fixtures=[],
            selection="",
            installed_service=False,
        )
    for row in consumers:
        row["revision"] = git("rev-parse", "HEAD")
    pins = tmp_path / "boundary-pins.json"
    pins.write_text(json.dumps(manifest))
    mutate = False

    def process_fixture(argv, *, cwd, log, env, records):
        # No provider build, pip install or pytest child is executed. Keep the
        # runner's actual source checks, fixture transport and XML admission.
        log.write_text("bounded process transport fixture\n")
        records.append({"argv": argv, "returncode": 0, "log": str(log)})
        if "build" in argv:
            destination = Path(argv[argv.index("--outdir") + 1])
            with ZipFile(destination / "fixture-0-py3-none-any.whl", "w") as archive:
                archive.writestr("fixture/__init__.py", "")
        elif "--junitxml" in argv:
            junit = Path(argv[argv.index("--junitxml") + 1])
            skipped = int(mutate and junit.name == rows[0]["id"] + ".xml")
            junit.write_text(
                f'<testsuites><testsuite tests="1" failures="0" errors="0" skipped="{skipped}"/></testsuites>'
            )

    monkeypatch.setattr(gate, "run", process_fixture)
    sources = [value for row in [*rows, *consumers] for value in ("--source", f"{row['id']}={source}")]
    common = ["--wheel", str(main_wheel), "--pins", str(pins), "--python068", sys.executable, *sources]
    clean = tmp_path / "boundary-clean"
    assert gate.main([*common, "--output", str(clean)]) == 0
    report = json.loads((clean / "report.json").read_text())
    assert report["passed"] is True and len(report["lanes"]) == len(rows)
    assert all(row["counts"] == dict(tests=1, failures=0, errors=0, skipped=0) for row in report["lanes"])
    assert len(report["static_inputs"]) == len(consumers)
    for expected, observed in zip(consumers, report["static_inputs"], strict=True):
        assert {row["path"] for row in observed["files"]} == set(expected["files"])
    mutate = True
    changed = tmp_path / "boundary-skipped"
    assert gate.main([*common, "--output", str(changed)]) == 1
    report = json.loads((changed / "report.json").read_text())
    assert report["passed"] is False and "must execute without skips" in report["error"]
    assert not report["lanes"]  # The first mutated lane was never admitted.


def test_missing_reference_region_evidence_changes_the_real_readiness_verdict(tmp_path, monkeypatch):
    """Only the upstream analyzer transport is substituted; assess/main are real."""
    from merlin_experiments.phase1.feedback import reference_analysis

    gate = _load("check_reference_readiness")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "visible.dump").write_text("bounded analyzer transport fixture\n")
    facts, contract = tmp_path / "facts.json", tmp_path / "contract.json"
    facts.write_text("{}")
    contract.write_text("{}")
    report = dict(
        input=dict(path="visible.dump", sha256=hashlib.sha256((workspace / "visible.dump").read_bytes()).hexdigest()),
        total=2,
        coverage=dict(accelerator_sites=7, selected_region_sites=7),
        regions=[
            dict(
                stable_id=f"region-{index}",
                accelerator_sites=sites,
                mechanism_projection=dict(
                    semantic_region_status="established",
                    source_binary_correspondence="established",
                    compatible_comparison="established",
                    unclassified_instruction_classes=[],
                ),
                cca=dict(
                    provenance=dict(
                        axis_evidence={
                            "dispatch.loop_offloaded": dict(status="inferred", evidence=["selected-isa-role-roster"])
                        }
                    )
                ),
            )
            for index, sites in enumerate((3, 4))
        ],
    )

    def analyzer_transport(request, **kwargs):
        assert kwargs["workspace"] == workspace and kwargs["reference_root"] is None
        if request.get("operation") == "inventory":
            return copy.deepcopy(report), {}
        selected = next(row for row in report["regions"] if row["stable_id"] == request["region"])
        return dict(report, regions=[copy.deepcopy(selected)]), {}

    monkeypatch.setattr(reference_analysis, "_analyze", analyzer_transport)
    common = [
        "--workspace",
        str(workspace),
        "--facts",
        str(facts),
        "--contract",
        str(contract),
        "--target",
        "fixture",
        "--reference",
        "visible.dump",
    ]
    clean = tmp_path / "reference-clean"
    assert gate.main([*common, "--output", str(clean)]) == 0
    ready = json.loads((clean / "readiness.json").read_text())
    assert ready["launch_ready"] is True and ready["ready_regions"] == 2
    # Keep every declared region/site in the denominator; remove its only
    # semantic/correspondence evidence rather than dropping the difficult row.
    report["regions"][1]["mechanism_projection"] = {}
    changed = tmp_path / "reference-missing"
    assert gate.main([*common, "--output", str(changed)]) == 2
    blocked = json.loads((changed / "readiness.json").read_text())
    assert blocked["launch_ready"] is False and blocked["region_count"] == 2
    assert blocked["ready_regions"] == 1
    assert blocked["missing"]["complete_upstream_semantic_region"] == 1
    assert blocked["missing"]["source_binary_correspondence"] == 1
    assert (changed / "0-region-1.json").is_file()
