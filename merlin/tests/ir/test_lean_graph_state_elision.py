"""Pinned actual configuration omission; complete outputs and independent refusals."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from merlin.common.paths import repo_root
from merlin.verify import lean_graph_state as state
from merlin.verify import lean_graph_state_elision as checker
from merlin.verify.lean_graph_state_elision_kernel import check_query

REFERENCE = Path("/home/jeremy/Projects/merlin-graph-proof-intake")


def contract():
    common = dict(
        reads=[],
        writes=[],
        clobbers=[],
        ordered=True,
        pure_write=False,
        no_other_effects=False,
        non_failing=False,
        semantic_sha256=hashlib.sha256(b"PROVISIONAL explicit fixture operational meaning v1").hexdigest(),
    )

    def rule(name, arguments, **kwargs):
        return {**common, "instruction": name, "arguments": arguments, **kwargs}

    def arg(name):
        return {"argument": name}

    return {
        "schema": "merlin.graph-state-elision.decoder.v1",
        "configuration_identity": "independently-selected-fixture-persistent-fields",
        "qualification_identity": "PROVISIONAL-NOT-HARDWARE-APPROVED",
        "rules": [
            rule(
                "setup",
                ["value"],
                writes=[["mode", arg("value")]],
                pure_write=True,
                no_other_effects=True,
                non_failing=True,
            ),
            rule(
                "pair",
                ["mode", "width"],
                writes=[["mode", arg("mode")], ["width", arg("width")]],
                pure_write=True,
                no_other_effects=True,
                non_failing=True,
            ),
            rule(
                "slot",
                ["index", "value"],
                writes=[[{"prefix": "slot", "argument": "index"}, arg("value")]],
                pure_write=True,
                no_other_effects=True,
                non_failing=True,
            ),
            rule("work", ["required", "add"], reads=[["mode", arg("required")]]),
            rule("reset", [], clobbers=["mode"]),
            rule("pulse", ["value"], writes=[["mode", arg("value")]]),
            rule("unordered", [], ordered=False),
            rule("observe", ["value"]),
        ],
    }


def kernel(*calls):
    return "kernel commands()\n" + "".join("  " + call + "\n" for call in calls)


CASES = {
    "basic": kernel("setup(value=7)", "setup(value=7)", "work(required=7, add=3)"),
    "multiple-fields": kernel(
        "pair(mode=7, width=11)", "setup(value=7)", "pair(mode=7, width=11)", "work(required=7, add=5)"
    ),
    "partial-redundancy": kernel(
        "setup(value=7)",
        "pair(mode=7, width=11)",
        "pair(mode=7, width=11)",
        "pair(mode=7, width=13)",
        "work(required=7, add=5)",
    ),
    "clobber": kernel("setup(value=7)", "reset()", "setup(value=7)", "setup(value=7)", "work(required=7, add=3)"),
    "unknown-barrier": kernel(
        "setup(value=7)", "opaque(value=19)", "setup(value=7)", "setup(value=7)", "work(required=7, add=3)"
    ),
    "unordered-barrier": kernel("setup(value=7)", "unordered()", "setup(value=7)", "setup(value=7)"),
    "retained-observers": kernel(
        "observe(value=19)",
        "setup(value=7)",
        "pulse(value=7)",
        "setup(value=7)",
        "observe(value=23)",
        "work(required=7, add=3)",
    ),
    "dynamic-fields": kernel(
        "slot(index=2, value=-17)",
        "slot(index=3, value=18446744073709551617)",
        "slot(index=2, value=-17)",
        "slot(index=3, value=18446744073709551617)",
    ),
    "unknown-read": kernel("work(required=7, add=3)", "setup(value=7)", "setup(value=7)"),
    "annotations": kernel("setup(value=7) on engine", "setup(value=7)", "setup(value=7)"),
    "dependencies": kernel(
        "t = opaque(value=19)",
        "setup(value=7) after t",
        "setup(value=7) in scratch[0:2]w",
        "setup(value=7) sets extra=11",
        "setup(value=7) needs extra=11",
        "setup(value=7)",
        "setup(value=7)",
    ),
    "identity": kernel("setup(value=7)", "work(required=7, add=3)"),
    "floating-observer": kernel("observe(value=-0x0.0p+0)", "setup(value=7)", "setup(value=7)"),
    "caller-memory": "kernel commands(input: i8[8] read, output: i8[8] write)\n"
    "  attrs {invocation=persistent}\n"
    "  copy(input=&input[0], output=&output[0], count=8)\n"
    "  setup(value=7)\n  setup(value=7)\n"
    "  copy(input=&input[1], output=&output[1], count=7)\n",
}


def produce(root, text=None, selected=None, disabled=False):
    root.mkdir(parents=True, exist_ok=True)
    before, selection = root / "source.mk", root / "contract.json"
    before.write_text(CASES["basic"] if text is None else text)
    selection.write_text(json.dumps(contract() if selected is None else selected, indent=2) + "\n")
    destination = root / "actual"
    command = [
        sys.executable,
        str(repo_root() / "src/merlin/verify/lean_graph_state_elision_capture.py"),
        "--reference",
        str(REFERENCE),
        "--source",
        str(before),
        "--contract",
        str(selection),
        "--destination",
        str(destination),
    ]
    if disabled:
        command.append("--disabled")
    p = subprocess.run(
        command,
        cwd=REFERENCE,
        env={**os.environ, "PYTHONPATH": str(REFERENCE / "src"), "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    (root / "producer.stdout").write_text(p.stdout)
    (root / "producer.stderr").write_text(p.stderr)
    assert p.returncode == 0, p.stderr
    return before, destination / "after.mk", selection, json.loads((destination / "projection.json").read_text())


def checked(root, before, after, selected, **kwargs):
    return checker.check_boundary(
        before,
        after,
        root / "checked",
        reference_root=REFERENCE,
        contract=selected,
        original_contract_sha256=state.file_ref(selected)["sha256"],
        **kwargs,
    )


@pytest.mark.parametrize("case", list(CASES))
def test_actual_complete_elision(case, tmp_path):
    before, after, selected, projection = produce(tmp_path, CASES[case])
    assert projection["projection_checked"], projection
    report = checked(tmp_path, before, after, selected)
    assert report["status"] == "conditional_model_checked", report
    assert report["genuine_changed_boundary"] is (case != "identity")
    if case == "unknown-read":
        assert report["read_establishment"]["status"] == "not_established"
    data = json.loads(Path(report["kernel"]["path"]).read_text())
    assert data["audit"]["raw"]["violations"] == []
    assert set(data["audit"]["raw"]["axioms_used"]) <= {"propext", "Classical.choice", "Quot.sound"}
    for flag in ("public_acceptance_issued", "admission_approved", "native_fidelity_proved", "source_success_proved"):
        assert report[flag] is False


def test_disabled_is_identity_separate_from_changed(tmp_path):
    before, after, selected, projection = produce(tmp_path, disabled=True)
    assert projection["selected_enabled"] is False and projection["removed_calls"] == 0
    assert checked(tmp_path, before, after, selected)["genuine_changed_boundary"] is False


MUTATIONS = {
    "first-setter": ("basic", lambda s: s.replace("  setup(value=7)\n", "", 1)),
    "different-value": ("basic", lambda s: s.replace("setup(value=7)", "setup(value=8)")),
    "changed-read": ("basic", lambda s: s.replace("required=7", "required=8")),
    "changed-operand": ("basic", lambda s: s.replace("add=3", "add=4")),
    "operand-order": ("basic", lambda s: s.replace("required=7, add=3", "add=3, required=7")),
    "missing-operand": ("basic", lambda s: s.replace("required=7, add=3", "required=7")),
    "missing-user": ("basic", lambda s: s.replace("  work(required=7, add=3)\n", "")),
    "different-function": ("basic", lambda s: s.replace("work(", "other(")),
    "interface": ("basic", lambda s: s.replace("commands()", "changed()")),
    "extra-observer": ("basic", lambda s: s + "  observe(value=23)\n"),
    "partial-write": ("partial-redundancy", lambda s: s.replace("pair(mode=7, width=13)", "setup(value=7)")),
    "changed-multifield": ("partial-redundancy", lambda s: s.replace("width=13", "width=11")),
    "missing-clobber": ("clobber", lambda s: s.replace("  reset()\n", "")),
    "cross-clobber": ("clobber", lambda s: s.replace("  reset()\n  setup(value=7)\n", "  reset()\n")),
    "missing-unknown": ("unknown-barrier", lambda s: s.replace("  opaque(value=19)\n", "")),
    "cross-unknown": (
        "unknown-barrier",
        lambda s: s.replace("  opaque(value=19)\n  setup(value=7)\n", "  opaque(value=19)\n"),
    ),
    "cross-unordered": (
        "unordered-barrier",
        lambda s: s.replace("  unordered()\n  setup(value=7)\n", "  unordered()\n"),
    ),
    "missing-pulse": ("retained-observers", lambda s: s.replace("  pulse(value=7)\n", "")),
    "outside-observer": ("retained-observers", lambda s: s.replace("value=23", "value=24")),
    "retained-swap": (
        "retained-observers",
        lambda s: s.replace(
            "  observe(value=23)\n  work(required=7, add=3)\n", "  work(required=7, add=3)\n  observe(value=23)\n"
        ),
    ),
    "unit-dependency": ("annotations", lambda s: s.replace(" on engine", "")),
    "token-dependency": ("dependencies", lambda s: s.replace(" after t", "")),
    "stage-dependency": ("dependencies", lambda s: s.replace(" in scratch[0:2]w", "")),
    "set-dependency": ("dependencies", lambda s: s.replace(" sets extra=11", "")),
    "assumes-dependency": ("dependencies", lambda s: s.replace(" needs extra=11", "")),
    "attributes": (
        "basic",
        lambda s: s.replace("kernel commands()\n", "kernel commands()\n  attrs {policy=changed}\n"),
    ),
    "floating-observer": ("floating-observer", lambda s: s.replace("value=-0x0.0p+0", "value=0x0.0p+0")),
    "memory-address": ("caller-memory", lambda s: s.replace("output=&output[1]", "output=&output[2]")),
    "memory-signature": ("caller-memory", lambda s: s.replace("input: i8[8] read", "input: i8[9] read")),
    "memory-access": ("caller-memory", lambda s: s.replace("i8[8] write", "i8[8] read")),
    "memory-attrs": ("caller-memory", lambda s: s.replace("invocation=persistent", "invocation=cleared")),
}


@pytest.mark.parametrize("kind", list(MUTATIONS))
def test_valid_actual_output_mutation_refused(kind, tmp_path):
    case, mutate = MUTATIONS[kind]
    before, after, selected, projection = produce(tmp_path, CASES[case])
    assert projection["projection_checked"], projection
    changed = tmp_path / "mutation.mk"
    changed.write_text(mutate(after.read_text()))
    assert changed.read_bytes() != after.read_bytes()
    report = checked(tmp_path, before, changed, selected)
    assert report["status"] == "unsupported", report
    detail = json.loads((tmp_path / "checked/projection/projection.json").read_text())
    assert "structurally valid" not in detail["reason"]["detail"], detail
    if kind == "floating-observer":
        assert "production_conformance_before_independent_refusal" in detail


@pytest.mark.parametrize(
    "kind", ["first", "wrong-value", "operand", "field", "unapproved", "unknown", "duplicate", "reorder"]
)
def test_lean_checker_independently_refuses_valid_data_mutations(kind, tmp_path):
    _, _, _, data = produce(tmp_path)
    projection = copy.deepcopy(data)
    if kind == "first":
        projection["after"] = projection["after"][1:]
    elif kind == "wrong-value":
        projection["before"][1]["effect"]["writes"][0][1] = 8
    elif kind == "operand":
        projection["after"][-1]["token"] += " altered"
    elif kind == "field":
        projection["before"][0]["effect"]["writes"][0][0] = 1
    elif kind == "unapproved":
        projection["before"][1]["eraseAllowed"] = False
    elif kind == "unknown":
        projection["before"][0]["effect"] = None
        projection["after"][0]["effect"] = None
    elif kind == "duplicate":
        projection["after"].append(projection["after"][-1])
    else:
        projection["after"].reverse()
    report = check_query(checker.emit_query(projection), tmp_path / "refused")
    assert report["returncode"] != 0
    assert "decide" in (tmp_path / "refused/kernel.stdout").read_text()


@pytest.mark.parametrize("kind", ["known-read-conflict", "duplicate-field", "float-value", "loop", "malformed"])
def test_explicit_unsupported_and_malformed_intake(kind, tmp_path):
    data, text = contract(), CASES["basic"]
    if kind == "known-read-conflict":
        text = text.replace("required=7", "required=8")
    elif kind == "duplicate-field":
        data["rules"][0]["writes"] *= 2
    elif kind == "float-value":
        text = text.replace("value=7", "value=7.0")
    elif kind == "loop":
        text = "kernel commands()\n  for i in 0..3:\n    setup(value=7)\n"
    else:
        text = "not a kernel"
    _, _, _, projection = produce(tmp_path, text, data)
    assert projection["projection_checked"] is False


def test_original_contract_cannot_be_rebound(tmp_path):
    before, after, selected, _ = produce(tmp_path)
    identity = state.file_ref(selected)["sha256"]
    selected.write_text(selected.read_text().replace('"non_failing": true', '"non_failing": false'))
    with pytest.raises(ValueError, match="original contract identity"):
        checker.check_boundary(
            before,
            after,
            tmp_path / "checked",
            reference_root=REFERENCE,
            contract=selected,
            original_contract_sha256=identity,
        )


@pytest.mark.parametrize("kind", ["boolean-index", "float-index", "integer-flag", "boolean-value"])
def test_exact_typed_certificate_intake(kind, tmp_path):
    _, _, _, projection = produce(tmp_path)
    if kind == "boolean-index":
        projection["before"][0]["id"] = True
    elif kind == "float-index":
        projection["before"][0]["id"] = 0.0
    elif kind == "integer-flag":
        projection["before"][0]["eraseAllowed"] = 1
    else:
        projection["before"][0]["effect"]["writes"][0][1] = True
    with pytest.raises(ValueError):
        checker.emit_query(projection)


def test_pure_metadata_cannot_supply_missing_total_overwrite_admission(tmp_path):
    selected = contract()
    selected["rules"][0]["non_failing"] = False
    before, after, contract_path, projection = produce(tmp_path, selected=selected)
    assert projection["projection_checked"] and projection["removed_calls"] == 1
    # Actual production trusts pure_write; independent checker also needs the
    # explicit local totality/whole-observation law. No approval is manufactured.
    result = checked(tmp_path, before, after, contract_path)
    assert result["status"] == "unavailable"
    assert "decide" in (tmp_path / "checked/kernel/kernel.stdout").read_text()
