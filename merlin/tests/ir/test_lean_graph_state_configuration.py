"""Actual persistent-configuration hoisting, complete correspondence and refusals."""

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
from merlin.verify import lean_graph_state_configuration as checker

REFERENCE = Path("/home/jeremy/Projects/merlin-graph-proof-intake")


def contract():
    common = {
        "deterministic_total": True,
        "complete_state_effects": True,
        "sequential_atomic_complete": True,
        "ordered": True,
        "semantic_sha256": hashlib.sha256(b"provisional exact fixture state semantics").hexdigest(),
        "configuration_reads": [],
        "configuration_writes": [],
        "clobbers": [],
        "state_reads": [],
        "state_writes": [],
        "pure_write": False,
    }
    return {
        "configuration_identity": "fixture-independent-persistent-field-mode",
        "qualification_identity": "PROVISIONAL-model-only-not-hardware-qualified",
        "rules": [
            {
                **common,
                "instruction": "setup",
                "arguments": ["value"],
                "configuration_writes": [["mode", "value"]],
                "state_writes": ["configuration:mode"],
                "pure_write": True,
            },
            {
                **common,
                "instruction": "compute",
                "arguments": ["address"],
                "configuration_reads": [["mode", 7]],
                "state_reads": ["configuration:mode", "observations"],
                "state_writes": ["observations"],
            },
            {
                **common,
                "instruction": "surrounding",
                "arguments": ["value"],
                "state_reads": ["observations"],
                "state_writes": ["observations"],
            },
            {
                **common,
                "instruction": "reset",
                "arguments": [],
                "clobbers": ["mode"],
                "state_writes": ["configuration:mode"],
            },
        ],
    }


def source(extent=3, stride=4):
    return f"""kernel commands()
  surrounding(value=19)
  for i in 0..{extent}:
    setup(value=7)
    compute(address=(i * {stride}))
  surrounding(value=23)
"""


def write_contract(root, data=None):
    path = root / "contract.json"
    path.write_text(json.dumps(data or contract(), indent=2) + "\n")
    return path


def produce(root, text=None, selected=None):
    before = root / "source.mk"
    before.write_text(text or source())
    selected = selected or write_contract(root)
    destination = root / "actual"
    p = subprocess.run(
        [
            sys.executable,
            str(repo_root() / "src/merlin/verify/lean_graph_state_configuration_capture.py"),
            "--reference",
            str(REFERENCE),
            "--source",
            str(before),
            "--contract",
            str(selected),
            "--destination",
            str(destination),
        ],
        cwd=REFERENCE,
        env={**os.environ, "PYTHONPATH": str(REFERENCE / "src"), "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
    )
    (root / "producer.stdout").write_text(p.stdout)
    (root / "producer.stderr").write_text(p.stderr)
    assert p.returncode == 0, p.stderr
    data = json.loads((destination / "capture.json").read_text())
    return before, destination / "after.mk", selected, data


def checked(root, before, after, selected, **kwargs):
    return checker.check_boundary(
        before, after, root / "checked", reference_root=REFERENCE, contract=selected, **kwargs
    )


@pytest.mark.parametrize("extent,stride", [(1, 1), (3, 4), (17, 16)])
def test_actual_configuration_prefix_whole_program_checked(extent, stride, tmp_path):
    before, after, selected, data = produce(tmp_path, source(extent, stride))
    assert data["projection_checked"]
    assert data["producer"].endswith("hoist_loop_configuration")
    result = checked(tmp_path, before, after, selected, original_contract_sha256=state.file_ref(selected)["sha256"])
    assert result["status"] == "graph_model_checked"
    assert result["transitive_audit"]["status"] == "checked"
    assert not result["transitive_audit"]["raw"]["violations"]
    assert result["producer_verification"]["removed_dynamic_configuration_calls"] == extent - 1
    assert not result["source_contract_approved"] and not result["public_acceptance_issued"]


@pytest.mark.parametrize("kind", ["setup-value", "body-operand", "surrounding", "domain", "drop", "extra", "signature"])
def test_valid_actual_configuration_mutation_refused(kind, tmp_path):
    before, after, selected, _ = produce(tmp_path)
    text = after.read_text()
    if kind == "setup-value":
        text = text.replace("setup(value=7)", "setup(value=8)")
    elif kind == "body-operand":
        text = text.replace("address=(i * 4)", "address=(i * 8)")
    elif kind == "surrounding":
        text = text.replace("value=23", "value=24")
    elif kind == "domain":
        text = text.replace("0..3:", "0..2:")
    elif kind == "drop":
        text = text.replace("  surrounding(value=23)\n", "")
    elif kind == "extra":
        text += "  surrounding(value=29)\n"
    else:
        text = text.replace("kernel commands()", "kernel changed()")
    assert text != after.read_text(), kind
    changed = tmp_path / "changed.mk"
    changed.write_text(text)
    assert checked(tmp_path, before, changed, selected)["status"] == "unsupported"


@pytest.mark.parametrize("kind", ["zero", "clobber", "unknown", "variant", "async"])
def test_configuration_hoist_requires_nonempty_invariant_complete_domain(kind, tmp_path):
    text = source()
    if kind == "zero":
        text = source(0)
    elif kind == "clobber":
        text = text.replace("    compute", "    reset()\n    compute")
    elif kind == "unknown":
        text = text.replace("    compute", "    unknown()\n    compute")
    elif kind == "variant":
        text = text.replace("setup(value=7)", "setup(value=i)")
    else:
        text = text.replace("setup(value=7)", "ready = setup(value=7)")
    _, _, _, data = produce(tmp_path, text)
    assert not data["projection_checked"]


def test_configuration_contract_cannot_be_rebound(tmp_path):
    before, after, selected, _ = produce(tmp_path)
    original = state.file_ref(selected)["sha256"]
    data = contract()
    data["rules"][1]["state_writes"] = []
    write_contract(tmp_path, data)
    with pytest.raises(ValueError, match="contract differs"):
        checked(tmp_path, before, after, selected, original_contract_sha256=original)


@pytest.mark.parametrize("kind", ["empty-loop", "clobber", "wrong-target", "wrong-fields"])
def test_kernel_configuration_checker_rejects_bad_certificate(kind, tmp_path):
    _, _, _, captured = produce(tmp_path)
    assert captured["projection_checked"]
    projection = copy.deepcopy(captured)
    if kind == "empty-loop":
        projection["move"]["extent"] = 0
        projection["before"][1]["loop"] = 0
        projection["after"][2]["loop"] = 0
    elif kind == "clobber":
        projection["footprints"][2]["writes"] = projection["move"]["fields"]
    elif kind == "wrong-fields":
        projection["footprints"][1]["writes"] = []
    else:
        projection["after"].pop()
    query = tmp_path / "BadConfiguration.lean"
    query.write_text(checker.emit_query(projection))
    frozen = state.prepare()
    p = subprocess.run(
        [str(state.LEAN), str(query)],
        cwd=tmp_path,
        env={**os.environ, "LEAN_PATH": str(frozen["lib"])},
        capture_output=True,
        text=True,
    )
    (tmp_path / "kernel-refusal.stdout").write_text(p.stdout)
    (tmp_path / "kernel-refusal.stderr").write_text(p.stderr)
    assert p.returncode != 0 and "decide" in p.stdout
