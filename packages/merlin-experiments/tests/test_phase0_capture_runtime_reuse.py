"""Ordinary Phase 0 freezes an explicit existing runtime selection without new authority."""

import copy
import hashlib
import json
import shutil

import pytest
from merlin_experiments.capture_execution import readonly_runtime as R
from merlin_experiments.capture_execution import runtime_store as STORE
from merlin_experiments.capture_execution.sealed_m2m import _snapshot_tree
from merlin_experiments.phase0 import freeze
from merlin_experiments.phase0 import m2m_runtime as M


def fixture(tmp_path, monkeypatch):
    entry = tmp_path / "store/initial"
    for name in (
        "opt/capture-venv/bin/python",
        "opt/capture-venv/lib/python3.12/site-packages/torch/_Cfixture.so",
        "opt/capture-venv/lib/python3.12/site-packages/numpy/_core/_multiarray_umathfixture.so",
        "opt/base/lib/python3.12/selected.py",
    ):
        path = entry / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic runtime member; not executed")
    venv = tmp_path / "selected-venv"
    shutil.copytree(entry / "opt/capture-venv", venv)
    python = venv / "bin/python"

    def runtime(_python):
        assert _python == python
        return {
            "base": "/opt/base",
            "venv": _snapshot_tree(venv),
            "base_python": _snapshot_tree(entry / "opt/base"),
            "python_sha256": hashlib.sha256(python.read_bytes()).hexdigest(),
        }

    initial = runtime(python)
    plan = {
        "base": initial["base"],
        "system_libs": [],
        "selected_trees": {"venv": initial["venv"], "base": initial["base_python"]},
    }
    dest = entry.parent / STORE._identity(plan)
    entry.rename(dest)
    entry = dest
    marker = entry.parent / (entry.name + ".complete.json")
    marker.write_text(json.dumps({"identity": entry.name, "base": plan["base"], "system_libs": []}))
    selection = R.select(plan, entry=entry, marker_sha256=M._file_digest(marker), destination=tmp_path)
    selected = {"root": str(tmp_path / "m2m"), "python": str(python), "package": {}, **runtime(python)}
    monkeypatch.setattr(M, "_runtime", runtime)
    from merlin_experiments.capture_execution import sealed_m2m

    monkeypatch.setattr(sealed_m2m, "_system_libs", lambda *args: ())
    return selected, selection, entry, marker, venv


def test_exact_existing_runtime_forwarding_preserves_legacy_config(tmp_path, monkeypatch):
    selected, selection, *_ = fixture(tmp_path, monkeypatch)
    destination = tmp_path / "fresh/run/phase0"
    default = M.sealed_capture_config(selected, destination)
    assert "runtime_reuse" not in default
    actual = M.sealed_capture_config(selected, destination, runtime_reuse=selection)
    assert actual == {**default, "runtime_reuse": selection}
    assert actual["runtime_reuse"] is not selection
    assert not destination.exists()  # resolve/preflight never creates a cache/run


@pytest.mark.parametrize("mutation", ["source", "marker", "cache", "selection", "python"])
def test_stale_foreign_or_mutated_runtime_refuses(tmp_path, monkeypatch, mutation):
    selected, selection, entry, marker, venv = fixture(tmp_path, monkeypatch)
    if mutation == "source":
        (venv / "new.py").write_bytes(b"unselected namespace member")
    elif mutation == "marker":
        marker.write_bytes(marker.read_bytes() + b" ")
    elif mutation == "cache":
        (entry / "opt/capture-venv/bin/python").write_bytes(b"changed")
    elif mutation == "selection":
        selection["inventory_sha256"] = "0" * 64
    else:
        selected["python_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        M.sealed_capture_config(selected, tmp_path / "new", runtime_reuse=selection)


def pinned_plan(path, value):
    path.write_text(json.dumps(value))
    key = "phase0:capture-runtime-reuse-selection"
    return {
        "input_paths": {key: str(path)},
        "inputs": {key: {"path": str(path), "sha256": freeze.fingerprint(path)}},
        "phases": {"0": {"phase0_capture_runtime_reuse": copy.deepcopy(value)}},
    }


@pytest.mark.parametrize("copied", [False, True])
def test_frozen_selection_data_reopens_its_exact_ordinary_input(tmp_path, copied):
    value = {"schema": R.SCHEMA, "entry": "selected"}
    path = tmp_path / ("snapshot/inputs/selection.json" if copied else "selection.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    plan = pinned_plan(path, value)
    assert freeze._runtime_reuse(plan, plan["phases"]["0"]) == value
    plan["phases"]["0"]["phase0_capture_runtime_reuse"]["entry"] = "substitution"
    plan["phases"]["0"]["env"] = {"MERLIN_PHASE0_SEALED_CAPTURE": "matching substituted config"}
    with pytest.raises(ValueError, match="pinned input"):
        freeze._verify_frozen_sources(plan)  # refuses before even reading source/CAS


@pytest.mark.parametrize("defect", ["missing_pin", "missing_data", "raw_change", "path_change"])
def test_selection_file_pin_is_mandatory_in_both_directions(tmp_path, defect):
    plan = pinned_plan(tmp_path / "selection.json", {"schema": R.SCHEMA})
    key = "phase0:capture-runtime-reuse-selection"
    if defect == "missing_pin":
        plan["inputs"].pop(key)
    elif defect == "missing_data":
        plan["phases"]["0"].pop("phase0_capture_runtime_reuse")
    elif defect == "raw_change":
        (tmp_path / "selection.json").write_bytes(b"{}")
    else:
        plan["input_paths"][key] = str(tmp_path / "foreign.json")
    with pytest.raises(ValueError, match="pinned input"):
        freeze._runtime_reuse(plan, plan["phases"]["0"])
    assert freeze._runtime_reuse({}, {}) is None


@pytest.mark.parametrize("raw", [b'{"a":1,"a":2}', b'{"x":NaN}', b"[]", b" " * (1024 * 1024 + 1)])
def test_selection_reader_stays_strict_and_bounded(tmp_path, raw):
    path = tmp_path / "selection.json"
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        M.read_runtime_reuse(path)


def test_selection_reader_rejects_indirect_inputs(tmp_path):
    path = tmp_path / "selection.json"
    path.write_bytes(b"{}")
    alias = tmp_path / "alias.json"
    alias.symlink_to(path)
    with pytest.raises(ValueError, match="ordinary"):
        M.read_runtime_reuse(alias)


def functional_profile(tmp_path):
    import yaml
    from test_public_functional_sources import derive

    entries, products, _ = derive()
    for name, raw in products.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw if isinstance(raw, bytes) else raw.encode())
    profile = tmp_path / "synthesis.yaml"
    profile.write_text(yaml.safe_dump({"capsules": entries}))
    return profile, entries


def test_generated_functional_loaders_belong_to_pinned_input_tree(tmp_path):

    from merlin_experiments.phase0 import freeze

    profile, entries = functional_profile(tmp_path)
    root = tmp_path / "selected-m2m"
    assert M._workload_names(root, profile) == ()
    selected = freeze._materialized_inputs({"inputs": {"synth_profile": str(profile)}})
    for entry in entries:
        assert tmp_path / entry["loader"] in selected
        assert tmp_path / entry["public_boundary"]["source_context"]["path"] in selected
    # Ordinary external model loaders still require a selected workload owner.
    import yaml

    profile.write_text(yaml.safe_dump({"capsules": [{"kind": "model", "loader": str(tmp_path / "other.py")}]}))
    with pytest.raises(ValueError, match="outside selected Model2MLIR"):
        M._workload_names(root, profile)


@pytest.mark.parametrize("changed", ["loader", "source_context", "program", "policy"])
def test_generated_functional_loader_never_skips_its_source_checks(tmp_path, changed):
    import yaml

    profile, entries = functional_profile(tmp_path)
    member = entries[0]["public_boundary"]
    if changed == "loader":
        (tmp_path / entries[0]["loader"]).write_bytes(b"unrelated source")
    elif changed == "source_context":
        (tmp_path / member["source_context"]["path"]).write_bytes(b"{}")
    elif changed == "program":
        member["program"]["nodes"][0]["target"] = "aten.neg.default"
    else:
        member["selection"]["max_nodes"] += 1
    profile.write_text(yaml.safe_dump({"capsules": entries}))
    with pytest.raises(ValueError):
        M._workload_names(tmp_path / "selected-m2m", profile)
