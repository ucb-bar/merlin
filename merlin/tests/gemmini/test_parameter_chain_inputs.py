"""Pure original-input/reference refusals; no compiler, capture or simulator."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import packing_execution_inputs as P
import parameter_chain_execution_inputs as E
import parameter_chain_reference as R
import pytest


def test_ordered_integer_reference_and_complete_roster():
    x = np.array([[-128, 127]], dtype=np.int8)
    weights = {f"w{i}": np.array([[127, -128], [-1, 2]], dtype=np.int8) for i in range(3)}
    expected, state = None, x.tolist()[0]
    for name in ("w0", "w1", "w2"):
        wide = [sum(state[k] * int(weights[name][k, j]) for k in range(2)) for j in range(2)]
        state = [min(127, max(-128, (v + 4) // 8)) for v in wide]
        expected = (np.array([wide], dtype=np.int32), np.array([state], dtype=np.int8))
    actual, prefixes = R.ordered_reference(x, weights)
    R.exact_outputs(actual, expected)
    assert len(prefixes) == 3 and all(p["prefix_pairs"] == 4 for p in prefixes)
    for bad in (actual[:1], actual[::-1], (actual[0].astype(np.int64), actual[1])):
        with pytest.raises(AssertionError):
            R.exact_outputs(bad, expected)


@pytest.mark.parametrize("change", ["input", "wide", "state", "missing_output", "missing_input"])
def test_captured_values_are_not_coerced_or_truncated(change):
    x = np.array([[1]], dtype=np.int8)
    wanted = (np.array([[2]], dtype=np.int32), np.array([[3]], dtype=np.int8))
    captured = SimpleNamespace(inputs=[x.tolist()], golden=[a.tolist() for a in wanted])
    if change == "missing_input":
        captured.inputs = []
    elif change == "missing_output":
        captured.golden.pop()
    elif change == "input":
        captured.inputs[0][0][0] += 256
    else:
        captured.golden[0 if change == "wide" else 1][0][0] += 1 << (32 if change == "wide" else 8)
    with pytest.raises(AssertionError):
        R.captured_values(captured, x, wanted)


@pytest.fixture
def operator_selection(tmp_path, monkeypatch):
    def asset(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(data)
        return P.pin(path)

    quiet = asset(tmp_path / "quiet.yaml", "name: selected_target\n")
    execution_pin = asset(tmp_path / "execution.json", "{}")
    monkeypatch.setattr(P, "read_selection", lambda p: ({"quiet_contract": quiet, "facts": quiet}, P.pin(Path(p))))
    original = {"run_dir": str(tmp_path / "original"), "plan": {"m2m_root": str(tmp_path / "m2m")}}
    selection_pin = asset(tmp_path / "original-selection.json", json.dumps(original))
    asset(tmp_path / "m2m/m2m/api.py", "# selected source\n")
    asset(tmp_path / "original/snapshots/source/workload/loader.py", "# sealed loader\n")
    attestation = asset(
        tmp_path / "attestation.json", json.dumps({"schema": "not_an_original_execution", "selection": selection_pin})
    )
    public = {name: asset(tmp_path / "public" / name, "public fixture") for name in R.PUBLIC_FILES}
    python = tmp_path / "venv/bin/python"
    worker = {
        "python": str(python),
        "binary": asset(python, "explicit selected tool"),
        "config": asset(tmp_path / "venv/pyvenv.cfg", "version_info = 3.12.13\n"),
        "torch_version": asset(
            tmp_path / "venv/lib/python3.12/site-packages/torch/version.py", "__version__ = 'selected'\n"
        ),
        "torch_init": asset(tmp_path / "venv/lib/python3.12/site-packages/torch/__init__.py", "# selected torch\n"),
    }
    profile = asset(
        tmp_path / "profile.yaml", "datapath:\n  required_oracle_tiers:\n  - L0\n  - L1\n  - L2\n  - L3\ncapsules: []\n"
    )
    document = dict(
        schema=E.SCHEMA,
        execution=execution_pin,
        attestation=attestation,
        source_member="loader.py",
        public_case=public,
        worker=worker,
        obligations_profile=profile,
    )
    path = tmp_path / "operator.json"
    path.write_text(json.dumps(document))
    return path, document


def test_closed_selection_is_data_not_capture_authority(operator_selection, tmp_path, monkeypatch):
    path, _ = operator_selection
    selected = E.read_selection(path)
    E.recheck(selected)
    from merlin_experiments.phase0 import sealed_native_program as N

    calls = []
    monkeypatch.setattr(N, "read", lambda *args, **kwargs: calls.append("must not read unverified capture"))
    with pytest.raises((ValueError, KeyError, AssertionError)):
        E.prepare(selected, tmp_path / "intake", monkeypatch)
    assert calls == []


@pytest.mark.parametrize("role", ["source", "weights", "reference", "capture", "worker", "profile"])
def test_changed_original_input_refuses(operator_selection, role):
    path, doc = operator_selection
    selected = E.read_selection(path)
    row = {
        "source": doc["public_case"]["capsule.interface.mlir"],
        "weights": doc["public_case"]["capsule.weights.safetensors"],
        "reference": doc["public_case"]["independent-reference.npz"],
        "capture": doc["attestation"],
        "worker": doc["worker"]["binary"],
        "profile": doc["obligations_profile"],
    }[role]
    Path(row["path"]).write_text("changed original bytes")
    with pytest.raises(AssertionError):
        E.recheck(selected)


@pytest.mark.parametrize(
    "change", ["unknown", "callback", "missing_public", "missing_worker", "capture_status", "source_escape"]
)
def test_no_callback_approval_or_incomplete_roster_selection(operator_selection, change):
    path, old = operator_selection
    doc = copy.deepcopy(old)
    if change in ("unknown", "callback", "capture_status"):
        doc[change] = "selected"
    elif change == "missing_public":
        doc["public_case"].pop("capsule.weights.safetensors")
    elif change == "missing_worker":
        doc["worker"].pop("binary")
    else:
        doc["source_member"] = "../loader.py"
    path.write_text(json.dumps(doc))
    with pytest.raises(AssertionError):
        E.read_selection(path)


def test_all_required_obligations_preserved(operator_selection):
    path, doc = operator_selection
    profile = Path(doc["obligations_profile"]["path"])
    profile.write_text(profile.read_text().replace("  - L3\n", ""))
    doc["obligations_profile"] = P.pin(profile)
    path.write_text(json.dumps(doc))
    with pytest.raises(AssertionError, match="obligations"):
        E.read_selection(path)


@pytest.mark.parametrize("name", ["prov.quantization", "prov.unknown_effect"])
def test_semantic_identity_refuses_unknown_provenance(tmp_path, name):
    weights = tmp_path / "weights.bin"
    weights.write_bytes(b"checked bytes")
    source = f'builtin.module attributes {{prov.weights_file = "weights.bin", {name} = "unknown"}} {{}}'
    with pytest.raises(ValueError, match="unknown provenance"):
        R.semantic_identity(source, weights_path=weights)


def test_only_inert_attribution_and_bound_weight_path_normalize(tmp_path):
    a, b = tmp_path / "public.bin", tmp_path / "capture.bin"
    a.write_bytes(b"same checked weights")
    b.write_bytes(a.read_bytes())

    def source(file, trace="a", family="contraction"):
        return (
            f'builtin.module attributes {{prov.weights_file = "{file}", '
            f'prov.region_id = "{trace}", prov.family = "{family}"}} {{}}'
        )

    expected = R.semantic_identity(source(a.name), weights_path=a)
    assert R.semantic_identity(source(str(b), trace="b"), weights_path=b) == expected
    assert R.semantic_identity(source(b.name, family="changed"), weights_path=b) != expected
    b.write_bytes(b"different checked weights")
    assert R.semantic_identity(source(b.name), weights_path=b) != expected
