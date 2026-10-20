"""Sealed model arithmetic selection is forwarded without supplier waivers."""

import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pytest
import yaml
from merlin_experiments.phase1.feedback import private_model_math_policy as P

from merlin.model_check import models as M
from merlin.runtime import linalg_numpy as LN
from merlin.runtime.host_math import host_math_recipe


def _fixture(tmp_path, *, policy="portable_exp_trig", own=None, receipt_policy=None, receipt=True):
    root = tmp_path / "bundle"
    root.mkdir(parents=True)
    config = {
        "schema": M.MODELS_SCHEMA,
        "target": "owned",
        "full": {"host_math_policy": policy},
        "models": {
            "toy": {
                "workload": "toy",
                "chain": {"kind": "single"},
                "programs": [{"name": "model"}],
                "judged": {"kind": "logits", "outputs": [{"program": "model", "output": 0}]},
            }
        },
    }
    if own is not None:
        config["models"]["toy"]["full"] = {"host_math_policy": own}
    path = root / "configuration/models.yaml"
    path.parent.mkdir()
    path.write_text(yaml.safe_dump(config))
    output = root / "toy/visible/v00/int8.npz"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"opaque output fixture: policy selection does not decode arrays")
    files = [path, output]
    if receipt:
        chosen = receipt_policy or own or policy
        math = host_math_recipe(chosen)
        contract = {"version": LN.NUMERIC_CONTRACT_VERSION}
        if chosen != "native":
            contract["host_math"] = {
                "policy": chosen,
                "scalar_source_sha256": sha256(math.source.encode()).hexdigest(),
                "compile_flags": list(math.compile_flags),
            }
        sidecar = output.with_name("reference_numeric_contract.json")
        sidecar.write_text(
            json.dumps(
                {
                    "schema": "merlin.model_check.reference_numeric_contract.v1",
                    "outputs_sha256": sha256(output.read_bytes()).hexdigest(),
                    "numeric_contract": contract,
                    "evaluator_sha256": "e" * 64,
                }
            )
        )
        files.append(sidecar)
    sums = root / "SHA256SUMS"
    sums.write_text("".join(f"{sha256(p.read_bytes()).hexdigest()}  {p.relative_to(root)}\n" for p in files))
    manifest = {
        "schema": "merlin.model_check.bundle.v1",
        "target": "owned",
        "models": {"toy": {}},
        "seal": {"sums": "SHA256SUMS", "sums_sha256": sha256(sums.read_bytes()).hexdigest()},
    }
    (root / "MANIFEST.json").write_text(json.dumps(manifest))
    doc = {"numerics_bundle": {"path": str(root)}}
    spec = tmp_path / "private.yaml"
    spec.write_text(yaml.safe_dump(doc))
    row = {"id": "toy", "capture": str(tmp_path / "capture")}
    return doc, spec, row, root


def _select(fixture):
    d, s, r, _ = fixture
    return P.select(d, s, r, target="owned", programs=("model",))


@pytest.mark.parametrize(
    ("common", "own", "selected"),
    [
        ("native", None, "native"),
        ("portable_exp_trig", None, "portable_exp_trig"),
        ("portable_exp_trig", "portable_neural_f32", "portable_neural_f32"),
    ],
)
def test_selected_policy_forwards_to_actual_compile_api(tmp_path, monkeypatch, common, own, selected):
    selection = _select(_fixture(tmp_path, policy=common, own=own))
    from merlin.compile import baremetal_model

    seen = {}

    def compile(**kwargs):
        seen.update(kwargs)
        return {"inputs": {"host_math_policy": kwargs["host_math_policy"]}}

    monkeypatch.setattr(baremetal_model, "compile_saved_model", compile)
    baremetal_model.compile_saved_model(**P.build_options(selection, {"readback": "full"}, ()))
    assert seen == {"host_math_policy": selected, "readback": "full"}
    assert selection["reference_receipts_verified"] == 1


def test_native_legacy_and_unreceipted_native_do_not_claim_reference_proofs(tmp_path):
    assert P.select({}, tmp_path / "absent", {}, target="owned", programs=())["policy"] == "native"
    selected = _select(_fixture(tmp_path, policy="native", receipt=False))
    assert selected["reference_receipts_verified"] == 0


@pytest.mark.parametrize("kind", ["policy", "missing", "config", "unknown", "program"])
def test_selected_policy_or_seal_mismatch_refuses(tmp_path, kind):
    fixture = _fixture(
        tmp_path,
        policy="native" if kind == "policy" else "portable_exp_trig",
        receipt_policy="portable_exp_trig" if kind == "policy" else None,
        receipt=kind != "missing",
    )
    d, s, r, root = fixture
    if kind == "config":
        (root / "configuration/models.yaml").write_text("changed")
    if kind == "unknown":
        fixture = _fixture(tmp_path / "other", policy="bogus", receipt=False)
    with pytest.raises((ValueError, M.ModelsFileError)):
        if kind == "program":
            P.select(d, s, r, target="owned", programs=("different",))
        else:
            _select(fixture)


def test_reference_and_recipe_mutations_refuse_before_build(tmp_path, monkeypatch):
    selected = _select(_fixture(tmp_path))
    original = P.host_math_recipe
    monkeypatch.setattr(P, "host_math_recipe", lambda policy: replace(original(policy), source="changed"))
    with pytest.raises(ValueError, match="scalar source/flags changed"):
        P.build_options(selected, {}, ())
    monkeypatch.setattr(P, "host_math_recipe", original)
    path = Path(selected["pins"][-1]["path"])
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="bytes changed"):
        P.build_options(selected, {}, ())


def test_exact_portable_archive_conflict_and_f64_preservation(tmp_path):
    selected = _select(_fixture(tmp_path, policy="portable_neural_f32"))
    assert P.build_options(selected, {}, ("sin", "cos", "pow"))["host_math_policy"] == "portable_neural_f32"
    with pytest.raises(ValueError, match="matching portable-provider contract"):
        P.build_options(selected, {}, ("sinf",))
    with pytest.raises(ValueError, match="conflicts with selected"):
        P.build_options(selected, {"host_math_policy": "native"}, ())


def test_snapshot_loader_does_not_read_changed_original_path(tmp_path):
    _, _, _, root = _fixture(tmp_path)
    path = root / "configuration/models.yaml"
    snapshot = path.read_bytes()
    path.write_text("broken")
    assert M.load(path, source_bytes=snapshot).models["toy"].name == "toy"


def test_real_source_preflight_refuses_portable_archive_conflict_before_candidate(tmp_path, monkeypatch):
    import test_private_source_preflight as fixtures
    from merlin_experiments.phase1.feedback import private_full_models as gate
    from merlin_experiments.phase1.feedback import private_linkage_support as linkage

    from merlin.targetgen.host_linkage_contract import SCHEMA, SUPPLIER

    options = fixtures._arguments(tmp_path, monkeypatch)
    selected = {"policy": "portable_trig", "pins": [], "recipe": P.recipe_identity("portable_trig")}
    monkeypatch.setattr(P, "select", lambda *a, **k: selected)
    source = {
        "eligible_groups": [0],
        "accelerator_source_demand": {},
        linkage.FIELD: {
            "status": linkage.PENDING,
            "count": 1,
            "occurrences": [
                {
                    "schema": "merlin.static_f32_math_source_body.v1",
                    "operation": "math.sin",
                    "contract": {
                        "schema": SCHEMA,
                        "supplier": SUPPLIER,
                        "archive_sha256": "a" * 64,
                        "symbols": ["sinf"],
                    },
                }
            ],
        },
    }
    monkeypatch.setattr(gate, "_source_obligations", lambda *a, **k: source)
    report = gate.run(None, **options, source_only=True)
    assert report["source_preflight_passed"] is False
    assert all("matching portable-provider contract" in row["reason"] for row in report["models"])
    assert all(row["checks"]["host_math_policy"]["policy"] == "portable_trig" for row in report["models"])
