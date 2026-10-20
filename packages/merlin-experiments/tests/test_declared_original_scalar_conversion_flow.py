"""Explicit converter inputs stay distinct from live registered source authority."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import test_declared_original_reference_flow as REF
from merlin_experiments.phase0 import original_call_sources as C
from merlin_experiments.phase0 import original_scalar_conversion_flow as F
from test_original_integer_scalar_binary_plan import conversion_budget

declared = REF.declared


def inputs(declared, tmp_path):
    _, _, standard = REF._request(declared, tmp_path)
    selected = json.loads(standard.read_bytes())
    selected.pop("execution_budget")
    selected.update(schema=F.SCHEMA, budget=conversion_budget())
    path = tmp_path / "scalar-selection.json"
    path.write_text(json.dumps(selected))
    return F._pin(path), path, selected


def test_selected_inputs_reopen_complete_source_parser_and_public_commit(declared, tmp_path):
    pin, _, _ = inputs(declared, tmp_path)
    selected = F.read_selection(pin, forbidden=())
    selected.verify()
    assert len(selected.source_pins) == 3
    assert len(selected.paths) == 4
    with pytest.raises(ValueError, match="actual live"):
        F.prepare(
            selected,
            schema_intake={},
            basis=None,
            source_record={},
            numerical_semantics={},
            destination=tmp_path / "native",
        )
    assert not (tmp_path / "native").exists()


@pytest.mark.parametrize(
    "change",
    ["schema", "saved_source", "saved_schema", "saved_basis", "factory", "missing_limit", "bool_limit", "timeout"],
)
def test_input_declaration_cannot_import_statuses_or_remove_complete_budgets(declared, tmp_path, change):
    _, path, raw = inputs(declared, tmp_path)
    if change == "schema":
        raw["schema"] = F.V.INTEGER_SELECTION_SCHEMA
    elif change.startswith("saved_"):
        raw[
            {
                "saved_source": "source_record_sha256",
                "saved_schema": "operator_schema_intake_sha256",
                "saved_basis": "semantic_basis_sha256",
            }[change]
        ] = "a" * 64
    elif change == "factory":
        raw["factory"] = "caller_hook"
    elif change == "missing_limit":
        del raw["budget"]["max_total_promotion_tensor_elements"]
    else:
        raw["budget"]["timeout_s"] = True if change == "bool_limit" else 181
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        F.read_selection(F._pin(path), forbidden=())


@pytest.mark.parametrize("change", ["selection_bytes", "parser_bytes", "source_bytes", "source_added", "source_alias"])
def test_each_verification_detects_input_tool_and_complete_source_drift(declared, tmp_path, change):
    pin, path, raw = inputs(declared, tmp_path)
    selected = F.read_selection(pin, forbidden=())
    root = Path(raw["capture_checkout"])
    if change == "selection_bytes":
        path.write_text("{}")
    elif change == "parser_bytes":
        Path(raw["mlir_opt"]["path"]).write_bytes(b"changed")
    elif change == "source_bytes":
        (root / "m2m/__init__.py").write_bytes(b"changed")
    elif change == "source_added":
        # Exact tracked source membership, not only the original entrypoint hash.
        import subprocess

        (root / "m2m/added.py").write_text("# new public source\n")
        subprocess.run(["git", "-C", str(root), "add", "m2m/added.py"], check=True)
    else:
        source = root / "m2m/__init__.py"
        source.unlink()
        source.symlink_to(path)
    with pytest.raises(ValueError):
        selected.verify()


def test_excluded_checkout_refuses_before_upstream_source_reader(declared, tmp_path, monkeypatch):
    pin, _, raw = inputs(declared, tmp_path)
    monkeypatch.setattr(F.P, "capture_sources", lambda value: pytest.fail("opened an excluded checkout"))
    with pytest.raises(ValueError):
        F.read_selection(pin, forbidden=(Path(raw["capture_checkout"]),))


def test_preparation_derives_current_live_identities_and_invokes_existing_v2_converter(declared, tmp_path, monkeypatch):
    pin, _, _ = inputs(declared, tmp_path)
    selected = F.read_selection(pin, forbidden=())
    source = {"schema": C.INTEGER_SCALAR_SCHEMA, "members": []}

    class Schema:
        sha256 = "a" * 64

    class Basis:
        source = SimpleNamespace(sha256="b" * 64)

    monkeypatch.setattr(F.V, "IndependentOperatorSchemaIntake", Schema)
    monkeypatch.setattr(F.V, "ComponentSemanticBasis", Basis)
    calls = []
    owner = object()
    monkeypatch.setattr(F.V, "prepare", lambda **kwargs: calls.append(kwargs) or owner)
    schema, basis, policy = Schema(), Basis(), {"original_policy_pending": True}
    assert (
        F.prepare(
            selected,
            schema_intake=schema,
            basis=basis,
            source_record=source,
            numerical_semantics=policy,
            destination=tmp_path / "fresh",
        )
        is owner
    )
    assert len(calls) == 1 and calls[0]["source_record"] is source
    assert calls[0]["schema_intake"] is schema and calls[0]["basis"] is basis
    assert calls[0]["numerical_semantics"] is policy
    actual = json.loads(calls[0]["selection"].read_bytes())
    assert actual["schema"] == F.V.INTEGER_SELECTION_SCHEMA
    assert actual["source_record_sha256"] == F.V._digest(source)
    assert actual["operator_schema_intake_sha256"] == schema.sha256
    assert actual["semantic_basis_sha256"] == basis.source.sha256
    assert calls[0]["destination"] == tmp_path / "fresh/registered"


def test_new_flow_refuses_v6_source_even_with_selected_v2_converter(declared, tmp_path, monkeypatch):
    pin, _, _ = inputs(declared, tmp_path)
    selected = F.read_selection(pin, forbidden=())

    class Schema:
        sha256 = "a" * 64

    class Basis:
        source = SimpleNamespace(sha256="b" * 64)

    monkeypatch.setattr(F.V, "IndependentOperatorSchemaIntake", Schema)
    monkeypatch.setattr(F.V, "ComponentSemanticBasis", Basis)
    monkeypatch.setattr(F.V, "prepare", lambda **kwargs: pytest.fail("reached native observer"))
    with pytest.raises(ValueError, match="widen"):
        F.prepare(
            selected,
            schema_intake=Schema(),
            basis=Basis(),
            source_record={"schema": C.SCALAR_BINARY_SCHEMA},
            numerical_semantics={},
            destination=tmp_path / "native",
        )
    assert not (tmp_path / "native").exists()
