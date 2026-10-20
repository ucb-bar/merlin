"""Source portability never changes the selected original facts or semantics."""

import copy
import hashlib
import json

import pytest

from merlin.compile import model_execution_inputs as MI
from merlin.runtime.backends import base as backends
from merlin.runtime.backends.chipyard_rocc_bound import BoundRoCCBackend
from merlin.targetgen.rocc import operand_selection_recipe as R
from merlin.targetgen.rocc import selected_operand_protocol as P
from merlin.targetgen.rocc.semantics import BoundSemantics
from merlin.targetgen.rtl.source_relocations import relocate_operand_sources

pytest_plugins = ("test_operand_selection_recipe",)


def pin(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.fixture
def relocated(selection, tmp_path):
    original, blob, recipe, _recipe_data, firrtl, isa = selection
    originals = (pin(isa), pin(firrtl))
    recovered = tmp_path / "recovered"
    recovered.mkdir()
    new_isa, new_firrtl = recovered / isa.name, recovered / firrtl.name
    isa.rename(new_isa)
    firrtl.rename(new_firrtl)
    rows = [
        {"original": old, "replacement": pin(new)} for old, new in zip(originals, (new_isa, new_firrtl), strict=True)
    ]
    return original, blob, recipe, rows, new_isa, new_firrtl


def derive(case):
    original, blob, recipe, rows, *_ = case
    return R.derive(original, original_facts_bytes=blob, recipe_path=recipe, source_relocations=rows)


def test_rederive_preserves_original_bytes_and_reopens_source_roster(relocated, tmp_path):
    original, blob, _recipe, rows, isa, firrtl = relocated
    before = copy.deepcopy(original._facts)
    sidecar = derive(relocated)
    assert sidecar["original_facts_sha256"] == hashlib.sha256(blob).hexdigest()
    assert sidecar["layouts"]["source"] == str(firrtl)
    assert pin(isa) in sidecar["source_pins"]
    assert all(x["status"] == "derived" for x in sidecar["layouts"]["fields"])
    facts_path, sidecar_path = tmp_path / "facts.json", tmp_path / "sidecar.json"
    facts_path.write_bytes(blob)
    sidecar_path.write_text(json.dumps(sidecar))
    selection = {"original_facts": pin(facts_path), "sidecar": pin(sidecar_path)}
    observer = P.bind_selected_observer(original, selection)
    assert observer.decode_instruction(
        4, {}, {"kind": "const", "raw": (17 << 20) | 128}, observer.isa_constants("array")
    ) == ("MVIN", {"rows": 17, "selected": True})
    assert P.selected_source_relocations(original, selection) == rows
    assert str(isa) in P.selected_inputs(selection) and str(firrtl) in P.selected_inputs(selection)
    assert original._facts == before and facts_path.read_bytes() == blob
    assert original.operand_observer is None
    firrtl.write_bytes(firrtl.read_bytes().replace(b"24", b"23", 1))
    with pytest.raises(ValueError, match="bytes changed"):
        P.bind_selected_observer(original, selection)


@pytest.mark.parametrize(
    "defect",
    [
        "duplicate",
        "unmatched",
        "old_digest",
        "new_digest",
        "extra",
        "symlink",
        "ancestor",
        "relative",
        "missing",
        "bytes",
        "identity",
        "budget",
        "replaced",
    ],
)
def test_closed_source_roster_refuses_unproved_relocations(relocated, tmp_path, defect):
    original, _blob, _recipe, rows, isa, _firrtl = relocated
    rows = copy.deepcopy(rows)
    if defect == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif defect == "unmatched":
        rows[0]["original"]["path"] = str(tmp_path / "not-recorded")
    elif defect in {"old_digest", "new_digest"}:
        rows[0]["original" if defect == "old_digest" else "replacement"]["sha256"] = "0" * 64
    elif defect == "extra":
        rows[0]["qualification"] = True
    elif defect in {"symlink", "ancestor"}:
        link = tmp_path / "link"
        link.symlink_to(isa if defect == "symlink" else isa.parent, target_is_directory=defect == "ancestor")
        rows[0]["replacement"]["path"] = str(link if defect == "symlink" else link / isa.name)
    elif defect == "relative":
        rows[0]["replacement"]["path"] = "relative.scala"
    elif defect == "missing":
        isa.unlink()
    elif defect == "bytes":
        isa.write_bytes(b"different source")
    elif defect == "identity":
        rows[0]["replacement"] = copy.deepcopy(rows[0]["original"])
    elif defect == "budget":
        rows = rows * 9
    else:
        isa.unlink()
        isa.symlink_to(tmp_path / "unavailable")
    with pytest.raises(ValueError):
        relocate_operand_sources(original._facts, rows)


@pytest.mark.parametrize("reverse", [False, True])
def test_chained_relocations_cannot_retarget_original_source_ownership(tmp_path, reverse):
    a, b, c = [tmp_path / name for name in ("a", "b", "c")]
    for path in (a, b, c):
        path.write_bytes(b"same source bytes")
    facts = {"inputs": {"isa_path": str(a), "isa_sha256": pin(a)["sha256"], "firrtl_inputs": [pin(b)]}}
    rows = [{"original": pin(a), "replacement": pin(b)}, {"original": pin(b), "replacement": pin(c)}]
    with pytest.raises(ValueError, match="uniquely preserve"):
        relocate_operand_sources(facts, list(reversed(rows)) if reverse else rows)


def test_duplicate_recorded_source_ownership_refuses_even_identical_digest(relocated):
    original, _blob, _recipe, rows, *_ = relocated
    facts = copy.deepcopy(original._facts)
    facts["inputs"]["firrtl_inputs"].append(copy.deepcopy(facts["inputs"]["firrtl_inputs"][0]))
    with pytest.raises(ValueError, match="ambiguous"):
        relocate_operand_sources(facts, rows)


def test_original_facts_mutation_during_source_hash_refuses(relocated, monkeypatch):
    from merlin.targetgen.rtl import source_relocations as S

    original, _blob, _recipe, rows, *_ = relocated
    native = S.hashlib.file_digest

    def mutate(*args):
        result = native(*args)
        original._facts["inputs"]["target"] = "other"
        return result

    monkeypatch.setattr(S.hashlib, "file_digest", mutate)
    with pytest.raises(ValueError, match="original facts or roster changed"):
        relocate_operand_sources(original._facts, rows)


@pytest.mark.parametrize("defect", ["source", "roster", "original"])
def test_recipe_rechecks_sources_and_original_after_observation(relocated, monkeypatch, defect):
    original, _blob, _recipe, rows, isa, _firrtl = relocated
    native = R.derive_operand_layouts

    def mutate(*args, **kwargs):
        result = native(*args, **kwargs)
        if defect == "source":
            isa.write_bytes(b"changed after observation")
        elif defect == "roster":
            rows.clear()
        else:
            original._facts["inputs"]["target"] = "other"
        return result

    monkeypatch.setattr(R, "derive_operand_layouts", mutate)
    with pytest.raises(ValueError):
        derive(relocated)


def test_binder_rechecks_source_after_observer_construction(relocated, monkeypatch):
    sidecar = derive(relocated)
    original, blob, _recipe, _rows, isa, _firrtl = relocated
    native = P.BoundOperandObserver

    def mutate(**kwargs):
        observer = native(**kwargs)
        isa.write_bytes(b"changed during observer construction")
        return observer

    monkeypatch.setattr(P, "BoundOperandObserver", mutate)
    raw = json.dumps(sidecar).encode()
    with pytest.raises(ValueError, match="changed"):
        P.bind_original_operand_selection(
            original,
            original_facts_bytes=blob,
            selection_bytes=raw,
            expected_selection_sha256=hashlib.sha256(raw).hexdigest(),
        )


def test_selected_execution_uses_only_bound_observer_relocations(relocated, tmp_path, monkeypatch):
    _original, blob, recipe, rows, _isa, firrtl = relocated
    facts = json.loads(blob)
    facts["facts"]["source"] = {"config": "selected_config"}
    facts["inputs"]["fir_sha256"] = pin(firrtl)["sha256"]
    contract = {
        "name": "array",
        "rtl_checks": ["decode_clean", "legal_funct"],
        "rocc_operand_roles": {
            "version": 1,
            "word_bits": 64,
            "instructions": [
                {"funct": 0, "class": "CONFIG_CMD", "operands": {}},
                {"funct": 4, "class": "INPUT", "operands": {}},
            ],
        },
        "runner": {"chipyard_rocc": {"config": "selected_config"}},
    }
    raw = json.dumps(facts).encode()
    original = BoundSemantics(target="array", contract=contract, facts=facts)
    sidecar = R.derive(original, original_facts_bytes=raw, recipe_path=recipe, source_relocations=rows)
    facts_path, sidecar_path = tmp_path / "facts.json", tmp_path / "sidecar.json"
    facts_path.write_bytes(raw)
    sidecar_path.write_text(json.dumps(sidecar))
    contract["rocc_operand_selection"] = {"original_facts": pin(facts_path), "sidecar": pin(sidecar_path)}
    bound = BoundRoCCBackend("array", json.dumps({"contract": contract, "facts": facts}), original)
    monkeypatch.setattr(backends, "get_backend", lambda target: bound)
    selected = MI.selected_execution_firrtl(facts_path, target="array")
    assert selected["firrtl"] == str(firrtl) and selected["sha256"] == pin(facts_path)["sha256"]
    assert facts_path.read_bytes() == raw
    with pytest.raises(MI.ModelExecutionInputError, match="byte-verified"):
        MI.selected_firrtl(facts_path, target="array", config="selected_config")
    sidecar_path.write_text(json.dumps({**sidecar, "source_relocations": []}))
    with pytest.raises(ValueError, match="byte pin"):
        MI.selected_execution_firrtl(facts_path, target="array")


def test_firrtl_selection_reopens_original_facts_after_hashing(relocated, tmp_path, monkeypatch):
    _original, blob, _recipe, rows, _isa, firrtl = relocated
    facts = json.loads(blob)
    facts["facts"]["source"] = {"config": "selected_config"}
    facts["inputs"]["fir_sha256"] = pin(firrtl)["sha256"]
    facts_path = tmp_path / "facts.json"
    facts_path.write_text(json.dumps(facts))
    native = MI.file_sha256

    def mutate(path):
        result = native(path)
        if path == firrtl:
            facts["facts"]["source"]["config"] = "other"
            facts_path.write_text(json.dumps(facts))
        return result

    monkeypatch.setattr(MI, "file_sha256", mutate)
    with pytest.raises(MI.ModelExecutionInputError, match="changed before publication"):
        MI.selected_firrtl(facts_path, target="array", config="selected_config", source_relocations=rows)
