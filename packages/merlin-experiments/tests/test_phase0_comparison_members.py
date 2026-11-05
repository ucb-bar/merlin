"""A comparison group needs every member: a definitely refused member of any role withdraws the family."""

from __future__ import annotations

from types import SimpleNamespace

from merlin_experiments.phase0 import comparison_screen as CSCREEN
from merlin_experiments.phase0 import program_admission as PA

from merlin.targetgen import corpus_spec as CS


def _entries():
    return [
        {"name": "PB00_island", "op": "matmul", "comparison_group": {"name": "g", "role": "island"}},
        {"name": "PB01_no_island", "op": "matmul", "comparison_group": {"name": "g", "role": "no_island"}},
    ]


def test_a_refused_member_of_any_role_withdraws_the_comparison(monkeypatch):
    monkeypatch.setattr(CS, "entry_binding", lambda entry, binding: (None, binding))
    monkeypatch.setattr(CS, "build", lambda entry, binding: ({"semantic": {}}, f"// {entry['name']}"))
    monkeypatch.setattr(PA, "account_interface_text", lambda mlir, **_: [{"program": mlir}])

    def summarize(observed, **_):
        refused = "island" in observed[0]["program"] and "no_island" not in observed[0]["program"]
        return {
            "status": "unsupported" if refused else "unknown",
            "reason": "requant: operand_dtype 'int32' not in ['int8']" if refused else "",
            "decisions": [],
        }

    monkeypatch.setattr(PA, "summarize", summarize)
    binding = SimpleNamespace(target="fixture")
    refusal = CSCREEN.refused_emitted_part(_entries(), binding=binding, evidence=object())
    assert refusal is not None and refusal["name"] == "PB00_island" and "requant" in refusal["reason"]
    # Members that are only unresolved are left to the written program's own screen.
    monkeypatch.setattr(PA, "summarize", lambda observed, **_: {"status": "unknown", "reason": "", "decisions": []})
    assert CSCREEN.refused_emitted_part(_entries(), binding=binding, evidence=object()) is None
