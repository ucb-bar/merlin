from __future__ import annotations

from merlin.common.paths import repo_root
from merlin.targetgen import capability_backing as CB

_DOC = """name: {name}
compute_units:
  - name: unit0
    dtypes: [int8, int16]
    ops: [matmul]
    semantic_capabilities:
{comment}      - {entry}
"""


def _doc(entry: str, comment_lines: tuple[str, ...] = (), *, name: str = "t") -> str:
    comment = "".join(f"      # {line}\n" for line in comment_lines)
    return _DOC.format(name=name, comment=comment, entry=entry)


def _gate_script():
    import importlib.util
    import sys

    path = repo_root() / "build_tools" / "checks" / "check_capability_backing.py"
    spec = importlib.util.spec_from_file_location("check_capability_backing", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_the_gate_fails_on_a_reintroduced_unbacked_narrowing(tmp_path, monkeypatch) -> None:
    """THE GATE'S OWN FAILING DIRECTION, driven through its ``main``.

    One synthetic target whose only contract document carries the defect line with prose and no
    citation. Nothing else can satisfy the gate: the ledger is ignored, the target declares no pin, and
    the comment names no file, test or module -- so a pass could only mean the gate stopped reading
    claims. The same document with an in-repo citation passes, so a gate that failed unconditionally
    would not satisfy this test either.
    """
    gate = _gate_script()
    contract = tmp_path / "contract.yaml"
    monkeypatch.setattr(CB, "gated_targets", lambda: ("t",))
    monkeypatch.setattr(CB, "contract_documents", lambda target: ((contract, contract.read_text()),))
    monkeypatch.setattr(CB, "_pin_read_paths", lambda target: {})
    entry = "{family: elementwise_map, dtypes: [int8, int16], composed_with: [contraction]}"
    contract.write_text(_doc(entry, ("An elementwise map exists ONLY fused behind a contraction.",)))
    assert gate.main(["--no-ratchet"]) == 1
    contract.write_text(_doc(entry, ("Derived by merlin.targetgen.eligibility.is_eligible.",)))
    assert gate.main(["--no-ratchet"]) == 0
