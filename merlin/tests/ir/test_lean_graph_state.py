"""Actual shared scheduling output, composed state proofs and valid-IR refusals."""

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
from merlin.verify import lean_graph_state as checker
from merlin.verify import lean_graph_state_replay as replay

REFERENCE = Path("/home/jeremy/Projects/merlin-graph-proof-intake")


def source(width=8, tail=False):
    extra = f"%tail=arith.addi %work,%result:i{width}" if tail else ""
    returned = "%tail" if tail else "%result"
    return f"""builtin.module {{
      func.func @kernel(%a:i{width},%b:i{width})->i{width} {{
        %first=arith.addi %a,%b:i{width}
        "state.config"(%a):(i{width})->()
        %work=arith.muli %a,%b:i{width}
        %seen="state.read"():()->i{width}
        %result=arith.addi %first,%seen:i{width}
        {extra}
        func.return {returned}:i{width}
      }}
      func.func @untouched(%arg:i{width})->i{width} {{func.return %arg:i{width}}}
    }}"""


def operation(name, inputs=(), outputs=(), reads=(), writes=(), observable=False):
    return {
        "name": name,
        "operand_types": list(inputs),
        "result_types": list(outputs),
        "reads": list(reads),
        "writes": list(writes),
        "semantic_sha256": hashlib.sha256(b"explicit original fixture total sequential frame model").hexdigest(),
        "deterministic_total": True,
        "complete_read_write_frames": True,
        "sequential_atomic_complete": True,
        "observable": observable,
    }


def contract(width=8):
    return {
        "operations": [
            operation("state.config", [f"i{width}"], writes=["configuration"]),
            operation("state.read", outputs=[f"i{width}"], reads=["configuration"]),
        ]
    }


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")
    return path


def actual(root, width=8, tail=False, order=None):
    before = root / "source.mlir"
    before.write_text(source(width, tail))
    original_contract = write(root / "contract.json", contract(width))
    order = order or ([0, 2, 1, 3, 4, 5, 6] if tail else [0, 2, 1, 3, 4, 5])
    capture = root / "actual"
    command = [
        sys.executable,
        str(repo_root() / "src/merlin/verify/lean_graph_state_capture.py"),
        "--reference",
        str(REFERENCE),
        "--source",
        str(before),
        "--destination",
        str(capture),
        "--contract",
        str(original_contract),
        "--order",
        *map(str, order),
    ]
    p = subprocess.run(
        command,
        cwd=REFERENCE,
        env={**os.environ, "PYTHONPATH": str(REFERENCE / "src"), "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        check=True,
    )
    (root / "producer.stdout").write_text(p.stdout)
    data = json.loads((capture / "capture.json").read_text())
    assert data["projection_checked"]
    return before, capture / "after.mlir", original_contract, data


def checked(root, before, after, original_contract, **kwargs):
    return checker.check_boundary(
        before, after, root / "checked", reference_root=REFERENCE, contract=original_contract, **kwargs
    )


@pytest.mark.parametrize("width,tail", [(8, False), (16, False), (32, True)])
def test_actual_shared_pass_composed_kernel_proof(width, tail, tmp_path):
    before, after, original_contract, capture = actual(tmp_path, width, tail)
    result = checked(
        tmp_path,
        before,
        after,
        original_contract,
        original_contract_sha256=capture["contract_sha256"],
        expected_before_sha256=checker.file_ref(before)["sha256"],
    )
    assert result["status"] == "graph_model_checked"
    assert result["composed_kernel_theorem_checked"] and result["actual_exchanges"] == 1
    assert result["actual_operations"] == (7 if tail else 6)
    assert not result["public_acceptance_issued"] and not result["source_contract_approved"]
    assert result["numerical_gates_discharged"] is False
    assert "unavailable" in result["full_LeanMLIR_route"]
    assert result["transitive_audit"]["status"] == "checked"
    assert not result["transitive_audit"]["raw"]["violations"]


def test_actual_two_exchanges_composed(tmp_path):
    before, after, original_contract, _ = actual(tmp_path, order=[2, 0, 1, 3, 4, 5])
    result = checked(tmp_path, before, after, original_contract)
    assert result["status"] == "graph_model_checked" and result["actual_exchanges"] == 2


@pytest.mark.parametrize("kind", ["operand", "binding", "untouched", "dependent"])
def test_valid_changed_ir_refused(kind, tmp_path):
    before, after, original_contract, _ = actual(tmp_path)
    target = after.read_text()
    if kind == "operand":
        target = target.replace('"state.config"(%a)', '"state.config"(%b)')
    elif kind == "binding":
        target = target.replace('"arith.addi"(%first, %seen)', '"arith.addi"(%work, %seen)')
    elif kind == "untouched":
        target = target.replace('sym_name = "untouched"', 'sym_name = "changed"')
    else:
        target = (
            source()
            .replace('"state.config"(%a):(i8)->()', "")
            .replace('%seen="state.read"():()->i8', '%seen="state.read"():()->i8\n"state.config"(%a):(i8)->()')
        )
    after = tmp_path / "mutation.mlir"
    after.write_text(target)
    result = checked(tmp_path, before, after, original_contract)
    assert result["status"] == "unsupported"
    assert not result.get("composed_kernel_theorem_checked", False)


@pytest.mark.parametrize("kind", ["missing-op", "partial", "incomplete", "non-atomic"])
def test_missing_semantic_admission_refused(kind, tmp_path):
    before, after, original_contract, _ = actual(tmp_path)
    data = contract()
    if kind == "missing-op":
        data["operations"].pop(0)
    else:
        field = {
            "partial": "deterministic_total",
            "incomplete": "complete_read_write_frames",
            "non-atomic": "sequential_atomic_complete",
        }[kind]
        data["operations"][0][field] = False
    write(original_contract, data)
    assert checked(tmp_path, before, after, original_contract)["status"] == "unsupported"


def test_rebound_contract_cannot_select_original_identity(tmp_path):
    before, after, original_contract, original = actual(tmp_path)
    data = contract()
    data["operations"][0]["writes"] = []
    write(original_contract, data)
    with pytest.raises(ValueError, match="contract differs"):
        checked(tmp_path, before, after, original_contract, original_contract_sha256=original["contract_sha256"])


def test_stale_before_identity_refused(tmp_path):
    before, after, original_contract, _ = actual(tmp_path)
    with pytest.raises(ValueError, match="actual source differs"):
        checked(tmp_path, before, after, original_contract, expected_before_sha256="b" * 64)


def test_caller_alias_refuses_actual_valid_store_exchange(tmp_path):
    original = """module {func.func @kernel(%a:memref<7xi8>,%b:memref<7xi8>,%x:i8,%y:i8) {
      %zero=arith.constant 0:index
      memref.store %x,%a[%zero]:memref<7xi8>
      memref.store %y,%b[%zero]:memref<7xi8>
      func.return
    }}"""
    target = (
        original.replace("memref.store %x,%a[%zero]:memref<7xi8>", "REPLACE")
        .replace("memref.store %y,%b[%zero]:memref<7xi8>", "memref.store %x,%a[%zero]:memref<7xi8>")
        .replace("REPLACE", "memref.store %y,%b[%zero]:memref<7xi8>")
    )
    before, after = tmp_path / "before.mlir", tmp_path / "after.mlir"
    before.write_text(original)
    after.write_text(target)
    selected = write(
        tmp_path / "contract.json", {"operations": [], "standard_memory_sequential": True, "index_bits": 64}
    )
    result = checked(tmp_path, before, after, selected)
    assert result["status"] == "unsupported" and "dependenc" in result["reason"]["detail"]


@pytest.mark.parametrize("kind", ["clobber", "distinct-names", "observer", "admitted-disjoint"])
def test_complete_effect_domain_not_resource_names(kind, tmp_path):
    first = '"state.first"():()->()'
    second = '"state.second"():()->()'
    before, after = tmp_path / "before.mlir", tmp_path / "after.mlir"
    before.write_text(f"module {{func.func @kernel() {{{first}\n{second}\nfunc.return}}}}")
    after.write_text(f"module {{func.func @kernel() {{{second}\n{first}\nfunc.return}}}}")
    data = {
        "operations": [
            operation("state.first", writes=["first"], observable=kind == "observer"),
            operation(
                "state.second", writes=["first" if kind == "clobber" else "second"], observable=kind == "observer"
            ),
        ],
        "disjoint_resources": [] if kind in ("clobber", "distinct-names") else [["first", "second"]],
    }
    result = checked(tmp_path, before, after, write(tmp_path / "contract.json", data))
    assert result["status"] == ("graph_model_checked" if kind == "admitted-disjoint" else "unsupported")
    # Disjointness is an original semantic premise; observer ordering still conflicts.
    assert not result["source_contract_approved"]


def test_hidden_observer_cannot_inherit_original_contract(tmp_path):
    before, after, original_contract, original = actual(tmp_path)
    data = contract()
    data["operations"][0]["observable"] = True
    write(original_contract, data)
    with pytest.raises(ValueError, match="contract differs"):
        checked(tmp_path, before, after, original_contract, original_contract_sha256=original["contract_sha256"])


@pytest.mark.parametrize("kind", ["unknown", "call", "region"])
def test_unknown_effect_call_or_nested_region_is_refused(kind, tmp_path):
    if kind == "call":
        declaration = "func.func private @foreign()"
        command = "func.call @foreign():()->()"
    elif kind == "region":
        declaration = ""
        command = '%yes=arith.constant true\nscf.if %yes {"state.hidden"():()->()\nscf.yield}'
    else:
        declaration, command = "", '"state.hidden"():()->()'
    before, after = tmp_path / "before.mlir", tmp_path / "after.mlir"
    text = f"module {{{declaration}\nfunc.func @kernel() {{{command}\nfunc.return}}}}"
    before.write_text(text)
    after.write_text(text)
    result = checked(tmp_path, before, after, write(tmp_path / "contract.json", {"operations": []}))
    assert result["status"] == "unsupported"


@pytest.mark.parametrize("change", ["position", "first", "missing-exchange"])
def test_kernel_rejects_untrusted_exchange_hint(change, tmp_path):
    _, _, _, captured = actual(tmp_path)
    projection = copy.deepcopy(captured["projection"])
    if change == "missing-exchange":
        projection["exchanges"] = []
    else:
        projection["exchanges"][0][change] += 1
    query = tmp_path / "BadCertificate.lean"
    query.write_text(checker.emit_query(projection))
    frozen = checker.prepare()
    p = subprocess.run(
        [str(checker.LEAN), str(query)],
        cwd=tmp_path,
        env={**os.environ, "LEAN_PATH": str(frozen["lib"])},
        capture_output=True,
        text=True,
    )
    (tmp_path / "rejection.stdout").write_text(p.stdout)
    (tmp_path / "rejection.stderr").write_text(p.stderr)
    assert p.returncode != 0 and "decide" in p.stdout


@pytest.mark.parametrize("change", ["empty", "boolean-index", "missing-cell", "duplicate-op"])
def test_incomplete_projection_refused(change, tmp_path):
    _, _, _, captured = actual(tmp_path)
    projection = copy.deepcopy(captured["projection"])
    if change == "empty":
        projection["nodes"] = []
    elif change == "boolean-index":
        projection["nodes"][0]["operation_index"] = False
    elif change == "missing-cell":
        projection["cells"] = []
    else:
        projection["order"][-1] = projection["order"][0]
    with pytest.raises(ValueError):
        checker.emit_query(projection)


def test_fresh_replay_and_stale_dependency_refusal(tmp_path):
    before, after, original_contract, _ = actual(tmp_path)
    result = checked(tmp_path, before, after, original_contract)
    assert result["status"] == "graph_model_checked"
    report = tmp_path / "checked/report.json"
    measured = replay.replay(report, tmp_path / "replayed", reference_root=REFERENCE)
    assert measured["same_claim"]
    changed = json.loads(report.read_text())
    changed["source_closure"][0]["original"]["sha256"] = "0" * 64
    selected = write(tmp_path / "stale-code-report.json", changed)
    with pytest.raises(ValueError, match="identity is stale"):
        replay.replay(selected, tmp_path / "stale-refused", reference_root=REFERENCE)


def test_wrong_selected_lean_identity_is_unavailable(monkeypatch):
    checker.prepare.cache_clear()
    with monkeypatch.context() as context:
        context.setattr(checker, "LEAN_SHA256", "0" * 64)
        with pytest.raises(ValueError, match="Lean bytes differ"):
            checker.prepare()
    checker.prepare.cache_clear()
