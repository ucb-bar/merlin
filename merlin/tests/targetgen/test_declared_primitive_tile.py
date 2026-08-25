"""A backend may declare the ONE tile it implements; the runtime then drives the loop nest.

The residency tiler already existed and already worked -- it just asked the wrong question. It was keyed
on "how much fits in the operand store", which needs a CLASSIFIED store, and on a device whose SRAMs
cannot be classified there is nothing to compute. Measured: ``_operand_store_capacity_elems`` returns
262144 for one target and ``None`` for the other, so on the second the loop nest at the bottom of
``run_matmul_on_mesh`` was not merely disabled but structurally unreachable, and every multi-tile layer
went whole to a backend that implements a single tile.

The declared tile is the question a backend can always answer: not "how much fits" but "what did you
build". The guardrail that makes this honest rather than flattering is that a pass produced this way is
attributed to the RUNTIME, and the score reports the backend's own coverage beside it.
"""
from __future__ import annotations

import yaml

from merlin.compile_cli import declared_primitive_tile
from merlin.targetgen import capsule_grade as CG
from merlin.targetgen.contract import schemas


def _pkg(tmp_path, **manifest):
    d = tmp_path / "pkg"
    d.mkdir(exist_ok=True)
    (d / "manifest.yaml").write_text(yaml.safe_dump(manifest))
    return d


# --------------------------------------------------------------- the declaration

def test_a_declared_tile_is_read_back(tmp_path):
    p = _pkg(tmp_path, primitive_tile={"m": 32, "k": 32, "n": 32, "dtype": "fp8_e4m3"})
    assert declared_primitive_tile(p) == (32, 32, 32)


def test_no_declaration_keeps_the_previous_behaviour(tmp_path):
    """Declaring nothing is legal: a backend that emits its own loop nest should NOT declare a tile."""
    assert declared_primitive_tile(_pkg(tmp_path, target="t")) is None
    assert declared_primitive_tile(None) is None
    assert declared_primitive_tile(tmp_path / "missing") is None


def test_a_malformed_declaration_is_not_a_declaration(tmp_path):
    """Fail closed toward the old behaviour rather than inventing a tile from a partial one."""
    assert declared_primitive_tile(_pkg(tmp_path, primitive_tile={"m": 32, "k": 32})) is None
    assert declared_primitive_tile(_pkg(tmp_path, primitive_tile={"m": 0, "k": 1, "n": 1})) is None


def test_the_manifest_schema_accepts_it():
    cmds = {k: {"argv": ["{tool}", "{input_mlir}"]}
            for k in ("parse", "lower_interface_to_target", "emit_command_buffer",
                      "lower_target_to_llvm")}
    schemas.validate({"artifact_type": "mlir_oot_target_backend", "target": "t", "language": "python",
                      "authoring": {"mode": "agent_generated_from_rtl_facts"}, "integrity_exempt": False,
                      "entrypoints": {"tool": "t"}, "commands": cmds,
                      "primitive_tile": {"m": 32, "k": 32, "n": 32}}, "manifest")


# --------------------------------------------------------------- the guardrail

def _score(monkeypatch, results):
    monkeypatch.setattr(CG, "load_package", lambda *a, **k: type("P", (), {"integrity_exempt": False})())
    monkeypatch.setattr(CG, "integrity_scan", lambda *a, **k: None)
    monkeypatch.setattr(CG, "build_package", lambda *a, **k: None)
    monkeypatch.setattr(CG, "source_experiment_env", lambda *a, **k: None)
    monkeypatch.setattr(CG.CR, "discover_capsules", lambda *a, **k: [{"name": r["capsule"]}
                                                                     for r in results])
    monkeypatch.setattr(CG.CR, "run_suite", lambda *a, **k: results)
    return CG.grade("pkg", capsules_root=["root"], runs_root="runs", target="atlas", max_workers=1)


def _cap(name, status="pass", tiled_by=None):
    r = {"capsule": name, "kind": "op", "label": "public", "status": status,
         "tiers": {"L4": {"status": status, "derived_from_rtl": True}},
         "numeric": {"status": status}, "trace_check": {"status": status}}
    if tiled_by:
        r["contract_obligations"] = {"capacity_fit": {"tiled_by": tiled_by,
                                                      "discharged_by": "merlin runtime"}}
    return r


def test_a_runtime_driven_pass_is_reported_separately_from_the_backends_own(monkeypatch):
    """This is the entire reason the runtime is allowed to own the loop: both numbers, always.

    Reporting only ``with_runtime_loop`` is how "our runtime covered for them" becomes "their compiler
    generalizes over shape" in a citation.
    """
    s = _score(monkeypatch, [_cap("A0"), _cap("A1"),
                             _cap("A2", tiled_by="declared_primitive_tile")])
    bc = s["backend_coverage"]
    assert bc["with_runtime_loop"] == "3/3"
    assert bc["unblocked"] == "2/3", "A2 only passed because the runtime drove the loop"
    assert bc["runtime_tiled_capsules"] == ["A2"]
    assert "Quote `unblocked`" in bc["note"]


def test_a_capacity_split_is_not_counted_as_a_declared_tile_loop(monkeypatch):
    """The two tilers answer different questions and the attribution must keep them apart."""
    s = _score(monkeypatch, [_cap("A0"), _cap("A1", tiled_by="capacity")])
    assert "backend_coverage" not in s


def test_no_runtime_tiling_leaves_the_score_shape_unchanged(monkeypatch):
    s = _score(monkeypatch, [_cap("A0")])
    assert "backend_coverage" not in s
    assert s["n_passed"] == 1
