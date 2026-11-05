"""The host/accelerator split of a model capsule comes from its frozen Phase 0 placement."""

from __future__ import annotations

import copy
import hashlib
from types import SimpleNamespace

import pytest
import yaml

from merlin.targetgen import frozen_placement as FP
from merlin.targetgen import native_model_execution as native
from merlin.targetgen import owned_host_model_execution as owned
from merlin.targetgen.owned_host_model_selection import owned_model_required

_HOST = "host"
_PROGRAM = "module { func.func @forward() { return } }\n"


def _decision(operation, placement, count=1):
    return {
        "operation": operation,
        "mlir_operation": "linalg.generic",
        "frontend_op": operation,
        "count": count,
        "placement": placement,
        "status": "admitted",
    }


def _capsule(tmp_path, *, decisions=None, program=_PROGRAM, **screen):
    directory = tmp_path / "SY_mixed_model"
    directory.mkdir()
    (directory / "capsule.interface.mlir").write_text(program)
    decisions = (
        [_decision("aten._int_mm.default", "accelerator", 2), _decision("aten.mul.Tensor", "host", 3)]
        if decisions is None
        else decisions
    )
    rows = {}
    for row in decisions:
        rows[row["placement"]] = rows.get(row["placement"], 0) + 1
    declaration = {
        "name": directory.name,
        "kind": "model",
        "interface_mlir": "capsule.interface.mlir",
        "software_screen": {
            "status": "admitted",
            "decisions": decisions,
            "placements": rows,
            "program_sha256": hashlib.sha256(program.encode()).hexdigest(),
            **screen,
        },
    }
    (directory / "capsule.yaml").write_text(yaml.safe_dump(declaration))
    return declaration, directory


def _group(placement, *names):
    return SimpleNamespace(placement=placement, members=[SimpleNamespace(name=name) for name in names])


def test_frozen_placement_is_read_and_bound_to_the_program_bytes(tmp_path):
    declaration, directory = _capsule(tmp_path)
    frozen = FP.read(declaration, directory)
    assert frozen["accelerator_operations"] == {"linalg.generic": 2}
    assert frozen["host_operations"] == {"linalg.generic": 3}
    assert frozen["program_sha256"] == hashlib.sha256(_PROGRAM.encode()).hexdigest()


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (lambda d: d.pop("software_screen"), "no frozen Phase 0 placement"),
        (lambda d: d["software_screen"].update(status="unknown"), "not admitted"),
        (lambda d: d["software_screen"].update(program_sha256="0" * 64), "other program bytes"),
        (lambda d: d["software_screen"].update(placements={"accelerator": 5, "host": 1}), "disagree"),
        (lambda d: d["software_screen"]["decisions"][0].update(placement=None), "admitted lane"),
        (lambda d: d["software_screen"]["decisions"][0].update(count=0), "no operation or count"),
    ],
)
def test_an_unbound_or_malformed_frozen_placement_is_refused(tmp_path, mutate, reason):
    declaration, directory = _capsule(tmp_path)
    mutate(declaration)
    with pytest.raises(FP.FrozenPlacementRefusal, match=reason):
        FP.read(declaration, directory)


def test_a_host_only_frozen_placement_has_nothing_to_split(tmp_path):
    declaration, directory = _capsule(tmp_path, decisions=[_decision("aten.mul.Tensor", "host")])
    with pytest.raises(FP.FrozenPlacementRefusal, match="no operation on the accelerator"):
        FP.read(declaration, directory)


def test_the_split_must_match_the_frozen_accelerator_operations(tmp_path):
    declaration, directory = _capsule(tmp_path)
    frozen = FP.read(declaration, directory)
    digest = frozen["program_sha256"]
    matched = [_group("mesh", "linalg.generic"), _group("mesh", "linalg.generic"), _group(_HOST, "linalg.generic")]
    record = FP.require_split(
        frozen, source_sha256=digest, device=FP.device_operations(matched, _HOST), eligible={"linalg.generic": 2}
    )
    assert record["status"] == "matched" and record["authority"] == "frozen Phase 0 software_screen"
    # Accelerator work left on Merlin's host: one frozen accelerator operation sits in a host group.
    on_host = [_group("mesh", "linalg.generic"), _group(_HOST, "linalg.generic", "linalg.generic")]
    with pytest.raises(FP.FrozenPlacementRefusal, match="may not move to the host"):
        FP.require_split(frozen, source_sha256=digest, device=FP.device_operations(on_host, _HOST))
    # Merlin may absorb supporting work, provided it comes from the original source.
    fused = [_group("mesh", "linalg.generic", "linalg.generic"), _group("mesh", "linalg.generic")]
    with pytest.raises(FP.FrozenPlacementRefusal, match="original source inventory"):
        FP.require_split(frozen, source_sha256=digest, device=FP.device_operations(fused, _HOST))
    record = FP.require_split(
        frozen, source_sha256=digest, device=FP.device_operations(fused, _HOST),
        eligible={"linalg.generic": 2}, source_operations={"linalg.generic": 5},
    )
    assert record["accelerator_operations"] == {"linalg.generic": 2}
    assert record["supporting_device_operations"] == {"linalg.generic": 1}
    with pytest.raises(FP.FrozenPlacementRefusal, match="outside the original source"):
        FP.require_split(
            frozen, source_sha256=digest, device={"linalg.generic": 6},
            source_operations={"linalg.generic": 5},
        )
    with pytest.raises(FP.FrozenPlacementRefusal, match="independent eligibility"):
        FP.require_split(
            frozen, source_sha256=digest, device=FP.device_operations(matched, _HOST), eligible={"linalg.generic": 1}
        )
    with pytest.raises(FP.FrozenPlacementRefusal, match="routed source differs"):
        FP.require_split(frozen, source_sha256="1" * 64, device=FP.device_operations(matched, _HOST))


def test_the_owned_route_reads_the_pinned_declaration_not_the_callers_copy(tmp_path):
    program = 'module { func.func @forward() { "linalg.generic"() : () -> ()\n' \
        '"linalg.generic"() : () -> ()\n return } }\n'
    declaration, directory = _capsule(tmp_path, program=program)
    source = directory / "capsule.interface.mlir"
    groups = [_group("mesh", "linalg.generic"), _group("mesh", "linalg.generic")]
    edited = copy.deepcopy(declaration)
    edited["software_screen"]["decisions"][1]["placement"] = "accelerator"
    record = owned._frozen_split({**edited, "__dir__": str(directory)}, source, target="fixture", groups=groups)
    assert record["accelerator_operations"] == {"linalg.generic": 2}
    source.write_text(_PROGRAM + "// changed\n")
    with pytest.raises(FP.FrozenPlacementRefusal, match="other program bytes"):
        owned._frozen_split({**declaration, "__dir__": str(directory)}, source, target="fixture", groups=groups)


def test_a_model_with_frozen_accelerator_work_always_takes_the_owned_route(tmp_path, monkeypatch):
    declaration, _ = _capsule(tmp_path)
    monkeypatch.setattr(
        native, "independent_frozen_source_eligibility", lambda *a, **k: pytest.fail("fallback census consulted")
    )
    assert FP.requires_merlin_host(declaration)
    assert owned_model_required(declaration, target="fixture")


def test_a_model_without_frozen_placement_keeps_the_eligibility_selection(monkeypatch):
    monkeypatch.setattr(native, "independent_frozen_source_eligibility", lambda *a, **k: {"n_eligible": 0})
    assert not FP.requires_merlin_host({"kind": "model"})
    assert not owned_model_required({"kind": "model"}, target="fixture")
    layer = {"kind": "layer", "software_screen": {"placements": {"accelerator": 1}}}
    assert not FP.requires_merlin_host(layer)
